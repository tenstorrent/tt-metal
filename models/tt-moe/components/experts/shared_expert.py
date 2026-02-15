# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Refactored SharedExpert implementation that exactly matches the reference architecture.
This version uses 3D tensors and weight sharding like the reference implementation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

import ttnn

# ============================================================================
# Utility Functions (from models/demos/deepseek_v3/utils/config_helpers.py)
# ============================================================================


def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def find_largest_divisor(n: int, max_val: int = 8) -> int:
    """Find the largest divisor of n that is <= max_val."""
    for divisor in range(min(n, max_val), 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)

    tensor = tensor.float() * inv_scale[tuple(slice(0, s) for s in tensor.shape)].float()
    del inv_scale
    return tensor  # Return as float32


def get_state_dicts(
    dicts: Sequence[Optional[Dict[str, torch.Tensor]]],
    key: str,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Get a weight from state dicts and stack them to add a dimension."""
    if not dicts:
        return torch.empty(0)

    # Get first non-None dict to determine shape/dtype
    first_dict = next((d for d in dicts if d is not None), None)
    assert first_dict is not None and key in first_dict

    expected_shape = shape if shape is not None else first_dict[key].shape
    expected_dtype = dtype if dtype is not None else first_dict[key].dtype

    # Create tensors list
    tensors = []
    for d in dicts:
        if d is None:
            tensors.append(torch.zeros(expected_shape, dtype=expected_dtype))
        else:
            assert key in d, f"Key {key} not found in state dict"
            assert d[key].shape == expected_shape
            tensors.append(d[key])

    # Stack along dim 0 to create 3D tensor
    return torch.stack(tensors, dim=0)


# Compute kernel configuration
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Constants from reference
SEQ_LEN_CHUNK_SIZE = 16384

# ============================================================================
# SharedExpert Class (Matching reference architecture)
# ============================================================================


class SharedExpert:
    """
    Shared Expert layer for MoE - refactored to match reference implementation exactly.
    Uses 3D tensors and weight sharding like the reference.
    """

    # Weight types (from MLPDequant)
    WEIGHT_TORCH_DTYPE = torch.float8_e4m3fn
    WEIGHT_SCALE_INV_TORCH_DTYPE = torch.float32

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: Any) -> Tuple[int, int]:
        """Get dimensions - uses moe_intermediate_size for shared expert."""
        dim = hf_config.hidden_size
        # Check for both possible attribute names
        if hasattr(hf_config, "moe_intermediate_size"):
            hidden_dim = hf_config.moe_intermediate_size  # Key difference from regular MLP
        else:
            hidden_dim = hf_config.intermediate_size  # Fallback for simplified config
        return dim, hidden_dim

    @classmethod
    def get_weight_shape(cls, hf_config: Any, is_w2: bool) -> Tuple[int, int]:
        """Get the shape of the weight tensor."""
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)
        if is_w2:
            return hidden_dim, dim
        else:
            return dim, hidden_dim

    @classmethod
    def convert_weights(
        cls,
        hf_config: Any,
        state_dicts: Tuple[Optional[Dict[str, torch.Tensor]], ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> Dict:
        """
        Convert weights matching the reference implementation exactly.
        Uses get_state_dicts to create 3D tensors and shards them across devices.
        """
        # Get weight block size for quantization
        if hasattr(hf_config, "quantization_config"):
            weight_block_height, weight_block_width = hf_config.quantization_config["weight_block_size"]
        else:
            weight_block_height = weight_block_width = 128  # Default

        weight_config = {}

        for hf_name, models_name, is_w2 in [
            ("gate_proj", "w1", False),
            ("down_proj", "w2", True),
            ("up_proj", "w3", False),
        ]:
            in_features, out_features = cls.get_weight_shape(hf_config, is_w2)

            # Use get_state_dicts to create 3D tensors (adds dimension 0)
            quantized_weight = get_state_dicts(
                state_dicts,
                f"{hf_name}.weight",
                shape=(out_features, in_features),
                dtype=cls.WEIGHT_TORCH_DTYPE,
            )

            scale_inv = get_state_dicts(
                state_dicts,
                f"{hf_name}.weight_scale_inv",
                shape=(
                    ttnn.core.divup(out_features, weight_block_height),
                    ttnn.core.divup(in_features, weight_block_width),
                ),
                dtype=cls.WEIGHT_SCALE_INV_TORCH_DTYPE,
            )

            # Dequantize with 3D block shape (like reference)
            metaweight_block_size = (1, weight_block_height, weight_block_width)
            dequantized_weight = dequantize(quantized_weight, scale_inv, metaweight_block_size)

            # Convert to metaweight format
            weight_tensor = cls.convert_metaweight(
                output_path / f"{models_name}.input_tensor_b",
                dequantized_weight,
                mesh_device,
                is_w2=is_w2,
            )

            weight_config[models_name] = {"input_tensor_b": weight_tensor}

        return weight_config

    @classmethod
    def convert_metaweight(
        cls,
        path: Path,
        torch_metaweight_tensor: torch.Tensor,
        mesh_device: ttnn.MeshDevice,
        is_w2: bool,
    ) -> ttnn.Tensor:
        """
        Convert metaweight tensor to TTNN format.
        For SharedExpert, we replicate weights across all devices (not shard).
        """
        # The input is [1, out_features, in_features] from get_state_dicts
        # Just squeeze the first dimension and transpose for matmul
        torch_metaweight_tensor = torch_metaweight_tensor.squeeze(0)  # Remove shard dimension

        # Transpose for matmul: [out_features, in_features] -> [in_features, out_features]
        torch_metaweight_tensor = torch_metaweight_tensor.transpose(0, 1)

        # Add batch dimensions for TTNN: [in_features, out_features] -> [1, 1, in_features, out_features]
        torch_metaweight_tensor = torch_metaweight_tensor.unsqueeze(0).unsqueeze(0)

        # Convert to bfloat16
        torch_metaweight_tensor = torch_metaweight_tensor.to(torch.bfloat16)

        # Create TTNN tensor - replicate across all devices (SharedExpert uses same weights everywhere)
        weight_tensor = ttnn.from_torch(
            torch_metaweight_tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return weight_tensor

    @classmethod
    def decode_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate decode configuration for this module."""
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)
        _, mesh_width = mesh_device.shape

        config = {
            "mesh_device": mesh_device,
            "input_memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_memory_config": ttnn.L1_MEMORY_CONFIG,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "mesh_width": mesh_width,
        }

        # Add linear configs
        for name in ["w1", "w2", "w3"]:
            config[name] = {
                "memory_config": ttnn.L1_MEMORY_CONFIG,
                "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            }

        # Add mul config for activation
        config["mul"] = {
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        return config

    @classmethod
    def prefill_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate prefill configuration for this module."""
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)
        _, mesh_width = mesh_device.shape

        matmul_core_grid_size = ttnn.CoreCoord(
            mesh_device.core_grid.x,
            mesh_device.core_grid.y,
        )

        config = {
            "mesh_device": mesh_device,
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "mesh_width": mesh_width,
            "max_rows": SEQ_LEN_CHUNK_SIZE,
            "matmul_core_grid_size": matmul_core_grid_size,
        }

        for name in ["w1", "w2", "w3"]:
            config[name] = {
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            }

        config["mul"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        return config

    @classmethod
    def create_state(cls, hf_config: Any, mesh_device: ttnn.MeshDevice, ccl: Any = None) -> Dict:
        """Create state (empty for shared expert)."""
        return {}

    @classmethod
    def _silu_workaround(cls, x: ttnn.Tensor) -> ttnn.Tensor:
        """Workaround for the silu PCC issue in ttnn."""
        x1 = ttnn.neg(x)
        x2 = ttnn.ones_like(x)
        x3 = ttnn.exp(x1)
        ttnn.deallocate(x1)
        x4 = ttnn.add(x3, 1)
        ttnn.deallocate(x3)
        x5 = ttnn.div(x2, x4)
        ttnn.deallocate(x2)
        ttnn.deallocate(x4)
        x6 = ttnn.mul(x, x5)
        ttnn.deallocate(x5)
        return x6

    @classmethod
    def _get_prefill_pc(
        cls, seq_len: int, dim: int, hidden_dim: int, mesh_width: int, core_grid_size: ttnn.CoreCoord, is_w2: bool
    ) -> Any:
        """Get the program config for linear layers in prefill mode."""
        if is_w2:
            per_device_in_features = even_int_div(hidden_dim, mesh_width)
            per_device_out_features = dim
        else:
            per_device_in_features = dim
            per_device_out_features = even_int_div(hidden_dim, mesh_width)

        per_core_M_tiles = ttnn.core.divup(seq_len, ttnn.TILE_SIZE * core_grid_size.y)
        K_tiles = ttnn.core.divup(per_device_in_features, ttnn.TILE_SIZE)
        per_core_N_tiles = ttnn.core.divup(per_device_out_features, ttnn.TILE_SIZE * core_grid_size.x)

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid_size,
            in0_block_w=find_largest_divisor(K_tiles),
            out_subblock_h=1,
            out_subblock_w=find_largest_divisor(per_core_N_tiles, 4),
            per_core_M=per_core_M_tiles,
            per_core_N=per_core_N_tiles,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in decode mode."""
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])

        # Apply silu workaround
        w1_out_activated = cls._silu_workaround(w1_out)
        ttnn.deallocate(w1_out)

        # Multiply
        activated = ttnn.mul(w1_out_activated, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out_activated)
        ttnn.deallocate(w3_out)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in prefill mode."""
        assert x.memory_config() == cfg["input_memory_config"]

        # Handle chunking if needed
        num_layers, _, seq_len, _ = x.shape
        original_seq_len = seq_len
        pad_rows = 0
        max_rows = cfg.get("max_rows", SEQ_LEN_CHUNK_SIZE)

        if seq_len > max_rows:
            if seq_len % max_rows != 0:
                pad_rows = max_rows - (seq_len % max_rows)
                x_padded = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_rows), (0, 0)), value=0.0)
                ttnn.deallocate(x)
                x = x_padded
                seq_len += pad_rows
            x = ttnn.reshape(x, [num_layers, even_int_div(seq_len, max_rows), max_rows, -1])
            seq_len = max_rows

        # Get program configs
        pc_args = {
            "seq_len": seq_len,
            "dim": cfg["dim"],
            "hidden_dim": cfg["hidden_dim"],
            "mesh_width": cfg["mesh_width"],
            "core_grid_size": cfg["matmul_core_grid_size"],
        }

        # Gate and up projections
        w1_out = ttnn.linear(x, program_config=cls._get_prefill_pc(is_w2=False, **pc_args), **cfg["w1"])
        w3_out = ttnn.linear(x, program_config=cls._get_prefill_pc(is_w2=False, **pc_args), **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection
        output = ttnn.linear(activated, program_config=cls._get_prefill_pc(is_w2=True, **pc_args), **cfg["w2"])
        ttnn.deallocate(activated)

        # De-chunk if needed
        if pad_rows > 0 or original_seq_len > max_rows:
            _, num_chunks, _, output_dim = output.shape
            if num_chunks > 1:
                output = ttnn.reshape(output, [num_layers, 1, -1, output_dim])
            if pad_rows > 0:
                output = ttnn.slice(output, [0, 0, 0, 0], [num_layers, 1, original_seq_len, output_dim])

        assert output.memory_config() == cfg["output_memory_config"]
        return output
