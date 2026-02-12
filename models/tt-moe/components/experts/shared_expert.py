# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert implementation for MoE - direct port of models/demos/deepseek_v3/tt/mlp/shared_expert.py

This is a self-contained implementation with all utilities copied locally.
The SharedExpert uses moe_intermediate_size for its hidden dimension.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

import ttnn

# ============================================================================
# Utility Functions (copied from models/demos/deepseek_v3/utils/config_helpers.py)
# ============================================================================


def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)

    # Convert to float32 for multiplication
    tensor_float = tensor.float()
    scale_float = inv_scale[tuple(slice(0, s) for s in tensor.shape)].float()
    result = tensor_float * scale_float

    # Clamp values to prevent overflow when converting to bfloat16
    result = torch.clamp(result, min=-3.38e38, max=3.38e38)

    del inv_scale
    return result.to(torch.bfloat16)  # Return as bfloat16


def find_largest_divisor(n: int, max_val: int = 8) -> int:
    """Find the largest divisor of n that is <= max_val."""
    for divisor in range(min(n, max_val), 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


# Compute kernel configuration (copied from config_helpers.py)
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Constants from reference
SEQ_LEN_CHUNK_SIZE = 16384
USERS_PER_ROW = 32


# ============================================================================
# SharedExpert Class (Direct port following MLP -> MLPDequant -> SharedExpert hierarchy)
# ============================================================================


class SharedExpert:
    """
    Shared Expert layer for Mixture-of-Experts (MoE) models.

    This is a direct port of models/demos/deepseek_v3/tt/mlp/shared_expert.py
    The only difference from regular MLP is using moe_intermediate_size instead of intermediate_size.
    """

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: Any) -> Tuple[int, int]:
        """Get the dimensions - uses moe_intermediate_size for shared expert."""
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size  # Key difference from regular MLP
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
    def decode_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate decode configuration for this module."""
        # Extract dimensions
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        # Get mesh dimensions
        _, mesh_width = mesh_device.shape

        # Memory configurations based on mode
        input_memory_config = ttnn.L1_MEMORY_CONFIG
        output_memory_config = ttnn.L1_MEMORY_CONFIG

        # Create config dictionary
        config = {
            "mesh_device": mesh_device,
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "mesh_width": mesh_width,
        }

        # Add linear configs with proper structure (like MLP)
        for name in ["w1", "w2", "w3"]:
            config[name] = {
                "memory_config": output_memory_config,
                "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            }

        # Add mul config for activation
        config["mul"] = {
            "memory_config": output_memory_config,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        return config

    @classmethod
    def prefill_model_config(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Generate prefill configuration for this module."""
        # Extract dimensions
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        # Get mesh dimensions
        _, mesh_width = mesh_device.shape

        # Memory configurations
        input_memory_config = ttnn.DRAM_MEMORY_CONFIG
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Core grid for matmuls
        matmul_core_grid_size = ttnn.CoreCoord(
            mesh_device.core_grid.x,
            mesh_device.core_grid.y,
        )

        # Create config dictionary
        config = {
            "mesh_device": mesh_device,
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "mesh_width": mesh_width,
            "max_rows": SEQ_LEN_CHUNK_SIZE,
            "matmul_core_grid_size": matmul_core_grid_size,
        }

        # Add linear configs
        for name in ["w1", "w2", "w3"]:
            config[name] = {
                "memory_config": output_memory_config,
                "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
            }

        # Add mul config
        config["mul"] = {
            "memory_config": output_memory_config,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        return config

    @classmethod
    def create_state(cls, hf_config: Any, mesh_device: ttnn.MeshDevice) -> Dict:
        """Create state (empty for shared expert)."""
        return {}

    @classmethod
    def convert_weights(
        cls,
        hf_config: Any,
        state_dicts: Tuple[Optional[Dict[str, torch.Tensor]], ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> Dict:
        """
        Convert and prepare weights for the shared expert.

        This follows the MLPDequant pattern of dequantizing weights during conversion.
        """
        # Support both HF config objects and plain dicts
        if hasattr(hf_config, "moe_intermediate_size"):
            use_quantized = hasattr(hf_config, "quantization_config")
            weight_block_size = (
                hf_config.quantization_config.get("weight_block_size", [1, 1]) if use_quantized else [1, 1]
            )
        else:
            use_quantized = hf_config.get("use_quantized_weights", False)
            weight_block_size = hf_config.get("weight_block_size", [1, 1])

        (state_dict,) = state_dicts
        assert state_dict is not None

        weight_config = {}

        for hf_name, ttnn_name in [
            ("gate_proj", "w1"),
            ("down_proj", "w2"),
            ("up_proj", "w3"),
        ]:
            weight_key = f"{hf_name}.weight"
            if weight_key in state_dict:
                weight = state_dict[weight_key]

                # Check for quantization scale
                scale_key = f"{hf_name}.weight_scale_inv"
                if use_quantized and scale_key in state_dict:
                    scale_inv = state_dict[scale_key]
                    # Dequantize the weight
                    weight = dequantize(weight, scale_inv, weight_block_size)

                # Transpose for matmul: [out_features, in_features] -> [1, 1, in_features, out_features]
                weight = weight.unsqueeze(0).unsqueeze(0).transpose(-1, -2)

                # Convert to bfloat16 for TTNN
                weight = weight.to(torch.bfloat16)

                # Create TTNN tensor - replicate across all devices
                weight_tensor = ttnn.from_torch(
                    weight,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Store the weight tensor in the config
                weight_config[ttnn_name] = {"input_tensor_b": weight_tensor}

        return weight_config

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
    def _silu_workaround(cls, x: ttnn.Tensor) -> ttnn.Tensor:
        """Workaround for the silu PCC issue in ttnn."""
        # -x
        x1 = ttnn.neg(x)

        # 1
        x2 = ttnn.ones_like(x)

        # exp(-x)
        x3 = ttnn.exp(x1)
        ttnn.deallocate(x1)

        # 1 + exp(-x)
        x4 = ttnn.add(x3, 1)
        ttnn.deallocate(x3)

        # 1 / (1 + exp(-x))
        x5 = ttnn.div(x2, x4)
        ttnn.deallocate(x2)
        ttnn.deallocate(x4)

        # x * (1 / (1 + exp(-x)))
        x6 = ttnn.mul(x, x5)
        ttnn.deallocate(x5)

        return x6

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: Dict, mode: str) -> ttnn.Tensor:
        """
        Forward pass through shared expert.

        Args:
            x: Input tensor
            cfg: Configuration dictionary with weights and configs
            mode: "decode" or "prefill"

        Returns:
            Expert output
        """
        # Verify input memory config
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        if mode == "prefill":
            num_layers, _, seq_len, _ = x.shape
            original_seq_len = seq_len

            # Chunk the input if needed
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

            # Apply activation and multiply (SwiGLU)
            activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)

            # Down projection
            output = ttnn.linear(activated, program_config=cls._get_prefill_pc(is_w2=True, **pc_args), **cfg["w2"])
            ttnn.deallocate(activated)

            # De-chunk if needed
            _, num_chunks, _, output_dim = output.shape
            if num_chunks > 1:
                output = ttnn.reshape(output, [num_layers, 1, -1, output_dim])
                if pad_rows > 0:
                    output = ttnn.slice(output, [0, 0, 0, 0], [num_layers, 1, original_seq_len, output_dim])

        else:  # decode mode
            # Gate and up projections
            w1_out = ttnn.linear(x, **cfg["w1"])
            w3_out = ttnn.linear(x, **cfg["w3"])

            # Apply silu workaround
            w1_out_activated = cls._silu_workaround(w1_out)
            ttnn.deallocate(w1_out)

            # Apply activation and multiply
            activated = ttnn.mul(w1_out_activated, w3_out, **cfg["mul"])
            ttnn.deallocate(w1_out_activated)
            ttnn.deallocate(w3_out)

            # Down projection
            output = ttnn.linear(activated, **cfg["w2"])
            ttnn.deallocate(activated)

        # Verify output memory config
        assert output.memory_config() == cfg["output_memory_config"]

        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in decode mode."""
        return cls._forward(x, cfg, "decode")

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: Dict) -> ttnn.Tensor:
        """Forward pass in prefill mode."""
        return cls._forward(x, cfg, "prefill")
