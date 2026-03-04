# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn

# ============================================================================
# Checkpoint Helper Function for Debugging
# ============================================================================


def _save_expert_checkpoint(
    name: str, tensor: ttnn.Tensor, prefix: str = "reference", mesh_device: Optional[ttnn.MeshDevice] = None
) -> None:
    """
    Save activation checkpoints for debugging expert implementations.
    Only saves when SAVE_EXPERT_CHECKPOINTS=1 environment variable is set.

    Args:
        name: Name of the checkpoint (e.g., "w1_output", "silu_output")
        tensor: TTNN tensor to save
        prefix: Directory prefix ("our_impl" or "reference")
        mesh_device: Optional mesh device for distributed tensors
    """
    if os.environ.get("SAVE_EXPERT_CHECKPOINTS") != "1":
        return

    try:
        # Create directory if it doesn't exist
        checkpoint_dir = Path(f"/tmp/expert_debug/{prefix}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor to PyTorch
        if mesh_device is not None:
            try:
                tensor_torch = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            except:
                # Try alternative mesh composer
                tensor_torch = ttnn.to_torch(
                    tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
                )
        else:
            tensor_torch = ttnn.to_torch(tensor)

        # Save checkpoint with metadata
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        checkpoint_data = {
            "tensor": tensor_torch.cpu(),
            "shape": list(tensor_torch.shape),
            "dtype": str(tensor_torch.dtype),
            "bytes": tensor_torch.numel() * tensor_torch.element_size(),
            "min": float(tensor_torch.min()) if tensor_torch.numel() > 0 else 0.0,
            "max": float(tensor_torch.max()) if tensor_torch.numel() > 0 else 0.0,
            "mean": float(tensor_torch.mean()) if tensor_torch.numel() > 0 else 0.0,
            "std": float(tensor_torch.std()) if tensor_torch.numel() > 0 else 0.0,
            "has_nan": bool(torch.isnan(tensor_torch).any()),
            "has_inf": bool(torch.isinf(tensor_torch).any()),
        }
        torch.save(checkpoint_data, checkpoint_path)

        logger.info(
            f"[CHECKPOINT] Saved {name} to {checkpoint_path}: shape={checkpoint_data['shape']}, dtype={checkpoint_data['dtype']}, bytes={checkpoint_data['bytes']}"
        )

    except Exception as e:
        logger.warning(f"[CHECKPOINT] Failed to save {name}: {e}")


from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    dequantize,
    even_int_div,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Experts(AbstractModule):
    """Experts layer for Mixture-of-Experts (MoE) module."""

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert hf_config.n_routed_experts % mesh_device.get_num_devices() == 0, (
            f"Number of experts ({hf_config.n_routed_experts}) must be divisible by the number of devices "
            f"({mesh_device.get_num_devices()})"
        )
        (state_dict,) = state_dicts
        assert state_dict is not None

        return {
            ttnn_name: {
                "input_tensor_b": shard_and_save(
                    output_path / f"{ttnn_name}.input_tensor_b",
                    dequantize(
                        torch.stack(
                            [
                                state_dict[f"experts.{expert_id}.{hf_name}.weight"]
                                for expert_id in range(hf_config.n_routed_experts)
                            ]
                        ),
                        torch.stack(
                            [
                                state_dict[f"experts.{expert_id}.{hf_name}.weight_scale_inv"]
                                for expert_id in range(hf_config.n_routed_experts)
                            ]
                        ),
                        (1, *hf_config.quantization_config["weight_block_size"]),
                    )
                    .unsqueeze(0)
                    .transpose(-1, -2),
                    shard_dims=(1, 1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat8_b if hf_name == "up_proj" else ttnn.bfloat4_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
            for hf_name, ttnn_name in [
                ("gate_proj", "w1_experts"),
                ("down_proj", "w2_experts"),
                ("up_proj", "w3_experts"),
            ]
        }

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return mesh_device.shape[1] == 8

    @classmethod
    @classmethod
    def _create_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, mode: str
    ) -> ModelPrefillConfig | ModelDecodeConfig:
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Calculate dimensions
        hidden_size = hf_config.hidden_size
        moe_intermediate_size = hf_config.moe_intermediate_size

        # Calculate input and output memory configurations
        if mode == "decode":
            input_memory_config = ttnn.L1_MEMORY_CONFIG
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Construct the config
        return {
            "mesh_device": MeshDeviceStub(mesh_device.shape),
            "w1_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w3_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul_experts": MulConfig(
                memory_config=output_memory_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        _, _, num_tokens, hidden_size = x.shape

        debug_experts = os.getenv("DEEPSEEK_V3_DEBUG_EXPERTS") == "1" and num_tokens > 8192

        def _log_expert_stats(name: str, tensor: ttnn.Tensor) -> None:
            if not debug_experts:
                return
            try:
                mesh_device = cfg.get("mesh_device")
                if mesh_device is not None:
                    tensor_torch = ttnn.to_torch(
                        tensor,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape
                        ),
                    )
                else:
                    tensor_torch = ttnn.to_torch(tensor)
                finite_mask = torch.isfinite(tensor_torch)
                numel = tensor_torch.numel()
                finite_count = finite_mask.sum().item()
                nan_count = torch.isnan(tensor_torch).sum().item()
                inf_count = torch.isinf(tensor_torch).sum().item()
                logger.info(
                    f"DEBUG experts {name}: shape={tensor_torch.shape}, "
                    f"mean={tensor_torch.mean():.4f}, std={tensor_torch.std():.4f}, "
                    f"max={tensor_torch.abs().max():.4f}, "
                    f"finite={finite_count}/{numel}, nan={nan_count}, inf={inf_count}"
                )
            except Exception as exc:
                logger.warning(f"DEBUG experts {name}: failed to extract stats: {exc}")

        # Save input checkpoint (ACTIVATIONS ONLY)
        mesh_device = cfg.get("mesh_device")
        _save_expert_checkpoint("expert_input", x, prefix="reference", mesh_device=mesh_device)

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])

        # Save checkpoints after projections (ACTIVATIONS ONLY)
        _save_expert_checkpoint("w1_output", w1_out, prefix="reference", mesh_device=mesh_device)
        _save_expert_checkpoint("w3_output", w3_out, prefix="reference", mesh_device=mesh_device)

        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply activation and multiply
        # The mul operation has input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        # which applies SiLU to w1_out before multiplication
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])

        # Save checkpoint after the SiLU and multiply (combined operation)
        _save_expert_checkpoint("multiply_output", activated, prefix="reference", mesh_device=mesh_device)

        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        _log_expert_stats("activated", activated)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])

        # Save checkpoint after down projection
        _save_expert_checkpoint("w2_output", output, prefix="reference", mesh_device=mesh_device)

        ttnn.deallocate(activated)
        _log_expert_stats("w2_out", output)

        # Reshape for output
        output = ttnn.permute(output, (1, 0, 2, 3))
        output = ttnn.reshape(output, shape=(1, cfg["num_experts_per_device"], num_tokens, hidden_size))

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)
