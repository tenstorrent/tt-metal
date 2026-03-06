# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
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


class RoutedExperts(AbstractModule):
    """Experts layer for Mixture-of-Experts (MoE) module."""

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def _apply_activation(cls, w1_out: ttnn.Tensor, w3_out: ttnn.Tensor, activation_config) -> ttnn.Tensor:
        """Apply activation function based on configuration.

        Args:
            w1_out: Gate projection output tensor
            w3_out: Up projection output tensor
            activation_config: ExpertActivationConfig object or dict with activation settings

        Returns:
            Activated tensor
        """
        # Handle both ExpertActivationConfig objects and dict configs
        if hasattr(activation_config, "activation_type"):
            activation_type = activation_config.activation_type
            memory_config = activation_config.memory_config
            swiglu_alpha = getattr(activation_config, "swiglu_alpha", 1.702)
            swiglu_limit = getattr(activation_config, "swiglu_limit", 7.0)
        else:
            # Legacy dict config support
            activation_type = activation_config.get("activation_type", "silu")
            memory_config = activation_config.get("memory_config", ttnn.L1_MEMORY_CONFIG)
            swiglu_alpha = activation_config.get("swiglu_alpha", 1.702)
            swiglu_limit = activation_config.get("swiglu_limit", 7.0)

        if activation_type == "silu":
            # DeepSeek: SILU activation
            # Formula: w1 * silu(w1) * w3
            activated = ttnn.mul(
                w1_out, w3_out, memory_config=memory_config, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )

        elif activation_type == "clamped_swiglu":
            # GPT-OSS: Clamped SwiGLU
            # Formula: (up_clamped + 1) * (gate_clamped * sigmoid(gate * alpha))
            activated = cls._apply_clamped_swiglu(
                gate=w1_out, up=w3_out, alpha=swiglu_alpha, limit=swiglu_limit, memory_config=memory_config
            )

        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

        return activated

    @classmethod
    def _apply_clamped_swiglu(
        cls, gate: ttnn.Tensor, up: ttnn.Tensor, alpha: float, limit: float, memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """Apply clamped SwiGLU activation (GPT-OSS style).

        Args:
            gate: Gate projection tensor
            up: Up projection tensor
            alpha: Scaling factor for sigmoid (default 1.702)
            limit: Clamping limit (default 7.0)
            memory_config: Memory configuration for intermediate tensors

        Returns:
            Activated tensor with same shape as inputs
        """
        # Clamp gate (max only)
        gate_clamped = ttnn.clamp(gate, min=None, max=limit)
        ttnn.deallocate(gate)

        # Clamp up (both min and max)
        up_clamped = ttnn.clamp(up, min=-limit, max=limit)
        ttnn.deallocate(up)

        # Compute gate_alpha = gate * alpha
        gate_alpha = ttnn.mul(gate_clamped, alpha)

        # Compute gate_sigmoid = sigmoid(gate_alpha)
        gate_sigmoid = ttnn.sigmoid(gate_alpha, memory_config=memory_config)
        ttnn.deallocate(gate_alpha)

        # Compute glu = gate * gate_sigmoid
        glu = ttnn.mul(gate_clamped, gate_sigmoid, memory_config=memory_config)
        ttnn.deallocate(gate_clamped)
        ttnn.deallocate(gate_sigmoid)

        # Add 1 to up: up = up + 1
        ttnn.add(up_clamped, 1.0, output_tensor=up_clamped)

        # Multiply: result = up * glu
        result = ttnn.mul(up_clamped, glu, memory_config=memory_config)
        ttnn.deallocate(up_clamped)
        ttnn.deallocate(glu)

        return result

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

        # Check if we have quantization config (DeepSeek) or not (GPT-OSS)
        if hf_config.quantization_config is not None:
            # DeepSeek path with quantization
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
        else:
            # GPT-OSS path without quantization
            return {
                ttnn_name: {
                    "input_tensor_b": shard_and_save(
                        output_path / f"{ttnn_name}.input_tensor_b",
                        torch.stack(
                            [
                                state_dict[f"experts.{expert_id}.{hf_name}.weight"]
                                for expert_id in range(hf_config.n_routed_experts)
                            ]
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
        # Handle both DeepSeek (moe_intermediate_size) and GPT-OSS (intermediate_size)
        moe_intermediate_size = getattr(
            hf_config, "moe_intermediate_size", getattr(hf_config, "intermediate_size", None)
        )
        if moe_intermediate_size is None:
            raise ValueError("Config must have either moe_intermediate_size or intermediate_size")

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

        # Get unified config if available, otherwise use legacy config
        unified_config = cfg.get("unified_expert_config")

        if unified_config and getattr(unified_config, "use_fused_gate_up", False):
            # Fused gate_up projection (GPT-OSS optional)
            gate_up = ttnn.linear(x, **cfg["w1_w3_fused_experts"])
            w1_out = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], unified_config.intermediate_size],
            )
            w3_out = ttnn.slice(
                gate_up,
                [0, 0, 0, unified_config.intermediate_size],
                [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], 2 * unified_config.intermediate_size],
            )
            ttnn.deallocate(gate_up)
        else:
            # Separate gate and up projections (DeepSeek default)
            w1_out = ttnn.linear(x, **cfg["w1_experts"])
            w3_out = ttnn.linear(x, **cfg["w3_experts"])

        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply unified activation
        if unified_config:
            activated = cls._apply_activation(w1_out, w3_out, unified_config.activation)
        else:
            # Legacy path - use existing mul_experts config
            activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)

        _log_expert_stats("activated", activated)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])
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

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass for expert computation.

        This method handles only the expert MLP computation.
        All_to_all operations are handled externally in MoEBlock.

        Args:
            x: Input tensor (already dispatched by all_to_all if needed)
            cfg: Runtime configuration

        Returns:
            Output tensor (before all_to_all_combine if needed)
        """
        # Simply use the existing _forward method
        # All_to_all operations are handled by MoEBlock
        return cls._forward(x, cfg)


# Backward compatibility alias
Experts = RoutedExperts
