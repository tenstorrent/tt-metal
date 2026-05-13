# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from ttnn.experimental.moe_compute_utils import (
    determine_compute_matmul_cores,
    get_shared_experts_per_device,
    get_w0_w1_memory_config,
    get_w2_memory_config,
)

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_LOFI,
    even_int_div,
    get_dequantized_tensor,
    get_fabric_config,
    is_quad_mesh,
    is_ring_fabric,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.quad_ring_packing import prepare_quad_ring_packed_experts
from models.demos.deepseek_v3.utils.quad_ring_packing import (
    quad_ring_shared_expert_to_device_map as _quad_ring_shared_expert_to_device_map,
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

    WEIGHT_TORCH_DTYPE = torch.bfloat16
    _warned_legacy_expert_checkpoint = False
    _warned_missing_quad_ring_prepared_checkpoint = False
    _QUAD_RING_PREPARED_W0_W1_KEY = "experts_quad_ring.w0_w1.weight"
    _QUAD_RING_PREPARED_W2_KEY = "experts_quad_ring.w2.weight"

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
        if is_quad_mesh(mesh_device) and is_ring_fabric(get_fabric_config()):
            return cls._convert_weights_quad_ring(hf_config, state_dicts, output_path, mesh_device)
        return cls._convert_weights_default(hf_config, state_dicts, output_path, mesh_device)

    @classmethod
    def _load_expert_weight(
        cls, state_dict: dict[str, torch.Tensor], hf_name: str, root_name: str, n_experts: int
    ) -> torch.Tensor:
        weight_name = f"{hf_name}.weight"
        expert_weights: list[torch.Tensor] = []
        for expert_id in range(n_experts):
            if n_experts == 1:
                full_weight_name = f"{root_name}.{weight_name}"
            else:
                full_weight_name = f"{root_name}.{expert_id}.{weight_name}"
            expert_weights.append(get_dequantized_tensor(state_dict, full_weight_name, dtype=cls.WEIGHT_TORCH_DTYPE))
        return torch.stack(expert_weights)

    @classmethod
    def _convert_weights_default(
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
                    cls._load_expert_weight(state_dict, hf_name, "experts", hf_config.n_routed_experts)
                    .unsqueeze(0)
                    .transpose(-1, -2),
                    shard_dims=(1, 1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat8_b if hf_name == "down_proj" else ttnn.bfloat4_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
            for hf_name, ttnn_name in [
                ("gate_proj", "w1_experts"),
                ("down_proj", "w2_experts"),
                ("up_proj", "w3_experts"),
            ]
        }

    # Kept as a static method so external callers (e.g. ``MoEOptimized``) can use it without
    # importing the helper module directly. The single source of truth lives in
    # ``models.demos.deepseek_v3.utils.quad_ring_packing``.
    quad_ring_shared_expert_to_device_map = staticmethod(_quad_ring_shared_expert_to_device_map)

    @classmethod
    def _convert_weights_quad_ring(
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

        num_layers = 1
        num_routed_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)
        num_routed_experts = hf_config.n_routed_experts
        hidden_size = hf_config.hidden_size
        loaded_prepared_weights = cls._load_quad_ring_prepared_weights(state_dict)
        ring2cores, compute_matmul_dram_core_range_set = determine_compute_matmul_cores(mesh_device)

        num_devices = mesh_device.get_num_devices()
        num_shared_experts = hf_config.n_shared_experts
        shared_expert_ids_to_device = cls.quad_ring_shared_expert_to_device_map(
            num_routed_experts, num_shared_experts, num_devices
        )
        num_shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_device, num_devices)[0]
        num_total_experts_per_device = num_shared_experts_per_device + num_routed_experts_per_device
        num_total_experts_on_devices = num_total_experts_per_device * num_devices

        if loaded_prepared_weights is not None:
            # The prepacked quad-ring checkpoint format covers routed experts only. Refuse to
            # silently drop shared experts; regen the prepared checkpoint (or omit it and take
            # the slower fallback) once it learns to carry them.
            if num_shared_experts > 0:
                raise ValueError(
                    "Prepacked quad-ring MoE checkpoints do not yet include shared-expert weights. "
                    "Omit the prepacked checkpoint to fall back to on-the-fly repacking (which "
                    "supports shared experts), or extend the prepared checkpoint format."
                )
            prepared_w0_w1, prepared_w2 = loaded_prepared_weights
            cls._validate_quad_ring_prepared_weights_shapes(prepared_w0_w1, prepared_w2, num_routed_experts)
            logger.info("Using prepacked quad-ring MoE expert tensors from checkpoint.")
            matmul_N = hf_config.moe_intermediate_size
        else:
            if not cls._warned_missing_quad_ring_prepared_checkpoint:
                logger.warning(
                    "Quad-ring prepared expert tensors were not found in the checkpoint. "
                    "Falling back to on-the-fly expert repacking, which is significantly slower. "
                    "Generate a prepared checkpoint with "
                    "`python models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py "
                    "<stacked-model-path>` and point `DEEPSEEK_V3_HF_MODEL` or `--model-path` at the resulting "
                    "`*-quad-ring` directory."
                )
                cls._warned_missing_quad_ring_prepared_checkpoint = True

            routed_gate = cls._load_expert_weight(state_dict, "gate_proj", "experts", num_routed_experts)
            routed_up = cls._load_expert_weight(state_dict, "up_proj", "experts", num_routed_experts)
            routed_down = cls._load_expert_weight(state_dict, "down_proj", "experts", num_routed_experts)
            # HF stores the shared MLP as a single unindexed module (``shared_experts.{proj}.weight``).
            shared_gate = get_dequantized_tensor(
                state_dict, "shared_experts.gate_proj.weight", dtype=cls.WEIGHT_TORCH_DTYPE
            )
            shared_up = get_dequantized_tensor(
                state_dict, "shared_experts.up_proj.weight", dtype=cls.WEIGHT_TORCH_DTYPE
            )
            shared_down = get_dequantized_tensor(
                state_dict, "shared_experts.down_proj.weight", dtype=cls.WEIGHT_TORCH_DTYPE
            )
            # matmul_N must match the post-transpose shape used downstream by get_w2_memory_config.
            matmul_N = routed_gate.shape[-2]
            prepared_w0_w1, prepared_w2 = prepare_quad_ring_packed_experts(
                routed_gate=routed_gate,
                routed_up=routed_up,
                routed_down=routed_down,
                shared_gate=shared_gate,
                shared_up=shared_up,
                shared_down=shared_down,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                num_devices=num_devices,
                hidden_size=hidden_size,
                shared_expert_ids_to_devices=shared_expert_ids_to_device,
                ring2cores=ring2cores,
            )

        w0_w1_memory_config = get_w0_w1_memory_config(
            num_layers, num_total_experts_per_device, hidden_size, compute_matmul_dram_core_range_set
        )
        w2_memory_config = get_w2_memory_config(
            num_layers, num_total_experts_per_device, matmul_N, compute_matmul_dram_core_range_set
        )

        return {
            "quad_ring_w0_w1_experts": {
                "input_tensor_b": shard_and_save(
                    output_path / "quad_ring_w0_w1_experts.input_tensor_b",
                    prepared_w0_w1,
                    shard_dims=(2, 2),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    memory_config=w0_w1_memory_config,
                )
            },
            "quad_ring_w2_experts": {
                "input_tensor_b": shard_and_save(
                    output_path / "quad_ring_w2_experts.input_tensor_b",
                    prepared_w2,
                    shard_dims=(2, 2),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    memory_config=w2_memory_config,
                )
            },
        }

    @classmethod
    def _load_quad_ring_prepared_weights(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        w0_w1_key = cls._QUAD_RING_PREPARED_W0_W1_KEY
        w2_key = cls._QUAD_RING_PREPARED_W2_KEY

        has_w0_w1 = w0_w1_key in state_dict
        has_w2 = w2_key in state_dict
        if has_w0_w1 != has_w2:
            missing_keys = [key for key, exists in ((w0_w1_key, has_w0_w1), (w2_key, has_w2)) if not exists]
            raise ValueError(
                "Checkpoint contains partial quad-ring prepared expert tensors. "
                f"Missing {missing_keys}; regenerate the quad-ring prepared checkpoint."
            )
        if not has_w0_w1:
            return None

        prepared_w0_w1 = get_dequantized_tensor(state_dict, w0_w1_key, dtype=cls.WEIGHT_TORCH_DTYPE).contiguous()
        prepared_w2 = get_dequantized_tensor(state_dict, w2_key, dtype=cls.WEIGHT_TORCH_DTYPE).contiguous()
        return prepared_w0_w1, prepared_w2

    @classmethod
    def _validate_quad_ring_prepared_weights_shapes(
        cls,
        prepared_w0_w1: torch.Tensor,
        prepared_w2: torch.Tensor,
        num_routed_experts: int,
    ) -> None:
        expected_tile_width = 4 * ttnn.TILE_SIZE
        if prepared_w0_w1.ndim != 6:
            raise ValueError(
                f"Expected '{cls._QUAD_RING_PREPARED_W0_W1_KEY}' to have rank 6, got {prepared_w0_w1.ndim}"
            )
        if prepared_w2.ndim != 6:
            raise ValueError(f"Expected '{cls._QUAD_RING_PREPARED_W2_KEY}' to have rank 6, got {prepared_w2.ndim}")
        if prepared_w0_w1.shape[0] != 12 or prepared_w2.shape[0] != 12:
            raise ValueError(
                "Quad-ring prepared expert tensors must have 12 DRAM-bank groups in dim 0, got "
                f"{prepared_w0_w1.shape[0]} and {prepared_w2.shape[0]}"
            )
        if prepared_w0_w1.shape[2] != num_routed_experts or prepared_w2.shape[2] != num_routed_experts:
            raise ValueError(
                "Quad-ring prepared expert tensors have mismatched expert dimension: expected "
                f"{num_routed_experts}, got {prepared_w0_w1.shape[2]} and {prepared_w2.shape[2]}"
            )
        if prepared_w0_w1.shape[-1] != expected_tile_width or prepared_w2.shape[-1] != expected_tile_width:
            raise ValueError(
                "Quad-ring prepared expert tensors must have tile width "
                f"{expected_tile_width} in the last dim, got {prepared_w0_w1.shape[-1]} and {prepared_w2.shape[-1]}"
            )

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
        config = {
            "mesh_device": MeshDeviceStub(mesh_device.shape),
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
        }

        if is_quad_mesh(mesh_device) and is_ring_fabric(get_fabric_config()):
            config["quad_ring_w0_w1_experts"] = {
                "input_tensor_b": FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            }
            config["quad_ring_w2_experts"] = {
                "input_tensor_b": FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            }

        else:
            config["w1_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            )
            config["w2_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            )
            config["w3_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            )
            config["mul_experts"] = MulConfig(
                memory_config=output_memory_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )

        return config

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

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])
        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply activation and multiply
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
