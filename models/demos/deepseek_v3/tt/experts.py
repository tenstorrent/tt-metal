# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from ttnn.experimental.moe_compute_utils import (
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
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
        cls, state_dict: dict[str, torch.Tensor], hf_name: str, n_routed_experts: int
    ) -> torch.Tensor:
        view_with_prefix = getattr(state_dict, "view_with_prefix", None)
        stacked_state_dict = view_with_prefix("experts_stacked.") if callable(view_with_prefix) else state_dict
        stacked_lookup_names = {f"{name}.weight" for name in ("gate_proj", "down_proj", "up_proj")}
        if not callable(view_with_prefix):
            stacked_lookup_names = {f"experts_stacked.{name}" for name in stacked_lookup_names}
        present_stacked_lookup_names = {name for name in stacked_lookup_names if name in stacked_state_dict}

        stacked_weight_name = f"experts_stacked.{hf_name}.weight"
        stacked_lookup_name = f"{hf_name}.weight" if callable(view_with_prefix) else stacked_weight_name
        if stacked_lookup_name in stacked_state_dict:
            stacked_weight = get_dequantized_tensor(
                stacked_state_dict, stacked_lookup_name, dtype=cls.WEIGHT_TORCH_DTYPE
            )
            if stacked_weight.ndim != 3:
                raise ValueError(
                    f"Expected stacked expert weight '{stacked_weight_name}' to have rank 3, got {stacked_weight.ndim}"
                )
            if stacked_weight.shape[0] != n_routed_experts:
                raise ValueError(
                    f"Expected stacked expert weight '{stacked_weight_name}' to contain "
                    f"{n_routed_experts} experts, got {stacked_weight.shape[0]}"
                )
            return stacked_weight.contiguous()

        if present_stacked_lookup_names:
            raise ValueError(
                f"Checkpoint mixes stacked and legacy expert weights: missing '{stacked_weight_name}' while "
                "other stacked expert tensors are present. Regenerate the stacked checkpoint so all expert "
                "projections are exported together."
            )

        if not cls._warned_legacy_expert_checkpoint:
            logger.warning(
                "Stacked expert tensors were not found in the DeepSeek checkpoint. "
                "Falling back to the slower legacy per-expert compatibility path. "
                "Generate a stacked checkpoint with "
                "`python models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py "
                "<source-model-path> --stack-experts` "
                "and point `DEEPSEEK_V3_HF_MODEL` or `--model-path` at the resulting "
                "`*-dequantized-stacked` directory."
            )
            cls._warned_legacy_expert_checkpoint = True

        weight_name = f"{hf_name}.weight"
        expert_weights: list[torch.Tensor] = []
        for expert_id in range(n_routed_experts):
            full_weight_name = f"experts.{expert_id}.{weight_name}"
            expert_weights.append(get_dequantized_tensor(state_dict, full_weight_name, dtype=cls.WEIGHT_TORCH_DTYPE))
        return torch.stack(expert_weights).contiguous()

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
                    cls._load_expert_weight(state_dict, hf_name, hf_config.n_routed_experts).unsqueeze(0).contiguous(),
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
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)
        num_routed_experts = hf_config.n_routed_experts

        hidden_size = hf_config.hidden_size
        matmul_N = hf_config.moe_intermediate_size
        w0_w1_shard_map, w2_shard_map, compute_matmul_dram_core_range_set = get_weight_core_shard_maps(
            mesh_device, hidden_size, matmul_N
        )

        prepared_state_dict = state_dict
        prepared_key_pairs = (
            ("experts_quad_ring.w0_w1.weight", "experts_quad_ring.w2.weight"),
            ("w0_w1.weight", "w2.weight"),
        )
        prepared_w0_w1_key = prepared_key_pairs[0][0]
        prepared_w2_key = prepared_key_pairs[0][1]
        has_prepared_w0_w1 = False
        has_prepared_w2 = False
        for candidate_w0_w1_key, candidate_w2_key in prepared_key_pairs:
            candidate_has_w0_w1 = candidate_w0_w1_key in prepared_state_dict
            candidate_has_w2 = candidate_w2_key in prepared_state_dict
            if candidate_has_w0_w1 or candidate_has_w2:
                prepared_w0_w1_key = candidate_w0_w1_key
                prepared_w2_key = candidate_w2_key
                has_prepared_w0_w1 = candidate_has_w0_w1
                has_prepared_w2 = candidate_has_w2
                break

        if has_prepared_w0_w1 or has_prepared_w2:
            if not (has_prepared_w0_w1 and has_prepared_w2):
                raise ValueError(
                    "Checkpoint contains partial quad-ring prepared expert tensors. "
                    f"Expected both '{prepared_w0_w1_key}' and '{prepared_w2_key}'."
                )
            prepared_w0_w1 = get_dequantized_tensor(
                prepared_state_dict, prepared_w0_w1_key, dtype=cls.WEIGHT_TORCH_DTYPE
            )
            prepared_w2 = get_dequantized_tensor(prepared_state_dict, prepared_w2_key, dtype=cls.WEIGHT_TORCH_DTYPE)
        else:
            allow_repack = os.getenv("DEEPSEEK_V3_ALLOW_QUAD_RING_WEIGHT_REPACK", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not allow_repack:
                raise ValueError(
                    "Quad-ring MoE weight loading requires prepacked HF expert tensors "
                    f"'{prepared_w0_w1_key}' and '{prepared_w2_key}' so cold TTNN weight-cache loads stay fast. "
                    "Run `python models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py "
                    "<stacked-dequantized-model>` and point DEEPSEEK_V3_HF_MODEL at the generated "
                    "`*-quad-ring` checkpoint, or set DEEPSEEK_V3_ALLOW_QUAD_RING_WEIGHT_REPACK=1 "
                    "to allow slow in-process repacking."
                )

            w0 = cls._load_expert_weight(state_dict, "gate_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)
            w1 = cls._load_expert_weight(state_dict, "up_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)
            w2 = cls._load_expert_weight(state_dict, "down_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)

            prepared_w0_w1 = []
            prepared_w2 = []
            for i in range(0, num_routed_experts, num_experts_per_device):
                prepared_w0_w1_tensor = prepare_w0_w1_tensor_for_moe_compute(
                    w0[:, i : i + num_experts_per_device, :, :],
                    w1[:, i : i + num_experts_per_device, :, :],
                    num_layers,
                    num_experts_per_device,
                    hidden_size,
                    matmul_N,
                    w0_w1_shard_map,
                )
                prepared_w2_tensor = prepare_w2_tensor_for_moe_compute(
                    w2[:, i : i + num_experts_per_device, :, :],
                    num_layers,
                    num_experts_per_device,
                    matmul_N,
                    hidden_size,
                    w2_shard_map,
                    w0_w1_shard_map,
                )

                prepared_w0_w1.append(prepared_w0_w1_tensor)
                prepared_w2.append(prepared_w2_tensor)

            prepared_w0_w1 = torch.cat(prepared_w0_w1, dim=2)
            prepared_w2 = torch.cat(prepared_w2, dim=2)

        w0_w1_memory_config, w2_memory_config, _, _ = get_weight_mem_configs(
            num_layers,
            num_experts_per_device,
            hidden_size,
            matmul_N,
            w0_w1_shard_map,
            w2_shard_map,
            compute_matmul_dram_core_range_set,
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
                transpose_b=True,
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            )
            config["w2_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            )
            config["w3_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
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
