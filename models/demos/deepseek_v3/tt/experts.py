# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from ttnn.experimental.moe_compute_utils import (
    determine_compute_matmul_cores,
    get_w0_w1_memory_config,
    get_w2_memory_config,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
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
        loaded_prepared_weights = cls._load_quad_ring_prepared_weights(state_dict)
        ring2cores, compute_matmul_dram_core_range_set = determine_compute_matmul_cores(mesh_device)

        if loaded_prepared_weights is not None:
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

            w0 = cls._load_expert_weight(state_dict, "gate_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)
            w1 = cls._load_expert_weight(state_dict, "up_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)
            w2 = cls._load_expert_weight(state_dict, "down_proj", num_routed_experts).unsqueeze(0).transpose(-1, -2)

            matmul_N = w0.shape[-1]

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
                    ring2cores,
                )
                prepared_w2_tensor = prepare_w2_tensor_for_moe_compute(
                    w2[:, i : i + num_experts_per_device, :, :],
                    num_layers,
                    num_experts_per_device,
                    matmul_N,
                    hidden_size,
                    ring2cores,
                )

                prepared_w0_w1.append(prepared_w0_w1_tensor)
                prepared_w2.append(prepared_w2_tensor)

            prepared_w0_w1 = torch.cat(prepared_w0_w1, dim=2)
            prepared_w2 = torch.cat(prepared_w2, dim=2)

        w0_w1_memory_config = get_w0_w1_memory_config(
            num_layers, num_experts_per_device, hidden_size, compute_matmul_dram_core_range_set
        )
        w2_memory_config = get_w2_memory_config(
            num_layers, num_experts_per_device, matmul_N, compute_matmul_dram_core_range_set
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

    @staticmethod
    def _with_throttle_level(
        compute_kernel_config: ttnn.WormholeComputeKernelConfig,
        throttle_level: ttnn.ThrottleLevel,
        packer_l1_acc: bool | None = None,
        dst_full_sync_en: bool | None = None,
    ) -> ttnn.WormholeComputeKernelConfig:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=compute_kernel_config.math_fidelity,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
            packer_l1_acc=compute_kernel_config.packer_l1_acc if packer_l1_acc is None else packer_l1_acc,
            dst_full_sync_en=compute_kernel_config.dst_full_sync_en if dst_full_sync_en is None else dst_full_sync_en,
            throttle_level=throttle_level,
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
            # See tt-metal#44280: quad 8upr prefill can hang in the legacy MoE expert matmuls.
            # Throttle the prefill up/gate projections as well as W2; decode keeps the existing configs.
            w1_w3_compute_kernel_config = (
                cls._with_throttle_level(
                    COMPUTE_KERNEL_CONFIG_LOFI,
                    ttnn.ThrottleLevel.LEVEL_5,
                    packer_l1_acc=False,
                    dst_full_sync_en=True,
                )
                if mode == "prefill"
                else COMPUTE_KERNEL_CONFIG_LOFI
            )
            config["w1_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=output_memory_config,
                compute_kernel_config=w1_w3_compute_kernel_config,
            )
            # See tt-metal#44280: prefill W2 is sensitive to matmul di/dt on the quad 8upr legacy path.
            # Use FP16 destination accumulation plus throttle/sync knobs to reduce W2 pressure without changing
            # tensor shapes.
            w2_compute_kernel_config = (
                cls._with_throttle_level(
                    COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
                    ttnn.ThrottleLevel.LEVEL_5,
                    packer_l1_acc=False,
                    dst_full_sync_en=True,
                )
                if mode == "prefill"
                else COMPUTE_KERNEL_CONFIG_HIFI2
            )
            config["w2_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=output_memory_config,
                compute_kernel_config=w2_compute_kernel_config,
            )
            config["w3_experts"] = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=output_memory_config,
                compute_kernel_config=w1_w3_compute_kernel_config,
            )
            config["mul_experts"] = MulConfig(
                memory_config=output_memory_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )

        return config

    @classmethod
    def _get_prefill_w2_program_config(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        grid_size = cfg["mesh_device"].compute_with_storage_grid_size()
        # tt-metal#44280: keep the W2 prefill matmul on a small core grid with
        # one K tile per block. FP16 destination accumulation keeps this 2x7
        # layout within L1 while reducing simultaneous matmul pressure.
        target_grid_x = min(grid_size.x, 2)
        target_grid_y = min(grid_size.y, 7)
        num_cores = target_grid_x * target_grid_y
        _, _, num_tokens, _ = x.shape
        hidden_size = cfg["w2_experts"].input_tensor_b.shape[-2]

        per_core_M = math.ceil(num_tokens / ttnn.TILE_SIZE)
        per_core_N = math.ceil(math.ceil(hidden_size / num_cores) / ttnn.TILE_SIZE)

        in0_block_w = 1
        out_subblock_w = 1
        out_subblock_h = 1

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(target_grid_x, target_grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    @staticmethod
    @contextmanager
    def _prefill_w2_stagger_env():
        # See tt-metal#44280: add a small odd-row stagger only while compiling
        # prefill W2, further reducing simultaneous matmul pressure.
        overrides = {"TT_MM_STAGGER_TYPE": "2", "TT_MM_STAGGER_VALUE": "10000"}
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

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
        if cfg["input_memory_config"] == ttnn.DRAM_MEMORY_CONFIG:
            with cls._prefill_w2_stagger_env():
                output = ttnn.linear(
                    activated,
                    program_config=cls._get_prefill_w2_program_config(activated, cfg),
                    **cfg["w2_experts"],
                )
        else:
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
