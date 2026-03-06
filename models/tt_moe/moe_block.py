# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn

# GPT-OSS components (optional backend)
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.tt_moe.collectives.ccl import CCL

# DeepSeek components (default)
from models.tt_moe.components.experts.routed_experts import RoutedExperts as MoEExperts

# MoE Preamble - intermediate processing between router and experts
from models.tt_moe.components.moe_preamble import MoEPreamble, MoEPreambleConfig
from models.tt_moe.components.routers.grouped_topk_router import GroupedTopKRouter as MoEGate

# Unified expert configuration
from models.tt_moe.config import create_deepseek_expert_config, create_gptoss_expert_config
from models.tt_moe.config.moe_unified_config import (
    MoEUnifiedConfig,
    get_deepseek_config,
    get_gptoss_decode_config,
    get_gptoss_prefill_config,
)


class MoEBlock(SharedStateAddOn, AbstractModule):
    """MoE module supporting both DeepSeek and GPT-OSS backends.

    Args:
        backend: Either "deepseek" (default) or "gptoss" to select the implementation.
                DeepSeek uses GroupedTopKRouter + RoutedExperts.
                GPT-OSS uses TopKRouter + ThroughputExperts.

    See the `AbstractModule` docstring for additional usage info.
    """

    @classmethod
    def _convert_gptoss_to_deepseek_weights(cls, state_dict: dict, hf_config: PretrainedConfig) -> dict:
        """Convert GPT-OSS weight naming to DeepSeek format.

        GPT-OSS format:
        - gate_up_proj: [num_experts, hidden_size, 2*intermediate_size] (fused, interleaved)
        - down_proj: [num_experts, intermediate_size, hidden_size]

        DeepSeek format:
        - experts.{i}.gate_proj.weight: [intermediate_size, hidden_size]
        - experts.{i}.up_proj.weight: [intermediate_size, hidden_size]
        - experts.{i}.down_proj.weight: [hidden_size, intermediate_size]
        """
        if not state_dict or len(state_dict) == 0:
            return state_dict

        # INSTRUMENTATION: Log all keys and shapes in the state dict
        logger.info(f"GPT-OSS state_dict keys: {list(state_dict.keys())}")
        for key, value in state_dict.items():
            if hasattr(value, "shape"):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        # Check if conversion is needed - look for individual expert indices
        if any("experts.0." in key for key in state_dict.keys()):
            # Already in DeepSeek format (has individual expert indices)
            logger.info("State dict already in DeepSeek format")
            return state_dict

        logger.info("Converting GPT-OSS weights to DeepSeek format")
        converted = {}

        # Get dimensions
        num_experts = getattr(hf_config, "num_local_experts", hf_config.n_routed_experts)
        logger.info(f"Number of experts to convert: {num_experts}")

        # Handle GPT-OSS format with "experts." prefix
        if "experts.gate_up_proj" in state_dict:
            # Fused gate/up weights - need to unfuse and reshape
            gate_up = state_dict["experts.gate_up_proj"]
            # gate_up shape: [num_experts, hidden_size, 2*intermediate_size]
            logger.info(f"Processing fused experts.gate_up_proj with shape {gate_up.shape}")

            # The weights are NOT interleaved in GPT-OSS!
            # They are concatenated: [gate | up]
            intermediate_size = gate_up.shape[-1] // 2

            for i in range(num_experts):
                # Extract expert i's weights
                expert_gate_up = gate_up[i]  # [hidden_size, 2*intermediate_size]

                # Split concatenated weights: first half is gate, second half is up
                gate_weight = expert_gate_up[:, :intermediate_size]  # [hidden_size, intermediate_size]
                up_weight = expert_gate_up[:, intermediate_size:]  # [hidden_size, intermediate_size]

                # Transpose to DeepSeek format
                converted[f"experts.{i}.gate_proj.weight"] = gate_weight.T  # [intermediate_size, hidden_size]
                converted[f"experts.{i}.up_proj.weight"] = up_weight.T  # [intermediate_size, hidden_size]

                # Add weight_scale_inv with ones (DeepSeek expects these for quantization)
                converted[f"experts.{i}.gate_proj.weight_scale_inv"] = torch.ones(1)
                converted[f"experts.{i}.up_proj.weight_scale_inv"] = torch.ones(1)
        elif "experts.gate_proj" in state_dict:
            # Separate gate and up projections with experts prefix
            gate_proj = state_dict["experts.gate_proj"]  # [num_experts, hidden_size, intermediate_size]
            up_proj = state_dict["experts.up_proj"]  # [num_experts, hidden_size, intermediate_size]

            for i in range(num_experts):
                converted[f"experts.{i}.gate_proj.weight"] = gate_proj[i].T
                converted[f"experts.{i}.up_proj.weight"] = up_proj[i].T
                converted[f"experts.{i}.gate_proj.weight_scale_inv"] = torch.ones(1)
                converted[f"experts.{i}.up_proj.weight_scale_inv"] = torch.ones(1)

        # Handle down projection
        if "experts.down_proj" in state_dict:
            down_proj = state_dict["experts.down_proj"]  # [num_experts, hidden_size, intermediate_size]
            logger.info(f"Processing experts.down_proj with shape {down_proj.shape}")

            # GPT-OSS down_proj shape is [num_experts, intermediate_size, hidden_size]
            # But the actual shape is [num_experts, hidden_size, intermediate_size] based on the log
            # Let me verify and handle both cases

            for i in range(num_experts):
                # The weights are already transposed in GPT-OSS
                # GPT-OSS: [num_experts, hidden_size, intermediate_size] (based on log showing [128, 2880, 2880])
                # DeepSeek expects: [hidden_size, intermediate_size]
                # So we just extract without transpose
                converted[f"experts.{i}.down_proj.weight"] = down_proj[i]  # [hidden_size, intermediate_size]
                converted[f"experts.{i}.down_proj.weight_scale_inv"] = torch.ones(1)
        elif "down_proj" in state_dict:
            # Handle case without experts prefix (shouldn't happen based on logs but keep for completeness)
            down_proj = state_dict["down_proj"]
            for i in range(num_experts):
                converted[f"experts.{i}.down_proj.weight"] = down_proj[i].T
                converted[f"experts.{i}.down_proj.weight_scale_inv"] = torch.ones(1)

        # Copy any other weights (like router weights)
        for key, value in state_dict.items():
            if not key.startswith("experts.") and key not in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"]:
                converted[key] = value
                logger.info(f"Copying non-expert weight: {key}")

        logger.info(f"Conversion complete. Converted keys: {list(converted.keys())[:10]}...")  # Show first 10 keys
        logger.info(f"Total converted weights: {len(converted)}")
        return converted

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
        backend: Literal["deepseek", "gptoss"] = "deepseek",
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"MoE expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts
        assert state_dict is not None

        if backend == "deepseek":
            # Use DeepSeek's GroupedTopKRouter and RoutedExperts
            return {
                "moe_gate": MoEGate.convert_weights(
                    hf_config, (state_dict,), output_path / "moe_gate", mesh_device, "gate."
                ),
                "moe_experts": MoEExperts.convert_weights(
                    hf_config, (state_dict,), output_path / "moe_experts", mesh_device
                ),
            }
        elif backend == "gptoss":
            # For GPT-OSS unified path, convert weight naming from GPT-OSS to DeepSeek format
            logger.info(
                f"[convert_weights] Processing GPT-OSS backend with state_dict keys: {list(state_dict.keys())[:5]}"
            )
            converted_state_dict = cls._convert_gptoss_to_deepseek_weights(state_dict, hf_config)
            logger.info(f"[convert_weights] After conversion, have {len(converted_state_dict)} weights")

            # Use the same expert weight converter as DeepSeek with converted weights
            return {
                "moe_gate": {},  # Empty dict for GPT-OSS router (handled separately)
                "moe_experts": MoEExperts.convert_weights(
                    hf_config, (converted_state_dict,), output_path / "moe_experts", mesh_device
                )
                if converted_state_dict
                else {},  # Use converted weights or empty if no state_dict
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        backend: Literal["deepseek", "gptoss"] = "deepseek",
    ) -> ModelState:
        """Create shared model state containing tensors that are constant across all instances.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            backend: Either "deepseek" or "gptoss" to select implementation
        Returns:
            ModelState containing shared tensors
        """
        num_devices = mesh_device.get_num_devices()

        # Get num_experts_per_device based on backend
        if backend == "deepseek":
            num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        else:  # gptoss
            # GPT-OSS distributes experts across devices
            # Use n_routed_experts if num_local_experts is not available
            num_experts = getattr(hf_config, "num_local_experts", hf_config.n_routed_experts)
            num_experts_per_device = num_experts // num_devices
        num_dispatch_device_rows = mesh_device.shape[0]

        expert_mapping_tensors = ttnn.from_torch(
            torch.eye(num_devices, dtype=torch.int32)
            .repeat_interleave(num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, hf_config.n_routed_experts), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return {
            "expert_mapping_tensors": expert_mapping_tensors,
            "remap_topk_mask": remap_topk_mask,
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        ccl: CCL,
    ) -> ModelState:
        """Create model state containing CCL-related communication configurations.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL instance for communication configuration
        Returns:
            ModelState containing CCL configurations
        """
        # Store CCL object for runtime semaphore initialization
        return {
            # CCL-specific parameters (semaphores and num_links)
            "all_to_all_dispatch": {
                "num_links": 4,
            },
            "all_to_all_combine": {
                "num_links": 4,
            },
            "ccl": ccl,
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
        topk_fallback: bool = False,
        backend: Literal["deepseek", "gptoss"] = "deepseek",
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            mode: Either "decode" or "prefill"
            topk_fallback: Whether to use fallback for topk operation
            backend: Either "deepseek" or "gptoss" to select implementation
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        # Get num_experts_per_device based on backend
        if backend == "deepseek":
            num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        else:  # gptoss
            # GPT-OSS distributes experts across devices
            # Use n_routed_experts if num_local_experts is not available
            num_experts = getattr(hf_config, "num_local_experts", hf_config.n_routed_experts)
            num_experts_per_device = num_experts // mesh_device.get_num_devices()

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            USERS_PER_ROW = 32
            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            # For GPT-OSS backend, use DRAM memory config to avoid core grid issues
            if backend == "gptoss":
                input_output_memory_config = ttnn.DRAM_MEMORY_CONFIG
            else:  # deepseek
                input_output_memory_config = ttnn.create_sharded_memory_config(
                    shape=(USERS_PER_ROW, HIDDEN_SIZE // TP_SIZE),
                    core_grid=ttnn.CoreGrid(y=7, x=4),
                    strategy=ttnn.ShardStrategy.WIDTH,
                )

            # Construct the config
            config = {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "backend": backend,  # Store backend for runtime
            }

            if backend == "deepseek":
                # DeepSeek-specific configuration
                config.update(
                    {
                        "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
                        "all_to_all_dispatch_output_memory_config": memory_config,
                        "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                        "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
                        "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                        "all_to_all_combine_output_memory_config": memory_config,
                        "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
                        "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
                        "input_memory_config": input_output_memory_config,
                        "output_memory_config": input_output_memory_config,
                        "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
                        "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                        "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                            cluster_axis=1,
                            dim=3,
                            memory_config=input_output_memory_config,
                        ),
                        "revert_tp": AllGatherAsyncConfig(
                            mesh_device=MeshDeviceStub(mesh_device.shape),
                            dim=-1,  # Last dimension
                            # memory_config=ttnn.create_sharded_memory_config(  # Bad PCC
                            #     shape=(USERS_PER_ROW, HIDDEN_SIZE),
                            #     core_grid=ttnn.CoreGrid(y=7, x=8),
                            #     strategy=ttnn.ShardStrategy.WIDTH,
                            # ),
                            memory_config=memory_config,
                            cluster_axis=1,
                        ),
                    }
                )
            else:  # gptoss
                # GPT-OSS-specific configuration
                # For unified path, we need the expert configs similar to DeepSeek
                config.update(
                    {
                        "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                        "use_throughput_experts": True,
                        "input_memory_config": input_output_memory_config,
                        "output_memory_config": input_output_memory_config,
                    }
                )

            return config
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            # Construct the config
            config = {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "backend": backend,  # Store backend for runtime
            }

            if backend == "deepseek":
                # DeepSeek-specific configuration
                config.update(
                    {
                        "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
                        "all_to_all_dispatch_output_memory_config": memory_config,
                        "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                        "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
                        "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                        "all_to_all_combine_output_memory_config": memory_config,
                        "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
                        "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
                        "input_memory_config": memory_config,
                        "output_memory_config": memory_config,
                        "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
                        "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                        "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                            cluster_axis=1,
                            dim=3,
                            memory_config=memory_config,
                        ),
                        "revert_tp": AllGatherAsyncConfig(
                            mesh_device=MeshDeviceStub(mesh_device.shape),
                            dim=-1,  # Last dimension
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            cluster_axis=1,
                        ),
                    }
                )
            else:  # gptoss
                # GPT-OSS-specific configuration for prefill
                # For unified path, include expert configs
                config.update(
                    {
                        "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                        "use_throughput_experts": True,
                        "input_memory_config": memory_config,
                        "output_memory_config": memory_config,
                    }
                )

            return config

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        backend: Literal["deepseek", "gptoss"] = "deepseek",
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode", topk_fallback=topk_fallback, backend=backend)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        backend: Literal["deepseek", "gptoss"] = "deepseek",
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill", topk_fallback=topk_fallback, backend=backend)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
        num_dispatch_devices = int(cfg.get("num_dispatch_devices", 1))
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            return cls._forward_chunked_prefill(x, cfg, chunk_size)
        return cls._forward_impl(x, cfg)

    @classmethod
    def _forward_chunked_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig, chunk_size: int) -> ttnn.Tensor:
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks: list[ttnn.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            output_chunks.append(cls._forward_impl(x_chunk, cfg))
            ttnn.deallocate(x_chunk)

        if len(output_chunks) == 1:
            return output_chunks[0]
        output = ttnn.concat(output_chunks, dim=2)
        for chunk in output_chunks:
            ttnn.deallocate(chunk)
        return output

    @classmethod
    def _forward_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Validate input dimensions
        hidden_size = cfg["hidden_size"]
        mesh_device = cfg.get("mesh_device")
        tp_size = mesh_device.shape[1] if mesh_device else 1

        x_dim = x.shape[-1]
        expected_dims = [hidden_size, hidden_size // tp_size] if tp_size > 1 else [hidden_size]

        if x_dim not in expected_dims:
            raise ValueError(
                f"MoE: Unexpected input dimension {x_dim}. Expected one of {expected_dims}. "
                f"(hidden_size={hidden_size}, tp_size={tp_size})"
            )

        # breakpoint()
        ccl = cfg["ccl"]  # CCL runtime initialization in execution order
        backend = cfg.get("backend", "deepseek")

        # Shape handling differs between backends:
        # - DeepSeek: [1, 1, batch/seq, hidden] - uses x.shape[-2] for batch_size_per_device
        # - GPT-OSS decode: [batch, 1, 1, hidden] - uses x.shape[0] for batch_size_per_device
        # - GPT-OSS prefill: [1, 1, seq, hidden] - uses x.shape[-2] for seq_len
        if backend == "gptoss":
            if x.shape[0] > 1:
                # GPT-OSS decode mode with high throughput (batch dimension at dim 0)
                batch_size_per_device = x.shape[0]
                seq_len = 1
            else:
                # GPT-OSS prefill mode or decode with batch=1
                # Check if this is prefill by looking at shape[-2]
                if x.shape[-2] > 1:
                    # Prefill: sequence dimension is at x.shape[-2]
                    # For all-to-all ops, we treat seq as batch
                    seq_len = x.shape[-2]  # Actual sequence length for mode detection
                    batch_size_per_device = x.shape[-2]  # Treat seq as batch for all-to-all ops
                else:
                    # Decode with batch=1
                    seq_len = 1
                    batch_size_per_device = 1
        else:
            # DeepSeek: always treats dimension -2 as batch/seq
            seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
            batch_size_per_device = x.shape[
                -2
            ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device

        # For GPT-OSS prefill, batch_size_per_device is actually seq_len (we treat seq as batch for all-to-all)
        # But the actual batch size is still 1
        if backend == "gptoss" and seq_len > 1:
            # Prefill mode - actual batch is 1, but we treat seq as batch for all-to-all
            batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # This is really total seq_len
            actual_batch_size = 1  # True batch size for prefill
        else:
            batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size
            actual_batch_size = batch_size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # MoE Preamble - backend-specific preprocessing
        preamble_config = MoEPreambleConfig(
            backend=cfg.get("backend", "deepseek"),
            hidden_size=cfg["hidden_size"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            num_experts_per_device=cfg["num_experts_per_device"],
            num_devices=cfg["num_devices"],
            num_dispatch_devices=cfg["num_dispatch_devices"],
            batch_size_per_device=batch_size_per_device,
            seq_len=seq_len,
            topk_weights_repeat=cfg.get("topk_weights_repeat")
            if cfg.get("backend", "deepseek") == "deepseek"
            else None,
            use_throughput_experts=cfg.get("backend", "deepseek") == "gptoss",
        )

        x_processed, topk_experts_weights_processed, topk_experts_indices_processed = MoEPreamble.forward(
            x, topk_experts_weights, topk_experts_indices, preamble_config
        )

        # MOE
        post_combine_output_tensor = cls._fwd_moe(
            x_processed,
            topk_experts_indices_processed,
            topk_experts_weights_processed,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )

        # Note: reduce_scatter is handled by the caller (decoder block or test)

        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Unified router interface for both backends.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_size]
            cfg: Runtime configuration dict

        Returns:
            Tuple of (weights, indices) both of shape [1, 1, seq_len, K]
        """
        backend = cfg.get("backend", "deepseek")

        if backend == "deepseek":
            # Use existing GroupedTopKRouter (already has correct interface)
            weights, indices = MoEGate.forward(x, cfg["moe_gate"])
        else:  # gptoss
            # Get or create router instance
            if "_gptoss_router" not in cfg:
                # Get router state dict from config
                experts_config = cfg.get("moe_experts", {})
                router_state_dict = experts_config.get("router_state_dict", {})

                if not router_state_dict:
                    # Create mock router for testing
                    logger.warning("Creating mock GPT-OSS router with random weights for testing")
                    num_experts = cfg["num_experts_per_device"] * cfg["num_devices"]
                    hidden_size = x.shape[-1]
                    router_state_dict = {
                        "weight": torch.randn(num_experts, hidden_size),
                        "bias": torch.randn(num_experts),
                    }

                # Create router using factory method
                cfg["_gptoss_router"] = TopKRouter.from_config(cfg, x.device(), router_state_dict)

            # Call unified forward interface
            weights, indices = cfg["_gptoss_router"].forward(x, cfg)

        return weights, indices

    @classmethod
    def _fwd_moe_unified(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: dict,
        config: MoEUnifiedConfig,
    ) -> ttnn.Tensor:
        """Unified MoE forward pass supporting both GPT-OSS and DeepSeek backends.

        This function provides a single configurable implementation for MoE forward pass
        that can handle both GPT-OSS and DeepSeek requirements through the MoEUnifiedConfig.

        Args:
            x: Input tensor after preamble processing
            topk_experts_indices: Expert routing indices
            topk_experts_weights: Expert routing weights
            cfg: Runtime configuration dictionary
            config: Unified MoE configuration

        Returns:
            Output tensor after MoE processing
        """
        from loguru import logger

        # Determine dimensions based on expert type
        if config.expert_type.startswith("throughput"):
            # GPT-OSS mode: x is [batch, 1, 1, hidden] for decode or [1, 1, seq, hidden] for prefill
            batch_size_per_device = x.shape[0] if x.shape[0] > 1 else x.shape[2]
            seq_len = 1 if x.shape[2] == 1 else x.shape[2]
            actual_hidden_size = x.shape[-1]
        else:
            # DeepSeek mode: x is [batch, 1, seq, hidden]
            batch_size_per_device = x.shape[0]
            seq_len = x.shape[2]
            actual_hidden_size = x.shape[-1]

        total_tokens = batch_size_per_device * seq_len
        output_chunks = []

        # Determine chunking strategy
        if config.enable_chunking:
            chunk_size = config.chunk_size or batch_size_per_device
            if config.chunk_dim == 0:  # Batch chunking (DeepSeek)
                num_chunks = (batch_size_per_device + chunk_size - 1) // chunk_size
            else:  # Sequence chunking (GPT-OSS prefill)
                num_chunks = (seq_len + chunk_size - 1) // chunk_size
        else:
            num_chunks = 1
            chunk_size = batch_size_per_device if config.chunk_dim == 0 else seq_len

        # Main processing loop (single iteration if no chunking)
        for chunk_idx in range(num_chunks):
            # Step 1: Prepare chunk inputs
            if config.enable_chunking:
                chunk_start = chunk_idx * chunk_size
                if config.chunk_dim == 0:
                    # Batch chunking (DeepSeek style)
                    chunk_end = min((chunk_idx + 1) * chunk_size, batch_size_per_device)
                    x_chunk = ttnn.slice(x, [chunk_start, 0, 0, 0], [chunk_end, 1, seq_len, actual_hidden_size])
                    indices_chunk = ttnn.slice(
                        topk_experts_indices,
                        [chunk_start, 0, 0, 0] if len(topk_experts_indices.shape) == 4 else [chunk_start, 0],
                        [chunk_end, 1, seq_len, topk_experts_indices.shape[-1]]
                        if len(topk_experts_indices.shape) == 4
                        else [chunk_end, topk_experts_indices.shape[-1]],
                    )
                else:  # Sequence chunking
                    chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
                    x_chunk = ttnn.slice(x, [0, 0, chunk_start, 0], [1, 1, chunk_end, actual_hidden_size])
                    indices_chunk = topk_experts_indices  # Indices don't change for seq chunking
            else:
                x_chunk = x
                indices_chunk = topk_experts_indices

            # Step 2: Prepare for all_to_all_dispatch
            if config.expert_type.startswith("throughput"):
                # GPT-OSS format: reshape to [1, 1, tokens, hidden]
                if len(x_chunk.shape) == 4 and x_chunk.shape[0] > 1:
                    # Decode: [batch, 1, 1, hidden] -> [1, 1, batch, hidden]
                    x_dispatch = ttnn.reshape(x_chunk, [1, 1, x_chunk.shape[0], actual_hidden_size])
                else:
                    # Prefill: already [1, 1, seq, hidden]
                    x_dispatch = x_chunk
            else:
                # DeepSeek format: [batch, 1, seq, hidden] -> [1, 1, batch*seq, hidden]
                batch_chunk = x_chunk.shape[0]
                x_dispatch = ttnn.reshape(x_chunk, [1, 1, batch_chunk * seq_len, actual_hidden_size])

            # Prepare indices for dispatch (ensure 4D and match x_dispatch shape)
            if len(indices_chunk.shape) == 2:
                # 2D indices: [batch, K] -> [1, 1, batch, K]
                # Convert to uint32 for reshaping
                indices_chunk = ttnn.typecast(indices_chunk, dtype=ttnn.uint32)
                indices_dispatch = ttnn.reshape(
                    indices_chunk, shape=(1, 1, indices_chunk.shape[0], indices_chunk.shape[1])
                )
                # Convert back to uint16 for all_to_all
                indices_dispatch = ttnn.typecast(indices_dispatch, dtype=ttnn.uint16)
            elif len(indices_chunk.shape) == 4:
                # 4D indices: need to reshape to match x_dispatch
                if config.expert_type == "routed":  # DeepSeek
                    # DeepSeek: [batch, 1, seq, K] -> [1, 1, batch*seq, K]
                    batch_chunk = indices_chunk.shape[0]
                    num_experts_per_tok = indices_chunk.shape[-1]
                    # Reshape directly without typecast since indices are already uint16
                    indices_dispatch = ttnn.reshape(
                        indices_chunk, shape=(1, 1, batch_chunk * seq_len, num_experts_per_tok)
                    )
                else:
                    # GPT-OSS: keep as-is or reshape accordingly
                    indices_dispatch = indices_chunk
            else:
                indices_dispatch = indices_chunk

            # Convert to ROW_MAJOR for all_to_all
            x_dispatch = ttnn.to_layout(x_dispatch, ttnn.ROW_MAJOR_LAYOUT)
            indices_dispatch = ttnn.to_layout(indices_dispatch, ttnn.ROW_MAJOR_LAYOUT)

            # Step 3: all_to_all_dispatch
            dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
                x_dispatch,
                indices_dispatch,
                cfg["expert_mapping_tensors"],
                **config.dispatch_config,
            )
            ttnn.deallocate(x_dispatch)
            ttnn.deallocate(indices_dispatch)

            # Step 4: Prepare for expert computation
            total_tokens_global = dispatch_output.shape[2]
            dispatch_reshaped = ttnn.reshape(dispatch_output, [1, 1, total_tokens_global, dispatch_output.shape[-1]])

            # Repeat for all experts on device
            # Use memory config from moe_experts if available, otherwise fall back to intermediate_memory_config
            expert_memory_config = cfg["moe_experts"].get("input_memory_config", config.intermediate_memory_config)

            dispatch_repeated = ttnn.repeat(
                dispatch_reshaped,
                ttnn.Shape([1, cfg["num_experts_per_device"], 1, 1]),
                memory_config=expert_memory_config,
            )
            dispatch_tiled = ttnn.to_layout(dispatch_repeated, ttnn.TILE_LAYOUT, memory_config=expert_memory_config)

            ttnn.deallocate(dispatch_output)
            ttnn.deallocate(dispatch_reshaped)
            ttnn.deallocate(dispatch_repeated)

            # Step 5: Expert computation (backend-specific)
            if config.expert_type == "routed":
                # DeepSeek: Use MoEExperts
                if "moe_experts" not in cfg:
                    logger.warning("moe_experts not in cfg, creating it")
                    # This shouldn't happen in normal flow
                experts_output = MoEExperts._forward(dispatch_tiled, cfg["moe_experts"])

            elif config.expert_type in ["throughput_decode", "throughput_prefill"]:
                # GPT-OSS: Use expert-only computation
                # First check if we need to load GPT-OSS weights
                if "gptoss_expert_weights" not in cfg:
                    # Extract state dicts from config
                    experts_state_dict = cfg.get("moe_experts", {}).get("experts_state_dict", {})

                    if experts_state_dict:
                        from models.demos.gpt_oss.tt.experts_throughput.config import (
                            ThroughputExpertConfig,
                            ThroughputProgramConfig,
                        )
                        from models.demos.gpt_oss.tt.experts_throughput.weights import load_throughput_expert_weights

                        # Get unified config from cfg if available
                        unified_config = cfg.get("unified_expert_config")

                        # Create ThroughputExpertConfig
                        expert_config = ThroughputExpertConfig(
                            num_experts=cfg["num_devices"] * cfg["num_experts_per_device"],
                            num_devices=cfg["num_devices"],
                            hidden_size=cfg["hidden_size"],
                            intermediate_size=unified_config.intermediate_size
                            if unified_config
                            else cfg.get("intermediate_size", cfg["hidden_size"] * 4),
                            num_experts_per_tok=cfg["num_experts_per_tok"],
                            alpha=unified_config.activation.swiglu_alpha
                            if unified_config and unified_config.activation
                            else 1.702,
                            swiglu_limit=unified_config.activation.swiglu_limit
                            if unified_config and unified_config.activation
                            else 7.0,
                        )

                        # Load weights
                        cfg["gptoss_expert_weights"] = load_throughput_expert_weights(
                            mesh_device=x.device(),
                            config=expert_config,
                            state_dict=experts_state_dict,
                            weight_dtype=unified_config.weight_dtype
                            if unified_config and hasattr(unified_config, "weight_dtype")
                            else ttnn.bfloat4_b,
                            tensor_cache_path=cfg.get("moe_experts", {}).get("cache_path"),
                        )
                        cfg["gptoss_expert_config"] = expert_config
                        cfg["gptoss_program_config"] = ThroughputProgramConfig()

                # Use expert-only computation
                from models.demos.gpt_oss.tt.experts_throughput.expert_only import expert_mlp_compute_only

                experts_output = expert_mlp_compute_only(
                    dispatch_tiled,
                    cfg["gptoss_expert_weights"],
                    cfg["gptoss_expert_config"],
                    config.intermediate_memory_config,
                    cfg["gptoss_program_config"],
                )

            else:
                raise ValueError(f"Unknown expert_type: {config.expert_type}")

            # Step 6: Prepare for all_to_all_combine
            experts_output_rm = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output_reshaped = ttnn.reshape(
                experts_output_rm, [cfg["num_experts_per_device"], 1, total_tokens_global, experts_output.shape[-1]]
            )
            ttnn.deallocate(experts_output)
            ttnn.deallocate(experts_output_rm)

            # Reshape dispatch metadata if needed
            if dispatch_metadata.shape[0] != 1:
                dispatch_metadata = ttnn.reshape(
                    dispatch_metadata,
                    shape=(
                        1,
                        dispatch_metadata.shape[0] * dispatch_metadata.shape[1],
                        dispatch_metadata.shape[2],
                        dispatch_metadata.shape[3],
                    ),
                )

            # Step 7: all_to_all_combine
            combine_output = ttnn.all_to_all_combine(
                experts_output_reshaped,
                dispatch_metadata,
                cfg["expert_mapping_tensors"],
                **config.combine_config,
            )
            ttnn.deallocate(experts_output_reshaped)
            ttnn.deallocate(dispatch_metadata)

            # Step 8: Apply routing weights and sum
            # Reshape combine output for weight multiplication
            combine_reshaped = ttnn.reshape(
                combine_output, shape=(cfg["num_experts_per_tok"], 1, combine_output.shape[2], combine_output.shape[3])
            )
            combine_tiled = ttnn.to_layout(combine_reshaped, ttnn.TILE_LAYOUT)
            ttnn.deallocate(combine_output)

            # Get appropriate weights chunk
            if config.enable_chunking and config.chunk_dim == 0:
                # Extract weights for this batch chunk
                token_start = chunk_start * seq_len
                token_end = chunk_end * seq_len
                if len(topk_experts_weights.shape) == 4:
                    weights_chunk = ttnn.slice(
                        topk_experts_weights,
                        [0, 0, token_start, 0],
                        [topk_experts_weights.shape[0], 1, token_end, topk_experts_weights.shape[-1]],
                    )
                else:
                    # Handle 2D weights
                    weights_chunk = topk_experts_weights[token_start:token_end]
            else:
                weights_chunk = topk_experts_weights

            # Prepare weights for multiplication (ensure proper shape)
            if config.expert_type.startswith("throughput"):
                # For GPT-OSS, prepare weights similar to existing implementation
                # Weights need to be [K, 1, batch_size, 1] to broadcast
                if len(weights_chunk.shape) == 2:
                    weights_rm = ttnn.to_layout(weights_chunk, ttnn.ROW_MAJOR_LAYOUT)
                    weights_rm = ttnn.reshape(weights_rm, [1, 1, weights_rm.shape[0], weights_rm.shape[1]])
                    weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0))
                    weights_prepared = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(weights_rm)
                elif len(weights_chunk.shape) == 4:
                    # Handle 4D weights from unified router interface
                    # Input is [1, 1, batch_seq, K], need [K, 1, batch_seq, 1]
                    weights_rm = ttnn.to_layout(weights_chunk, ttnn.ROW_MAJOR_LAYOUT)
                    weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0))  # [K, 1, batch_seq, 1]
                    weights_prepared = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(weights_rm)
                else:
                    weights_prepared = weights_chunk
            else:
                weights_prepared = weights_chunk

            # Multiply by routing weights
            weighted_output = ttnn.mul(combine_tiled, weights_prepared, memory_config=config.output_memory_config)
            ttnn.deallocate(combine_tiled)
            if weights_prepared is not weights_chunk:
                ttnn.deallocate(weights_prepared)

            # Sum across experts dimension
            chunk_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
            ttnn.deallocate(weighted_output)
            output_chunks.append(chunk_output)

        # Step 9: Concatenate chunks if needed
        if len(output_chunks) == 1:
            output = output_chunks[0]
        else:
            concat_dim = 2  # Concatenate along token dimension
            output = ttnn.concat(output_chunks, dim=concat_dim)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        # Step 10: Optional all-reduce (GPT-OSS only)
        if config.enable_all_reduce:
            output = ttnn.all_reduce(
                output,
                **config.all_reduce_config,
            )

        # Step 11: Final reshape based on backend
        if config.expert_type.startswith("throughput") and seq_len == 1:
            # GPT-OSS decode: reshape from [1, 1, batch, hidden] to [batch, 1, 1, hidden]
            output = ttnn.reshape(output, [batch_size_per_device, 1, 1, output.shape[-1]])

        # Step 12: Convert to output memory config if needed
        if output.memory_config() != config.output_memory_config:
            output = ttnn.to_memory_config(output, config.output_memory_config)

        return output

    @classmethod
    def _fwd_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Route to unified implementation based on backend.

        This function creates the appropriate configuration for either GPT-OSS or DeepSeek
        backend and calls the unified MoE forward pass.
        """
        backend = cfg.get("backend", "deepseek")
        mode = "decode" if seq_len == 1 else "prefill"

        # Get or create hf_config
        hf_config = cfg.get("hf_config")

        # Create unified config based on backend and mode
        if backend == "gptoss":
            # Ensure hf_config has necessary fields
            if not hf_config:
                hf_config = type(
                    "obj",
                    (object,),
                    {
                        "hidden_size": cfg["hidden_size"],
                        "intermediate_size": cfg.get("intermediate_size", 2880),  # GPT-OSS default
                        "num_experts_per_tok": cfg["num_experts_per_tok"],
                    },
                )
                logger.warning(f"GPT-OSS: Using fallback config with intermediate_size={hf_config.intermediate_size}")

            # Create GPT-OSS expert config for loading weights
            unified_config = create_gptoss_expert_config(
                hf_config,
                mode=mode,
                num_experts_per_device=cfg["num_experts_per_device"],
                use_fused_weights=cfg.get("use_fused_gate_up", False),
            )
            cfg["unified_expert_config"] = unified_config

            # Get MoE unified config
            if mode == "decode":
                config = get_gptoss_decode_config()
            else:
                config = get_gptoss_prefill_config()

            # Use the unified forward path for GPT-OSS
            return cls._fwd_moe_unified(x, topk_experts_indices, topk_experts_weights, cfg, config)

        # DeepSeek backend - also use unified implementation
        if not hf_config:
            # Create a minimal config object for DeepSeek if not provided
            hf_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": cfg["hidden_size"],
                    "moe_intermediate_size": cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 4608)),
                    "num_experts_per_tok": cfg["num_experts_per_tok"],
                },
            )

        # Create DeepSeek expert config for loading weights
        unified_config = create_deepseek_expert_config(
            hf_config, mode=mode, num_experts_per_device=cfg["num_experts_per_device"]
        )
        cfg["unified_expert_config"] = unified_config

        # Get MoE unified config for DeepSeek
        config = get_deepseek_config(mode)

        # Adjust chunk size if specified in cfg
        if "moe_chunk_size" in cfg:
            config.chunk_size = cfg["moe_chunk_size"]

        # Use the unified forward path for DeepSeek
        return cls._fwd_moe_unified(x, topk_experts_indices, topk_experts_weights, cfg, config)

    @classmethod
    def _fwd_reduce_scatter(
        cls, post_combine_output_tensor: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig, ccl: CCL
    ) -> ttnn.Tensor:
        return ttnn.experimental.reduce_scatter_minimal_async(
            post_combine_output_tensor, **ccl.populate_reduce_scatter_runtime_args(cfg["final_output_reduce_scatter"])
        )

    @classmethod
    def _fwd_all_gather(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        return ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))

    @classmethod
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled (DeepSeek only)
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward(x, cfg)

        # Handle reduce_scatter if tensor parallel is enabled (DeepSeek only)
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled (DeepSeek only)
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward(x, cfg)

        # Handle reduce_scatter if tensor parallel is enabled (DeepSeek only)
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output


# Backward compatibility alias
MoE = MoEBlock
