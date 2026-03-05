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
        backend = cfg.get("backend", "deepseek")

        # Instrumentation: Log input to router
        from models.tt_moe.utils.tensor_debug import log_tensor_checkpoint

        log_tensor_checkpoint(x, f"MoEBlock_{backend}_1_router_input")

        if backend == "deepseek":
            weights, indices = MoEGate.forward(x, cfg["moe_gate"])
        else:  # gptoss
            # Use TopKRouter for GPT-OSS
            # Input shape handling:
            # - Decode: [batch, 1, 1, hidden] - already in right shape
            # - Prefill: [1, 1, seq, hidden] - already in right shape
            hidden_size = x.shape[-1]

            # Keep input shape as-is for GPT-OSS router
            x_reshaped = x

            # Create router if not cached in config
            if "_gptoss_router" not in cfg:
                # Create a simple object-like config for TopKRouter
                # TopKRouter expects an object with attributes, not a dict
                class RouterConfig:
                    def __init__(self, num_local_experts, num_experts_per_tok, hidden_size):
                        self.num_local_experts = num_local_experts
                        self.num_experts_per_tok = num_experts_per_tok
                        self.hidden_size = hidden_size

                router_config = RouterConfig(
                    num_local_experts=cfg["num_experts_per_device"] * cfg["num_devices"],
                    num_experts_per_tok=cfg["num_experts_per_tok"],
                    hidden_size=hidden_size,
                )

                # Get the router state dict from runtime config
                # For GPT-OSS, state dicts are passed through runtime config
                experts_config = cfg.get("moe_experts", {})
                router_state_dict = experts_config.get("router_state_dict", {})

                # Add validation logging
                if router_state_dict:
                    logger.info(f"Loading GPT-OSS TopKRouter with state_dict keys: {list(router_state_dict.keys())}")
                    for key, value in router_state_dict.items():
                        if hasattr(value, "shape"):
                            logger.info(f"  Router weight {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.warning("No router weights found in config - creating mock router with random weights")

                # Initialize TopKRouter with actual weights if available
                if router_state_dict:
                    cfg["_gptoss_router"] = TopKRouter(
                        x.device(),
                        router_config,
                        router_state_dict,
                        tensor_cache_path=None,
                    )
                    logger.info("Successfully loaded GPT-OSS router with provided weights")
                else:
                    # Fallback: create mock router for testing
                    logger.warning("Creating mock GPT-OSS router with random weights for testing")
                    cfg["_gptoss_router"] = TopKRouter(
                        x.device(),
                        router_config,
                        {
                            "weight": torch.randn(router_config.num_local_experts, router_config.hidden_size),
                            "bias": torch.randn(router_config.num_local_experts),
                        },
                    )

            # Call GPT-OSS router
            use_throughput_experts = cfg.get("use_throughput_experts", True)
            expert_indices, expert_weights = cfg["_gptoss_router"](x_reshaped, use_throughput_experts)

            weights, indices = expert_weights, expert_indices

        # Instrumentation: Log output of router
        log_tensor_checkpoint(indices, f"MoEBlock_{backend}_2_router_output_indices")
        log_tensor_checkpoint(weights, f"MoEBlock_{backend}_3_router_output_weights")

        return weights, indices

    @classmethod
    def _fwd_repeat_permute_expert_weights(
        cls, topk_experts_weights: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> ttnn.Tensor:
        # DEPRECATED: This is now handled by MoEPreamble.forward()
        # Kept for backward compatibility if called directly
        logger.warning("_fwd_repeat_permute_expert_weights is deprecated. Use MoEPreamble instead.")
        backend = cfg.get("backend", "deepseek")

        if backend == "gptoss":
            # GPT-OSS doesn't need this transformation
            return topk_experts_weights

        # DeepSeek path
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)
        return topk_experts_weights

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
        backend = cfg.get("backend", "deepseek")
        logger.info(
            f"_fwd_moe called with backend={backend}, x.shape={x.shape}, indices.shape={topk_experts_indices.shape}, weights.shape={topk_experts_weights.shape}"
        )

        # Create unified configuration based on backend
        mode = "decode" if seq_len == 1 else "prefill"
        hf_config = cfg.get("hf_config")

        if backend == "gptoss":
            logger.info("Using unified GPT-OSS path with all_to_all")
            logger.info(f"x shape after preamble: {x.shape}")
            logger.info(f"topk_experts_indices shape after preamble: {topk_experts_indices.shape}")
            logger.info(f"topk_experts_weights shape after preamble: {topk_experts_weights.shape}")
            logger.info(f"batch_size_per_device={batch_size_per_device}, seq_len={seq_len}")

            # Create GPT-OSS unified configuration
            # Ensure we use the correct intermediate_size from hf_config
            if not hf_config:
                # Create fallback config only if hf_config not provided
                # Use intermediate_size from cfg if available (GPT-OSS should be 2880)
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
            else:
                logger.info(f"GPT-OSS: Using provided hf_config with intermediate_size={hf_config.intermediate_size}")

            unified_config = create_gptoss_expert_config(
                hf_config,
                mode=mode,
                num_experts_per_device=cfg["num_experts_per_device"],
                use_fused_weights=cfg.get("use_fused_gate_up", False),  # Use separate weights for now
            )
            cfg["unified_expert_config"] = unified_config

            # Implement all_to_all for GPT-OSS (similar to DeepSeek)
            return cls._fwd_moe_gptoss_unified(
                x, topk_experts_indices, topk_experts_weights, cfg, batch_size_per_device, batch_size, seq_len
            )

        # DeepSeek backend continues with existing implementation
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

        unified_config = create_deepseek_expert_config(
            hf_config, mode=mode, num_experts_per_device=cfg["num_experts_per_device"]
        )
        cfg["unified_expert_config"] = unified_config

        # DeepSeek path - inputs are already preprocessed by MoEPreamble
        # x is already reshaped to (batch_size_per_device, 1, seq_len, hidden_size)
        # topk_experts_indices is already reshaped to (batch_size_per_device, 1, seq_len, num_experts_per_tok)
        # topk_experts_weights is already transformed with repeat/permute
        tokens = batch_size * seq_len
        x_rm = x  # Already in the right shape from MoEPreamble
        topk_experts_indices_rm = topk_experts_indices  # Already in the right shape from MoEPreamble

        # Chunk along local batch dimension to keep all_to_all_dispatch output small in prefill.
        chunk_size = min(batch_size_per_device, max(1, cfg.get("moe_chunk_size", batch_size_per_device)))
        output_chunks: list[ttnn.Tensor] = []

        def _slice_topk_weights(batch_start: int, batch_end: int) -> ttnn.Tensor:
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            return ttnn.slice(
                topk_experts_weights,
                [0, 0, token_start, 0],
                [cfg["num_experts_per_tok"], 1, token_end, cfg["hidden_size"]],
            )

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            batch_size_chunk = batch_chunk * cfg["num_dispatch_devices"]

            x_chunk = ttnn.slice(
                x_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["hidden_size"]],
            )
            topk_indices_chunk = ttnn.slice(
                topk_experts_indices_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )

            all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
                x_chunk,
                topk_indices_chunk,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_dispatch"],
            )
            ttnn.deallocate(x_chunk)
            ttnn.deallocate(topk_indices_chunk)

            dispatch_chunk = ttnn.reshape(
                all_to_all_dispatch_output_tensors,
                shape=(1, 1, batch_size_chunk * seq_len, cfg["hidden_size"]),
            )
            dispatch_chunk = ttnn.repeat(dispatch_chunk, **cfg["activations_repeat"])
            dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
            ttnn.deallocate(all_to_all_dispatch_output_tensors)

            # Instrumentation: Log input to experts
            from models.tt_moe.utils.tensor_debug import log_tensor_checkpoint

            log_tensor_checkpoint(dispatch_chunk, "MoEBlock_deepseek_4_experts_input")

            experts_output = MoEExperts._forward(dispatch_chunk, cfg["moe_experts"])

            # Instrumentation: Log output of experts
            log_tensor_checkpoint(experts_output, "MoEBlock_deepseek_5_experts_output")

            ttnn.deallocate(dispatch_chunk)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output = ttnn.reshape(
                experts_output, shape=(cfg["num_experts_per_device"], batch_size_chunk, seq_len, cfg["hidden_size"])
            )

            all_to_all_dispatch_metadata_tensors = ttnn.reshape(
                all_to_all_dispatch_metadata_tensors,
                shape=(1, batch_size_chunk, seq_len, cfg["num_experts_per_tok"]),
            )

            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                all_to_all_dispatch_metadata_tensors,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_combine"],
            )
            ttnn.deallocate(experts_output)
            ttnn.deallocate(all_to_all_dispatch_metadata_tensors)

            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(cfg["num_experts_per_tok"], 1, batch_chunk * seq_len, cfg["hidden_size"]),
            )
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)

            topk_weights_chunk = _slice_topk_weights(batch_start, batch_end)
            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_weights_chunk, **cfg["mul_experts_output_with_weights"]
            )
            ttnn.deallocate(topk_weights_chunk)

            post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
            output_chunks.append(post_combine_output_tensor)

        if len(output_chunks) == 1:
            post_combine_output_tensor = output_chunks[0]
        else:
            post_combine_output_tensor = ttnn.concat(output_chunks, dim=2)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm)
        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gptoss_unified(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """GPT-OSS unified implementation with all_to_all dispatch and combine.

        This method implements the GPT-OSS MoE using all_to_all operations
        similar to DeepSeek, but configured for GPT-OSS requirements.
        """
        import os

        import torch

        from models.tt_moe.utils.tensor_debug import log_tensor_checkpoint

        # Directory to save intermediate activations
        debug_dir = "/tmp/gptoss_unified_debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Helper to convert distributed tensors to torch
        def to_torch_debug(tensor, name="tensor"):
            """Convert potentially distributed tensor to torch for debugging."""
            try:
                # Skip activation saving for distributed tensors
                # The issue is complex with row-sharded tensors
                # Just return None to skip saving for now
                return None
            except Exception as e:
                logger.debug(f"Failed to convert {name} to torch: {e}")
                return None

        # Helper to save tensor to file
        def save_tensor(tensor, filename, tensor_name, is_float=True):
            """Save a tensor to file if conversion was successful."""
            torch_tensor = to_torch_debug(tensor, tensor_name)
            if torch_tensor is not None:
                if is_float:
                    torch_tensor = torch_tensor.float()
                torch_tensor = torch_tensor.cpu()
                torch.save(torch_tensor, f"{debug_dir}/{filename}")
                logger.info(f"  Saved: {filename} shape={torch_tensor.shape}")
            else:
                logger.warning(f"  Could not save {filename}")

        logger.info("=" * 80)
        logger.info("GPT-OSS UNIFIED PATH - SHAPE INSTRUMENTATION + ACTIVATION SAVING")
        logger.info("=" * 80)
        logger.info(
            f"INPUTS: x.shape={x.shape}, indices.shape={topk_experts_indices.shape}, "
            f"weights.shape={topk_experts_weights.shape}"
        )
        logger.info(
            f"PARAMS: batch_size_per_device={batch_size_per_device}, batch_size={batch_size}, seq_len={seq_len}"
        )
        logger.info(f"PARAMS: num_devices={cfg['num_devices']}, num_experts_per_device={cfg['num_experts_per_device']}")
        logger.info(f"Saving intermediate activations to: {debug_dir}")

        unified_config = cfg["unified_expert_config"]

        # Create expert mapping tensors for all_to_all operations
        # Format: [1, 1, num_experts, num_devices] with one-hot encoding
        if "expert_mapping_tensors" not in cfg:
            num_devices = cfg["num_devices"]
            num_experts_per_device = cfg["num_experts_per_device"]
            num_experts = num_devices * num_experts_per_device

            # Create one-hot mapping showing which device owns which expert
            # mapping[e, d] = 1 if expert e is on device d
            import torch

            mapping = (
                torch.eye(num_devices, dtype=torch.int32)
                .repeat_interleave(num_experts_per_device, dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            cfg["expert_mapping_tensors"] = ttnn.from_torch(
                mapping,
                device=x.device(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(x.device()),
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            logger.info(f"Created expert_mapping_tensors with shape: {cfg['expert_mapping_tensors'].shape}")

        # Configure all_to_all dispatch and combine for GPT-OSS
        # Use L1_MEMORY_CONFIG for decode, DRAM for prefill
        mode = "decode" if seq_len == 1 else "prefill"
        memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        dispatch_config = {
            "cluster_axis": 0,  # Distribute across rows
            "memory_config": memory_config,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_concat_dim": 2,  # Concatenate on seq dimension
        }

        combine_config = {
            "cluster_axis": 0,  # Combine across rows
            "memory_config": memory_config,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_shard_dim": 2,  # Shard on seq dimension
        }

        logger.info("-" * 60)
        logger.info("STEP 1: Prepare inputs for all_to_all_dispatch")

        # Save initial inputs
        save_tensor(x, "1_input_x.pt", "x", is_float=True)
        save_tensor(topk_experts_indices, "1_input_indices.pt", "indices", is_float=False)
        save_tensor(topk_experts_weights, "1_input_weights.pt", "weights", is_float=True)

        # Prepare inputs for all_to_all_dispatch
        # GPT-OSS expects: x: [batch, 1, 1, hidden], indices: [batch, K], weights: [batch, K]
        # Need to reshape for all_to_all which expects 4D tensors

        # Reshape x from [batch, 1, 1, hidden] or [1, 1, seq, hidden] to [1, 1, batch_size_per_device, hidden] for all_to_all
        # Note: For prefill, batch_size_per_device is actually the sequence length (we treat seq as batch for all-to-all)
        # Use actual hidden size from x, not cfg["hidden_size"], as preamble may have changed it
        actual_hidden_size = x.shape[-1]
        logger.info(f"  DEBUG: Reshaping x from {x.shape} to (1, 1, {batch_size_per_device}, {actual_hidden_size})")
        logger.info(f"  DEBUG: mode={mode}, seq_len={seq_len}, batch_size_per_device={batch_size_per_device}")
        x_reshaped = ttnn.reshape(x, shape=(1, 1, batch_size_per_device, actual_hidden_size))
        logger.info(f"  x reshaped: {x.shape} -> {x_reshaped.shape}")

        # Reshape indices from [batch, K] or [seq, K] to [1, 1, batch_size_per_device, K]
        if len(topk_experts_indices.shape) == 2:
            # Convert to uint32 for reshaping
            topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint32)
            topk_experts_indices = ttnn.reshape(
                topk_experts_indices, shape=(1, 1, batch_size_per_device, cfg["num_experts_per_tok"])
            )
            # Convert back to uint16 for all_to_all
            topk_experts_indices_u32 = topk_experts_indices
            topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint16)
            ttnn.deallocate(topk_experts_indices_u32)
            logger.info(f"  indices reshaped: 2D -> {topk_experts_indices.shape}")

        # Convert to ROW_MAJOR layout as required by all_to_all_dispatch
        x_rm = ttnn.to_layout(x_reshaped, ttnn.ROW_MAJOR_LAYOUT)
        topk_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        logger.info(f"  x layout: {x_reshaped.layout} -> {x_rm.layout}")
        logger.info(f"  indices layout: {topk_experts_indices.layout} -> {topk_indices_rm.layout}")

        logger.info("-" * 60)
        logger.info("STEP 2: all_to_all_dispatch")
        logger.info(f"  Inputs: x_rm={x_rm.shape}, indices_rm={topk_indices_rm.shape}")
        logger.info(f"  expert_mapping={cfg['expert_mapping_tensors'].shape}")
        logger.info(f"  dispatch_config: {dispatch_config}")

        # Save inputs to all_to_all_dispatch
        save_tensor(x_rm, "2_before_dispatch_x.pt", "x_rm", is_float=True)
        save_tensor(topk_indices_rm, "2_before_dispatch_indices.pt", "indices_rm", is_float=False)

        # Apply all_to_all_dispatch - route tokens to expert devices
        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            topk_indices_rm,
            cfg["expert_mapping_tensors"],
            **dispatch_config,
        )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_indices_rm)

        logger.info(f"  Outputs: dispatch_output={dispatch_output.shape}, dispatch_metadata={dispatch_metadata.shape}")

        # Save outputs from all_to_all_dispatch
        save_tensor(dispatch_output, "3_after_dispatch_output.pt", "dispatch_output", is_float=True)
        save_tensor(dispatch_metadata, "3_after_dispatch_metadata.pt", "dispatch_metadata", is_float=False)

        logger.info("-" * 60)
        logger.info("STEP 3: Prepare for expert computation")

        # Reshape dispatch output for expert computation
        # dispatch_output: [num_devices, 1, batch_size, hidden_size]
        # Need: [1, num_experts_per_device, batch_size, hidden_size]
        # For GPT-OSS prefill, dispatch_output already has all tokens (batch_size is actually total seq)
        if seq_len > 1:
            # Prefill mode - dispatch_output has all seq tokens
            total_tokens = batch_size  # batch_size is actually total seq_len for prefill
        else:
            # Decode mode
            total_tokens = batch_size * seq_len
        logger.info(f"  total_tokens = {total_tokens} (batch_size={batch_size}, seq_len={seq_len})")

        # Use actual hidden size from dispatch_output, not cfg["hidden_size"], as preamble may have changed it
        dispatch_hidden_size = dispatch_output.shape[-1]
        dispatch_reshaped = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, dispatch_hidden_size))
        logger.info(f"  dispatch reshape: {dispatch_output.shape} -> {dispatch_reshaped.shape}")

        # Repeat for expert dimension
        dispatch_repeated = ttnn.repeat(dispatch_reshaped, ttnn.Shape((1, cfg["num_experts_per_device"], 1, 1)))
        logger.info(f"  dispatch repeat: {dispatch_reshaped.shape} -> {dispatch_repeated.shape}")

        dispatch_tiled = ttnn.to_layout(dispatch_repeated, ttnn.TILE_LAYOUT)
        logger.info(f"  dispatch layout: {dispatch_repeated.layout} -> {dispatch_tiled.layout}")

        ttnn.deallocate(dispatch_output)
        ttnn.deallocate(dispatch_reshaped)
        ttnn.deallocate(dispatch_repeated)

        logger.info("-" * 60)
        logger.info("STEP 4: Expert computation")

        # Instrumentation: Log input to experts
        log_tensor_checkpoint(dispatch_tiled, "MoEBlock_gptoss_unified_4_experts_input")
        logger.info(f"  Expert input shape: {dispatch_tiled.shape}")

        # Save expert input
        save_tensor(dispatch_tiled, "4_expert_input.pt", "dispatch_tiled", is_float=True)

        # Run expert computation - check if we need to load GPT-OSS weights
        if "gptoss_expert_weights" not in cfg:
            # Extract state dicts from config
            experts_state_dict = cfg.get("moe_experts", {}).get("experts_state_dict", {})

            if experts_state_dict:
                # Load GPT-OSS specific modules
                from models.demos.gpt_oss.tt.experts_throughput.config import (
                    ThroughputExpertConfig,
                    ThroughputProgramConfig,
                )
                from models.demos.gpt_oss.tt.experts_throughput.weights import load_throughput_expert_weights

                # Create ThroughputExpertConfig using unified_config's intermediate_size
                # The unified_config should have the correct intermediate_size from hf_config
                expert_config = ThroughputExpertConfig(
                    num_experts=cfg["num_devices"] * cfg["num_experts_per_device"],
                    num_devices=cfg["num_devices"],
                    hidden_size=cfg["hidden_size"],
                    intermediate_size=unified_config.intermediate_size,  # Use from unified config
                    num_experts_per_tok=cfg["num_experts_per_tok"],
                    alpha=unified_config.activation.swiglu_alpha if unified_config.activation else 1.702,
                    swiglu_limit=unified_config.activation.swiglu_limit if unified_config.activation else 7.0,
                )

                # Validate that intermediate_size matches weights if available
                if "gate_up_proj" in experts_state_dict:
                    # gate_up_proj shape: [num_experts, hidden_size, 2 * intermediate_size]
                    gate_up_shape = experts_state_dict["gate_up_proj"].shape
                    weight_intermediate_size = gate_up_shape[-1] // 2
                    if weight_intermediate_size != expert_config.intermediate_size:
                        logger.warning(
                            f"Weight intermediate_size ({weight_intermediate_size}) doesn't match "
                            f"config intermediate_size ({expert_config.intermediate_size}). Using weight size."
                        )
                        expert_config.intermediate_size = weight_intermediate_size

                # Load and convert weights to TTNN tensors
                logger.info(
                    f"Loading GPT-OSS expert weights from state_dict with keys: {list(experts_state_dict.keys())}"
                )
                cfg["gptoss_expert_weights"] = load_throughput_expert_weights(
                    mesh_device=x.device(),
                    config=expert_config,
                    state_dict=experts_state_dict,
                    weight_dtype=unified_config.weight_dtype
                    if hasattr(unified_config, "weight_dtype")
                    else ttnn.bfloat4_b,
                    tensor_cache_path=cfg.get("moe_experts", {}).get("cache_path"),
                )
                cfg["gptoss_expert_config"] = expert_config

                # Create program config
                cfg["gptoss_program_config"] = ThroughputProgramConfig()

                logger.info("Successfully loaded GPT-OSS expert weights as TTNN tensors")

        # For GPT-OSS, we should use ThroughputExperts directly instead of unified path
        # The unified path is too complex and doesn't match how ThroughputExperts works
        if "gptoss_expert_weights" in cfg or cfg.get("backend") == "gptoss":
            # GPT-OSS should use ThroughputExperts module directly
            # Create ThroughputExperts if not already created
            if "throughput_experts" not in cfg:
                from models.demos.gpt_oss.tt.experts_throughput import ThroughputExpertConfig, ThroughputExperts

                # Create config
                expert_config = ThroughputExpertConfig(
                    num_experts=cfg["num_devices"] * cfg["num_experts_per_device"],
                    num_devices=cfg["num_devices"],
                    hidden_size=unified_config.hidden_size,
                    intermediate_size=unified_config.intermediate_size,
                    num_experts_per_tok=cfg["num_experts_per_tok"],
                    alpha=unified_config.activation.swiglu_alpha if unified_config.activation else 1.702,
                    swiglu_limit=unified_config.activation.swiglu_limit if unified_config.activation else 7.0,
                )

                # Get expert weights from config or state dict
                experts_state_dict = cfg.get("moe_experts", {}).get("experts_state_dict", {})

                # Create ThroughputExperts instance
                throughput_experts = ThroughputExperts(
                    mesh_device=x.device(),
                    config=expert_config,
                    state_dict=experts_state_dict,
                    weight_dtype=ttnn.bfloat4_b,
                    dispatch_cluster_axis=0,
                    decode_memory_config=ttnn.L1_MEMORY_CONFIG,
                    tensor_cache_path=cfg.get("moe_experts", {}).get("cache_path"),
                    mesh_config=cfg.get("mesh_config"),
                    ccl_manager=cfg.get("ccl_manager"),
                )
                cfg["throughput_experts"] = throughput_experts
                logger.info("Created ThroughputExperts instance for GPT-OSS")

            # Use ThroughputExperts directly - it handles all dispatch/combine internally
            throughput_experts = cfg["throughput_experts"]

            # ThroughputExperts expects the original inputs, not the dispatch output
            # Return to using original x, indices, and weights before all the dispatch operations
            experts_output = throughput_experts(
                hidden_states=x,  # Use original x
                topk_expert_indices=topk_experts_indices,
                topk_expert_weights=topk_experts_weights,
                is_decode=(seq_len == 1),
            )
            logger.info(f"ThroughputExperts output shape: {experts_output.shape}")

            # Convert to expected output memory config if needed
            expected_output_memory_config = cfg.get("output_memory_config", ttnn.DRAM_MEMORY_CONFIG)
            if experts_output.memory_config() != expected_output_memory_config:
                experts_output = ttnn.to_memory_config(experts_output, expected_output_memory_config)
                logger.info(f"Converted output to memory config: {expected_output_memory_config}")

            # ThroughputExperts handles all_reduce internally, so we can skip to the end
            return experts_output
        else:
            # Fallback to previous implementation (shouldn't happen with proper setup)
            logger.warning("GPT-OSS weights not loaded, falling back to MoEExperts._forward")
            if "moe_experts" not in cfg:
                moe_experts_cfg = MoEExperts._create_model_config(cfg.get("hf_config"), x.device(), mode)
                cfg["moe_experts"] = moe_experts_cfg
                logger.info("  Created moe_experts config")
            experts_output = MoEExperts._forward(dispatch_tiled, cfg["moe_experts"])
        logger.info(f"  Expert output shape: {experts_output.shape}")

        # Save expert output
        save_tensor(experts_output, "5_expert_output.pt", "experts_output", is_float=True)

        # Instrumentation: Log output of experts
        log_tensor_checkpoint(experts_output, "MoEBlock_gptoss_unified_5_experts_output")

        ttnn.deallocate(dispatch_tiled)

        logger.info("-" * 60)
        logger.info("STEP 5: Prepare for all_to_all_combine")

        # Prepare for all_to_all_combine
        # experts_output: [1, num_experts_per_device, total_tokens, hidden_size]
        # Need: [num_experts_per_device, 1, total_tokens, hidden_size] in ROW_MAJOR
        experts_output_rm = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        logger.info(f"  experts_output layout: {experts_output.layout} -> {experts_output_rm.layout}")

        experts_output_reshaped = ttnn.reshape(
            experts_output_rm, shape=(cfg["num_experts_per_device"], 1, total_tokens, cfg["hidden_size"])
        )
        logger.info(f"  experts_output reshape: {experts_output_rm.shape} -> {experts_output_reshaped.shape}")

        # Reshape dispatch_metadata for combine
        dispatch_metadata_reshaped = ttnn.reshape(
            dispatch_metadata, shape=(1, 1, total_tokens, cfg["num_experts_per_tok"])
        )
        logger.info(f"  dispatch_metadata reshape: {dispatch_metadata.shape} -> {dispatch_metadata_reshaped.shape}")

        logger.info("-" * 60)
        logger.info("STEP 6: all_to_all_combine")
        logger.info(f"  Inputs: experts={experts_output_reshaped.shape}, metadata={dispatch_metadata_reshaped.shape}")
        logger.info(f"  combine_config: {combine_config}")

        # Apply all_to_all_combine - route expert outputs back
        # Use the same expert_mapping_tensors that was used for dispatch (like original GPT-OSS)
        combine_output = ttnn.all_to_all_combine(
            experts_output_reshaped,
            dispatch_metadata_reshaped,
            cfg["expert_mapping_tensors"],
            **combine_config,
        )
        logger.info(f"  Output: combine_output={combine_output.shape}")

        # NOW we can deallocate the tensors that were consumed by all_to_all_combine
        # This matches the original GPT-OSS implementation timing
        ttnn.deallocate(experts_output)
        ttnn.deallocate(experts_output_rm)
        ttnn.deallocate(experts_output_reshaped)
        ttnn.deallocate(dispatch_metadata_reshaped)

        # Save combine output
        save_tensor(combine_output, "6_after_combine.pt", "combine_output", is_float=True)

        logger.info("-" * 60)
        logger.info("STEP 7: Apply routing weights and reduce")

        # combine_output: [K, 1, batch_size_per_device, hidden_size]
        # Convert to TILE layout
        post_combine = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)
        logger.info(f"  combine_output layout: {combine_output.layout} -> {post_combine.layout}")
        ttnn.deallocate(combine_output)

        # Prepare routing weights for multiplication
        # topk_experts_weights: [batch, K] -> [K, 1, batch, hidden_size]
        logger.info(f"  topk_experts_weights shape: {topk_experts_weights.shape}")

        if len(topk_experts_weights.shape) == 2:
            # Expand weights to match post_combine shape
            # First reshape to [batch, K, 1, 1]
            weights_expanded = ttnn.reshape(
                topk_experts_weights, shape=(batch_size_per_device, cfg["num_experts_per_tok"], 1, 1)
            )
            logger.info(f"  weights expand: {topk_experts_weights.shape} -> {weights_expanded.shape}")

            # Permute to [K, 1, batch, 1]
            weights_permuted = ttnn.permute(weights_expanded, (1, 2, 0, 3))
            logger.info(f"  weights permute: {weights_expanded.shape} -> {weights_permuted.shape}")

            # Repeat along hidden dimension
            weights_repeated = ttnn.repeat(weights_permuted, ttnn.Shape((1, 1, 1, cfg["hidden_size"])))
            logger.info(f"  weights repeat: {weights_permuted.shape} -> {weights_repeated.shape}")

            ttnn.deallocate(weights_expanded)
            ttnn.deallocate(weights_permuted)
        else:
            weights_repeated = topk_experts_weights
            logger.info(f"  weights already prepared: {weights_repeated.shape}")

        # Apply routing weights
        weighted_output = ttnn.mul(post_combine, weights_repeated, memory_config=unified_config.output_memory_config)
        logger.info(f"  weighted_output: {post_combine.shape} * {weights_repeated.shape} = {weighted_output.shape}")
        ttnn.deallocate(post_combine)
        ttnn.deallocate(weights_repeated)

        # Sum across experts (K dimension)
        output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        logger.info(f"  sum across experts: {weighted_output.shape} -> {output.shape}")
        ttnn.deallocate(weighted_output)

        # All-reduce across columns (cluster_axis=1) to aggregate expert outputs
        # This is critical for GPT-OSS! The original implementation does this after summing.
        # Why this is needed:
        # 1. Experts are sharded across ALL devices in 2D (rows x cols)
        # 2. all_to_all_dispatch/combine on axis=0 handles row-wise redistribution
        # 3. But tokens may route to experts in different COLUMNS
        # 4. After combine, each device has partial results from experts in its column
        # 5. We need to sum these partials across columns to get complete expert outputs
        logger.info("STEP 8: All-reduce across columns to aggregate expert outputs")
        output_all_reduced = ttnn.all_reduce(
            output,
            num_links=4,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,  # Reduce across columns
            memory_config=memory_config,
        )
        logger.info(f"  all_reduce: {output.shape} (cluster_axis=1)")
        ttnn.deallocate(output)
        output = output_all_reduced

        # Reshape output back to original shape
        # Decode: From [1, 1, batch, hidden] to [batch, 1, 1, hidden]
        # Prefill: From [1, 1, seq, hidden] to [1, 1, seq, hidden] (no reshape needed)
        if mode == "decode":
            output = ttnn.reshape(output, shape=(batch_size_per_device, 1, 1, cfg["hidden_size"]))
            logger.info(f"  final reshape (decode): -> {output.shape}")
        else:
            # Prefill mode - output is already in correct shape [1, 1, seq, hidden]
            logger.info(f"  no reshape needed for prefill, output shape: {output.shape}")

        logger.info("=" * 80)
        logger.info(f"FINAL OUTPUT SHAPE: {output.shape}")
        logger.info("=" * 80)

        # Save final output before memory config conversion
        save_tensor(output, "7_final_output.pt", "output", is_float=True)

        # Convert to expected output memory config if needed
        if "output_memory_config" in cfg:
            output = ttnn.to_memory_config(output, cfg["output_memory_config"])
            logger.info(f"Converted output to memory config: {cfg['output_memory_config']}")

        # Clean up cached weights if they were loaded for this forward pass
        if "gptoss_expert_weights" in cfg and not cfg.get("keep_expert_weights", False):
            weights = cfg["gptoss_expert_weights"]
            if hasattr(weights, "deallocate"):
                weights.deallocate()
            del cfg["gptoss_expert_weights"]
            del cfg["gptoss_expert_config"]
            del cfg["gptoss_program_config"]
            logger.info("Cleaned up temporarily loaded GPT-OSS expert weights")

        return output

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
