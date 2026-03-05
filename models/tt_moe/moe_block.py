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
from models.demos.gpt_oss.tt.experts_throughput import ThroughputExpertConfig, ThroughputExperts

# GPT-OSS components (optional backend)
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.tt_moe.collectives.ccl import CCL

# DeepSeek components (default)
from models.tt_moe.components.experts.routed_experts import RoutedExperts as MoEExperts

# MoE Preamble - intermediate processing between router and experts
from models.tt_moe.components.moe_preamble import MoEPreamble, MoEPreambleConfig
from models.tt_moe.components.routers.grouped_topk_router import GroupedTopKRouter as MoEGate


class MoEBlock(SharedStateAddOn, AbstractModule):
    """MoE module supporting both DeepSeek and GPT-OSS backends.

    Args:
        backend: Either "deepseek" (default) or "gptoss" to select the implementation.
                DeepSeek uses GroupedTopKRouter + RoutedExperts.
                GPT-OSS uses TopKRouter + ThroughputExperts.

    See the `AbstractModule` docstring for additional usage info.
    """

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
            # Use GPT-OSS's TopKRouter and ThroughputExperts weight conversion
            # Note: GPT-OSS uses different weight organization

            # For GPT-OSS, we don't convert weights here since the components
            # handle their own weight loading. We just return empty configs
            # to satisfy the weight validation. The actual state dict will be
            # passed through the config at runtime.
            return {
                "moe_gate": {},  # Empty dict for GPT-OSS router
                "moe_experts": {},  # Empty dict for GPT-OSS experts
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
                # ThroughputExperts handles all_to_all operations internally
                # so we don't need to set dispatch/combine configs here
                config.update(
                    {
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
                # ThroughputExperts handles all_to_all operations internally
                config.update(
                    {
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
        if backend == "gptoss" and x.shape[0] > 1:
            # GPT-OSS decode mode with high throughput
            batch_size_per_device = x.shape[0]
            seq_len = 1
        else:
            # DeepSeek or GPT-OSS prefill
            seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
            batch_size_per_device = x.shape[
                -2
            ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device

        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

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

                # Initialize TopKRouter with actual weights if available
                if router_state_dict:
                    cfg["_gptoss_router"] = TopKRouter(
                        x.device(),
                        router_config,
                        router_state_dict,
                        tensor_cache_path=None,
                    )
                else:
                    # Fallback: create mock router for testing
                    logger.warning("GPT-OSS router weights not found, creating mock router")
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

        if backend == "gptoss":
            # Use GPT-OSS ThroughputExperts path
            logger.info("Entering GPT-OSS ThroughputExperts path")
            return cls._fwd_moe_gptoss(
                x, topk_experts_indices, topk_experts_weights, cfg, batch_size_per_device, batch_size, seq_len
            )

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
    def _fwd_moe_gptoss(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """GPT-OSS implementation using ThroughputExperts."""

        logger.info(
            f"GPT-OSS _fwd_moe_gptoss: x.shape={x.shape}, batch_size_per_device={batch_size_per_device}, seq_len={seq_len}"
        )

        # Create ThroughputExperts if not cached in config
        if "_gptoss_experts" not in cfg:
            logger.info("Creating ThroughputExperts for GPT-OSS backend...")
            # Get expert state dict from config (stored during convert_weights)
            experts_config = cfg.get("moe_experts", {})
            experts_state_dict = experts_config.get("experts_state_dict", {})

            # Create configuration for ThroughputExperts
            # Get intermediate_size from state dict if available
            intermediate_size = cfg.get("intermediate_size", 10560)  # GPT-OSS default
            if experts_state_dict and "gate_up_proj" in experts_state_dict:
                # Infer intermediate size from weight shape
                gate_up_shape = experts_state_dict["gate_up_proj"].shape
                intermediate_size = gate_up_shape[-1] // 2  # Fused gate_up has 2x intermediate size
                logger.info(f"Inferred intermediate_size={intermediate_size} from gate_up_proj shape={gate_up_shape}")

            num_experts = cfg["num_experts_per_device"] * cfg["num_devices"]
            logger.info(
                f"Creating ThroughputExpertConfig: intermediate_size={intermediate_size}, num_experts={num_experts}, hidden_size={cfg['hidden_size']}, num_experts_per_tok={cfg['num_experts_per_tok']}, num_devices={cfg['num_devices']}"
            )

            # ThroughputExperts requires num_experts to be divisible by num_devices
            assert num_experts % cfg["num_devices"] == 0, (
                f"ThroughputExperts requires num_experts ({num_experts}) to be divisible by "
                f"num_devices ({cfg['num_devices']}). Got remainder: {num_experts % cfg['num_devices']}"
            )

            throughput_config = ThroughputExpertConfig(
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                hidden_size=cfg["hidden_size"],
                num_experts_per_tok=cfg["num_experts_per_tok"],
                num_devices=cfg["num_devices"],
            )

            # Get mesh config from cfg if available
            mesh_config = cfg.get("mesh_config")
            if mesh_config is None:
                # Create a simple mesh config for GPT-OSS
                from models.demos.gpt_oss.config import MeshConfig, ModeConfig

                mesh_shape = x.device().shape
                # Simple config: EP on rows, TP on cols
                decode_config = ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0], sp=1)
                prefill_config = ModeConfig(tp=mesh_shape[1], ep=1, sp=mesh_shape[0])

                mesh_config = MeshConfig(
                    mesh_shape=mesh_shape,
                    decode=decode_config,
                    prefill=prefill_config,
                    tp_axis=1,  # TP on columns
                )

            # Create ThroughputExperts with actual weights
            if experts_state_dict:
                logger.info("Creating ThroughputExperts with actual weights")

                # Debug logging
                ccl_manager = cfg.get("ccl_manager") or cfg.get("ccl")
                cache_path = cfg.get("cache_path")
                logger.info(f"ThroughputExperts params:")
                logger.info(f"  mesh_device: {x.device()}")
                logger.info(f"  config: {throughput_config}")
                logger.info(f"  ccl_manager: {ccl_manager}")
                logger.info(f"  mesh_config: {mesh_config}")
                logger.info(f"  cache_path: {cache_path}")
                logger.info(f"  dispatch_cluster_axis: 0")
                logger.info(f"  weight_dtype: {ttnn.bfloat16}")
                logger.info(f"  state_dict keys: {list(experts_state_dict.keys())}")
                logger.info(
                    f"  state_dict shapes: gate_up_proj={experts_state_dict.get('gate_up_proj', torch.tensor([])).shape}, down_proj={experts_state_dict.get('down_proj', torch.tensor([])).shape}"
                )

                logger.info("Calling ThroughputExperts constructor...")
                cfg["_gptoss_experts"] = ThroughputExperts(
                    mesh_device=x.device(),
                    config=throughput_config,
                    state_dict=experts_state_dict,
                    ccl_manager=ccl_manager,
                    mesh_config=mesh_config,
                    weight_dtype=ttnn.bfloat4_b,  # Use bfloat4_b like test_modules.py
                    dispatch_cluster_axis=0,
                    decode_memory_config=ttnn.L1_MEMORY_CONFIG,  # Use L1 like test_modules.py
                    tensor_cache_path=cache_path,
                )
                logger.info("ThroughputExperts created successfully with actual weights")
            else:
                # Fallback: create mock experts for testing
                logger.warning("GPT-OSS expert weights not found, creating mock experts")
                mock_experts_state_dict = {
                    "gate_up_proj": torch.randn(
                        throughput_config.num_experts,
                        throughput_config.hidden_size,
                        2 * throughput_config.intermediate_size,
                    ),
                    "down_proj": torch.randn(
                        throughput_config.num_experts,
                        throughput_config.intermediate_size,
                        throughput_config.hidden_size,
                    ),
                }
                logger.info(
                    f"Mock weights created: gate_up_proj shape={mock_experts_state_dict['gate_up_proj'].shape}, down_proj shape={mock_experts_state_dict['down_proj'].shape}"
                )
                logger.info(f"Creating ThroughputExperts with mesh_device={x.device()}, mesh_config={mesh_config}")

                try:
                    cfg["_gptoss_experts"] = ThroughputExperts(
                        mesh_device=x.device(),
                        config=throughput_config,
                        state_dict=mock_experts_state_dict,
                        ccl_manager=cfg.get("ccl_manager") or cfg.get("ccl"),  # Support both key names
                        mesh_config=mesh_config,
                        weight_dtype=ttnn.bfloat4_b,  # Use bfloat4_b like test_modules.py
                        dispatch_cluster_axis=0,
                        decode_memory_config=ttnn.L1_MEMORY_CONFIG,  # Use L1 like test_modules.py
                        tensor_cache_path=cfg.get("cache_path"),
                    )
                    logger.info("ThroughputExperts created successfully with mock weights")
                except Exception as e:
                    logger.error(f"Failed to create ThroughputExperts: {e}")
                    raise

        # ThroughputExperts expects different input format
        is_decode = seq_len == 1

        # Instrumentation: Log input to experts
        from models.tt_moe.utils.tensor_debug import log_tensor_checkpoint

        log_tensor_checkpoint(x, "MoEBlock_gptoss_4_experts_input")
        log_tensor_checkpoint(topk_experts_indices, "MoEBlock_gptoss_4b_expert_indices")
        log_tensor_checkpoint(topk_experts_weights, "MoEBlock_gptoss_4c_expert_weights")

        logger.info(
            f"Calling ThroughputExperts with is_decode={is_decode}, x.shape={x.shape}, indices.shape={topk_experts_indices.shape}, weights.shape={topk_experts_weights.shape}"
        )

        try:
            # Call ThroughputExperts
            output = cfg["_gptoss_experts"](
                hidden_states=x,
                topk_expert_indices=topk_experts_indices,
                topk_expert_weights=topk_experts_weights,
                is_decode=is_decode,
            )
            logger.info(
                f"ThroughputExperts returned successfully, output.shape={output.shape if hasattr(output, 'shape') else 'unknown'}"
            )
        except Exception as e:
            logger.error(f"ThroughputExperts failed with error: {e}")
            raise

        # Instrumentation: Log output of experts
        log_tensor_checkpoint(output, "MoEBlock_gptoss_5_experts_output")

        # Convert output to the expected memory config
        # ThroughputExperts may return L1, but we need to match the expected output config
        if "output_memory_config" in cfg:
            output = ttnn.to_memory_config(output, cfg["output_memory_config"])

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
