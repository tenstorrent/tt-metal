# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn


# DEBUG: Helper for hang debugging
def _debug_print(msg: str, flush: bool = True):
    """Print debug message with timestamp and flush to ensure immediate output."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {msg}", file=sys.stderr, flush=flush)


from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.experts import Experts as MoEExperts
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    DeepseekMoEReduceScatterConfig,
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
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


class MoE(SharedStateAddOn, AbstractModule):
    """MoE module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"MoE expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts
        assert state_dict is not None

        return {
            "moe_gate": MoEGate.convert_weights(
                hf_config, (state_dict,), output_path / "moe_gate", mesh_device, "gate."
            ),
            "moe_experts": MoEExperts.convert_weights(
                hf_config, (state_dict,), output_path / "moe_experts", mesh_device
            ),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create shared model state containing tensors that are constant across all instances.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelState containing shared tensors
        """
        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        num_dispatch_device_rows = mesh_device.shape[0]

        logger.info(
            "Creating MoE shared state: expert mapping tensor "
            f"(num_devices={num_devices}, experts_per_device={num_experts_per_device})..."
        )
        expert_mapping_start = perf_counter()
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
        logger.info(f"Created MoE expert mapping tensor in {perf_counter() - expert_mapping_start:.2f}s")

        logger.info(
            "Creating MoE shared state: remap topk mask "
            f"(dispatch_rows={num_dispatch_device_rows}, experts={hf_config.n_routed_experts})..."
        )
        remap_mask_start = perf_counter()
        remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, hf_config.n_routed_experts), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        logger.info(f"Created MoE remap topk mask in {perf_counter() - remap_mask_start:.2f}s")

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
        fabric_config: ttnn.FabricConfig,
        mode: str,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            shard_core_grid = ttnn.CoreGrid(y=7, x=4)
            per_core_width = (HIDDEN_SIZE // TP_SIZE) // shard_core_grid.num_cores
            input_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(
                    ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE),
                    ttnn.core.roundup(per_core_width, ttnn.TILE_SIZE),
                ),
                core_grid=shard_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "fabric_config": fabric_config,
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
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
                "sum_experts_output_memory_config": memory_config,
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=input_output_memory_config,
                ),
                "ring_sum_experts_output_memory_config": DeepseekMoEReduceScatterConfig.create_default_input_memory_config(
                    USERS_PER_ROW, HIDDEN_SIZE, TP_SIZE
                ),
                "ring_final_output_reduce_scatter": DeepseekMoEReduceScatterConfig(
                    cluster_axis=1,
                    dim=3,
                    output_memory_config=input_output_memory_config,
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
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "fabric_config": fabric_config,
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
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
                "sum_experts_output_memory_config": memory_config,
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

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, fabric_config, "decode", topk_fallback=topk_fallback)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, fabric_config, "prefill", topk_fallback=topk_fallback)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        _debug_print(f"MoE.forward: START (input shape={x.shape})")
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
        num_dispatch_devices = int(cfg.get("num_dispatch_devices", 1))
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            _debug_print(
                f"MoE.forward: Using chunked prefill (global_tokens={global_tokens} > chunk_tokens={chunk_tokens})"
            )
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            result = cls._forward_chunked_prefill(x, cfg, chunk_size)
            _debug_print("MoE.forward: END (chunked prefill)")
            return result
        result = cls._forward_impl(x, cfg)
        _debug_print("MoE.forward: END")
        return result

    @classmethod
    def _forward_chunked_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig, chunk_size: int) -> ttnn.Tensor:
        _debug_print(f"MoE._forward_chunked_prefill: START (chunk_size={chunk_size})")
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks: list[ttnn.Tensor] = []
        num_chunks = 0
        for start in range(0, seq_len, chunk_size):
            _debug_print(f"MoE._forward_chunked_prefill: Processing chunk {num_chunks} (start={start})")
            end = min(start + chunk_size, seq_len)
            _debug_print(f"MoE._forward_chunked_prefill: ttnn.slice chunk START")
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            _debug_print(f"MoE._forward_chunked_prefill: ttnn.slice chunk DONE")
            output_chunks.append(cls._forward_impl(x_chunk, cfg))
            _debug_print(f"MoE._forward_chunked_prefill: deallocate chunk START")
            ttnn.deallocate(x_chunk)
            _debug_print(f"MoE._forward_chunked_prefill: deallocate chunk DONE")
            num_chunks += 1

        if len(output_chunks) == 1:
            _debug_print("MoE._forward_chunked_prefill: END (single chunk)")
            return output_chunks[0]
        _debug_print(f"MoE._forward_chunked_prefill: ttnn.concat {len(output_chunks)} chunks START")
        output = ttnn.concat(output_chunks, dim=2)
        _debug_print("MoE._forward_chunked_prefill: ttnn.concat DONE")
        for i, chunk in enumerate(output_chunks):
            _debug_print(f"MoE._forward_chunked_prefill: deallocate output chunk {i} START")
            ttnn.deallocate(chunk)
            _debug_print(f"MoE._forward_chunked_prefill: deallocate output chunk {i} DONE")
        _debug_print("MoE._forward_chunked_prefill: END")
        return output

    @classmethod
    def _forward_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        _debug_print(f"MoE._forward_impl: START (input shape={x.shape})")
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
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size
        _debug_print(f"MoE._forward_impl: batch_size={batch_size}, batch_size_per_device={batch_size_per_device}")

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        _debug_print("MoE._forward_impl: _fwd_moe_gate START")
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)
        _debug_print("MoE._forward_impl: _fwd_moe_gate DONE")

        # Repeat + Permute Expert weights
        _debug_print("MoE._forward_impl: _fwd_repeat_permute_expert_weights START")
        topk_experts_weights = cls._fwd_repeat_permute_expert_weights(topk_experts_weights, cfg)
        _debug_print("MoE._forward_impl: _fwd_repeat_permute_expert_weights DONE")

        # MOE
        _debug_print("MoE._forward_impl: _fwd_moe START")
        post_combine_output_tensor = cls._fwd_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )
        _debug_print("MoE._forward_impl: _fwd_moe DONE")

        # Note: sum_experts and reduce_scatter is handled by the caller (decoder block or test)

        _debug_print("MoE._forward_impl: END")
        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_repeat_permute_expert_weights(
        cls, topk_experts_weights: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> ttnn.Tensor:
        _debug_print("MoE._fwd_repeat_permute_expert_weights: START")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: to_layout ROW_MAJOR START")
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        _debug_print("MoE._fwd_repeat_permute_expert_weights: to_layout ROW_MAJOR DONE")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: repeat START")
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])
        _debug_print("MoE._fwd_repeat_permute_expert_weights: repeat DONE")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: permute START")
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        _debug_print("MoE._fwd_repeat_permute_expert_weights: permute DONE")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: to_layout TILE START")
        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        _debug_print("MoE._fwd_repeat_permute_expert_weights: to_layout TILE DONE")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: deallocate START")
        ttnn.deallocate(topk_experts_weights_rm)
        _debug_print("MoE._fwd_repeat_permute_expert_weights: deallocate DONE")
        _debug_print("MoE._fwd_repeat_permute_expert_weights: END")
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
        _debug_print(f"MoE._fwd_moe: START (batch_size={batch_size}, seq_len={seq_len})")
        tokens = batch_size * seq_len
        _debug_print("MoE._fwd_moe: to_layout x ROW_MAJOR START")
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _debug_print("MoE._fwd_moe: to_layout x ROW_MAJOR DONE")
        _debug_print("MoE._fwd_moe: reshape x START")
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )
        _debug_print("MoE._fwd_moe: reshape x DONE")

        _debug_print("MoE._fwd_moe: to_layout topk_experts_indices ROW_MAJOR START")
        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        _debug_print("MoE._fwd_moe: to_layout topk_experts_indices ROW_MAJOR DONE")
        _debug_print("MoE._fwd_moe: reshape topk_experts_indices START")
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
        )
        _debug_print("MoE._fwd_moe: reshape topk_experts_indices DONE")

        # Chunk along local batch dimension to keep all_to_all_dispatch output small in prefill.
        chunk_size = min(batch_size_per_device, max(1, cfg.get("moe_chunk_size", batch_size_per_device)))
        output_chunks: list[ttnn.Tensor] = []
        num_chunks = 0
        _debug_print(f"MoE._fwd_moe: Processing {batch_size_per_device} tokens in chunks of {chunk_size}")

        def _slice_topk_weights(batch_start: int, batch_end: int) -> ttnn.Tensor:
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            return ttnn.slice(
                topk_experts_weights,
                [0, 0, token_start, 0],
                [cfg["num_experts_per_tok"], 1, token_end, cfg["hidden_size"]],
            )

        for batch_start in range(0, batch_size_per_device, chunk_size):
            _debug_print(f"MoE._fwd_moe: Processing chunk {num_chunks} (batch_start={batch_start})")
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            batch_size_chunk = batch_chunk * cfg["num_dispatch_devices"]

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - slice x_chunk START")
            x_chunk = ttnn.slice(
                x_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["hidden_size"]],
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - slice x_chunk DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - slice topk_indices_chunk START")
            topk_indices_chunk = ttnn.slice(
                topk_experts_indices_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - slice topk_indices_chunk DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - all_to_all_dispatch START")
            all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
                x_chunk,
                topk_indices_chunk,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_dispatch"],
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - all_to_all_dispatch DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate x_chunk, topk_indices_chunk START")
            ttnn.deallocate(x_chunk)
            ttnn.deallocate(topk_indices_chunk)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate x_chunk, topk_indices_chunk DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape dispatch_chunk START")
            dispatch_chunk = ttnn.reshape(
                all_to_all_dispatch_output_tensors,
                shape=(1, 1, batch_size_chunk * seq_len, cfg["hidden_size"]),
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape dispatch_chunk DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - repeat dispatch_chunk START")
            dispatch_chunk = ttnn.repeat(dispatch_chunk, **cfg["activations_repeat"])
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - repeat dispatch_chunk DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - to_layout TILE dispatch_chunk START")
            dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - to_layout TILE dispatch_chunk DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate all_to_all_dispatch_output_tensors START")
            ttnn.deallocate(all_to_all_dispatch_output_tensors)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate all_to_all_dispatch_output_tensors DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - MoEExperts._forward START")
            experts_output = MoEExperts._forward(dispatch_chunk, cfg["moe_experts"])
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - MoEExperts._forward DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate dispatch_chunk START")
            ttnn.deallocate(dispatch_chunk)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate dispatch_chunk DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - to_layout experts_output ROW_MAJOR START")
            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - to_layout experts_output ROW_MAJOR DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape experts_output START")
            experts_output = ttnn.reshape(
                experts_output, shape=(cfg["num_experts_per_device"], batch_size_chunk, seq_len, cfg["hidden_size"])
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape experts_output DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape dispatch_metadata START")
            all_to_all_dispatch_metadata_tensors = ttnn.reshape(
                all_to_all_dispatch_metadata_tensors,
                shape=(1, batch_size_chunk, seq_len, cfg["num_experts_per_tok"]),
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape dispatch_metadata DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - all_to_all_combine START")
            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                all_to_all_dispatch_metadata_tensors,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_combine"],
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - all_to_all_combine DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate experts_output, dispatch_metadata START")
            ttnn.deallocate(experts_output)
            ttnn.deallocate(all_to_all_dispatch_metadata_tensors)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate experts_output, dispatch_metadata DONE")

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - reshape post_combine_output START")
            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(cfg["num_experts_per_tok"], 1, batch_chunk * seq_len, cfg["hidden_size"]),
            )
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)

            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - _slice_topk_weights START")
            topk_weights_chunk = _slice_topk_weights(batch_start, batch_end)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - _slice_topk_weights DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - mul with weights START")
            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_weights_chunk, **cfg["mul_experts_output_with_weights"]
            )
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - mul with weights DONE")
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate topk_weights_chunk START")
            ttnn.deallocate(topk_weights_chunk)
            _debug_print(f"MoE._fwd_moe: chunk {num_chunks} - deallocate topk_weights_chunk DONE")

            output_chunks.append(post_combine_output_tensor)
            num_chunks += 1

        if len(output_chunks) == 1:
            _debug_print("MoE._fwd_moe: single output chunk, no concat needed")
            post_combine_output_tensor = output_chunks[0]
        else:
            _debug_print(f"MoE._fwd_moe: ttnn.concat {len(output_chunks)} output chunks START")
            post_combine_output_tensor = ttnn.concat(output_chunks, dim=2)
            _debug_print("MoE._fwd_moe: ttnn.concat output chunks DONE")
            for i, chunk in enumerate(output_chunks):
                _debug_print(f"MoE._fwd_moe: deallocate output chunk {i} START")
                ttnn.deallocate(chunk)
                _debug_print(f"MoE._fwd_moe: deallocate output chunk {i} DONE")

        _debug_print("MoE._fwd_moe: deallocate x_rm START")
        ttnn.deallocate(x_rm)
        _debug_print("MoE._fwd_moe: deallocate x_rm DONE")
        _debug_print("MoE._fwd_moe: deallocate topk_experts_indices_rm START")
        ttnn.deallocate(topk_experts_indices_rm)
        _debug_print("MoE._fwd_moe: deallocate topk_experts_indices_rm DONE")
        _debug_print("MoE._fwd_moe: END")
        return post_combine_output_tensor

    @classmethod
    def _fwd_all_gather(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        return ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))

    @classmethod
    def _fwd_reduce_scatter(
        cls, post_combine_output_tensor: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig, ccl: CCL
    ) -> ttnn.Tensor:
        # Use standard reduce_scatter (composite fallback) to avoid shard shape constraints
        # encountered by the minimal async path in some decode configurations.
        rs_cfg = cfg["final_output_reduce_scatter"]
        rs_kwargs = {
            "dim": rs_cfg["dim"],
            "cluster_axis": rs_cfg.get("cluster_axis"),
            "subdevice_id": rs_cfg.get("subdevice_id"),
            "memory_config": rs_cfg.get("memory_config"),
            "intermediate_memory_config": rs_cfg.get("intermediate_memory_config"),
            "num_links": rs_cfg.get("num_links"),
            "topology": rs_cfg.get("topology"),
            "chunks_per_sync": rs_cfg.get("chunks_per_sync"),
            "num_workers_per_link": rs_cfg.get("num_workers_per_link"),
            "num_buffers_per_channel": rs_cfg.get("num_buffers_per_channel"),
        }
        rs_kwargs = {k: v for k, v in rs_kwargs.items() if v is not None}
        return ttnn.reduce_scatter(post_combine_output_tensor, **rs_kwargs)

    @classmethod
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        _debug_print(f"MoE.forward_prefill: START (handle_tensor_parallel={handle_tensor_parallel})")
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            _debug_print("MoE.forward_prefill: _fwd_all_gather START")
            x = cls._fwd_all_gather(x, cfg)
            _debug_print("MoE.forward_prefill: _fwd_all_gather DONE")

        # Run the forward pass
        _debug_print("MoE.forward_prefill: calling forward START")
        output = cls.forward(x, cfg)
        _debug_print("MoE.forward_prefill: calling forward DONE")

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            _debug_print("MoE.forward_prefill: sum_experts and reduce_scatter START")
            ccl = cfg["ccl"]
            _debug_print("MoE.forward_prefill: ttnn.sum START")
            output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
            _debug_print("MoE.forward_prefill: ttnn.sum DONE")
            _debug_print("MoE.forward_prefill: _fwd_reduce_scatter START")
            output = cls._fwd_reduce_scatter(output, cfg, ccl)
            _debug_print("MoE.forward_prefill: _fwd_reduce_scatter DONE")
            _debug_print("MoE.forward_prefill: sum_experts and reduce_scatter DONE")

        _debug_print("MoE.forward_prefill: END")
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        _debug_print(f"MoE.forward_decode: START (handle_tensor_parallel={handle_tensor_parallel})")
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            _debug_print("MoE.forward_decode: _fwd_all_gather START")
            x = cls._fwd_all_gather(x, cfg)
            _debug_print("MoE.forward_decode: _fwd_all_gather DONE")

        # Run the forward pass
        _debug_print("MoE.forward_decode: calling forward START")
        output = cls.forward(x, cfg)
        _debug_print("MoE.forward_decode: calling forward DONE")

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            _debug_print("MoE.forward_decode: sum_experts and reduce_scatter START")
            ccl = cfg["ccl"]
            tp_size = cfg["mesh_device"].shape[1]

            if cfg["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING and tp_size == 8:
                _debug_print("MoE.forward_decode: fast_reduce_nc path START")
                output = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                    output,
                    dim=0,
                    split_size=output.shape[-1] // tp_size,
                    output_memory_config=cfg["ring_sum_experts_output_memory_config"],
                )
                _debug_print("MoE.forward_decode: deepseek_moe_reduce_scatter START")
                output = ttnn.experimental.deepseek_moe_reduce_scatter(
                    output, **cfg["ring_final_output_reduce_scatter"]
                )
                _debug_print("MoE.forward_decode: deepseek_moe_reduce_scatter DONE")
                _debug_print("MoE.forward_decode: fast_reduce_nc path DONE")
            else:
                _debug_print("MoE.forward_decode: standard sum + reduce_scatter path START")
                output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
                output = cls._fwd_reduce_scatter(output, cfg, ccl)
                _debug_print("MoE.forward_decode: standard sum + reduce_scatter path DONE")
            _debug_print("MoE.forward_decode: sum_experts and reduce_scatter DONE")

        _debug_print("MoE.forward_decode: END")
        return output
