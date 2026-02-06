# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.experts import Experts as MoEExperts
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
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
                "num_links": 1,
            },
            "all_to_all_combine": {
                "num_links": 1,
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

            USERS_PER_ROW = 32
            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            input_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(USERS_PER_ROW, HIDDEN_SIZE // TP_SIZE),
                core_grid=ttnn.CoreGrid(y=7, x=4),
                strategy=ttnn.ShardStrategy.WIDTH,
            )

            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
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
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=input_output_memory_config,
                    topology=ttnn.Topology.Linear,
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
                    topology=ttnn.Topology.Linear,
                ),
            }
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
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
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=memory_config,
                    topology=ttnn.Topology.Linear,
                ),
                "revert_tp": AllGatherAsyncConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dim=-1,  # Last dimension
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cluster_axis=1,
                    topology=ttnn.Topology.Linear,
                ),
            }

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, topk_fallback: bool = False
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode", topk_fallback=topk_fallback)

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, topk_fallback: bool = False
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill", topk_fallback=topk_fallback)

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
        # breakpoint()
        ccl = cfg["ccl"]  # CCL runtime initialization in execution order
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # Repeat + Permute Expert weights

        topk_experts_weights = cls._fwd_repeat_permute_expert_weights(topk_experts_weights, cfg)

        # MOE

        post_combine_output_tensor = cls._fwd_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )

        # Note: reduce_scatter is handled by the caller (decoder block or test)

        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_repeat_permute_expert_weights(
        cls, topk_experts_weights: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> ttnn.Tensor:
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
        tokens = batch_size * seq_len
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
        )

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

            experts_output = MoEExperts._forward(dispatch_chunk, cfg["moe_experts"])
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
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls.forward(x, cfg)
