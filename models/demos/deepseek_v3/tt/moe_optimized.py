# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
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
    AllToAllDispatchMetadataConfig,
    DeepseekMoEPostCombineTilizeConfig,
    DeepseekMoEReduceScatterConfig,
    MeshDeviceStub,
    MoEComputeConfig,
    MorehFullConfig,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
    SelectiveReduceCombineConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, is_ring_fabric
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


def _assert_quad_ring(fabric_config: ttnn.FabricConfig, mesh_device: ttnn.Device) -> None:
    if not is_ring_fabric(fabric_config):
        raise AssertionError("MoEOptimized requires ring fabric")
    if mesh_device.shape[0] != 16:
        raise AssertionError(
            f"MoEOptimized requires 16 dispatch-device rows (quad); got mesh_device.shape[0]={mesh_device.shape[0]}"
        )


def _decode_sharded_io_memory_config(
    hf_config: PretrainedConfig,
    mesh_device: ttnn.Device,
    batch_size_per_row: int,
) -> ttnn.MemoryConfig:
    hidden_size = hf_config.hidden_size
    tp_size = mesh_device.shape[1]
    shard_core_grid = ttnn.CoreGrid(y=7, x=4)
    per_core_width = (hidden_size // tp_size) // shard_core_grid.num_cores
    return ttnn.create_sharded_memory_config(
        shape=(
            ttnn.core.roundup(batch_size_per_row, ttnn.TILE_SIZE),
            ttnn.core.roundup(per_core_width, ttnn.TILE_SIZE),
        ),
        core_grid=shard_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class MoEOptimized(SharedStateAddOn, AbstractModule):
    """MoE forward path optimized for quad mesh (16 dispatch rows) with ring fabric only.

    Same external API shape as `MoE` (convert_weights, create_shared_state, model_config, forward_*).
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
        ), f"MoEOptimized expects exactly one non-padding state dict, got {len(state_dicts)}"
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

        logger.info(
            "Creating MoEOptimized shared state: expert mapping tensor "
            f"(num_devices={num_devices}, experts_per_device={num_experts_per_device})..."
        )

        num_experts = num_devices * num_experts_per_device
        torch_expert_mapping_tensor = (
            (torch.arange(num_experts) // num_experts_per_device).unsqueeze(0).repeat(num_devices, 1)
        )
        expert_mapping_tensor = ttnn.from_torch(
            torch_expert_mapping_tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        config = {
            "expert_mapping_tensor": expert_mapping_tensor,
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
        }

        # TODO: #42722 - preallocated tensors for all_to_all_dispatch_metadata
        if False:
            rows_per_block = USERS_PER_ROW
            batch = rows_per_block * mesh_device.shape[0]
            preallocated_all_to_all_dispatch_metadata_tensors = (
                AllToAllDispatchMetadataConfig.create_preallocated_dispatch_output_tensors(
                    mesh_device,
                    batch,
                    hf_config.hidden_size,
                    hf_config.num_experts_per_tok,
                )
            )
            config[
                "quad_ring_preallocated_all_to_all_dispatch_metadata_tensors"
            ] = preallocated_all_to_all_dispatch_metadata_tensors

        return config

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
        num_links = ccl.get_max_links(axis=0)
        return {
            "all_to_all_dispatch": {
                "num_links": num_links,
            },
            "all_to_all_combine": {
                "num_links": num_links,
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
        batch_size_per_row: int,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Build operator configuration for decode or prefill."""

        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        _assert_quad_ring(fabric_config, mesh_device)

        mesh_stub = MeshDeviceStub(mesh_device.shape)
        memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        if mode == "decode":
            io_memory_config = _decode_sharded_io_memory_config(hf_config, mesh_device, batch_size_per_row)
            reduce_scatter_memory_config = io_memory_config
        else:
            io_memory_config = memory_config
            reduce_scatter_memory_config = memory_config

        config: ModelDecodeConfig | ModelPrefillConfig = {
            "mesh_device": mesh_stub,
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
            "input_memory_config": io_memory_config,
            "output_memory_config": io_memory_config,
            "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
            "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
            "sum_experts_output_memory_config": memory_config,
            "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                cluster_axis=1,
                dim=3,
                memory_config=reduce_scatter_memory_config,
            ),
            "revert_tp": AllGatherAsyncConfig(
                mesh_device=mesh_stub,
                dim=-1,
                memory_config=memory_config,
                cluster_axis=1,
            ),
        }

        if mode == "decode":
            hidden_size = hf_config.hidden_size
            tp_size = mesh_device.shape[1]
            config[
                "ring_sum_experts_output_memory_config"
            ] = DeepseekMoEReduceScatterConfig.create_default_input_memory_config(
                batch_size_per_row, hidden_size, tp_size
            )
            config["ring_final_output_reduce_scatter"] = DeepseekMoEReduceScatterConfig(
                cluster_axis=1,
                dim=3,
                output_memory_config=io_memory_config,
            )

        batch = batch_size_per_row * mesh_device.shape[0]
        seq_len = 1

        # TODO: #41009
        config["quad_ring_all_to_all_dispatch_metadata"] = AllToAllDispatchMetadataConfig(
            worker_mode=ttnn.WorkerMode.DIRECT,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            drain_sync_tilizer_core=(6, 9),
            cluster_axis=0,
            num_links=4,
            cross_device_semaphore=None,
        )
        config[
            "quad_ring_all_to_all_dispatch_metadata_sharded_memory_config"
        ] = AllToAllDispatchMetadataConfig.get_metadata_sharded_memory_config(
            batch_size_per_row, hf_config.num_experts_per_tok
        )
        config["quad_ring_moreh_full"] = MorehFullConfig(
            shape=[hf_config.num_experts_per_tok, batch_size_per_row, hf_config.hidden_size],
            fill_value=0,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        config["quad_ring_moe_compute"] = MoEComputeConfig(
            output_height_shard_dim=4,
            output_width_shard_dim=4,
            cluster_axis=0,
        )
        config["quad_ring_selective_reduce_combine"] = SelectiveReduceCombineConfig(
            hidden_size=hf_config.hidden_size,
            batch_size=batch,
            seq_size=seq_len,
            select_experts_k=hf_config.num_experts_per_tok,
            experts=hf_config.n_routed_experts,
            cluster_axis=0,
            token_parallel_core_dim=4,
            data_parallel_core_dim=4,
            worker_cores=ttnn.experimental.get_moe_combine_cores(mesh_device),
            mux_core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 7))}),
        )
        config["quad_ring_deepseek_moe_post_combine_tilize_config"] = DeepseekMoEPostCombineTilizeConfig(
            output_memory_config=DeepseekMoEPostCombineTilizeConfig.get_sharded_memory_config(),
        )

        # TODO: temporary until optimized ops support prefill shapes or prefill reads decode weights
        config["moe_chunk_size"] = batch_size_per_row

        return config

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "decode",
            batch_size_per_row=batch_size_per_row,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelPrefillConfig:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "prefill",
            batch_size_per_row=USERS_PER_ROW,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def _forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward_impl(x, cfg, "decode")

    @classmethod
    def _forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
        num_dispatch_devices = int(cfg.get("num_dispatch_devices", 1))
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            return cls._forward_chunked_prefill(x, cfg, chunk_size)
        return cls._forward_impl(x, cfg, "prefill")

    @classmethod
    def _forward_chunked_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig, chunk_size: int) -> ttnn.Tensor:
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks: list[ttnn.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            output_chunks.append(cls._forward_impl(x_chunk, cfg, "prefill"))
            ttnn.deallocate(x_chunk)

        if len(output_chunks) == 1:
            return output_chunks[0]
        output = ttnn.concat(output_chunks, dim=2)
        for chunk in output_chunks:
            ttnn.deallocate(chunk)
        return output

    @classmethod
    def _forward_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig, mode: str) -> ttnn.Tensor:
        # Validate input dimensions
        hidden_size = cfg["hidden_size"]
        mesh_device = cfg.get("mesh_device")
        tp_size = mesh_device.shape[1] if mesh_device else 1

        x_dim = x.shape[-1]
        expected_dims = [hidden_size, hidden_size // tp_size] if tp_size > 1 else [hidden_size]

        if x_dim not in expected_dims:
            raise ValueError(
                f"MoEOptimized: Unexpected input dimension {x_dim}. Expected one of {expected_dims}. "
                f"(hidden_size={hidden_size}, tp_size={tp_size})"
            )

        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # MOE
        if mode == "decode":
            post_combine_output_tensor = cls._fwd_decode_moe(
                x,
                topk_experts_indices,
                topk_experts_weights,
                cfg,
                batch_size_per_device,
                batch_size,
                seq_len,
            )
        else:
            post_combine_output_tensor = cls._fwd_prefill_moe(
                x,
                topk_experts_indices,
                topk_experts_weights,
                cfg,
                batch_size_per_device,
                batch_size,
                seq_len,
            )

        # Note: sum_experts and reduce_scatter is handled by the caller (decoder block or test)
        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_decode_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        # TODO: #41009 — can move towards removing these TMs once optimized moe_gate is integrated
        topk_experts_indices_rm = ttnn.to_layout(
            topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        topk_experts_weights_rm = ttnn.to_layout(
            topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        ttnn.deallocate(topk_experts_indices)
        ttnn.deallocate(topk_experts_weights)

        topk_experts_indices_rm = ttnn.permute(
            topk_experts_indices_rm, (2, 0, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        topk_experts_weights_rm = ttnn.permute(
            topk_experts_weights_rm, (2, 0, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG
        )

        post_combine_output_tensor = cls._forward_moe_optimized_ring_impl(
            cfg,
            x_rm,
            topk_experts_indices_rm,
            topk_experts_weights_rm,
        )

        return post_combine_output_tensor

    @classmethod
    def _fwd_prefill_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        # Chunk along local batch dimension to keep all_to_all_dispatch output small in prefill.
        chunk_size = min(batch_size_per_device, max(1, cfg.get("moe_chunk_size", batch_size_per_device)))
        output_chunks: list[ttnn.Tensor] = []

        # TODO: #41009
        topk_experts_indices_rm = ttnn.to_layout(
            topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        topk_experts_weights_rm = ttnn.to_layout(
            topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        ttnn.deallocate(topk_experts_indices)
        ttnn.deallocate(topk_experts_weights)

        # NOTE: store in DRAM while chunking, as moe_compute requires just about all of L1
        topk_experts_indices_rm = ttnn.permute(
            topk_experts_indices_rm, (2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        topk_experts_weights_rm = ttnn.permute(
            topk_experts_weights_rm, (2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            pad_amount = cfg["moe_chunk_size"] - batch_chunk

            x_rm_chunk = ttnn.slice(
                x_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["hidden_size"]],
            )
            x_rm_chunk = ttnn.pad(
                x_rm_chunk,
                padding=((0, pad_amount), (0, 0), (0, 0), (0, 0)),
                value=0.0,
            )

            topk_experts_indices_rm_chunk = ttnn.slice(
                topk_experts_indices_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )
            topk_experts_indices_rm_chunk = ttnn.pad(
                topk_experts_indices_rm_chunk,
                padding=((0, pad_amount), (0, 0), (0, 0), (0, 0)),
                value=0.0,
            )

            topk_experts_weights_rm_chunk = ttnn.slice(
                topk_experts_weights_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )
            topk_experts_weights_rm_chunk = ttnn.pad(
                topk_experts_weights_rm_chunk,
                padding=((0, pad_amount), (0, 0), (0, 0), (0, 0)),
                value=0.0,
            )

            post_combine_output_tensor = cls._forward_moe_optimized_ring_impl(
                cfg,
                x_rm_chunk,
                topk_experts_indices_rm_chunk,
                topk_experts_weights_rm_chunk,
            )
            post_combine_output_tensor = ttnn.slice(
                post_combine_output_tensor,
                [0, 0, 0, 0],
                [cfg["num_experts_per_tok"], 1, batch_chunk, cfg["hidden_size"]],
            )

            output_chunks.append(post_combine_output_tensor)

        ttnn.deallocate(topk_experts_indices_rm)
        ttnn.deallocate(topk_experts_weights_rm)

        if len(output_chunks) == 1:
            post_combine_output_tensor = output_chunks[0]
        else:
            post_combine_output_tensor = ttnn.concat(output_chunks, dim=2)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        ttnn.deallocate(x_rm)
        return post_combine_output_tensor

    @classmethod
    def _forward_moe_optimized_ring_impl(
        cls,
        cfg: RunDecodeConfig | RunPrefillConfig,
        x_rm: ttnn.Tensor,
        topk_experts_indices_rm: ttnn.Tensor,
        topk_experts_weights_rm: ttnn.Tensor,
    ):
        ccl = cfg["ccl"]

        topk_experts_indices_rm_sharded = ttnn.to_memory_config(
            topk_experts_indices_rm,
            memory_config=cfg["quad_ring_all_to_all_dispatch_metadata_sharded_memory_config"],
        )
        topk_experts_weights_rm_sharded = ttnn.to_memory_config(
            topk_experts_weights_rm,
            memory_config=cfg["quad_ring_all_to_all_dispatch_metadata_sharded_memory_config"],
        )

        # NOTE: L1 sharded topk_experts_weights need to be deallocated before moe_compute,
        # configure weights for post combine scaling prior to that deallocation, and store in DRAM
        topk_experts_weights_for_scaling = ttnn.permute(
            topk_experts_weights_rm, (3, 1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        topk_experts_weights_for_scaling = ttnn.to_layout(
            topk_experts_weights_for_scaling, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        ttnn.deallocate(topk_experts_indices_rm)
        ttnn.deallocate(topk_experts_weights_rm)

        # NOTE: needs to run prior to all_to_all_dispatch_metadata
        preallocated_combine_output = ttnn.moreh_full(**cfg["quad_ring_moreh_full"])

        # TODO: #41009
        (
            dispatch_output_sparse_buffer,
            dispatch_output_expert_indices,
            dispatch_output_expert_scores,
        ) = ttnn.experimental.all_to_all_dispatch_metadata(
            x_rm,
            topk_experts_indices_rm_sharded,
            topk_experts_weights_rm_sharded,
            cfg["expert_mapping_tensor"],
            output_tensors=None,
            **cfg["quad_ring_all_to_all_dispatch_metadata"],
        )

        # deallocation required in order to free up L1 space for moe_compute
        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm_sharded)
        ttnn.deallocate(topk_experts_weights_rm_sharded)

        # NOTE: we are actively working on fusing moe_compute and selective_reduce_combine
        (
            compute_output_token_counts,
            compute_output_dense_expert_activation,
            compute_ouput_dense_e_t,
            _,  # tile layout output of selective tilize (same buffer as output)
            compute_output,
        ) = ttnn.experimental.moe_compute(
            dispatch_output_sparse_buffer,
            dispatch_output_expert_indices,
            dispatch_output_expert_scores,
            cfg["expert_mapping_tensor"],
            cfg["moe_experts"]["quad_ring_w0_w1_experts"]["input_tensor_b"],
            cfg["moe_experts"]["quad_ring_w2_experts"]["input_tensor_b"],
            layer_id=0,  # each layer is composed of distinct tensors, as apposed to all layers fused together
            **cfg["quad_ring_moe_compute"],
        )

        # NOTE: can't deallocate dispatch output tensors as they are preallocated and reused across layers

        combine_output = ttnn.experimental.selective_reduce_combine(
            compute_output,
            compute_output_dense_expert_activation,
            compute_ouput_dense_e_t,
            compute_output_token_counts,
            output_tensor=preallocated_combine_output,
            **ccl.populate_selective_reduce_combine_args(cfg["quad_ring_selective_reduce_combine"]),
        )

        ttnn.deallocate(compute_output)
        ttnn.deallocate(compute_output_dense_expert_activation)
        ttnn.deallocate(compute_ouput_dense_e_t)
        ttnn.deallocate(compute_output_token_counts)

        combine_output = ttnn.unsqueeze(combine_output, dim=1)

        if combine_output.shape[2] == ttnn.TILE_SIZE:
            combine_output = ttnn.experimental.deepseek_moe_post_combine_tilize(
                combine_output,
                **cfg["quad_ring_deepseek_moe_post_combine_tilize_config"],
            )

        else:
            combine_output_shape = list(combine_output.shape)
            combine_output_shape[2] = (
                (combine_output_shape[2] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
            ) * ttnn.TILE_SIZE
            combine_output = ttnn.tilize_with_val_padding(
                combine_output,
                output_tensor_shape=combine_output_shape,
                pad_value=0.0,
                memory_config=cfg["quad_ring_deepseek_moe_post_combine_tilize_config"]["output_memory_config"],
            )

        post_combine_output_tensor = ttnn.mul(
            combine_output, topk_experts_weights_for_scaling, **cfg["mul_experts_output_with_weights"]
        )

        ttnn.deallocate(combine_output)
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
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        num_tokens_per_row = x.shape[-2]
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls._forward_decode(x, cfg)

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            tp_size = cfg["mesh_device"].shape[1]

            if is_ring_fabric(cfg["fabric_config"]) and tp_size == 8 and num_tokens_per_row == ttnn.TILE_SIZE:
                output = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                    output,
                    dim=0,
                    split_size=output.shape[-1] // tp_size,
                    output_memory_config=cfg["ring_sum_experts_output_memory_config"],
                )
                output = ttnn.experimental.deepseek_moe_reduce_scatter(
                    output, **cfg["ring_final_output_reduce_scatter"]
                )
            else:
                output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
                output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output

    @classmethod
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls._forward_prefill(x, cfg)

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
            output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output
