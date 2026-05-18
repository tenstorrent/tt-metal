# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig


def _tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16 or tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    if tt_dtype == ttnn.float32:
        return torch.float32
    if tt_dtype == ttnn.uint16:
        return torch.uint16
    raise ValueError(f"Unsupported tt dtype: {tt_dtype}")


class _TTMoEDecodeExpertState:
    def _load_weights():
        # TODO eventually support caching and loading weights
        pass

    def _validate():
        # TODO
        pass

    def _init_expert_mapping(expert_mapping: mesh_shape):
        expert_mapping_dtype = ttnn.uint16
        return ttnn.from_torch(
            torch_expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=expert_mapping_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(
                mesh_device,
                dim=0,
            ),
        )

    @staticmethod
    def _device_map_reorder_weights(
        torch_w0: "torch.Tensor", torch_w1: "torch.Tensor", torch_w2: "torch.Tensor", expert_mapping: list[int]
    ):
        ind = torch.Tensor([expert_mapping])
        return tuple([t[:, ind, :, :] for t in (torch_w0, torch_w1, torch_w2)])

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        torch_w0: "torch.Tensor",
        torch_w1: "torch.Tensor",
        torch_w2: "torch.Tensor",
        *,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
        has_bias: bool,
        num_routed_experts: int,
        expert_mapping: list[int],
        num_shared_experts: int,
        shared_expert_ids_to_devices: dict[int, list[int]] | None = None,
        shared_id_to_torch_w0: dict[int, "torch.Tensor"] | None = None,
        shared_id_to_torch_w1: dict[int, "torch.Tensor"] | None = None,
        shared_id_to_torch_w2: dict[int, "torch.Tensor"] | None = None,
        torch_b0: "torch.Tensor" | None = None,
        torch_b1: "torch.Tensor" | None = None,
        torch_b2: "torch.Tensor" | None = None,
    ):
        self._validate()

        mapped_torch_w0, mapped_torch_w1, mapped_torch_w2 = self._device_map_reorder_weights(
            torch_w0, torch_w1, torch_w2, expert_mapping
        )

        if shared_expert_ids_to_devices is not None:
            total_torch_w0, total_torch_w1, total_torch_w2 = add_shared_expert_weights(
                mapped_torch_w0,
                mapped_torch_w1,
                mapped_torch_w2,
                shared_id_to_torch_w0,
                shared_id_to_torch_w1,
                shared_id_to_torch_w2,
                shared_expert_ids_to_devices,
                mesh_device.num_devices(),
            )

            total_expert_mapping = map_shared_experts(
                expert_mapping, shared_expert_ids_to_devices, mesh_shape, cluster_axis
            )

        else:
            total_torch_w0, total_torch_w1, total_torch_w2 = mapped_torch_w0, mapped_torch_w1, mapped_torch_w2
            total_expert_mapping = expert_mapping

        self.tt_expert_mapping = self._init_expert_mapping(
            total_expert_mapping, shared_expert_ids_to_devices, mesh_shape, cluster_axis
        )

        self.tt_w0_w1, self.tt_w2 = self._init_total_expert_weights_impl(
            total_torch_w0,
            total_torch_w1,
            total_torch_w2,
            cluster_axis,
            mesh_device,
            has_bias,
            torch_b0,
            torch_b1,
            torch_b2,
        )

    @staticmethod
    def _init_total_expert_weights_impl(
        torch_w0: "torch.Tensor",
        torch_w1: "torch.Tensor",
        torch_w2: "torch.Tensor",
        cluster_axis: int,
        mesh_device: ttnn.MeshDevice,
        has_bias: bool,
        torch_b0: "torch.Tensor" | None = None,
        torch_b1: "torch.Tensor" | None = None,
        torch_b2: "torch.Tensor" | None = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        # TODO validate these be comparing to explicit values in the config
        num_layers = torch_w0.shape[0]
        hidden_size = torch_w0.shape[-2]
        intermediate_size = torch_w0.shape[-1]
        total_experts_per_device = torch_w0.shape[1] // mesh_device.get_num_devices()

        w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
            mesh_device, hidden_size, intermediate_size
        )

        if has_bias:
            torch_w0_w1_reordered = prepare_w0_w1_tensor_with_bias(
                torch_w0,
                torch_w1,
                torch_b0,
                torch_b1,
                num_layers,
                total_experts_per_device,
                hidden_size,
                intermediate_size,
                w0_w1_shard_map,
            )

            torch_w2_reordered = prepare_w2_tensor_with_bias(
                torch_w2,
                torch_b2,
                num_layers,
                total_experts_per_device,
                intermediate_size,
                hidden_size,
                w2_shard_map,
                w0_w1_shard_map,
            )

        else:
            torch_w0_w1_reordered = prepare_w0_w1_tensor_for_moe_compute(
                torch_w0,
                torch_w1,
                num_layers,
                total_experts_per_device,
                hidden_size,
                intermediate_size,
                w0_w1_shard_map,
            )
            torch_w2_reordered = prepare_w2_tensor_for_moe_compute(
                torch_w2, num_layers, experts_per_device, intermediate_size, hidden_size, w2_shard_map, w0_w1_shard_map
            )

        w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total = get_weight_mem_configs(
            num_layers,
            total_experts_per_device,
            hidden_size,
            intermediate_size,
            w0_w1_shard_map,
            w2_shard_map,
            dram_core_range_set,
        )

        tt_w0_w1 = ttnn.from_torch(
            torch_w0_w1_reordered,
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device),
        )

        tt_w2 = ttnn.from_torch(
            torch_w2_reordered,
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device),
        )

        return tt_w0_w1, tt_w2


class _TTMoEDecodeBuffers:
    """Persistent buffers and semaphores for the MoE decode pipeline.

    Allocates the dispatch output triple (sparse buffer, expert indices, expert
    scores), the dispatch/combine cross-device semaphores, and the combine
    output buffer once in __init__, for reuse across forward() calls.

    Shapes and memory configs mirror those used in test_optimized_moe_decode_block.py.
    """

    SPARSE_BUFFER_DTYPE = ttnn.bfloat16
    INDICES_DTYPE = ttnn.uint16
    SCORES_DTYPE = ttnn.bfloat16
    COMBINE_OUTPUT_DTYPE = ttnn.bfloat16

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
        batch_per_device: int,
        hidden_size: int,
        effective_experts_k: int,
        shard_dim: int,
        compute_tilize_drain_core: ttnn.CoreCoord,
    ):
        # --- derived sizes (seq=1 for decode) ---
        num_dispatch_devices = mesh_shape[cluster_axis]
        total_tokens = batch_per_device * num_dispatch_devices
        tokens_per_device = batch_per_device

        shard_dims = (shard_dim, None) if cluster_axis == 0 else (None, shard_dim)

        # --- global semaphores (one each, no double buffering required —
        # combine syncs after reading dispatch output, dispatch syncs at end) ---
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        self.dispatch_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        self.combine_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

        # --- dispatch output buffers ---
        # Sparse buffer: DRAM, row-major, sharded along cluster axis
        sparse_buffer = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, hidden_size],
                dtype=_tt_to_torch_dtype(self.SPARSE_BUFFER_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.SPARSE_BUFFER_DTYPE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        # Indices / scores share an L1 height-sharded mem config on the drain core
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
            [total_tokens, effective_experts_k],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        l1_height_sharded = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        indices = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, effective_experts_k],
                dtype=_tt_to_torch_dtype(self.INDICES_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.INDICES_DTYPE,
            memory_config=l1_height_sharded,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        scores = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, effective_experts_k],
                dtype=_tt_to_torch_dtype(self.SCORES_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.SCORES_DTYPE,
            memory_config=l1_height_sharded,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        self.tt_dispatch_output_tensors = (sparse_buffer, indices, scores)
        self.tt_combine_output = ttnn.from_torch(
            torch.zeros(
                [effective_experts_k, tokens_per_device, hidden_size],
                dtype=_tt_to_torch_dtype(self.COMBINE_OUTPUT_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.COMBINE_OUTPUT_DTYPE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class TTMoEDecode:
    DEEPSEEK_RS_DP_DIM: int = 8
    SKIP_RS_DP_DIM: int = 1

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: TTMoEDecodeConfig,
        torch_w0: torch.Tensor,
        torch_w1: torch.Tensor,
        torch_w2: torch.Tensor,
        shared_id_to_torch_w0: dict[int, torch.Tensor] | None = None,
        shared_id_to_torch_w1: dict[int, torch.Tensor] | None = None,
        shared_id_to_torch_w2: dict[int, torch.Tensor] | None = None,
        torch_b0: torch.Tensor | None = None,
        torch_b1: torch.Tensor | None = None,
        torch_b2: torch.Tensor | None = None,
    ):
        self.config = config
        self.state = _TTMoEDecodeExpertState(
            mesh_device,
            torch_w0,
            torch_w1,
            torch_w2,
            **config.state.model_dump(),
            shared_id_to_torch_w0=shared_id_to_torch_w0,
            shared_id_to_torch_w1=shared_id_to_torch_w1,
            shared_id_to_torch_w2=shared_id_to_torch_w2,
            torch_b0=torch_b0,
            torch_b1=torch_b1,
            torch_b2=torch_b2,
        )
        self.buffers = _TTMoEDecodeBuffers(mesh_device, **config.buffers.model_dump())

    def _format_dispatch_inputs(
        self,
        tt_x: ttnn.Tensor,
        tt_indices: ttnn.Tensor,
        tt_scores: ttnn.Tensor,
    ):
        if tt_x.memory_config() != self.config.dispatch_input_memory_config:
            tt_dispatch_input_tensor_bundle = (
                ttnn.to_memory_config(tt_x, memory_config=self.config.dispatch_input_memory_config),
                True,
            )
        else:
            tt_dispatch_input_tensor_bundle = tt_x, False

        if tt_indices.memory_config() != self.config.dispatch_input_memory_config:
            tt_dispatch_input_expert_indices_tensor_bundle = (
                ttnn.to_memory_config(
                    tt_indices,
                    memory_config=self.config.dispatch_input_memory_config,
                ),
                True,
            )
        else:
            tt_dispatch_input_expert_indices_tensor_bundle = tt_indices, False

        if tt_scores.memory_config() != self.config.dispatch_input_expert_scores_memory_config:
            tt_dispatch_input_expert_scores_tensor_bundle = (
                ttnn.to_memory_config(
                    tt_scores,
                    memory_config=self.config.dispatch_input_expert_scores_memory_config,
                ),
                True,
            )
        else:
            tt_dispatch_input_expert_scores_tensor_bundle = tt_scores, False

        return (
            tt_dispatch_input_tensor_bundle,
            tt_dispatch_input_expert_indices_tensor_bundle,
            tt_dispatch_input_expert_scores_tensor_bundle,
        )

    def forward(
        self, tt_x: ttnn.Tensor, tt_scores: ttnn.Tensor, tt_indices: ttnn.Tensor, layer_id: int = 0
    ) -> ttnn.Tensor:
        (
            (tt_dispatch_input_tensor, dealloc_input),
            (tt_dispatch_input_expert_indices_tensor, dealloc_indices),
            (tt_dispatch_input_expert_scores_tensor, dealloc_scores),
        ) = self._format_dispatch_inputs(tt_x, tt_indices, tt_scores)

        (
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
        ) = ttnn.experimental.all_to_all_dispatch_metadata(
            tt_dispatch_input_tensor,
            tt_dispatch_input_expert_indices_tensor,
            tt_dispatch_input_expert_scores_tensor,
            self.state.tt_expert_mapping,
            layer_id=layer_id,
            **self.config.dispatch.model_dump(),
            # shared_expert_ids
            # cluster_axi
            # num_links
            # drain_sync_tilizer_core ???
            # worker_mode
            # dispatch_algorithm
            output_tensors=self.buffers.tt_dispatch_output_tensors,
            cross_device_semaphore=self.buffers.dispatch_global_semaphore,
        )

        if dealloc_input:
            ttnn.deallocate(tt_dispatch_input_tensor)

        if dealloc_scores:
            ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        (
            _,
            _,
            _,
            tt_l1_compute_output,
            _,
            tt_combine_output,
        ) = ttnn.experimental.moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            self.state.tt_expert_mapping,
            self.state.tt_w0_w1,
            self.state.tt_w2,
            layer_id=layer_id,
            # output_height_shard_dim=compute_output_height_shard_dim,
            # cluster_axis=cluster_axis,
            # mux_core_range_set=combine_mux_cores,
            # has_bias
            # activation_type
            **self.config.compute.model_dump(),
            optional_output_tensor=self.buffers.tt_combine_output,
            optional_cross_device_semaphore=self.buffers.combine_global_semaphore,
        )
        ttnn.deallocate(tt_l1_compute_output)

        # unsqueeze
        # [select_experts_k, tokens_per_device, hidden_size] -> [select_experts_k, 1, tokens_per_device, hidden_size]
        tt_unsqueezed_output = ttnn.unsqueeze(tt_combine_output, dim=1)

        if self.config.batch_per_device == ttnn.TILE_SIZE:
            tt_tilized_compute_output = ttnn.experimental.deepseek_moe_post_combine_tilize(
                tt_unsqueezed_output,
                # output_memory_config,
                **self.config.post_combine_tilize.model_dump(),
            )

        else:
            output_tensor_shape = list(tt_unsqueezed_output.shape)
            output_tensor_shape[2] = ((output_tensor_shape[2] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            tt_tilized_compute_output = ttnn.tilize_with_val_padding(
                tt_unsqueezed_output,
                output_tensor_shape=output_tensor_shape,
                pad_value=0.0,
                # memory_config=post_combine_tilize_output_memory_config,
                **self.config.tilize_with_val_padding.model_dump(),
            )

        # scale with scores and accumulate
        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            tt_tilized_compute_output,
            tt_dispatch_input_expert_indices_tensor,
            self.state.tt_expert_mapping,
            # reduce_dim=0,
            # cluster_axis=cluster_axis,
            # split_size=int(tt_tilized_compute_output.shape[-1] // num_replicated_devices),
            # output_memory_config=fast_reduce_output_memory_config,
            # num_shared_experts=num_shared_experts,
            # shared_expert_scale=shared_expert_score,
            **self.config.reduce.model_dump(),
            scores_tensor=tt_scores,
        )

        if dealloc_indices:
            ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)

        # [select_experts_k, tokens_per_device, hidden_size // num_replicated_devices] final per device shape
        if self.config.mesh_shape[1 - self.config.cluster_axis] == self.DEEPSEEK_RS_DP_DIM:
            tt_final_output = ttnn.experimental.deepseek_moe_reduce_scatter(
                tt_fast_reduce_output_tensors,
                # output_memory_config=rs_output_memory_config,
                # dim=-1,
                # num_links=4,
                # topology=ttnn.Topology.Ring,
                # cluster_axis=1,
                **self.config.deepseek_moe_reduce_scatter.model_dump(),
            )
        elif self.config.mesh_shape[1 - self.config.cluster_axis] == SKIP_RS_DP_DIM:
            tt_final_output = tt_fast_reduce_output_tensors[0]
        else:
            tt_final_output = ttnn.reduce_scatter(
                tt_fast_reduce_output_tensors, **self.config.reduce_scatter.model_dump()
            )

        return tt_final_output
