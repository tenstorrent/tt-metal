# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``TTMoEDecodeCustom``: ``models/common/moe`` ``TTMoEDecode`` whose decode step runs the gemma4-local
``MoeComputeCustom`` op (custom_ops/moe_compute, per-expert resident weights) followed by the standalone
``ttnn.experimental.selective_reduce_combine`` op instead of the fused ``ttnn.experimental.moe_compute``.

``TTMoEDecode.forward`` (tt_moe_decode.py) is the only place the Gemma 4 lane engine reaches ``moe_compute``, so this
subclass overrides ``forward`` with a copy of that method and swaps the one op call for the pair (same inputs, same
six outputs; the combine takes the fused op's CCL knobs: the config's cluster axis and mux range, the fabric topology
resolved as the fused op resolves it, every link the fabric reports). Selected by ``tt/experts/moe_decode.py`` under
``GEMMA4_MOE_COMPUTE_CUSTOM_OP=1``.
"""

from __future__ import annotations

import os

import ttnn
from models.common.moe.tt_moe_decode import TTMoEDecode
from models.demos.gemma4.custom_ops.moe_compute.op import MoeComputeCustom


def custom_op_enabled() -> bool:
    return os.environ.get("GEMMA4_MOE_COMPUTE_CUSTOM_OP", "0") == "1"


class TTMoEDecodeCustom(TTMoEDecode):
    def __init__(self, mesh_device, config, *args, **kwargs):
        super().__init__(mesh_device, config, *args, **kwargs)
        self._mesh_device = mesh_device
        cc = config.compute
        experts_per_device = int(self.expert_state.tt_w0_w1.shape[2])
        self._custom_op = MoeComputeCustom(
            mesh_device,
            hidden_size=config.hidden_size,
            intermediate_size=cc.intermediate_size,
            experts_per_device=experts_per_device,
            select_experts_k=config.select_experts_k,
            output_height_shard_dim=cc.output_height_shard_dim,
            mux_core_range_set=cc.mux_core_range_set,
            activation_type=cc.activation_type,
            cluster_axis=cc.cluster_axis,
            has_bias=cc.has_bias,
            num_shared_experts_per_device=cc.num_shared_experts_per_device,
            staging_layout="standalone",
        )
        rows, cols = mesh_device.shape
        axis_extent = (rows, cols)[cc.cluster_axis]
        # the fused op's get_usable_topology: an axis of extent 2 is a line whatever the fabric says
        self._combine_topology = ttnn.Topology.Linear if axis_extent <= 2 else config.topology
        self._combine_num_links = ttnn.get_num_links(mesh_device, cc.cluster_axis)
        self._combine_cores = ttnn.experimental.get_moe_combine_cores(
            mesh_device,
            cc.output_height_shard_dim,
            self._custom_op.ws,
            config.hidden_size,
            mux_core_range_set=cc.mux_core_range_set,
        )
        self._total_tokens = config.batch_per_device * axis_extent

    def _moe_compute(self, sparse, indices, scores, layer_id):
        """The fused op's six outputs from the custom op plus the standalone combine."""
        cc = self.config.compute
        counts, activation, e_t, tt_l1_compute_output, staging = self._custom_op(
            sparse,
            indices,
            scores,
            self.expert_state.tt_expert_mapping,
            self.expert_state.tt_w0_w1,
            self.expert_state.tt_w2,
            layer_id=layer_id,
        )
        tt_combine_output = ttnn.experimental.selective_reduce_combine(
            staging,
            activation,
            e_t,
            counts,
            self.config.hidden_size,
            1,
            self._total_tokens,
            self.config.select_experts_k,
            cc.cluster_axis,
            topology=self._combine_topology,
            num_links=self._combine_num_links,
            token_parallel_core_dim=cc.output_height_shard_dim,
            data_parallel_core_dim=self._custom_op.ws,
            worker_cores=self._combine_cores,
            mux_core_range_set=cc.mux_core_range_set,
            output_tensor=self.buffers.tt_combine_output,
            optional_cross_device_semaphore=self.buffers.combine_global_semaphore,
        )
        return counts, activation, e_t, tt_l1_compute_output, staging, tt_combine_output

    def forward(self, tt_x, tt_scores, tt_indices, layer_id: int = 0):
        # -- TTMoEDecode.forward with the moe_compute call swapped (see its docstring for the pipeline) ------------
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
            self.expert_state.tt_expert_mapping,
            **self.config.dispatch.model_dump(),
            output_tensors=self.buffers.tt_dispatch_output_tensors,
            cross_device_semaphore=self.buffers.dispatch_global_semaphore,
        )
        if dealloc_input:
            ttnn.deallocate(tt_dispatch_input_tensor)
        if dealloc_scores:
            ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        # -- the one swapped call ----------------------------------------------------------------------------------
        (_, _, _, tt_l1_compute_output, _, tt_combine_output) = self._moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            layer_id,
        )
        ttnn.deallocate(tt_l1_compute_output)

        tt_unsqueezed_output = ttnn.unsqueeze(tt_combine_output, dim=1)
        tt_unsqueezed_output = self._pad_for_fast_reduce(tt_unsqueezed_output)
        if self.config.use_post_combine_tilize:
            tt_tilized_compute_output = ttnn.experimental.deepseek_moe_post_combine_tilize(
                tt_unsqueezed_output, **self.config.post_combine_tilize.model_dump()
            )
        else:
            output_tensor_shape = list(tt_unsqueezed_output.shape)
            output_tensor_shape[2] = ((output_tensor_shape[2] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            tt_tilized_compute_output = ttnn.tilize_with_val_padding(
                tt_unsqueezed_output,
                output_tensor_shape=output_tensor_shape,
                pad_value=0.0,
                **self.config.tilize_with_val_padding.model_dump(),
            )
        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            tt_tilized_compute_output,
            tt_dispatch_input_expert_indices_tensor,
            self.expert_state.tt_expert_mapping,
            **self.config.reduce.model_dump(),
            scores_tensor=tt_scores,
        )
        ttnn.deallocate(tt_tilized_compute_output)
        if dealloc_indices:
            ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)

        if (
            self.config.mesh_shape[1 - self.config.cluster_axis] == self.DEEPSEEK_RS_DP_DIM
            and self.config.topology == ttnn.Topology.Ring
        ):
            tt_final_output = ttnn.experimental.deepseek_moe_reduce_scatter(
                tt_fast_reduce_output_tensors, **self.config.deepseek_moe_reduce_scatter.model_dump()
            )
            for t in tt_fast_reduce_output_tensors:
                ttnn.deallocate(t)
        elif self.config.mesh_shape[1 - self.config.cluster_axis] == self.SKIP_RS_DP_DIM:
            tt_final_output = tt_fast_reduce_output_tensors[0]
        else:
            tt_final_output = ttnn.reduce_scatter(
                tt_fast_reduce_output_tensors[0], **self.config.reduce_scatter.model_dump()
            )
            for t in tt_fast_reduce_output_tensors:
                ttnn.deallocate(t)
        return self._unpad_after_reduce_scatter(tt_final_output)
