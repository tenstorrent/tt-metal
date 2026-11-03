# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``TTMoEDecodeCustom``: ``models/common/moe`` ``TTMoEDecode`` whose decode step runs the gemma4-local
``MoeComputeCustom`` op (custom_ops/moe_compute, per-expert resident weights) followed by the standalone
``ttnn.experimental.selective_reduce_combine`` op instead of the fused ``ttnn.experimental.moe_compute``.

``TTMoEDecode.forward`` (tt_moe_decode.py) is the only place the Gemma 4 lane engine reaches ``moe_compute``, so this
subclass runs that method with the one op call redirected to the pair (same inputs, same six outputs; the combine
takes the fused op's CCL knobs: the config's cluster axis and mux range, the fabric topology resolved as the fused op
resolves it, every link the fabric reports). Selected by ``tt/experts/moe_decode.py`` under
``GEMMA4_MOE_COMPUTE_CUSTOM_OP=1`` when ``custom_reason`` is None.
"""

from __future__ import annotations

import os

import ttnn
from models.common.moe.tt_moe_decode import TTMoEDecode
from models.demos.gemma4.custom_ops.moe_compute.op import MoeComputeCustom


def custom_op_enabled() -> bool:
    return os.environ.get("GEMMA4_MOE_COMPUTE_CUSTOM_OP", "0") == "1"


def custom_reason(mesh_device, config, num_routed_experts: int):
    """None when ``MoeComputeCustom`` can serve this ``TTMoEDecodeConfig`` on ``mesh_device``, else why not (the
    wrapper then keeps ``TTMoEDecode``): the arch and the standalone staging capacity at the frame's tokens
    (``batch_per_device`` x the dispatch axis extent) for this chip's experts."""
    cc = config.compute
    rows, cols = mesh_device.shape
    tokens = int(config.batch_per_device) * int((rows, cols)[cc.cluster_axis])
    return MoeComputeCustom.unsupported_reason(
        mesh_device,
        hidden_size=int(config.hidden_size),
        experts_per_device=int(num_routed_experts) // (int(rows) * int(cols)),
        tokens=tokens,
        output_height_shard_dim=int(cc.output_height_shard_dim),
    )


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
        """``TTMoEDecode.forward`` with its one ``ttnn.experimental.moe_compute`` call served by ``_moe_compute`` (the
        custom op + the standalone combine). The shared pipeline is not copied: the op symbol is redirected for the
        duration of the call (same six outputs, the extra keyword arguments of the shared call are the config's)."""
        exp = ttnn.experimental
        shared_op = exp.moe_compute

        def _redirected(sparse, indices, scores, mapping, w0_w1, w2, *args, layer_id=layer_id, **kwargs):
            return self._moe_compute(sparse, indices, scores, layer_id)

        exp.moe_compute = _redirected
        try:
            return TTMoEDecode.forward(self, tt_x, tt_scores, tt_indices, layer_id=layer_id)
        finally:
            exp.moe_compute = shared_op
