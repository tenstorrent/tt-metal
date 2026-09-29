# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1 decoder block, single-chunk prefill prototype (§4).

The residual is ``hc_mult`` fp32 streams ``[1, 1, S/sp, hc_mult * hidden/tp]`` (per chip: its hidden slice
of every stream), as in the in-tree V4 block. V4.1 lags the hyper-connection collapse by one sublayer:
attention collapses with the incoming ``pre_mix``, the FFN with the ``pre`` split before attention, and
the block returns the ``pre`` split before the FFN for the next block (``inference/model.py``
``Block.forward``).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v41.attention import TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS, V41MeshLayout
from models.demos.deepseek_v3_d_p.tt.v41.mhc import TtV41HyperConnections
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe


class TtV41Block(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        seq_len: int,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path=None,
    ):
        """``weights``: ``attn`` (TtV41Attention weights), ``attn_norm``, ``ffn_norm``, ``hc_attn`` / ``hc_ffn``
        (fn, base, scale), and the MoE state-dict entries ``gate_weights``, ``routed_expert_weights``,
        ``shared_expert_weights``."""
        V41MeshLayout.of(mesh_device).check_model(config)
        self.layer = layer
        topology = per_axis_topology()[TP_AXIS]  # the norms reduce over TP
        norm = lambda w: TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.EMB_SIZE,
            torch_weight=w,
            epsilon=config.RMS_NORM_EPS,
            cluster_axis=TP_AXIS,
            num_links=fabric_num_links(),
            topology=topology,
        )
        self.attn_norm = norm(weights["attn_norm"])
        self.ffn_norm = norm(weights["ffn_norm"])
        self.residual = TtV41HyperConnections(mesh_device, config, weights["hc_attn"], weights["hc_ffn"])
        self.attn = TtV41Attention(
            mesh_device,
            config,
            layer,
            weights["attn"] | {k: weights[k] for k in ("compressor", "indexer") if k in weights},
        )
        self.ffn = TtV41Moe(
            mesh_device,
            config,
            layer,
            weights,
            seq_len,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
        )

    def forward(self, x, pre_mix, state, length: int, image_mask=None):
        """x [1, 1, S/sp, hc*hidden/tp] fp32, pre_mix [1, 1, S/sp, hc] fp32 for the chunk at ``state.start``
        with ``length`` valid tokens -> (x, next pre_mix). ``image_mask`` [1, 1, S/sp, 1] (1 = image token; None =
        text only) selects the MoE gate's ``bias_vl`` (reference ``Block.forward``)."""
        return self.residual(
            x,
            pre_mix,
            attention=lambda h: self.attn(self.attn_norm(h), state, length),
            ffn=lambda h: self.ffn(self.ffn_norm(h), image_mask=image_mask),
        )
