# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1 decoder block, single-chunk prefill prototype (§4).

The residual is ``hc_mult`` fp32 streams ``[1, 1, S/sp, hc_mult * hidden/tp]`` (per chip: its hidden slice
of every stream), as in the in-tree V4 block. V4.1 lags the hyper-connection collapse by one sublayer:
attention collapses with the incoming ``pre_mix``, the FFN with the ``pre`` split before attention, and
the block returns the ``pre`` split before the FFN for the next block (``inference/model.py``
``Block.forward``).
"""

from types import SimpleNamespace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.attention import TtV41Attention

SUBLAYER_DTYPE = ttnn.bfloat16


class TtV41Block(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        seq_len: int,
        host,
        num_links: int = 1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: ``attn`` (TtV41Attention weights), ``attn_norm``, ``ffn_norm``, ``hc_attn`` / ``hc_ffn``
        (fn, base, scale), and the MoE state-dict entries ``gate_weights``, ``routed_expert_weights``,
        ``shared_expert_weights``."""
        tp_axis, sp_axis = 1, 0
        self.layer = layer
        norm = lambda w: TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.EMB_SIZE,
            torch_weight=w,
            epsilon=config.RMS_NORM_EPS,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
        )
        self.attn_norm = norm(weights["attn_norm"])
        self.ffn_norm = norm(weights["ffn_norm"])
        mhc_cfg = MHCConfig(
            dim=config.EMB_SIZE,
            n=config.HC_MULT,
            sinkhorn_iters=config.HC_SINKHORN_ITERS,
            eps=config.HC_EPS,
            norm_eps=config.RMS_NORM_EPS,
        )
        self.attn_res = TtMHCWrap(mesh_device, mhc_cfg, *weights["hc_attn"], tp_axis=tp_axis, topology=topology)
        self.ffn_res = TtMHCWrap(mesh_device, mhc_cfg, *weights["hc_ffn"], tp_axis=tp_axis, topology=topology)
        self.attn = TtV41Attention(mesh_device, config, layer, weights["attn"], seq_len, host, topology=topology)
        self.ffn = TtPrefillBlock._build_moe(
            mesh_device=mesh_device,
            model_cfg=config,
            config=SimpleNamespace(rms_norm_eps=config.RMS_NORM_EPS),
            state_dict=weights,
            seq_len=seq_len,
            sp_axis=sp_axis,
            emb_dim=config.EMB_SIZE,
            num_links=num_links,
            topology=topology,
            gate_fallback_mode=GateComputeMode.DEVICE_FP32,
            routed_expert_activations_dtype=ttnn.bfloat8_b,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_activations_dtype=ttnn.bfloat16,
            shared_expert_weights_dtype=ttnn.bfloat8_b,
            dispatch_buffer_capacity_factor=2,
            layer_idx=layer,
        )

    def _sublayer(self, h, norm, fn):
        cast = ttnn.typecast(h, SUBLAYER_DTYPE)
        out = fn(norm(cast))
        return ttnn.typecast(out, ttnn.float32)

    def _moe(self, h):
        # TtMoe works in 3D and the prototype uploads the sequence as a plain contiguous SP shard.
        out, _ = self.ffn(ttnn.squeeze(h, dim=0), actual_isl=None, padding_side="right", actual_start=0)
        return ttnn.unsqueeze(out, dim=0)

    def forward(self, x, pre_mix):
        """x [1, 1, S/sp, hc*hidden/tp] fp32, pre_mix [1, 1, S/sp, hc] fp32 -> (x, next pre_mix)."""
        attn_pre, attn_post, attn_comb = self.attn_res.split(x)
        h = self._sublayer(self.attn_res.collapse(x, pre_mix), self.attn_norm, self.attn)
        x = self.attn_res.hc_post(h, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_res.split(x)
        h = self._sublayer(self.ffn_res.collapse(x, attn_pre), self.ffn_norm, self._moe)
        x = self.ffn_res.hc_post(h, x, ffn_post, ffn_comb)
        return x, ffn_pre
