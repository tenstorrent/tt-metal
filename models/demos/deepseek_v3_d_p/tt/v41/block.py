# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One DeepSeek-V4.1-Flash block for the prefill (eager), ``model.py`` ``Block.forward``:

    attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)        # this block's attention site
    x = hc_post(attn(attn_norm(hc_pre(x, pre_mix))), x, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
    x = hc_post(ffn(ffn_norm(hc_pre(x, attn_pre))), x, ffn_post, ffn_comb)
    return x, ffn_pre                                              # ffn_pre collapses the NEXT block's attention input

SINGLE-PASS mHC: each site's collapse uses the PREVIOUS site's ``pre`` (layer 0's is the identity one-hot [1, 0, 0, 0]),
where V4-Flash collapsed each site with its own. The mixes themselves are V4-Flash's (``tt/v4/mhc_math.py``), so
``V41HyperConnection`` is ``TtHyperConnection`` with the mix row and the collapse split apart; the ``pre`` handed between
sites is the fp32 ``[1, 1, S_l, 32]`` mix row (columns 0..3 meaningful).

Streams: a LIST of 4 bf16 ``[1, 1, S_l, D_l]`` (SP on S, TP on D), as V4-Flash.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v4 import mhc_math as M
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import TtHyperConnection

from .attention import V41CSA, V41SWA, V41CSAConsumer
from .moe import build_v41_moe

HC = M.HC


class V41HyperConnection(TtHyperConnection):
    """``mixes(streams) -> (pre_row, post_row, comb_row)`` and ``collapse(streams, pre_row) -> bf16 [1, 1, S_l, D_l]``;
    ``mix`` (the hc_post) is inherited."""

    def mixes(self, streams: list):
        assert len(streams) == HC and not self.stacked
        xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
        mix = self._mix_row(xf)
        for x in xf:
            ttnn.deallocate(x)
        aff = ttnn.add(ttnn.multiply(mix, self.SV), self.BV)
        sig = ttnn.sigmoid(aff)
        pre = ttnn.add(sig, self.hc_eps)
        post = ttnn.multiply(sig, 2.0)
        E = ttnn.add(ttnn.multiply(ttnn.exp(ttnn.clamp(aff, -M.EXP_CLAMP, M.EXP_CLAMP)), self.COMB_MASK), self.ONE_COL)
        X = ttnn.div(E, self._mm(E, self.R_SOFT))
        X = ttnn.add(X, self.EPS_COMB)
        X = ttnn.div(X, self._mm(X, self.C_AUG))
        for _ in range(self.sinkhorn_iters - 1):
            X = ttnn.div(X, self._mm(X, self.R_AUG))
            X = ttnn.div(X, self._mm(X, self.C_AUG))
        return pre, post, X

    def collapse(self, streams: list, pre_row):
        xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
        out = ttnn.typecast(self._weighted_sum(pre_row, M.PRE0, xf), streams[0].dtype)
        for x in xf:
            ttnn.deallocate(x)
        return out


def identity_pre_row(mesh_device, seq_local: int, sp_axis: int = 0):
    """Layer 0's ``pre_mix`` (``make_identity_pre_mix``): column 0 = 1, as an fp32 [1, 1, S_l, 32] row per chip."""
    row = torch.zeros(1, 1, seq_local, M.ROW)
    row[..., M.PRE0] = 1.0
    return ttnn.from_torch(
        row,
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_v41_attention(mesh_device, cfg, layer: int, w: dict, rotary_emb, *, sources: dict, **kw):
    """``sources``: {layer: module} of the KV sources built so far (a consumer attends to its source's entries)."""
    r = cfg.role(layer)
    if r.mode == "swa":
        return V41SWA.from_weights(mesh_device, cfg, layer, w, rotary_emb, **kw)
    if r.mode == "full":
        return V41CSA.from_weights(mesh_device, cfg, layer, w, rotary_emb, **kw)
    assert r.mode == "reuse", f"layer {layer}: {r.mode} attention is a decoder layer (not built by the prefill)"
    return V41CSAConsumer.from_weights(mesh_device, cfg, layer, w, rotary_emb, source=sources[r.kv_source], **kw)


class V41PrefillBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        layer: int,
        w: dict,
        *,
        rotary_emb,
        seq_len_per_chip: int,
        sources: dict,
        ck=None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        weight_cache_path=None,
        load_routed_from_cache: bool = False,
    ):
        super().__init__()
        self.mesh_device, self.cfg, self.layer = mesh_device, cfg, int(layer)
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        D = cfg.dim
        hc = dict(
            hidden=D,
            rms_eps=cfg.norm_eps,
            hc_eps=cfg.hc_eps,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        self.attn_hc = V41HyperConnection(
            mesh_device, fn=w["hc_attn_fn"], base=w["hc_attn_base"], scale=w["hc_attn_scale"], **hc
        )
        self.ffn_hc = V41HyperConnection(
            mesh_device, fn=w["hc_ffn_fn"], base=w["hc_ffn_base"], scale=w["hc_ffn_scale"], **hc
        )

        def norm(name):
            return TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=D,
                epsilon=cfg.norm_eps,
                torch_weight=w[f"{name}.weight"].float(),
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"v41_layer_{layer}.{name}",
            )

        self.attn_norm, self.ffn_norm = norm("attn_norm"), norm("ffn_norm")
        self.attn = build_v41_attention(
            mesh_device,
            cfg,
            layer,
            w,
            rotary_emb,
            sources=sources,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            weight_cache_path=weight_cache_path,
        )
        if cfg.role(layer).mode == "full":
            sources[self.layer] = self.attn
        self.moe = build_v41_moe(
            mesh_device,
            cfg,
            layer,
            w,
            seq_len_per_chip=seq_len_per_chip,
            ck=ck,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            load_routed_from_cache=load_routed_from_cache,
        )
        self.state = None

    def alloc_state(self, max_seq_len: int, chunk_tokens: int):
        self.state = self.attn.alloc_state(max_seq_len, chunk_tokens=chunk_tokens)
        return self.state

    def forward(self, streams: list, pre_row, *, real_len: int, on_hidden: Optional[callable] = None):
        """-> (streams, ffn_pre_row). ``on_hidden(name, tensor)`` (debug) sees the attention / MoE outputs."""
        assert len(streams) == HC
        S_l, D_l = streams[0].shape[2], streams[0].shape[3]
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(streams)
        h = self.attn_norm(self.attn_hc.collapse(streams, pre_row))
        y = self.attn(h, seq_len_actual=real_len, state=self.state)
        ttnn.deallocate(h)
        if on_hidden is not None:
            on_hidden(f"{self.layer}:attn.y", y)
        streams = self.attn_hc.mix(streams, y, attn_post, attn_comb)
        ttnn.deallocate(y)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(streams)
        h = self.ffn_norm(self.ffn_hc.collapse(streams, attn_pre))
        h3 = ttnn.reshape(h, [1, S_l, D_l])
        moe_out, _ = self.moe(h3, actual_isl=real_len, actual_start=0)
        moe_out = ttnn.reshape(moe_out, [1, 1, S_l, D_l])
        if on_hidden is not None:
            on_hidden(f"{self.layer}:moe.out", moe_out)
        streams = self.ffn_hc.mix(streams, moe_out, ffn_post, ffn_comb)
        ttnn.deallocate(moe_out)
        return streams, ffn_pre
