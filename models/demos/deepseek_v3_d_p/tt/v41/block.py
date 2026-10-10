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

import os
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v4 import mhc_math as M
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import TtHyperConnection
from models.demos.deepseek_v3_d_p.tt.v4.trace_island import TraceIsland, copy_into

from .attention import V41CSA, V41SWA, V41CSAConsumer
from .moe import build_v41_moe

HC = M.HC


class V41HyperConnection(TtHyperConnection):
    """``mixes(streams) -> (pre_row, post_row, comb_row)`` and ``collapse(streams, pre_row) -> bf16 [1, 1, S_l, D_l]``;
    ``mix`` (the hc_post) is inherited.

    DS41F-0037 C-P1 (the prefill is host-issue bound): STACKED is the default (``V41_MHC_STACKED=0`` = per-stream ops;
    a warm 1536-token chunk 2.7 -> 1.9 s, worst export PCC 0.978315 -> 0.978125). ``V41_MHC_HOST_SINKHORN=1``: the
    device computes the TP-reduced logit row, the host the rest of ``mixes`` (affine, sigmoid, softmax, 20 Sinkhorn
    iterations: ~90 tiny programs on the device) on the 8 SP shards, and uploads pre / post / comb."""

    def __init__(self, device, *, fn, base, scale, hidden, hc_eps=1e-6, **kw):
        super().__init__(device, fn=fn, base=base, scale=scale, hidden=hidden, hc_eps=hc_eps, **kw)
        self.stacked = os.environ.get("V41_MHC_STACKED", "1") == "1"
        self.host_sinkhorn = os.environ.get("V41_MHC_HOST_SINKHORN", "0") == "1"
        if self.host_sinkhorn:
            W = M.prep_hyper_connection(fn, base, scale, int(hidden))
            K = M.sinkhorn_constants(float(hc_eps))
            self._h = {k: W[k].float() for k in ("SV", "BV", "COMB_MASK", "ONE_COL", "R_SOFT")}
            self._h.update({k: K[k].float() for k in ("R_AUG", "C_AUG", "EPS_COMB")})

    def _host_rows(self, t) -> torch.Tensor:
        """[S, 32] fp32: the SP shards of a TP-replicated [1, 1, S_l, 32] row tensor, in SP order (TP replica 0)."""
        rows, cols = (int(v) for v in self.device.shape)
        devs = ttnn.get_device_tensors(t)
        if self.sp_axis == 0:
            pick = [devs[r * cols] for r in range(rows)]
        else:
            pick = [devs[c] for c in range(cols)]
        return torch.cat([ttnn.to_torch(d).float().reshape(-1, M.ROW) for d in pick], dim=0)

    def _upload_rows(self, x: torch.Tensor):
        """[S, 32] fp32 host -> [1, 1, S_l, 32] fp32 TILE, SP-sharded on the rows, replicated over TP."""
        return ttnn.from_torch(
            x.reshape(1, 1, -1, M.ROW).contiguous(),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=self._mesh_mapper(sp_dim=2),
        )

    def _mixes_host(self, mix):
        """The device formulas of ``mixes`` below on the host, in fp32 (same constants, same order)."""
        h = self._h
        aff = self._host_rows(mix) * h["SV"] + h["BV"]
        ttnn.deallocate(mix)
        sig = torch.sigmoid(aff)
        pre = sig + self.hc_eps
        post = sig * 2.0
        E = torch.exp(aff.clamp(-M.EXP_CLAMP, M.EXP_CLAMP)) * h["COMB_MASK"] + h["ONE_COL"]
        X = E / (E @ h["R_SOFT"])
        X = X + h["EPS_COMB"]
        X = X / (X @ h["C_AUG"])
        for _ in range(self.sinkhorn_iters - 1):
            X = X / (X @ h["R_AUG"])
            X = X / (X @ h["C_AUG"])
        return self._upload_rows(pre), self._upload_rows(post), self._upload_rows(X)

    def mixes(self, streams: list):
        assert len(streams) == HC
        if self.stacked:  # PREFILL_MHC_STACKED=1: the norm-folded logit row in 5 programs instead of 5 per stream
            Xf = self._stack(streams)
            mix = self._mix_row_stacked(Xf)
            ttnn.deallocate(Xf)
        else:
            xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
            mix = self._mix_row(xf)
            for x in xf:
                ttnn.deallocate(x)
        if self.host_sinkhorn:
            return self._mixes_host(mix)
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
        if (
            self.stacked
        ):  # one broadcast multiply + one stream-dim reduce instead of a slice / multiply / add per stream
            Xf = self._stack(streams)
            w = self._select_cols(pre_row, list(range(M.PRE0, M.PRE0 + HC)))
            out = ttnn.typecast(self._weighted_sum_stacked(w, Xf), streams[0].dtype)
            ttnn.deallocate(Xf)
            return out
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
        self.state = None  # the CURRENT slot's attention state (use_slot)
        self.states: dict = {}
        self._islands = None  # (A, B, S_in, pre_in, y_buf) once enable_trace_islands ran

    def alloc_state(self, max_seq_len: int, chunk_tokens: int, slot: int = 0):
        self.states[slot] = self.attn.alloc_state(max_seq_len, chunk_tokens=chunk_tokens)
        if self.state is None:
            self.state = self.states[slot]
        return self.states[slot]

    def use_slot(self, slot: int) -> None:
        self.state = self.states[slot]

    def forward(self, streams: list, pre_row, *, real_len: int, on_hidden: Optional[callable] = None, prof=None):
        """-> (streams, ffn_pre_row). ``on_hidden(name, tensor)`` (debug) sees the attention / MoE outputs. ``prof(name)``
        (V41_PREFILL_PROFILE, ``prefill.PhaseProfile``) closes a timed phase."""
        assert len(streams) == HC
        mark = prof if prof is not None else (lambda name: None)
        S_l, D_l = streams[0].shape[2], streams[0].shape[3]
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(streams)
        h = self.attn_norm(self.attn_hc.collapse(streams, pre_row))
        mark("hc.attn")
        y = self.attn(h, seq_len_actual=real_len, state=self.state)
        ttnn.deallocate(h)
        mark(f"attn.{type(self.attn).__name__}")
        if on_hidden is not None:
            on_hidden(f"{self.layer}:attn.y", y)
        streams = self.attn_hc.mix(streams, y, attn_post, attn_comb)
        ttnn.deallocate(y)
        mark("hc.mix")

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(streams)
        h = self.ffn_norm(self.ffn_hc.collapse(streams, attn_pre))
        h3 = ttnn.reshape(h, [1, S_l, D_l])
        mark("hc.ffn")
        moe_out, _ = self.moe(h3, actual_isl=real_len, actual_start=0)
        moe_out = ttnn.reshape(moe_out, [1, 1, S_l, D_l])
        mark("moe")
        if on_hidden is not None:
            on_hidden(f"{self.layer}:moe.out", moe_out)
        streams = self.ffn_hc.mix(streams, moe_out, ffn_post, ffn_comb)
        ttnn.deallocate(moe_out)
        mark("hc.mix")
        return streams, ffn_pre

    # ---- trace islands (V41_PREFILL_ISLANDS, DS41F-0037 C-P1; V4-Flash's DS4F-0246/0247 pattern) ---------------------------
    # The eager prefill is host-issue bound and the mHC sites + MoE are ~60% of its programs and know nothing about the
    # chunk position: island A = attn mixes + collapse + attn norm, island B = attn mix + ffn mixes + collapse + ffn norm +
    # MoE + ffn mix, both captured once (full chunks) and replayed every chunk; the attention between them stays eager.
    # Rules (v4/trace_island.py): inputs are block-owned persistent buffers fed by ttnn.copy; outputs are persistent; no
    # eager tensor may live across a replay (the CSA source's selection mask goes through persistent buffers, attention.py);
    # every lazily cached constant must exist before the capture (the runtime captures after its warm-all).
    def enable_trace_islands(self, streams: list, pre_row, chunk_tokens: int):
        """Capture A and B with ``streams`` / ``pre_row`` as shape templates (their contents do not matter) ->
        (out_streams, ffn_pre): B's persistent outputs, which serve as the next block's templates."""
        assert self._islands is None, "islands already enabled"
        S_in = [ttnn.clone(t) for t in streams]
        pre_in = ttnn.clone(pre_row)
        S_l, D_l = int(streams[0].shape[2]), int(streams[0].shape[3])

        def island_a(*args):
            s = list(args[:HC])
            attn_pre, attn_post, attn_comb = self.attn_hc.mixes(s)
            h = self.attn_norm(self.attn_hc.collapse(s, args[HC]))
            return attn_pre, attn_post, attn_comb, h

        A = TraceIsland(self.mesh_device, island_a, S_in + [pre_in], name=f"v41.L{self.layer}.A")
        attn_pre, attn_post, attn_comb, h = A.capture()
        y_buf = ttnn.clone(h)  # the attention output has the stream shape (o-proj back to D_l)

        def island_b(*args):
            s = list(args[:HC])
            streams2 = self.attn_hc.mix(s, args[HC], attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(streams2)
            hh = self.ffn_norm(self.ffn_hc.collapse(streams2, attn_pre))
            moe_out, _ = self.moe(ttnn.reshape(hh, [1, S_l, D_l]), actual_isl=chunk_tokens, actual_start=0)
            moe_out = ttnn.reshape(moe_out, [1, 1, S_l, D_l])
            out = self.ffn_hc.mix(streams2, moe_out, ffn_post, ffn_comb)
            ttnn.deallocate(moe_out)
            for t in streams2:
                ttnn.deallocate(t)
            return tuple(out) + (ffn_pre,)

        B = TraceIsland(self.mesh_device, island_b, S_in + [y_buf], moe=self.moe, name=f"v41.L{self.layer}.B")
        outs = B.capture()
        self._islands = (A, B, S_in, pre_in, y_buf)
        self._island_chunk = int(chunk_tokens)
        return list(outs[:HC]), outs[HC]

    def forward_islands(self, streams: list, pre_row, *, real_len: int, owned: bool, owned_pre: bool = False):
        """The traced forward: copy the inputs in, replay A, run the attention eagerly, replay B. ``owned`` /
        ``owned_pre``: the streams / the pre row are eager tensors of the caller's, freed after the copy -- never a
        previous island's persistent outputs (an Engram layer gets eager streams but the previous block's persistent pre
        row: freeing that one freed block L-1's island-B output, 19:23)."""
        A, B, S_in, pre_in, y_buf = self._islands
        assert real_len == self._island_chunk, f"islands run full chunks only ({real_len} of {self._island_chunk})"
        for dst, src in zip(S_in, streams):
            copy_into(dst, src)
        copy_into(pre_in, pre_row)
        if owned:
            for t in streams:
                ttnn.deallocate(t)
        if owned_pre:
            ttnn.deallocate(pre_row)
        _attn_pre, _post, _comb, h = A.replay()
        y = self.attn(h, seq_len_actual=real_len, state=self.state)
        if y.dtype != y_buf.dtype:
            y2 = ttnn.typecast(y, y_buf.dtype)
            ttnn.deallocate(y)
            y = y2
        copy_into(y_buf, y)
        ttnn.deallocate(y)
        outs = B.replay()
        return list(outs[:HC]), outs[HC]

    def release_islands(self) -> None:
        if self._islands is not None:
            self._islands[0].release()
            self._islands[1].release()
            self._islands = None
