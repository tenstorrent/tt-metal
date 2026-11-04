# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 (PRODUCTION_PATH.md §7, P2): batched TP prefill of the persistent stack (tt/stack_model.py StackBackend) on
the decode stack's OWN per-chip weight shards, one trace per physical bucket.

Layout per chip (mesh of N = TP chips, rank r = row-major mesh index = ShardTensorToMesh order):
  residual X: SEQUENCE-PARALLEL when the bucket allows it (C >= 32 N): chip r holds rows [r C/N, (r+1) C/N) of the
      chunk, [1, 1, C/N, 16384] fp32 (4 mHC streams flat); mHC pre / post, the norms and the router run on those rows
      only. Small buckets (C < 32 N) keep X replicated ("rep" mode).
  sublayer input: h = RMSNorm(mhc_pre(X)) bf16 -> ALL-GATHER over the mesh -> [1, 1, C, 4096] on every chip;
  attention (TP heads, the decode shards): KDA = the layer's OptimizedKDA (8 heads per chip at TP8, the DS weights the
      megakernel streams) through a per-layer scratch state (slot-agnostic trace); MLA = q_a | kv_a (replicated DS) ->
      latent (replicated: every chip fills its own copy of the paged latent cache) + q_b / W_uk / SDPA / W_uv / o_proj on
      the chip's heads; -> o_proj PARTIAL [1, 1, C, 4096];
  FFN: dense = the layer's OptimizedSwiGLU shards (column gate|up, row down) -> PARTIAL; MoE = router (replicated, on the
      local rows, gathered) + the chip's EP experts on every row routed to them + the TP shared expert -> PARTIAL;
  PARTIAL -> REDUCE-SCATTER (sp) / ALL-REDUCE (rep) -> mhc_post on the local rows.
Per layer: 2 gathers + 2 reduce-scatters (= the 2 all-reduces of the TP layer, split around the sequence-parallel mHC).

State written in the decode stack's layout: KDA S [chunk, H_r, 128, 128] fp32 / conv C [1, 3, chunk, 3 H_r 128] bf16 of
the slot (copied in / out of the per-layer scratch with SlotOps around the replay), the latent cache pages of the slot
through paged_fill_cache (chunk page table + chunk start: device inputs). The last valid row's final residual is
broadcast into the decode residual rows, and the stack's own head (MkHead: final norm, lm_head shard, device top-K /
sampler) produces the first token inside the same trace.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import os
import time
from typing import Optional

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH


def dsa_dense_window() -> bool:
    """GLM53_DSA_DENSE_WINDOW=1 (doc/ttnn/14-production-tp8-megakernel/dsa/dense_window_design.md): every sparse_dsa
    layer keeps a REPLICATED window of each slot's latent rows 0 .. 2,111 (positions < DENSE_MAX 2,051 matter): the
    decode's all-dense launches attend it in one pass (kernels/mla/att_dsa_shard_dm0.cpp)"""
    return os.environ.get("GLM53_DSA_DENSE_WINDOW", "0") == "1"


def window_idx(t: int, segs: list, wt: int = 512 // 32) -> torch.Tensor:
    """the dense window's tile scatter of a prefill's latent [1, 1, t, 512] TILE (tiles_to, SKIP = 0xFFFFFFFF), the
    same on every chip: segs = [(row offset o in the bucket, position of its row 0, n_valid, slot)] on 32-row
    boundaries; tile row k of a segment (position p0 + 32 k, a valid row) -> window tile row slot x 2 WB + (p0 + 32 k) / 32
    while p0 + 32 k < WB x 64 (the window's [S x WB, 1, 64, 512] pages). Returns [1, t / 32 x wt] int64."""
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import DSA_WIN_BLOCKS as WB

    out = torch.full((1, t // 32 * wt), DSH.SKIP, dtype=torch.int64)
    for o, p0, nv, slot in segs:
        assert o % 32 == 0 and p0 % 32 == 0, (o, p0)
        for k in range(-(-int(nv) // 32)):
            pos = int(p0) + 32 * k
            if pos >= WB * 64:
                break
            dst = int(slot) * WB * 2 + pos // 32
            src = o // 32 + k
            out[0, src * wt : (src + 1) * wt] = torch.arange(dst * wt, (dst + 1) * wt)
    return out


from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA, SPARSE_KDA
from models.autoports.zai_org_glm_5_3_flash.tt.generator import slice_plan as _slice_plan
from models.autoports.zai_org_glm_5_3_flash.tt.generator import slice_safe_bucket

FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
HIDDEN = 4096
NSTREAM = 4
BLOCK = 64  # latent page (tokens)


def _ckc_fp32(dev):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)  # fmt: skip


def trace_view(mesh) -> dict:
    """GLM53_PF_CAPTURE_REPORT: the DRAM view (dynamic trace mode, GLM53_TRACE_REGION=0, puts trace buffers in DRAM)
    and the TRACE region view (region mode) per bank, for a per-capture size table"""
    out = {}
    for name in ("DRAM", "TRACE"):
        bt = getattr(ttnn.BufferType, name, None)
        if bt is None:
            continue
        try:
            v = ttnn.get_memory_view(mesh, bt)
            out[name] = (
                int(v.total_bytes_allocated_per_bank),
                int(v.largest_contiguous_bytes_free_per_bank),
                int(v.num_banks),
            )
        except Exception:  # a view this build does not offer
            pass
    return out


def report_capture(mesh, what, before) -> dict:
    """print one CAPTURE_REPORT line: the per-bank delta of each view since `before` (and the totals); returns now"""
    now = trace_view(mesh)
    parts = []
    for k, (al, lf, nb) in now.items():
        b = (before or {}).get(k, (al, lf, nb))[0]
        parts.append(
            f"{k} +{(al - b) / 2**20:.2f} MiB/bank (x{nb} = +{(al - b) * nb / 1e6:.1f} MB) alloc {al / 2**20:.1f} MiB/bank largest free {lf / 2**20:.1f} MiB/bank"
        )
    print(f"CAPTURE_REPORT {what}: " + "; ".join(parts), flush=True)
    return now


class _TraceReport:
    def __init__(self, mesh):
        self.on = os.environ.get("GLM53_PF_CAPTURE_REPORT", "0") == "1" and os.environ.get("GLM53_MOCK") != "1"
        self.mesh, self.last = mesh, (trace_view(mesh) if self.on else None)

    def __call__(self, what):
        if self.on:
            self.last = report_capture(self.mesh, what, self.last)


@dataclasses.dataclass
class ChunkInputs:
    """Persistent device inputs of one bucket's trace (written by the host before every replay)."""

    bucket: int
    tokens: ttnn.Tensor  # [1, C] uint32 ROW_MAJOR (sp: sharded on dim -1 -> [1, C/N] per chip)
    valid_end: ttnn.Tensor  # [1] uint32 (KDA actual_end)
    valid_end_f: ttnn.Tensor  # [1, 1, 1, 1] fp32
    chunk_table: ttnn.Tensor  # [1, C / 64] int32 (the chunk's latent pages)
    user_table: ttnn.Tensor  # [1, blocks_per_seq] int32 (the slot's page-table row)
    chunk_start: ttnn.Tensor  # [1] int32
    last_mask: ttnn.Tensor  # [1, 1, C_local, 1] fp32 one-hot of the last valid row on its owner chip (zeros elsewhere)
    chunk_start_f: ttnn.Tensor = None  # [1, 1, 1, 1] fp32 (sparse_dsa: positions of the chunk's rows)
    pool_table: ttnn.Tensor = None  # [1, C / 128] int32: the chunk's pooled-key store pages (sparse_dsa layers)
    pool_user: ttnn.Tensor = None  # [1, max_seq / 128] int32: the slot's pooled-key store pages
    ring_rows: ttnn.Tensor = None  # [1, 32] uint32: chunk rows of the DSA ring's [k | g] rows 0..3 (C = a zero row)
    ring_pages: ttnn.Tensor = None  # [1, 8] int32: the slot's ring tiles (8 per slot)
    chunk_start_u: ttnn.Tensor = None  # [1] uint32: indexer_score_dsa's device-word chunk start (GLM53_PF_IX_BOUND)
    # GLM53_PF_IX_TOPK_BOUND: [1] uint32 PER CHIP (ShardTensorToMesh dim 0, rank order): the top-k's pool-column bound =
    # indexer_score_dsa's written extent ceil((chunk_start + (rank + 1) C_l) / 128) x 32 under GLM53_PF_IX_BOUND
    vis_cols_u: ttnn.Tensor = None
    # GLM53_PF_IX_GATHER_BOUND: [1] uint32 PER CHIP: the local shard units this chip compacts for the pooled-key
    # all_gather = #{u < ceil((chunk_start + c) / 128) : u % C == rank} (the last chip's visible units)
    pool_units_u: ttnn.Tensor = None
    # GLM53_DSA_SHARD (the sequence-sharded pool, tt/dsa_shard.py): per-chip tables (ShardTensorToMesh dim 0) -- the
    # chunk's latent tiles -> this chip's local blocks (tiles_to; 0xFFFFFFFF = not this chip's unit / a padding tile
    # row), its pooled-key tile rows -> this chip's local store pages; user_table / pool_user are then the slot's LOCAL
    # page lists (local blocks [1, bps_local] per chip; local pooled-key pages [1, pps_local], the same on every chip)
    lat_idx: ttnn.Tensor = None  # [1, C / 32 x 16] uint32 per chip
    win_idx: ttnn.Tensor = None  # GLM53_DSA_DENSE_WINDOW: [1, C / 32 x 16] uint32, the same on every chip (window_idx)
    pool_idx: ttnn.Tensor = None  # [1, C / 128 x 4] uint32 per chip
    # GLM53_PF_KDA_SNAPSHOT (one-replay store A, doc/ttnn/09-vllm/perf/one_replay_store/design.md): [1] uint32 = the
    # chunk's rows before the KDA checkpoint cut; 32 = no cut in this chunk (a one-chunk snapshot nobody reads)
    snap_end_u: ttnn.Tensor = None


class PrefillLayer:
    """One layer's prefill pieces on the mesh (built from the StackBackend's resident layer objects)."""

    def __init__(self, tpp: "TPPrefill", e: dict, kind: str):
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import (
            CompositeHyperConnection,
            FusedDecoderConfig,
            StreamMajorHyperConnection,
        )
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
            OptimizedCompositeHyperConnection,
            OptimizedDecoderConfig,
        )

        mesh, cfg, source, L = tpp.mesh, tpp.cfg, tpp.source, e["L"]
        self.L, self.kind, self.e = L, kind, e
        dev = lambda n: source.device(L, n, mesh)  # noqa: E731
        host = lambda n: source.host(L, n)  # noqa: E731
        hc_kw = dict(hc_mult=cfg.hc_mult, hidden_size=cfg.hidden_size, sinkhorn_iters=cfg.hc_sinkhorn_iters,
                     hc_eps=cfg.hc_eps, norm_eps=cfg.rms_norm_eps)  # fmt: skip
        ocfg = OptimizedDecoderConfig()
        self.hc = {}
        for sub in ("attn", "ffn"):
            if tpp.mhc_mode == "sm":  # stream-major residual [1, 4, T, 4096]: ~24 ops per sublayer, fp32 throughout
                hc = StreamMajorHyperConnection(mesh, dev(f"hc_{sub}_fn"), host(f"hc_{sub}_base"),
                                                host(f"hc_{sub}_scale"), **hc_kw)  # fmt: skip
                hc.fcfg = FusedDecoderConfig()
            else:
                hc = CompositeHyperConnection(mesh, dev(f"hc_{sub}_fn"), host(f"hc_{sub}_base"),
                                              host(f"hc_{sub}_scale"), **hc_kw)  # fmt: skip
                hc.__class__ = OptimizedCompositeHyperConnection
                hc.ocfg = ocfg
            self.hc[sub] = hc
        row = lambda t: ttnn.reshape(t, (1, 1, 1, t.shape[-1]))  # noqa: E731
        self.norm = {"attn": row(dev("input_layernorm.weight")), "ffn": row(dev("post_attention_layernorm.weight"))}
        self.eps = cfg.rms_norm_eps
        # GLM53_PF_MHC_PRE=fused7a (Stage 7a lever A, tt/mhc_pre_pf.py): per sublayer ONE program for the coefficients,
        # the pre-mix and the sublayer's RMSNorm (comp mHC + fused post only); its output is already normalised, so
        # rms() hands it through once (_normed)
        self.mpf, self._normed = None, {}
        if os.environ.get("GLM53_PF_MHC_PRE", "") == "fused7a" and tpp.mhc_mode == "comp":
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf import MhcPrePf

            self.mpf = {sub: MhcPrePf.from_source(mesh, source, cfg, L, sub, nm) for sub, nm in
                        (("attn", "input_layernorm.weight"), ("ffn", "post_attention_layernorm.weight"))}  # fmt: skip
        self.ckc = _ckc_fp32(mesh)
        tp = e["tp"]
        if kind in (DENSE_KDA, SPARSE_KDA):
            self.kda = e["kda"]  # OptimizedKDA on the chip's head shard (the megakernel's DS weights)
            self.scratch = tpp.kda_scratch(self.kda)
            # GLM53_PF_KDA_SNAPSHOT: the state at the checkpoint cut (same layout as the scratch: S [1, H, 128, 128]
            # fp32, C [1, 3, 1, W] bf16 row 0) -- the one-replay store's staging, read by the KDA export
            self.cut_scratch = tpp.kda_scratch(self.kda) if getattr(tpp, "kda_snapshot", False) else None
        else:
            self.mla = MlaPrefill(tpp, tp, L)
        if kind == DENSE_KDA:
            self.mlp = tp.mlp  # OptimizedSwiGLU, column gate|up / row down shards
        else:
            self.moe = tp.mlp if kind == SPARSE_KDA else tp.moe.mlp  # OptimizedMoE (router, EP experts)
            # the served stack's batched expert TP (GLM53_TP_XSL=1 + GLM53_TP_XSL_SLICED=1, tt/tp8_sparse.py): the MoE
            # object carries no routed weights (routed=False: [None] * E); the layer holds every expert sliced, chip k
            # = slice k (I / TP of each, fused_decoder.load_xsl_experts) -> (tp, gate_up list, down list)
            w = getattr(tp, "w", None)
            self.xsl = w.xsl_experts if kind == SPARSE_KDA and getattr(tp, "xsl_sliced", False) else None
            self.shared = getattr(getattr(tp, "w", None), "shared_tp", None) if kind == SPARSE_KDA else None
            if self.shared is None:  # sparse_dsa (and sparse_kda without GLM53_TP_SHARED): build the TP shard
                self.shared = tpp.tp_shared(L, self.moe)
            self.ep = tpp.ep_bounds()

    # ------------------------------------------------------------------ sublayers
    def rms(self, h, sub: str, dtype=BF16):
        if self._normed.get(sub) is h:  # the fused 7a mhc_pre's output: normalised already
            self._normed[sub] = None
            return h if h.dtype == dtype else ttnn.typecast(h, dtype)
        out = ttnn.rms_norm(h, weight=self.norm[sub], epsilon=self.eps, compute_kernel_config=self.ckc)
        return ttnn.typecast(out, dtype) if out.dtype != dtype else out


class MlaPrefill:
    """sparse_dsa attention prefill on the chip's head shard (absorbed MLA, the latent cache as K = V):
    q_a | kv_a (replicated DS) -> RMSNorms -> latent (every chip) + q_b (DS shard) -> per-head W_uk -> chunked SDPA over
    the paged latent cache (flash: online softmax, no materialised scores) -> per-head W_uv -> o_proj (DS shard)."""

    def __init__(self, tpp: "TPPrefill", d, L: int):
        mesh = tpp.mesh
        attn, shard = d.attn, d.mla.shard
        self.attn, self.shard = attn, shard
        self.tpp = tpp
        self.idx = DsaIndexer(tpp, L) if tpp.dsa else None  # GLM53_PF_DSA=1: pooled keys + sparse chunks (P3)
        self.h = shard.h  # local heads
        self.q_lora, self.kv_lora, self.dqk, self.dv = attn.q_lora, attn.kv_lora, attn.dqk, attn.dv
        self.eps, self.scale = attn.eps, attn.scale
        self.ckc_fp32 = _ckc_fp32(mesh)
        hw = shard.host_weights
        ndev = tpp.ndev
        mapper = ttnn.ShardTensorToMesh(mesh, dim=1) if ndev > 1 else None
        # per-head W_uk [1, H_r, 256, 512] / W_uv [1, H_r, 512, 256] bf16 (the decode program keeps them in its grouped
        # bfp8 layout; the prefill's batched head matmuls take these small copies: 2 x 2 MB per chip per layer at TP8)
        cat = lambda key: torch.cat([w.reshape(1, *w.shape).float() for w in hw[key]], dim=1)  # noqa: E731
        # hard assert (coordinator 09:49Z): the host copies uploaded here are made at construction; a DRAM-image replay
        # once handed back zeros for them (tp_image _host0) -> W_uk = W_uv = 0 in every prefill MLA layer
        if os.environ.get("GLM53_PF_ZERO_ASSERT", "1") == "1":
            zero = [f"{k}[{i}]" for k in ("w_uk", "w_uv") for i, w in enumerate(hw[k]) if not bool(w.abs().max() > 0)]
            assert (
                not zero
            ), f"layer {L}: prefill MLA host weights all-zero {zero[:4]} (an image replay's _host0 zeros?)"
        self.w_uk = ttnn.from_torch(cat("w_uk"), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    memory_config=DRAM, mesh_mapper=mapper)  # fmt: skip
        self.w_uv = ttnn.from_torch(cat("w_uv"), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    memory_config=DRAM, mesh_mapper=mapper)  # fmt: skip
        # GLM53_PF_HEAD_L1ACC=0: no packer L1 accumulation in the head matmuls -- a smaller intermediate CB (the absorbed
        # W_uv matmul's static CBs end 6 KB over the L1 floor of a B = 4 or a 9-layer decode stack, 01:55Z / 02:19Z)
        self.ckc_head = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                               math_approx_mode=False, fp32_dest_acc_en=True,
                                                               packer_l1_acc=os.environ.get("GLM53_PF_HEAD_L1ACC", "1") == "1")  # fmt: skip
        self.sdpa_prog, self.sdpa_ckc = sdpa_config(mesh, attn)
        self.w_k = self.w_v = None
        if tpp.mla_form == "mha":
            self.enable_mha()

    def _mha(self, q, src, mask=None):
        """the non-absorbed attention: K_h / V_h = src W_k / W_v ([1, H_r, R, 256], R = src rows), SDPA of q
        [1, H_r, C, 256] (causal without a mask, else the additive mask [1, 1, C, R]), o_proj; frees q and src"""
        kv = []
        for w_ in (self.w_k, self.w_v):
            f = ttnn.matmul(src, w_, compute_kernel_config=self.ckc_head, dtype=BF16, memory_config=DRAM)
            t_, _, _ = ttnn.experimental.nlp_create_qkv_heads(f, num_heads=self.h, num_kv_heads=0,
                                                              transpose_k_heads=False, memory_config=DRAM)  # fmt: skip
            ttnn.deallocate(f)
            kv.append(t_)
        ttnn.deallocate(src)
        c = int(q.shape[-2])
        prog = self.mha_prog
        if prog.q_chunk_size > c:  # a bucket narrower than the q chunk (the pair's 128-row sparse key)
            q_, k_, fid, *rest = self.mha_spec.split(",")
            prog = self.mha_progs.setdefault(c, sdpa_config(self.tpp.mesh, self.attn,
                                                            ",".join([str(c), k_, fid] + rest))[0])  # fmt: skip
        # heads written side by side by the SDPA itself: no nlp_concat_heads (whose CBs for H_r = 32 rows of 8,192 end
        # 6 KB over the L1 floor of a B = 4 / 9-layer decode stack, 01:55Z / 02:19Z) and one op fewer
        o2 = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv[0],
            kv[1],
            attn_mask=mask,
            is_causal=mask is None,
            scale=self.scale,
            program_config=prog,
            compute_kernel_config=self.mha_ckc,
            memory_config=DRAM,
            output_concat_heads=True,
        )  # fmt: skip  # [1,1,C,H_r 256]
        for t_ in kv + [q]:
            ttnn.deallocate(t_)
        out = pf_mm(self.shard.ds_o, o2)
        ttnn.deallocate(o2)
        return out

    def _a2a(self, x, heads_to_rows: bool):
        """heads -> sequence (in_dim 1 gathered, out_dim 2 split; mesh axis 0 then 1: chip (r, c) ends with the rows of
        sequence rank r C + c, the sp shard order) or the inverse (axis 1 then 0)"""
        mesh = self.tpp.mesh
        axes = [a for a in (0, 1) if int(mesh.shape[a]) > 1]
        for a in axes if heads_to_rows else axes[::-1]:
            y = ttnn.experimental.all_to_all_async_generic(x, in_dim=1 if heads_to_rows else 2,
                                                            out_dim=2 if heads_to_rows else 1, cluster_axis=a,
                                                            memory_config=DRAM)  # fmt: skip
            ttnn.deallocate(x)
            x = y
        return x

    def _sparse_sdpa(self, q_lat, ids, kv, kchunk: int | None = None):
        """the selected keys only (ttnn.transformer.sparse_sdpa, absorbed: K = V = the ROW_MAJOR latent store kv, row =
        token position): q_lat [1, H_r, C, 512] bf16 TILE, ids [1, 1, C (or C / N with the a2a reshard), SPARSE_W]
        uint32 -> o_lat [1, H_r, C, 512] bf16 TILE (q_lat kept). The op takes the heads of one token as one tile: H_r <
        32 (TP8: 8) is zero-padded to 32 heads (in blocks of GLM53_PF_SPARSE_ROWS query rows: the padded q / output
        copies are 4x the chip's q_lat, 64 MiB each at TP8 C = 2,048 in one block), or (GLM53_PF_SPARSE_RESHARD=a2a)
        resharded heads -> sequence and back"""
        if self.tpp.sparse_reshard == "a2a" and self.tpp.ndev > 1:
            q_seq = self._a2a(ttnn.clone(q_lat, memory_config=DRAM), True)  # [1, H, C / N, 512]
            q_rm = ttnn.to_layout(q_seq, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(q_seq)
            o = ttnn.transformer.sparse_sdpa(q_rm, kv, ids, 512,
                                             kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=self.scale,
                                             k_chunk_size=kchunk or sparse_kchunk(),
                                             compute_kernel_config=self.ckc_fp32)  # fmt: skip
            ttnn.deallocate(q_rm)
            o_t = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(o)
            return self._a2a(o_t, False)  # [1, H_r, C, 512]
        h, c = int(q_lat.shape[1]), int(q_lat.shape[2])
        hp = -(-h // TILE) * TILE
        rb = c if hp == h else min(c, self.tpp.sparse_rows)  # unpadded (TP2): one call over all rows, as before
        outs = []
        for r0 in range(0, c, rb):
            r1 = min(c, r0 + rb)
            qb = q_lat if rb == c else ttnn.slice(q_lat, (0, 0, r0, 0), (1, h, r1, 512))
            qp = ttnn.pad(qb, [(0, 0), (0, hp - h), (0, 0), (0, 0)], 0.0) if hp != h else qb
            if qb is not q_lat and qb is not qp:
                ttnn.deallocate(qb)
            q_rm = ttnn.to_layout(qp, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            if qp is not q_lat:
                ttnn.deallocate(qp)
            ib = ids if rb == c else ttnn.slice(ids, (0, 0, r0, 0), (1, 1, r1, SPARSE_W))
            o = ttnn.transformer.sparse_sdpa(q_rm, kv, ib, 512,
                                             kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=self.scale,
                                             k_chunk_size=sparse_kchunk(), compute_kernel_config=self.ckc_fp32)  # fmt: skip
            ttnn.deallocate(q_rm)
            if ib is not ids:
                ttnn.deallocate(ib)
            o_lat = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(o)
            if hp != h:
                o_h = ttnn.slice(o_lat, (0, 0, 0, 0), (1, h, r1 - r0, 512))
                ttnn.deallocate(o_lat)
                o_lat = o_h
            outs.append(o_lat)
        if len(outs) == 1:
            return outs[0]
        o_lat = ttnn.concat(outs, dim=2, memory_config=DRAM)
        for o_ in outs:
            ttnn.deallocate(o_)
        return o_lat

    def _spa_blocked(self, q_lat, kv, ids, sp: "SparseCtx"):
        """GLM53_PMK_SPA in row blocks (SparseCtx.spa_blocks): q heads -> rows once; per block of spa_rows rows the union
        launch (SpaUnion row_off: that block's union into the shared compact store, its rows of ids_new) and sparse_sdpa
        on the block's q rows; the blocks concatenate in row order, then rows -> heads (= _sparse_sdpa's a2a form)"""
        q_seq = self._a2a(ttnn.clone(q_lat, memory_config=DRAM), True)  # [1, H, C / N, 512]
        q_rm = ttnn.to_layout(q_seq, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(q_seq)
        h, rb, outs = int(q_rm.shape[1]), sp.spa_rows, []
        for j in range(sp.spa_blocks):
            sp.spu(kv, ids, sp.ids_new, row_off=j * rb)
            qb = ttnn.slice(q_rm, (0, 0, j * rb, 0), (1, h, (j + 1) * rb, 512))
            ib = ttnn.slice(sp.ids_new, (0, 0, j * rb, 0), (1, 1, (j + 1) * rb, SPARSE_W))
            outs.append(ttnn.transformer.sparse_sdpa(qb, sp.cmp, ib, 512, kv_format=ttnn.transformer.SparseKVFormat.BF16,
                                                     scale=self.scale, k_chunk_size=spa_kchunk(),
                                                     compute_kernel_config=self.ckc_fp32))  # fmt: skip
            ttnn.deallocate(qb)
            ttnn.deallocate(ib)
        ttnn.deallocate(q_rm)
        o = ttnn.concat(outs, dim=2, memory_config=DRAM)
        for o_ in outs:
            ttnn.deallocate(o_)
        o_t = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(o)
        return self._a2a(o_t, False)  # [1, H_r, C, 512]

    def _bq(self, q_lat, ids_loc, cache, ci: ChunkInputs, sp: "SparseCtx"):
        """lever B-q (GLM53_DSA_BQ): this chip attends the selected keys it OWNS (ids_loc: expand_rows SHARD mode, local
        row ids of its latent slice; sp.counts' valid flag per row) with the q of ALL heads (all-gathered: head order =
        rank order), ttnn.transformer.sparse_sdpa_stats -> (o_c normalised, m_c, l_c); the owner merge: M = max_c m_c,
        w_c = exp((m_c - M) scale) l_c / sum_c' (an invalid partial -- no owned key, the dummy -- has l_c = 0, m_c =
        -1e30), o = sum_c w_c o_c summed in fp32 by a head-dim reduce-scatter (chip r keeps its heads [8 r, 8 r + 8)).
        Same math as kernels/mla/att_c.cpp's partial tree combine / tt/dsa_shard.py bq_merge (host reference).
        GLM53_PF_BQ_HEAD_BLOCK=hb (2026-10-23 08:02Z, 14's n29h7 mock: the all-gathered q of 64 heads at the 8,192 bucket
        is [1, 64, 8192, 512] bf16 = 512 MiB per chip + its row-major copy + the output, ~1.5 GB transient beside the 1M
        KV pool -> DRAM OOM): the same B-q in head BLOCKS of hb heads -- block j gathers local heads [j p, (j + 1) p) of
        every chip (p = hb / ndev; block head order = rank order, chip r's heads at [p r, p (r + 1))), attends, merges and
        reduce-scatters them (chip r keeps exactly its own p heads of the block), and the blocks concatenate in local head
        order. Every head's softmax, stats, merge weights and owner-chip reduce are those of the full form (per-head math,
        the same owner rank in the reduce-scatter) -> bitwise the full gather; the transient drops by H / hb."""
        tpp = self.tpp
        C, h_r, n = int(q_lat.shape[2]), int(q_lat.shape[1]), tpp.ndev
        hb = int(os.environ.get("GLM53_PF_BQ_HEAD_BLOCK", "0") or 0)
        # GLM53_PF_BQ_ROW_BLOCK=rb (2026-10-23 08:24:55Z ruling, with HEAD_BLOCK=32): the chunk's q rows in blocks of rb
        # rows as well -- every row's attention and owner merge are independent of the other rows (sparse_sdpa_stats per
        # (head, row); the merge per row; the reduce-scatter per element by owner) -> bitwise by construction; the q
        # transient at the 8,192 bucket drops 512 MiB -> 64 MiB per call with hb 32 + rb 2,048
        rb = int(os.environ.get("GLM53_PF_BQ_ROW_BLOCK", "0") or 0)
        kv = sp.local_latents_rm(cache, ci)  # [1, 1, T_local, 512] bf16 ROW_MAJOR
        if hb and hb < h_r * n:
            p_ = hb // n
            assert p_ * n == hb and h_r % p_ == 0, f"GLM53_PF_BQ_HEAD_BLOCK={hb}: a multiple of {n} dividing {h_r * n}"
            # sparse_sdpa(_stats) takes the heads as tile rows: H % 32 == 0 and H >= 32 (sparse_sdpa_device_operation.cpp)
            assert hb % TILE == 0, f"GLM53_PF_BQ_HEAD_BLOCK={hb}: sparse_sdpa needs a multiple of {TILE} heads per call"
        else:
            p_ = h_r
        if rb and rb < C:
            assert rb % TILE == 0 and C % rb == 0, f"GLM53_PF_BQ_ROW_BLOCK={rb}: a multiple of {TILE} dividing {C}"
            row_blocks = [(r0, r0 + rb) for r0 in range(0, C, rb)]
        else:
            row_blocks = [(0, C)]
        rows_out = []
        for r0, r1 in row_blocks:
            whole = (r0, r1) == (0, C)
            q_r = q_lat if whole else ttnn.slice(q_lat, (0, 0, r0, 0), (1, h_r, r1, int(q_lat.shape[3])))
            ids_r = ids_loc if whole else ttnn.slice(ids_loc, (0, 0, r0, 0), (1, 1, r1, int(ids_loc.shape[3])))
            cnt_r = sp.counts if whole else ttnn.slice(sp.counts, (0, 0, r0, 0), (1, 1, r1, int(sp.counts.shape[3])))
            outs = []
            for j in range(0, h_r, p_):
                q_b = q_r if p_ == h_r else ttnn.slice(q_r, (0, j, 0, 0), (1, j + p_, r1 - r0, int(q_r.shape[3])))
                outs.append(self._bq_block(q_b, kv, ids_r, cnt_r, r1 - r0))  # -> [1, p, rows, 512]: own heads, merged
                if q_b is not q_r:
                    ttnn.deallocate(q_b)
            if not whole:
                for t_ in (q_r, ids_r, cnt_r):
                    ttnn.deallocate(t_)
            if len(outs) == 1:
                rows_out.append(outs[0])
            else:
                rows_out.append(ttnn.concat(outs, dim=1))  # [1, H_r, rows, 512] in local head order
                for t_ in outs:
                    ttnn.deallocate(t_)
        ttnn.deallocate(kv)
        if len(rows_out) == 1:
            return rows_out[0]
        o_lat = ttnn.concat(rows_out, dim=2)  # [1, H_r, C, 512] in row order
        for t_ in rows_out:
            ttnn.deallocate(t_)
        return o_lat

    def _bq_block(self, q_part, kv, ids_loc, counts, C: int):
        """B-q over this chip's local heads q_part [1, p, C, 512] (all of them, or one GLM53_PF_BQ_HEAD_BLOCK block; C =
        the chunk's rows or one GLM53_PF_BQ_ROW_BLOCK block, with ids_loc / counts the same rows):
        gather the block's heads of every chip, sparse_sdpa_stats over the owned keys, the owner merge, the head-dim
        reduce-scatter -> [1, p, C, 512] bf16 = this chip's p heads, summed over the owners; kv stays alive"""
        tpp = self.tpp
        H = int(q_part.shape[1]) * tpp.ndev
        q_all = tpp.gather_dim(q_part, 1)  # [1, H, C, 512] bf16 TILE (every chip's heads, rank order)
        q_rm = ttnn.to_layout(q_all, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if q_all is not q_part:
            ttnn.deallocate(q_all)
        out, st = ttnn.transformer.sparse_sdpa_stats(q_rm, kv, ids_loc, 512, kv_format=ttnn.transformer.SparseKVFormat.BF16,
                                                     scale=self.scale, k_chunk_size=sparse_kchunk(),
                                                     compute_kernel_config=self.ckc_fp32)  # fmt: skip
        ttnn.deallocate(q_rm)
        o_c = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, H, C, 512] bf16
        ttnn.deallocate(out)
        st_t = ttnn.to_layout(st, ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 1, C, 2 H] fp32: [m | l]
        ttnn.deallocate(st)
        m_c = ttnn.slice(st_t, (0, 0, 0, 0), (1, 1, C, H))
        l_c = ttnn.slice(st_t, (0, 0, 0, H), (1, 1, C, 2 * H))
        ttnn.deallocate(st_t)
        # the valid flag (fp32 1 / 0 at word 1 of the counts page) broadcast over the heads
        cnt = ttnn.to_layout(counts, ttnn.TILE_LAYOUT, memory_config=DRAM)
        valid1 = ttnn.slice(cnt, (0, 0, 0, 1), (1, 1, C, 2))  # [1, 1, C, 1]
        ttnn.deallocate(cnt)
        valid = ttnn.repeat(valid1, ttnn.Shape((1, 1, 1, H)))  # [1, 1, C, H]
        ttnn.deallocate(valid1)
        # an invalid partial: m -> -1e30 (never the max), l -> 0 (weight 0)
        if os.environ.get("GLM53_PF_BQ_MERGE_OLD", "0") == "1":  # the pre-fix form (A/B only): NOT exact, see below
            m_v = ttnn.add(ttnn.multiply(ttnn.subtract(m_c, NEG), valid), NEG)
        else:  # valid ? m : NEG as an exact select (2026-10-24 01:20Z): (m - NEG) x valid + NEG rounds m + 1e30 to 1e30 in
            # fp32 (ULP ~7.6e22) -> every valid partial's max became 0 and the weights lost exp((m_c - M) scale): the
            # sharded construction's ~0.003 PCC loss per sparse chunk at L3 (the 32K first-token miss lead)
            m_v = ttnn.where(valid, m_c, NEG)
        l_v = ttnn.multiply(l_c, valid)
        for t_ in (m_c, l_c):
            ttnn.deallocate(t_)
        m_all = tpp.gather_dim(m_v, 1)  # [1, N, C, H]
        l_all = tpp.gather_dim(l_v, 1)
        M = ttnn.max(m_all, dim=1, keepdim=True)  # [1, 1, C, H]
        n = tpp.ndev
        M_n = ttnn.repeat(M, ttnn.Shape((1, n, 1, 1)))
        e_all = ttnn.multiply(ttnn.exp(ttnn.multiply(ttnn.subtract(m_all, M_n), self.scale)), l_all)  # [1, N, C, H]
        for t_ in (m_all, l_all, M_n):
            ttnn.deallocate(t_)
        Z = ttnn.sum(e_all, dim=1, keepdim=True)  # [1, 1, C, H]
        ttnn.deallocate(e_all)
        e_c = ttnn.multiply(ttnn.exp(ttnn.multiply(ttnn.subtract(m_v, M), self.scale)), l_v)  # this chip's numerator
        for t_ in (m_v, l_v, M):
            ttnn.deallocate(t_)
        w_c = ttnn.divide(e_c, Z)  # [1, 1, C, H] fp32 (0 on an invalid partial)
        for t_ in (e_c, Z, valid):
            ttnn.deallocate(t_)
        w_t = ttnn.permute(w_c, (0, 3, 2, 1))  # [1, H, C, 1]
        ttnn.deallocate(w_c)
        o32 = ttnn.typecast(o_c, FP32)
        ttnn.deallocate(o_c)
        o_w = ttnn.multiply(o32, w_t)  # [1, H, C, 512] fp32 (the dummy partial reads a finite row: x 0 = 0)
        for t_ in (o32, w_t):
            ttnn.deallocate(t_)
        o_sum = tpp.reduce_scatter_dim(o_w, 1)  # [1, H_r, C, 512] fp32: this chip's heads, summed over the chips
        o_lat = ttnn.typecast(o_sum, BF16)
        ttnn.deallocate(o_sum)
        return o_lat

    def enable_mha(self, spec=None):
        """the non-absorbed (MHA) form of a chunk at position 0: K_h = latents W_uk_h^T, V_h = latents W_uv_h as
        [512, H_r 256] per chip (2 MB each at TP8) + its SDPA config (spec, default GLM53_PF_MHA_SDPA)"""
        mesh, ndev, h_r = self.tpp.mesh, self.tpp.ndev, self.h
        hw = self.shard.host_weights
        mapper = ttnn.ShardTensorToMesh(mesh, dim=1) if ndev > 1 else None

        def flat(w):  # [H, 512, 256] (heads in the shard order) -> [1, ndev, 512, H_r 256]
            return w.reshape(ndev, h_r, 512, 256).permute(0, 2, 1, 3).reshape(1, ndev, 512, h_r * 256)

        if self.w_k is None:
            for name, w in (
                ("w_k", torch.cat([w.float() for w in hw["w_uk"]]).transpose(-2, -1)),
                ("w_v", torch.cat([w.float() for w in hw["w_uv"]])),
            ):  # fmt: skip  # per chip [H_r, ...]
                setattr(self, name, ttnn.from_torch(flat(w), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                                    memory_config=DRAM, mesh_mapper=mapper))  # fmt: skip
        spec = spec or os.environ.get("GLM53_PF_MHA_SDPA", "256,128,HiFi2")
        self.mha_prog, self.mha_ckc = sdpa_config(mesh, self.attn, spec)
        self.mha_spec = spec
        self.mha_progs = {}  # narrower buckets' programs (q chunk = the bucket)

    def __call__(
        self, h, cache, ci: ChunkInputs, h_local=None, sparse: Optional["SparseCtx"] = None, fresh=False, window=None
    ):
        """h [1, 1, C, 4096] bf16 (all rows) -> o_proj partial [1, 1, C, 4096] bf16; fills the chunk's latent rows and
        pooled indexer keys. sparse (chunks with a query past position 2,050): DSA selection on the local rows h_local,
        then ONE flash SDPA over the slot's latents (pool-major order) with the selection as the additive mask
        (GLM53_PF_SPARSE_ATTN=mask), or sparse_sdpa over each row's selected ids in the request's ROW_MAJOR latent
        store (=sdpa)."""
        t = int(h.shape[-2])
        if self.idx is not None:
            self.idx.keys(h, ci)
        sel = self.idx.select(self.attn, h_local, ci, sparse) if sparse is not None else None
        qk = pf_mm(self.attn.ds_qa_kva, h)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.attn.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.attn.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        ttnn.deallocate(c)
        if self.idx is not None and self.idx.lat_rm is not None:
            self.idx.write_latents(latent, ci)
        q = pf_mm(self.shard.ds_qb, q_r)  # [1, 1, C, H_r * 256]
        ttnn.deallocate(q_r)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.h, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
        )  # fmt: skip  # [1, H_r, C, 256]
        lat = ttnn.typecast(latent, cache.dtype) if latent.dtype != cache.dtype else latent
        shard = self.tpp.shard is not None
        if shard:  # GLM53_DSA_SHARD: every tile row to its page's OWNER chip only (ci.lat_idx per chip; SKIP elsewhere)
            tiles_to(lat, ci.lat_idx, cache)
            if window is not None and ci.win_idx is not None:  # GLM53_DSA_DENSE_WINDOW: rows < 2,112 on EVERY chip
                tiles_to(lat, ci.win_idx, window)
        else:
            ttnn.experimental.paged_fill_cache(cache, lat, ci.chunk_table, batch_idx=0)
        sdpa = sparse is not None and sparse.sdpa
        # the sharded cache holds only this chip's units: a dense chunk (no cached context: the host plan routes every
        # continuation chunk of a sharded cache through the sparse / gather path) attends the chunk's OWN latents
        keep_lat = shard and sparse is None
        if self.w_k is not None and (fresh or (sparse is not None and not sdpa)):  # MHA form (GLM53_PF_MLA_FORM=mha)
            if lat is not latent:
                ttnn.deallocate(lat)
            if fresh:  # chunk at position 0: causal over the chunk's own latents
                return self._mha(q, latent)
            ttnn.deallocate(latent)
            kv = sparse.latents(cache, ci)  # [1, 1, T, 512] pool-major
            mask = self.idx.token_mask(sel, sparse)  # [1, 1, C, T] bf16 additive
            out = self._mha(q, kv, mask)
            ttnn.deallocate(mask)
            return out
        if not keep_lat:
            ttnn.deallocate(latent)
        q_lat = ttnn.matmul(q, self.w_uk, compute_kernel_config=self.ckc_head, dtype=BF16, memory_config=DRAM)
        ttnn.deallocate(q)
        prog = self.sdpa_prog if self.sdpa_prog.q_chunk_size <= t else self.attn.prefill_program
        if sdpa and sparse.spa:  # GLM53_PMK_SPA: this chip's rows over the union of their selections, gathered
            kv = sparse.local_latents_rm(cache, ci)  # [1, 1, T_local, 512] the owner's local rows
            if sparse.spa_blocks == 1:
                sparse.spu(kv, sel, sparse.ids_new)  # union + exchange + owner push -> sparse.cmp; ids -> compact rows
                ttnn.deallocate(kv)
                o_lat = self._sparse_sdpa(q_lat, sparse.ids_new, sparse.cmp, kchunk=spa_kchunk())  # the a2a form
            else:
                o_lat = self._spa_blocked(q_lat, kv, sel, sparse)
                ttnn.deallocate(kv)
        elif sdpa and sparse.bq:  # lever B-q: split-K over the owner chips' local latents
            o_lat = self._bq(q_lat, sel, cache, ci, sparse)  # [1, H_r, C, 512]
        elif sdpa:  # gather: the slot's latents from the paged cache (this chunk's rows included: filled above)
            kv = sparse.latents_rm(cache, ci) if sparse.gather else self.idx.lat_rm
            o_lat = self._sparse_sdpa(q_lat, sel, kv)  # [1, H_r, C, 512]
            if sparse.gather:
                ttnn.deallocate(kv)
        elif keep_lat:  # a fresh chunk on the sharded cache: causal over its own latents (the packed path's form)
            o_lat = ttnn.transformer.scaled_dot_product_attention(
                q_lat,
                latent,
                latent,
                is_causal=True,
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=self.sdpa_ckc,
                memory_config=DRAM,
            )  # fmt: skip  # [1, H_r, C, 512]
            ttnn.deallocate(latent)
        elif sparse is None:
            o_lat = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_lat,
                cache,
                cache,
                ci.user_table,
                chunk_start_idx_tensor=ci.chunk_start,
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=self.sdpa_ckc,
                memory_config=DRAM,
            )  # fmt: skip  # [1, H_r, C, 512]
        else:
            kv = sparse.latents(cache, ci)  # [1, 1, T, 512] pool-major (column j P + p = token 4 p + j)
            mask = self.idx.token_mask(sel, sparse)  # [1, 1, C, T] bf16 additive
            o_lat = ttnn.transformer.scaled_dot_product_attention(
                q_lat, kv, kv, attn_mask=mask, is_causal=False, scale=self.scale,
                program_config=prog, compute_kernel_config=self.sdpa_ckc,
                memory_config=DRAM)  # fmt: skip
            ttnn.deallocate(mask)
            ttnn.deallocate(kv)
        ttnn.deallocate(q_lat)
        o = ttnn.matmul(o_lat, self.w_uv, compute_kernel_config=self.ckc_head, dtype=BF16, memory_config=DRAM)
        ttnn.deallocate(o_lat)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)  # [1, 1, C, H_r * 256]
        out = pf_mm(self.shard.ds_o, o)
        ttnn.deallocate(o)
        return out

    def packed(self, h, cache, lat_idx, attn_mask, pk=None, k=0, window=None):  # noqa: C901
        """PACKED prefill (fresh segments back to back, coordinator 2026-10-22 03:15Z): h [1, 1, T, 4096] bf16 (all rows)
        -> o_proj partial [1, 1, T, 4096] bf16. Every segment's latent rows go to ITS slot's pages (tiles_to by lat_idx:
        32-row tile rows, segments start on 32-row boundaries); the attention is ONE SDPA in the absorbed form over the
        chunk's own latents with the additive block-diagonal causal mask attn_mask [1, 1, T, T] (a fresh segment sees
        only its own earlier rows: no cached context). GLM53_PF_DSA=1 (the served env): every segment's pooled indexer
        keys + ring go to its slot (DsaIndexer.keys_packed; pk = the packed inputs, k segments). Fresh segments of <=
        2,048 rows are dense: no selection."""
        t = int(h.shape[-2])
        if self.idx is not None:
            assert self.idx.lat_rm is None, "packed + GLM53_PF_SPARSE_ATTN=sdpa: one request's row-major latent store"
            self.idx.keys_packed(h, pk, k)
        qk = pf_mm(self.attn.ds_qa_kva, h)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.attn.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.attn.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        ttnn.deallocate(c)
        q = pf_mm(self.shard.ds_qb, q_r)
        ttnn.deallocate(q_r)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.h, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
        )  # fmt: skip  # [1, H_r, T, 256]
        lat = ttnn.typecast(latent, cache.dtype) if latent.dtype != cache.dtype else latent
        tiles_to(lat, lat_idx, cache)
        if window is not None and pk is not None and pk.get("win_idx") is not None:  # GLM53_DSA_DENSE_WINDOW
            tiles_to(lat, pk["win_idx"], window)
        if lat is not latent:
            ttnn.deallocate(lat)
        q_lat = ttnn.matmul(q, self.w_uk, compute_kernel_config=self.ckc_head, dtype=BF16, memory_config=DRAM)
        ttnn.deallocate(q)
        prog = self.sdpa_prog if self.sdpa_prog.q_chunk_size <= t else self.attn.prefill_program
        keys = latent
        if pk is not None and pk.get("cont"):  # continuation segments: [the gathered cached prefixes | the packed rows]
            npg = int(pk["pre_table"].shape[-1])
            pre = ttnn.empty((npg, 1, BLOCK, int(latent.shape[-1])), dtype=cache.dtype, layout=ttnn.TILE_LAYOUT,
                             device=h.device(), memory_config=DRAM)  # fmt: skip
            page_gather(cache, pk["pre_table"], pre, npg, (BLOCK // TILE) * (int(latent.shape[-1]) // TILE))
            pre_r = ttnn.reshape(pre, (1, 1, npg * BLOCK, int(latent.shape[-1])))
            pre_b = ttnn.typecast(pre_r, BF16) if pre_r.dtype != BF16 else pre_r
            keys = ttnn.concat([pre_b, latent], dim=2)
            for t_ in {id(v): v for v in (pre, pre_r, pre_b)}.values():
                ttnn.deallocate(t_)
        o_lat = ttnn.transformer.scaled_dot_product_attention(
            q_lat,
            keys,
            keys,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
            program_config=prog,
            compute_kernel_config=self.sdpa_ckc,
            memory_config=DRAM,
        )  # fmt: skip  # [1, H_r, T, 512]
        ttnn.deallocate(q_lat)
        if keys is not latent:
            ttnn.deallocate(keys)
        ttnn.deallocate(latent)
        o = ttnn.matmul(o_lat, self.w_uv, compute_kernel_config=self.ckc_head, dtype=BF16, memory_config=DRAM)
        ttnn.deallocate(o_lat)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        out = pf_mm(self.shard.ds_o, o)
        ttnn.deallocate(o)
        return out


def sdpa_config(mesh, attn, spec=None):
    """(program, compute config) of the prefill SDPA: spec (default GLM53_PF_SDPA) = "q,k,fidelity[,exp_approx]" (P3
    tuning, tests/s16/jobs/sdpa_bench.py; chunk starts must be multiples of q), empty: the decoder's q 64 / k 64 HiFi4
    """
    spec = os.environ.get("GLM53_PF_SDPA", "") if spec is None else spec
    if not spec:
        return attn.prefill_program, attn.ckc_sdpa
    q, k, fid, *rest = spec.split(",")
    prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
                                  q_chunk_size=int(q), k_chunk_size=int(k), exp_approx_mode=rest[:1] == ["1"])  # fmt: skip
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                 math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)  # fmt: skip
    return prog, ckc


CONV_ROWS = 3  # the KDA causal conv's history rows (CONV_K - 1)
NULL_BLOCK = 0  # vLLM's BlockPool.null_block: never a request's page (generator_vllm.vllm_page_rows)


def chunk_pages(row: torch.Tensor, chunk_start: int, n_valid: int, c: int) -> torch.Tensor:
    """The c // BLOCK latent pages a chunk's paged_fill_cache writes (every row of the bucket, padded rows included).
    A page that holds no valid row of the chunk must not alias a page with valid rows: the generator pads a slot's
    page-table row past its coverage with the last covered page (Glm53Generator._write_page_rows), and vLLM covers only
    the blocks the request holds, so a short request in a larger bucket (22 tokens in the 128 bucket: pages [A, A])
    wrote its padded page over its valid one. Pages past the valid end go to the slot's own page while the row covers
    them (rows there are written by decode before they are read), else to the row's last covered page when that page
    holds no valid row, else to the null block (a full-coverage identity row gets there only when the bucket runs past
    max_seq_len). The coverage is the row up to its repeated tail."""
    row = row.to(torch.int64).reshape(-1)
    step = torch.nonzero(row[1:] != row[:-1])
    cov = int(step[-1]) + 2 if step.numel() else 1
    end_p = -(-(chunk_start + n_valid) // BLOCK)  # pages b0 .. end_p - 1 hold the chunk's valid rows
    spare = int(row[cov - 1]) if cov - 1 >= end_p else NULL_BLOCK
    p = torch.arange(chunk_start // BLOCK, (chunk_start + c) // BLOCK)
    assert end_p <= cov, f"valid rows of [{chunk_start}, {chunk_start + n_valid}) past the row's {cov} covered pages"
    own = row[p.clamp(max=row.numel() - 1)]
    return torch.where(p < cov, own, torch.full_like(own, spare)).clamp(min=0)


FRESH = -1  # trace-key window of a dense chunk at position 0 with the MHA-form attention (TPPrefill.fresh_keys)
KPOOL = 4  # index_kpool: one pooled indexer key per 4 tokens
TOPK_POOLS = 512  # index_topk (2048) / index_kpool


def _ix_fidelity():
    """GLM53_PF_IX_FID (HiFi4 default; HiFi2 / LoFi): indexer_score_dsa's matmul fidelity (lever (a) of
    doc/ttnn/14-production-tp8-megakernel/dsa/indexer_scaling.md; not exact below HiFi4)"""
    return getattr(ttnn.MathFidelity, os.environ.get("GLM53_PF_IX_FID", "HiFi4"))


def _ix_program_config():
    """GLM53_PF_IX_CFG = "q_chunk,k_chunk,head_group[,serial[,pass_heads]]" (elements / heads; 0 heads = all; serial 1 =
    the EXACT per-head reduction at any head group: bitwise the default's scores with q / w resident; pass_heads = heads
    per matmul DEST pass in that mode, 1 default / 8 the stretch) -> IndexerScoreProgramConfig;
    unset = the op's default (32, 32, 1). Lever (c) of indexer_scaling.md: the work unit only -- exact by construction
    """
    v = os.environ.get("GLM53_PF_IX_CFG", "")
    if not v.strip():
        return None
    f = [int(x) for x in v.split(",")]
    q, k, hg = f[0], f[1], f[2]
    serial = bool(f[3]) if len(f) > 3 else False
    pass_heads = f[4] if len(f) > 4 else 1
    return ttnn.IndexerScoreProgramConfig(q_chunk_size=q, k_chunk_size=k, head_group_size=hg, serial_head_reduce=serial,
                                          serial_pass_heads=pass_heads)  # fmt: skip


DENSE_EXACT = KPOOL * TOPK_POOLS + KPOOL - 1  # 2051: a query at position p < 2051 selects every visible key
POOL_PAGE = 32  # pooled-key store page: 32 pools (128 tokens), one tile row
# GLM53_DSA_PAGED_POOLS=1 (Stage 12's paged pooled keys, doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3, the padded
# layout i-b): the pooled keys follow each 64-token LATENT BLOCK -- per sparse_dsa layer per chip ppool = [nb_local, 1,
# 32, 128], local block lb owns page lb, its 16 pooled rows in rows 0-15 (rows 16-31 unused). The prefill writes one
# padded tile row per block (paged_pool_src) with tiles_to by a BLOCK-indexed pool_idx (4 entries per block: page tiles
# 4 lb .. 4 lb + 3, SKIP = 0xFFFFFFFF; the host idx is Stage 12's prefill_pool_idx / packed_pool_idx)
PAGED_POOLS = os.environ.get("GLM53_DSA_PAGED_POOLS", "0") == "1"
POOL_BLOCK_ROWS = BLOCK // 4  # pooled rows per 64-token latent block (KPOOL = 4 tokens per pool)


def pool_idx_tiles(c: int, paged: bool) -> int:
    """entries of a c-row chunk's pooled-key tile index (tiles_to): 4 tiles per 128-token unit tile row today, 4 tiles
    per 64-token block's padded tile row under GLM53_DSA_PAGED_POOLS"""
    rows_per_tile_row = BLOCK if paged else 4 * POOL_PAGE  # tokens per pooled tile row
    return max(1, c // rows_per_tile_row) * (128 // TILE)


def paged_pool_src(pooled, c: int):
    """GLM53_DSA_PAGED_POOLS: the chunk's pooled keys [1, 1, c / 4, 128] bf16 TILE -> [1, c / 64, 16, 128] bf16 TILE =
    one [32, 128] tile row per 64-token block (its 16 pooled rows; rows 16-31 = the tile's padding, unused by every
    reader) whose 4 tile pages, in order (block-major, then column), are tiles_to's sources 4 b .. 4 b + 3; pooled stays
    the caller's"""
    nb = c // BLOCK
    rm_ = ttnn.to_layout(pooled, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    blk = ttnn.reshape(rm_, (1, nb, POOL_BLOCK_ROWS, int(pooled.shape[-1])))
    out = ttnn.to_layout(blk, ttnn.TILE_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(rm_)
    return out


INDEX_HEADS, INDEX_DIM = 32, 128
NEG = -1.0e30  # additive mask of an unselected key (finite: -inf + -inf stays -inf-like without NaN paths)
RING_TILES = 2 * INDEX_DIM // TILE  # the decode indexer's ring [S, 1, 32, 256]: 8 tiles per slot
# GLM53_PF_SPARSE_ATTN=sdpa: the selected ids of a query row (512 pools x 4 + the open pool's tail = 2,051) padded with
# sentinels to whole sparse_sdpa key chunks of 256 (fewer bf16 roundings of its running softmax state than 32-key
# chunks)
SPARSE_KCHUNK = 256
SPARSE_W = -(-(TOPK_POOLS * KPOOL + KPOOL - 1) // SPARSE_KCHUNK) * SPARSE_KCHUNK  # 2,304


def spa_kchunk() -> int:
    """GLM53_PMK_SPA_KCHUNK (default 64): the key chunk of the SP-row phase's sparse_sdpa (GLM53_PMK_SPA). Its rows carry
    all 64 heads (2 head tiles: the Q / O CBs are twice the pad form's), and the served construction's L1 floor left the
    128-key form 6,560 B short (14b's M5 B32 45-layer mock, 2026-10-25 10:26Z: program 2736 vs the first L1 buffer on
    0-0..10-9); 64 keys halve the K row-major and K tile CBs (~128 KiB per core). A multiple of 32 dividing SPARSE_W."""
    v = int(os.environ.get("GLM53_PMK_SPA_KCHUNK", "") or 64)
    if v % 32 or SPARSE_W % v:
        raise ValueError(f"GLM53_PMK_SPA_KCHUNK={v}: a multiple of 32 dividing SPARSE_W = {SPARSE_W}")
    return v


def sparse_kchunk() -> int:
    """GLM53_PF_SPARSE_KCHUNK (default SPARSE_KCHUNK = 256): the sparse_sdpa / sparse_sdpa_stats key chunk, read at call
    time (a long-lived server's knob). Its K / V CBs dominate the op's ~1 MB of static CBs on every core; the n29h7 1M
    sharded env's full stack leaves the op 240 KB of L1 above them, so 256 clashed by 83 KB (2026-10-23 07:58Z, d0 2x4
    mock): 128 halves the K chunk. The selection width stays SPARSE_W = 2,304 (a multiple of every allowed chunk)."""
    v = int(os.environ.get("GLM53_PF_SPARSE_KCHUNK", str(SPARSE_KCHUNK)))
    if v % 32 or SPARSE_W % v:
        raise ValueError(f"GLM53_PF_SPARSE_KCHUNK={v}: a multiple of 32 dividing SPARSE_W = {SPARSE_W}")
    return v


def page_gather(
    src,
    table,
    dst,
    n_pages: int,
    tiles_per_page: int,
    ncores: int = 16,
    scatter: bool = False,
    bound=None,
    bound_div: int = 1,
):
    """dst page i <- src page table[i] for i < n_pages (TILE pages; table = int32 [1, >= n_pages] ROW_MAJOR on device:
    the slot's page-table row, so a captured program serves every slot). kernels/prefill/page_gather.cpp.
    bound (GLM53_PF_IX_GATHER_BOUND): a one-element uint32 device word; only pages i < word / bound_div are copied."""
    tile_bytes = {BF16: 2048, FP32: 4096, ttnn.bfloat8_b: 1088}[src.dtype]
    table_bytes = int(table.shape[-1]) * 4
    aligned = (table_bytes + 63) // 64 * 64
    depth = 8
    grid = src.device().compute_with_storage_grid_size()
    ncores = max(1, min(ncores, n_pages))
    per = (n_pages + ncores - 1) // ncores
    cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        first = min(i * per, n_pages)
        rt[x][y] = [src.buffer_address(), dst.buffer_address(), table.buffer_address(), first,
                    min(per, n_pages - first)] + ([bound.buffer_address()] if bound is not None else [])  # fmt: skip
    kd = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, "prefill", "page_gather.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=[tiles_per_page, tile_bytes, table_bytes, aligned, depth] + ([int(bound_div)] if bound is not None else []),
        runtime_args=rt, defines=[("GATHER_CB", "0")] + ([("PAGE_SCATTER", "1")] if scatter else [])
        + ([("PAGE_BOUND", "1")] if bound is not None else []), config=ttnn.ReaderConfigDescriptor())  # fmt: skip
    size = aligned + depth * tile_bytes + (64 if bound is not None else 0)
    cb = ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])  # fmt: skip
    return ttnn.generic_op(
        [src, table, dst] + ([bound] if bound is not None else []),
        ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]),
    )


def pool_compact(src, table, dst, n_units: int, bound, tpp: int = 4, ncores: int = 16):
    """GLM53_PF_IX_GATHER_BOUND: this chip's pooled keys, compact (kernels/prefill/pool_compact.cpp): dst tile row lu =
    rows 0-15 of local blocks 2 lu, 2 lu + 1 (src pages via the slot's LOCAL table), for lu < the device word `bound`
    (this chip's visible local units) of n_units; dst [1, 1, n_units x 32, 128] TILE"""
    tile_bytes = {BF16: 2048, FP32: 4096}[src.dtype]
    table_bytes = int(table.shape[-1]) * 4
    aligned = (table_bytes + 63) // 64 * 64
    grid = src.device().compute_with_storage_grid_size()
    ncores = max(1, min(ncores, n_units))
    per = (n_units + ncores - 1) // ncores
    cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        first = min(i * per, n_units)
        rt[x][y] = [src.buffer_address(), dst.buffer_address(), table.buffer_address(), bound.buffer_address(), first,
                    min(per, n_units - first)]  # fmt: skip
    kd = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, "prefill", "pool_compact.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=[tpp, tile_bytes, table_bytes, aligned], runtime_args=rt, defines=[("GATHER_CB", "0")],
        config=ttnn.ReaderConfigDescriptor())  # fmt: skip
    size = 64 + aligned + 2 * tile_bytes
    cb = ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])  # fmt: skip
    return ttnn.generic_op([src, table, dst, bound], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))


def latents_rm_bounded(cache, table, out, n_blocks: int, bound, ncores: int = 32):
    """GLM53_PF_LAT_BOUND: out [1, 1, T_local, 512] bf16 ROW_MAJOR <- the visible local blocks of the paged bf16 TILE
    latent cache (local block lb = page table[lb]; visible: lb < BPU x the device word `bound`), untilized; blocks are
    strided over the cores. kernels/prefill/latent_rm_{reader,compute,writer}.cpp"""
    assert cache.dtype == BF16, cache.dtype
    tile_bytes = 2048
    tpr, trows = KV_LORA_W // TILE, BLOCK // TILE
    tpp, bpu = tpr * trows, DSH.UNIT // BLOCK
    table_bytes = int(table.shape[-1]) * 4
    aligned = (table_bytes + 63) // 64 * 64
    grid = cache.device().compute_with_storage_grid_size()
    nc = max(1, min(ncores, n_blocks, grid.x * grid.y))
    cores = [(i % grid.x, i // grid.x) for i in range(nc)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rr[x][y] = [cache.buffer_address(), table.buffer_address(), bound.buffer_address(), i, n_blocks, nc]
        rc[x][y] = [i]
        rw[x][y] = [out.buffer_address(), bound.buffer_address(), i, n_blocks, nc]
    src = lambda name: os.path.join(KDIR, "prefill", f"latent_rm_{name}.cpp")  # noqa: E731
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
    )
    kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct_, runtime_args=rt, defines=[], config=cfg)  # fmt: skip
    kernels = [kd("reader", [tile_bytes, table_bytes, aligned, tpr, trows, tpp, bpu], rr,
                  dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
               kd("compute", [tpr, trows], rc, cc),
               kd("writer", [KV_LORA_W * 2, trows, bpu, BLOCK, tpr], rw,
                  dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
    cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[  # noqa: E731
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
    cbs = [cb(0, BF16, tile_bytes, 2 * tpr), cb(1, BF16, tile_bytes, 2 * tpr), cb(2, ttnn.uint32, 64, 1),
           cb(3, ttnn.uint32, 64 + aligned, 1), cb(4, ttnn.uint32, 64, 1)]  # fmt: skip
    ttnn.generic_op([cache, table, out, bound], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


KV_LORA_W = 512  # the latent row width of the MLA cache (kv_lora_rank)


def _rows_program(name, tensors, rows: int, ct: list, rt_head: list, cb_bytes: int, ncores: int, rt_tail=()):
    """a one-kernel generic_op over `rows` ROW_MAJOR rows split across ncores cores (runtime args rt_head + [first,
    count] + rt_tail); kernels/prefill/<name>.cpp, CB 0 of cb_bytes"""
    grid = tensors[0].device().compute_with_storage_grid_size()
    ncores = max(1, min(ncores, rows, grid.x * grid.y))
    per = -(-rows // ncores)
    cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        first = min(i * per, rows)
        rt[x][y] = rt_head + [first, min(per, rows - first)] + list(rt_tail)
    kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "prefill", f"{name}.cpp"),
                               source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
                               compile_time_args=ct, runtime_args=rt, config=ttnn.WriterConfigDescriptor())  # fmt: skip
    cb = ttnn.CBDescriptor(total_size=cb_bytes, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_bytes)])  # fmt: skip
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))


def rows_at(src_rm, start, dst_rm, ncores: int = 16, depth: int = 8):
    """dst_rm row (start + r) <- src_rm row r for every row r of src_rm [.., R, W] ROW_MAJOR (one page per row); start
    = the one-element int32 ROW_MAJOR chunk-start tensor, so a captured program serves every chunk position.
    kernels/prefill/rows_at.cpp"""
    row_bytes = int(src_rm.shape[-1]) * src_rm.element_size()
    rows = int(src_rm.shape[-2]) * (int(src_rm.shape[-3]) if len(src_rm.shape) > 2 else 1)
    return _rows_program("rows_at", [src_rm, start, dst_rm], rows, [row_bytes, depth],
                         [src_rm.buffer_address(), dst_rm.buffer_address(), start.buffer_address()],
                         64 + depth * row_bytes, ncores)  # fmt: skip


def rows_to(src_rm, idx, dst_rm, depth: int = 2):
    """dst_rm row idx[r] <- src_rm row r (idx [1, R] uint32 ROW_MAJOR one page, 0xFFFFFFFF = skip; one page per row):
    a slot-agnostic row scatter (the host writes idx before every replay). kernels/prefill/rows_to.cpp, one core"""
    row_bytes = int(src_rm.shape[-1]) * src_rm.element_size()
    rows = int(src_rm.shape[-2])
    idx_bytes = int(idx.shape[-1]) * 4
    return _rows_program("rows_to", [src_rm, idx, dst_rm], rows, [row_bytes, depth, idx_bytes],
                         [src_rm.buffer_address(), dst_rm.buffer_address(), idx.buffer_address()],
                         (idx_bytes + 63) // 64 * 64 + depth * row_bytes, 1)  # fmt: skip


_TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}

# GLM53_PF_SYNC_EVERY_LAUNCH: the launch fence's process-wide state (installed once, by the first TPPrefill)
_LAUNCH_FENCE: Optional[dict] = None
_LAUNCH_FENCE_MESH = None


def install_launch_fence(mesh) -> dict:
    """GLM53_PF_SYNC_EVERY_LAUNCH=1 (coordinator 2026-10-23 21:19:41Z; the TP8 late-+1 hang): ttnn.synchronize_device
    (mesh) before EVERY program launch of an eager COMPILE pass -- every ttnn operation call (FastOperation) and every
    ttnn.generic_op (the PMK program A / B scopes, the generic-op fillers) -- so no program starts while an earlier one
    can still deliver a late write into its config-ring slot. ON from process start and again from every
    TPPrefill.quiesce() (each compile pass after trace replays starts with one); OFF from ttnn.begin_trace_capture
    until the next quiesce; never inside a capture (depth tracked by wrapping begin / end). Replays and the served
    steady state are untouched. The state dict counts fenced launches and launches seen inside captures."""
    global _LAUNCH_FENCE, _LAUNCH_FENCE_MESH
    _LAUNCH_FENCE_MESH = mesh
    if _LAUNCH_FENCE is not None:
        return _LAUNCH_FENCE
    import ttnn.decorators as _D  # noqa: PLC0415

    st = dict(on=True, depth=0, fenced=0, in_capture=0, captures=0)
    busy = [False]
    _begin, _end, _gop, _call = (
        ttnn.begin_trace_capture,
        ttnn.end_trace_capture,
        ttnn.generic_op,
        _D.FastOperation.__call__,
    )

    def _fence():
        if st["depth"]:
            st["in_capture"] += 1
            return
        if not st["on"] or busy[0]:
            return
        busy[0] = True
        try:
            ttnn.synchronize_device(_LAUNCH_FENCE_MESH)
            st["fenced"] += 1
        finally:
            busy[0] = False

    def begin_trace_capture(*a, **k):
        st["on"] = False  # the compile pass ends where the capture begins
        st["depth"] += 1
        st["captures"] += 1
        try:
            return _begin(*a, **k)
        except BaseException:
            st["depth"] -= 1
            raise

    def end_trace_capture(*a, **k):
        try:
            return _end(*a, **k)
        finally:
            st["depth"] = max(0, st["depth"] - 1)

    def fast_call(self, *a, **k):
        _fence()
        return _call(self, *a, **k)

    _D.FastOperation.__call__ = fast_call
    if not isinstance(_gop, _D.FastOperation):

        def generic_op(*a, **k):
            _fence()
            return _gop(*a, **k)

        ttnn.generic_op = generic_op
    ttnn.begin_trace_capture, ttnn.end_trace_capture = begin_trace_capture, end_trace_capture
    _LAUNCH_FENCE = st
    print(f"TPPrefill: GLM53_PF_SYNC_EVERY_LAUNCH=1: launch fence installed (generic_op is "
          f"{'a FastOperation' if isinstance(_gop, _D.FastOperation) else 'wrapped'})", flush=True)  # fmt: skip
    return st


def tiles_to(src, idx, dst, ncores: int = 16, depth: int = 4):
    """dst tile idx[i] <- src tile i (TILE tensors of the same dtype, interleaved DRAM: one page per tile; idx [1, N]
    uint32 ROW_MAJOR one page, 0xFFFFFFFF = skip): kernels/prefill/rows_to.cpp over tile pages -- the PACKED prefill's
    latent rows (32-row tile rows) into each segment's slot pages (the host writes idx before every replay)"""
    tb = _TILE_BYTES[str(src.dtype).split(".")[-1]]
    assert str(dst.dtype) == str(src.dtype), (src.dtype, dst.dtype)
    n = int(idx.shape[-1])
    idx_bytes = n * 4
    return _rows_program("rows_to", [src, idx, dst], n, [tb, depth, idx_bytes],
                         [src.buffer_address(), dst.buffer_address(), idx.buffer_address()],
                         (idx_bytes + 63) // 64 * 64 + depth * tb, ncores)  # fmt: skip


def expand_rows(pools, start, out, base, ncores: int = 64, shard=None, counts=None):
    """the sparse_sdpa ids of chunk rows: pools [1, 1, R, 512] uint32 ROW_MAJOR (best first) + the chunk start + base
    (one-element int32: the chip's first row in the chunk) -> out [1, 1, R, SPARSE_W] uint32 ROW_MAJOR (in place): row r
    (position p = start + base + r) = 4 pools[r][j] + {0..3} for j < min(n_closed, 512), then the open pool's tokens
    4 n_closed .. p, then 0xFFFFFFFF. kernels/prefill/expand_rows.cpp.
    shard (GLM53_DSA_BQ: the per-chip [rank, C] page, ShardTensorToMesh dim 0) + counts ([1, 1, R, 16] uint32): this
    chip's OWNED keys only, as LOCAL row ids of its latent slice, counts[r][0] = how many (0: one dummy key emitted)"""
    rows, width = int(pools.shape[-2]), int(out.shape[-1])
    sh = shard is not None
    tens = [pools, start, out, base] + ([shard, counts] if sh else [])
    rt = [pools.buffer_address(), start.buffer_address(), out.buffer_address(), base.buffer_address()]
    tail = [shard.buffer_address(), counts.buffer_address()] if sh else []
    return _rows_program("expand_rows", tens, rows, [TOPK_POOLS, KPOOL, width, int(sh)], rt,
                         192 + (TOPK_POOLS * 4 + 63) // 64 * 64 + width * 4 + 64, ncores, rt_tail=tail)  # fmt: skip


class SparseCtx:
    """The pieces one sparse trace (chunk C, key window T = 4 P tokens of the slot, T >= chunk_start + C) shares across
    its sparse_dsa layers. Keys are attended in POOL-MAJOR order (column j P + p = token 4 p + j: softmax attention is
    order-free), so the token mask of a pool selection [C, P] is the selection concatenated 4 times; the causal part (the
    open pool's tail, the future) is one additive [C, T] mask per forward."""

    def __init__(self, tpp: "TPPrefill", c: int, t: int):
        mesh, repl = tpp.mesh, tpp.repl
        assert tpp.mode(c) == "sp" or tpp.ndev == 1, f"sparse prefill needs a sequence-parallel bucket (C={c})"
        assert t % (KPOOL * POOL_PAGE) == 0 and t % BLOCK == 0 and t <= tpp.max_seq, (t, tpp.max_seq)
        self.tpp, self.c, self.t, self.p = tpp, c, t, t // KPOOL
        self.cl = c // tpp.ndev if tpp.ndev > 1 else c
        # indexer_score_dsa's causal bound is a host int (traced as a constant): score every pool a query of ANY chunk
        # start <= T - C can see (device r adds r C_l), then mask with the true bound from the device chunk start
        self.cs_fixed = t - c
        self.sdpa = tpp.sparse_attn in ("sdpa", "gather")  # sparse_sdpa over selected ids (no [C, T] tensor at all)
        # gather: sparse_sdpa's key store = the slot's latents gathered from the PAGED cache per layer into this window's
        # one shared [T, 512] buffer (no per-layer max_seq ROW_MAJOR store, no per-chunk write; rows a packed or another
        # request's chunk wrote are in the cache, so packed admission and interleaved requests stay correct)
        self.gather = tpp.sparse_attn == "gather"
        put =lambda v, dt, layout, mp=repl: ttnn.from_torch(v, dtype=dt, layout=layout, device=mesh,  # noqa: E731
                                                              memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
        self.perm = self.tau = self.iota_c = self.lat_buf = self.ids = self.row_base = None
        # GLM53_PF_SPARSE_RESHARD=a2a (heads -> sequence around sparse_sdpa): each chip expands only its own rows
        self.reshard = self.sdpa and tpp.sparse_reshard == "a2a" and tpp.ndev > 1
        # GLM53_PF_ARENA (default 1): every buffer below that is per-replay scratch (written before it is read inside one
        # replay) or a constant of its shape is taken from TPPrefill.arena, ONE tensor per (kind, shape): the keys of
        # both buckets at one window share the window's pool buffers, the windows of one bucket share its row buffers,
        # and W1's sub-chunk contexts share the TPPrefill's (keys and groups replay strictly in sequence)
        ar = tpp.arena
        if self.sdpa:  # the expanded ids of the chunk rows (expand_rows' output, persistent across replays) + the
            # chip's first row in the chunk (0: every chip expands all C rows; rank x C_l: its own rows)
            rows_ids = self.cl if self.reshard else c
            self.ids = ar("ids", rows_ids, lambda: ttnn.zeros((1, 1, rows_ids, SPARSE_W), dtype=ttnn.uint32,
                                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM))  # fmt: skip
            base = torch.tensor([r * self.cl if self.reshard else 0 for r in range(tpp.ndev)], dtype=torch.int32)
            self.row_base = ar("row_base", (c, self.reshard), lambda: put(
                base.reshape(tpp.ndev, 1) if tpp.ndev > 1 else base.reshape(1, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT,
                ttnn.ShardTensorToMesh(mesh, dim=0) if tpp.ndev > 1 else None))  # fmt: skip
        else:
            tok = torch.arange(self.p).reshape(1, self.p) * KPOOL + torch.arange(KPOOL).reshape(KPOOL, 1)
            tok = tok.reshape(1, t)
            self.perm = put(tok.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # [1, T]
            self.tau = put(tok.float().reshape(1, 1, 1, t), FP32, ttnn.TILE_LAYOUT)  # token of every pool-major column
            self.iota_c = put(torch.arange(c, dtype=torch.float32).reshape(1, 1, c, 1), FP32, ttnn.TILE_LAYOUT)
        rows = torch.arange(c, dtype=torch.float32).reshape(1, 1, c, 1)  # chip r: its rows r C_l .. (r + 1) C_l - 1
        self.rows_local = ar("rows_local", c, lambda: put(rows, FP32, ttnn.TILE_LAYOUT,
                                                          ttnn.ShardTensorToMesh(mesh, dim=2) if tpp.ndev > 1 else None))  # fmt: skip
        self.pool_iota = ar("pool_iota", self.p, lambda: put(torch.arange(self.p, dtype=torch.float32).reshape(1, 1, 1, self.p),
                                                             FP32, ttnn.TILE_LAYOUT))  # fmt: skip
        if not self.sdpa:  # (gather: latents_rm's staging buffer is transient)
            self.lat_buf = ttnn.zeros((1, 1, t, 512), dtype=tpp.latent_dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                      memory_config=DRAM)  # fmt: skip
        self.pool_buf = ar("pool_buf", self.p, lambda: ttnn.zeros((1, 1, self.p, INDEX_DIM), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                                                  device=mesh, memory_config=DRAM))  # fmt: skip
        # GLM53_DSA_SHARD: the cached context is read through the OWNER-AWARE local tables -- every chip gathers its
        # own local pages (page_gather over its local table), the chips' blocks are all-gathered in rank order inside
        # the trace, and these static tables put the units back in global order (tt/dsa_shard.py gather_table), so
        # pool_buf / the key store hold the same bytes as the replicated path's (bitwise selection / attention).
        # (Lever B-q -- score every row against the LOCAL pools, exact candidate merge, sparse_sdpa over the local slice
        # with (m, l) partials and a 64-head reduce-scatter -- needs a sparse_sdpa that emits (m, l): an op change.)
        self.shard = tpp.shard
        self.pool_loc = self.pool_g2l = self.lat_g2l = None
        if self.shard is not None:
            assert self.gather, (
                "GLM53_DSA_SHARD: the sparse prefill reads the cache through the owner-aware tables "
                "(GLM53_PF_SPARSE_ATTN=gather)"
            )
            sm = tpp.sm
            self.t_local = sm.local_cap(t)
            self.p_local = self.t_local // KPOOL
            self.pool_loc = ar("pool_loc", self.p_local, lambda: ttnn.zeros((1, 1, self.p_local, INDEX_DIM), dtype=BF16,
                                                                           layout=ttnn.TILE_LAYOUT, device=mesh,
                                                                           memory_config=DRAM))  # fmt: skip
            self.pool_g2l = ar("pool_g2l", t, lambda: put(sm.gather_table(self.p // POOL_PAGE, self.p_local // POOL_PAGE, 1),
                                                          ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))  # fmt: skip
            self.lat_g2l = ar("lat_g2l", t, lambda: put(sm.gather_table(t // BLOCK, self.t_local // BLOCK, DSH.UNIT // BLOCK),
                                                        ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))  # fmt: skip
        # GLM53_PF_IX_GATHER_BOUND=1 (stage 2 of the visible-pool bound; with GLM53_PF_IX_BOUND=1, whose score reads only
        # the pool tiles < the chip's vis_cols word): paged_pools compacts / gathers / un-permutes only the visible units
        self.gather_bound = (self.shard is not None and os.environ.get("GLM53_PF_IX_GATHER_BOUND", "0") == "1"
                             and os.environ.get("GLM53_PF_IX_BOUND", "0") == "1")  # fmt: skip
        # GLM53_PF_LAT_BOUND=1: local_latents_rm (the BQ key store) only for the visible local blocks
        self.lat_bound = self.shard is not None and os.environ.get("GLM53_PF_LAT_BOUND", "0") == "1"
        # GLM53_DSA_BQ: the per-chip [rank, C] page of expand_rows' SHARD mode and its per-row counts page (word 0 = the
        # keys this chip emitted, word 1 = the valid flag as fp32 bits: the merge's mask)
        self.bq = tpp.bq
        self.shard_page = self.counts = None
        if self.bq:
            pg = torch.zeros(tpp.ndev, 16, dtype=torch.int32)
            pg[:, 0] = torch.arange(tpp.ndev)
            pg[:, 1] = tpp.ndev
            self.shard_page = put(pg, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, ttnn.ShardTensorToMesh(mesh, dim=0))
            self.counts = ttnn.zeros((1, 1, c, 16), dtype=FP32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                     memory_config=DRAM)  # fmt: skip
        # GLM53_PMK_SPA: the compact key store (cap pools x 4 rows, one address on every chip), the remapped ids of this
        # chip's rows and the union program (its L1 sync / bitmap words on the lane core persist across replays)
        self.spa = bool(getattr(tpp, "spa", False)) and self.sdpa
        self.spu = self.cmp = self.ids_new = None
        self.spa_blocks, self.spa_rows = 1, self.cl
        if self.spa:
            assert self.reshard and self.shard is not None, "GLM53_PMK_SPA: the a2a reshard on the sharded cache"
            from models.autoports.zai_org_glm_5_3_flash.tt.spa_union import SpaUnion  # noqa: PLC0415

            # the worst-case union of R rows is min(P, R x 513) pools (every visible pool, or every row's 512 + the open
            # pool): row BLOCKS (GLM53_PMK_SPA_BLOCKS, else halved until a block's worst case fits GLM53_PMK_SPA_CAP) make
            # the bound static -- no overflow at any window (1M TP8: 256 rows = 131K pools = 513 MiB; 4 blocks of 64 rows
            # fit 32K pools = 128 MiB); every row's attention is independent of the blocking (bitwise)
            cap_max = int(os.environ.get("GLM53_PMK_SPA_CAP", "") or 0)  # (a "0" string is truthy: int first)
            nb = int(os.environ.get("GLM53_PMK_SPA_BLOCKS", "") or 0)
            if not nb:
                nb = 1
                while cap_max and min(self.p, (self.cl // nb) * (TOPK_POOLS + 1)) > cap_max and self.cl % (2 * nb) == 0 \
                        and (self.cl // (2 * nb)) % TILE == 0:  # fmt: skip
                    nb *= 2
            assert self.cl % nb == 0 and (self.cl // nb) % TILE == 0, ("GLM53_PMK_SPA_BLOCKS", nb, self.cl)
            self.spa_blocks, self.spa_rows = nb, self.cl // nb
            cap = min(self.p, self.spa_rows * (TOPK_POOLS + 1))
            if cap_max:
                cap = min(
                    cap, cap_max
                )  # only below the static bound when the halving bottomed out (the status flags it)
            self.cmp = tpp.spa_store(cap)
            self.ids_new = ar("ids_new", self.cl, lambda: ttnn.zeros((1, 1, self.cl, SPARSE_W), dtype=ttnn.uint32,
                                                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                                                     memory_config=DRAM))  # fmt: skip
            self.spu = SpaUnion(
                mesh,
                self.cmp,
                self.spa_rows,
                t,
                SPARSE_W,
                cap=cap,
                syncs=tpp.spa_syncs(),
                relay=tpp.spa_relay(self.spa_rows, t),
            )
        self.c_add = self.nvis = None

    def begin(self, ci: ChunkInputs):
        """per forward: the causal additive mask [1, 1, C, T] (pool-major columns; mask mode) and the local rows'
        visible pools"""
        if not self.sdpa:
            pos = ttnn.add(self.iota_c, ci.chunk_start_f)  # [1, 1, C, 1] fp32
            # the causal additive mask STRAIGHT in bf16 (2026-10-22 14:48Z: Stage 9's int27 open OOMed here at T = 32K: the
            # fp32 [C, T] comparison + the rsub / multiply temporaries = 3 x 256 MiB before the bf16 typecast). The compare
            # stays fp32 (token positions exact) and EMITS bf16 0 / 1; scaled to 0 / NEG in place -> one [C, T] bf16 buffer
            # (128 MiB at C = 2,048, T = 32K). Same values as (1 - (tau <= pos)) x NEG -> bf16: tau > pos <=> not tau <= pos
            self.c_add = ttnn.gt(self.tau, pos, dtype=BF16)  # [1, 1, C, T] bf16 (row x column broadcast)
            ttnn.multiply_(self.c_add, NEG)  # 0 / NEG
            ttnn.deallocate(pos)
        pos_l = ttnn.add(self.rows_local, ci.chunk_start_f)  # [1, 1, C_l, 1] global position of every local row
        self.nvis = ttnn.floor(ttnn.multiply(ttnn.add(pos_l, 1.0), 1.0 / KPOOL))  # complete pools visible
        ttnn.deallocate(pos_l)

    def end(self):
        for t_ in (self.c_add, self.nvis):
            if t_ is not None:
                ttnn.deallocate(t_)
        self.c_add = self.nvis = None

    def local_latents_rm(self, cache, ci: ChunkInputs):
        """GLM53_DSA_BQ: this chip's LOCAL latent rows (its local table of the slot), [1, 1, T_local, 512] bf16
        ROW_MAJOR in local order (the key store of its split-K partial); the caller frees it"""
        if (getattr(self, "lat_bound", False) and getattr(ci, "pool_units_u", None) is not None and cache.dtype == BF16
                and self.tpp.ndev > 1):  # fmt: skip
            # ZEROED, not empty (the cf3h-b hit regression, coordinator 2026-10-24 04:32:37Z): the bounded store writes only
            # the visible blocks; a ttnn.empty store left the rows past the bound as UNINITIALIZED DRAM (history-dependent)
            # where the unbounded path held the null page's rows. Zeroed, every visible row is bitwise the same and the rest
            # is deterministic. ON THE DEVICE (cf3h-c window 06:00Z: ttnn.zeros(device=) is a HOST write -> "Writes are not
            # supported during trace capture"): moreh_full is a device program (its DM kernels fill every 1 KB row page)
            out = ttnn.moreh_full([1, 1, self.t_local, 512], 0.0, self.tpp.mesh, dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  memory_config=DRAM)  # fmt: skip
            return latents_rm_bounded(cache, ci.user_table, out, self.t_local // BLOCK, ci.pool_units_u)
        loc = ttnn.empty((1, 1, self.t_local, 512), dtype=self.tpp.latent_dtype, layout=ttnn.TILE_LAYOUT,
                         device=self.tpp.mesh, memory_config=DRAM)  # fmt: skip
        page_gather(cache, ci.user_table, loc, self.t_local // BLOCK, (BLOCK // TILE) * (512 // TILE))
        src = loc if loc.dtype == BF16 else ttnn.typecast(loc, BF16)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if src is not loc:
            ttnn.deallocate(src)
        ttnn.deallocate(loc)
        return rm

    def _gather_local(self, src, table_local, n_local_pages: int, n_pages: int, tpp_: int, g2l, shape_local, dst):
        """GLM53_DSA_SHARD: dst (global page order) <- un-permute(all_gather(this chip's local pages of src))"""
        loc = ttnn.empty(shape_local, dtype=src.dtype, layout=ttnn.TILE_LAYOUT, device=self.tpp.mesh,
                         memory_config=DRAM)  # fmt: skip
        page_gather(src, table_local, loc, n_local_pages, tpp_)
        g = self.tpp.gather(loc, True) if self.tpp.ndev > 1 else loc  # [1, 1, C x local rows, W], block r = chip r
        if g is not loc:
            ttnn.deallocate(loc)
        page_gather(g, g2l, dst, n_pages, tpp_)
        ttnn.deallocate(g)
        return dst

    def paged_pools(self, store, ci: ChunkInputs, dst):
        """GLM53_DSA_PAGED_POOLS: the slot's pooled keys in global pool order -> dst [1, 1, P, 128] bf16 TILE. The pooled
        page IS the latent block (page lb of ppool = local block lb, its 16 pooled rows in rows 0-15), so the latent
        tables serve: this chip's local blocks (ci.user_table) -> every chip's -> global block order (lat_g2l), as
        latents_rm; then each block's rows 0-15 (the padding rows 16-31 dropped) -> P = T / 4 rows"""
        nb_l, nb = self.t_local // BLOCK, self.t // BLOCK
        if getattr(self, "gather_bound", False) and getattr(ci, "pool_units_u", None) is not None and self.tpp.ndev > 1:
            # stage 2: local unit lu = blocks 2 lu, 2 lu + 1 = one pool tile row, so the compact store is already in the
            # pooled layout and the un-permute is pool_g2l (one tile row per unit) straight into dst; pool tile rows
            # >= the chip's vis_cols word / 32 are left stale (the bounded score never reads them)
            nu_l = nb_l // (DSH.UNIT // BLOCK)
            assert nu_l * POOL_PAGE == self.p_local, (nu_l, self.p_local)
            comp = ttnn.empty((1, 1, nu_l * TILE, INDEX_DIM), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.tpp.mesh,
                              memory_config=DRAM)  # fmt: skip
            pool_compact(store, ci.user_table, comp, nu_l, ci.pool_units_u, tpp=INDEX_DIM // TILE)
            g = self.tpp.gather(comp, True)
            ttnn.deallocate(comp)
            page_gather(
                g, self.pool_g2l, dst, self.p // POOL_PAGE, INDEX_DIM // TILE, bound=ci.vis_cols_u, bound_div=POOL_PAGE
            )
            ttnn.deallocate(g)
            return dst
        pad = ttnn.empty((1, 1, nb * TILE, INDEX_DIM), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.tpp.mesh,
                         memory_config=DRAM)  # fmt: skip
        self._gather_local(store, ci.user_table, nb_l, nb, INDEX_DIM // TILE, self.lat_g2l,
                           (1, 1, nb_l * TILE, INDEX_DIM), pad)  # fmt: skip
        rm_ = ttnn.to_layout(pad, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(pad)
        top = ttnn.slice(ttnn.reshape(rm_, (1, nb, TILE, INDEX_DIM)), (0, 0, 0, 0), (1, nb, POOL_BLOCK_ROWS, INDEX_DIM))
        ttnn.deallocate(rm_)
        flat = ttnn.to_layout(ttnn.reshape(top, (1, 1, nb * POOL_BLOCK_ROWS, INDEX_DIM)), ttnn.TILE_LAYOUT,
                              memory_config=DRAM)  # fmt: skip
        ttnn.deallocate(top)
        ttnn.copy(flat, dst)
        ttnn.deallocate(flat)
        return dst

    def latents(self, cache, ci: ChunkInputs):
        """the slot's latent rows 0 .. T - 1 (paged cache -> contiguous -> pool-major) [1, 1, T, 512]"""
        assert (
            self.shard is None
        ), "GLM53_DSA_SHARD: the mask-mode latents (GLM53_PF_SPARSE_ATTN=mask) read a replicated cache"
        page_gather(cache, ci.user_table, self.lat_buf, self.t // BLOCK, (BLOCK // TILE) * (512 // TILE))
        rm = ttnn.to_layout(ttnn.reshape(self.lat_buf, (self.t, 512)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        kv = ttnn.embedding(self.perm, rm, layout=ttnn.TILE_LAYOUT, dtype=self.lat_buf.dtype)
        ttnn.deallocate(rm)
        return ttnn.reshape(kv, (1, 1, self.t, 512))

    def latents_rm(self, cache, ci: ChunkInputs):
        """gather mode: the slot's latent rows 0 .. T - 1 in TOKEN order (paged cache -> contiguous), [1, 1, T, 512] bf16
        ROW_MAJOR: sparse_sdpa's key store (ids = token positions), 32 MiB at T = 32K; the caller frees it"""
        # the TILE staging buffer is transient (every page written by the gather): no per-window resident 32 MiB at 32K
        buf = ttnn.empty((1, 1, self.t, 512), dtype=self.tpp.latent_dtype, layout=ttnn.TILE_LAYOUT, device=self.tpp.mesh,
                         memory_config=DRAM)  # fmt: skip
        tpp_ = (BLOCK // TILE) * (512 // TILE)
        if (
            self.shard is not None
        ):  # this chip's local blocks (ci.user_table = its LOCAL table) -> every chip's -> global
            self._gather_local(cache, ci.user_table, self.t_local // BLOCK, self.t // BLOCK, tpp_, self.lat_g2l,
                               (1, 1, self.t_local, 512), buf)  # fmt: skip
        else:
            page_gather(cache, ci.user_table, buf, self.t // BLOCK, tpp_)
        src = buf if buf.dtype == BF16 else ttnn.typecast(buf, BF16)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if src is not buf:
            ttnn.deallocate(src)
        ttnn.deallocate(buf)
        return rm

    def release(self):
        self.end()
        for t_ in (
            self.perm,
            self.tau,
            self.iota_c,
            self.rows_local,
            self.pool_iota,
            self.lat_buf,
            self.pool_buf,
            self.ids,
            self.row_base,
            self.pool_loc,
            self.pool_g2l,
            self.lat_g2l,
            self.shard_page,
            self.counts,
            self.ids_new,
        ) + (
            tuple(self.spu.owned()) if self.spu is not None else ()
        ):  # fmt: skip  # (cmp / syncs: the TPPrefill's)
            if t_ is not None and self.tpp.in_arena(t_):  # GLM53_PF_ARENA: the TPPrefill's (shared by other keys)
                continue
            if t_ is not None:
                ttnn.deallocate(t_)


class DsaIndexer:
    """sparse_dsa lightning indexer (k-pooled, kpool 4) on the prefill: every chunk writes its pooled keys into the
    slot's rows of the pooled-key store ([B P_max / 32, 1, 32, 128] bf16 TILE == the decode indexer's
    pools [B, 1, P_max, 128], prework/dsa/indexer_decode.py); a sparse chunk scores its LOCAL query rows against the
    slot's pools (indexer_score_dsa, key_compression_ratio 4), keeps the top-512 visible pools per row (threshold = the
    rank-512 score: ties keep every pool at the threshold) + the open pool (tail), and gathers the selection [C, P]."""

    def __init__(self, tpp: "TPPrefill", L: int):
        mesh, src = tpp.mesh, tpp.source
        dev = lambda n: src.device(L, f"self_attn.indexer.{n}", mesh)  # noqa: E731
        row = lambda t: ttnn.reshape(t, (1, 1, 1, t.shape[-1]))  # noqa: E731
        self.tpp = tpp
        self.w_kg = ttnn.concat([dev("wk.weight"), dev("index_kpool_compress_gate")], dim=-1, memory_config=DRAM)
        self.w_p = dev("weights_proj.weight")  # [4096, 32]
        self.wq_b = dev("wq_b.weight")  # [1536, 4096]
        self.k_w, self.k_b = row(dev("k_norm.weight")), row(dev("k_norm.bias"))
        ape = src.host(L, "self_attn.indexer.index_kpool_compress_ape").float().reshape(KPOOL, INDEX_DIM)
        self.ape = [ttnn.from_torch(ape[j].reshape(1, 1, 1, INDEX_DIM), dtype=FP32, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM, mesh_mapper=tpp.repl) for j in range(KPOOL)]  # fmt: skip
        # relu(q.k / sqrt(128)) * w * 32^-0.5 == relu(q.k) * (w * 32^-0.5 * 128^-0.5)
        self.w_scale = INDEX_HEADS**-0.5 * INDEX_DIM**-0.5
        self.ckc = _ckc_fp32(mesh)
        # indexer levers (doc/ttnn/14-production-tp8-megakernel/dsa/indexer_scaling.md): GLM53_PF_IX_FID (default HiFi4 =
        # today) and GLM53_PF_IX_CFG = "q_chunk,k_chunk,head_group" (default = the op's 32, 32, 1 = today)
        self.ckc_score = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=_ix_fidelity(),
                                                                math_approx_mode=False, fp32_dest_acc_en=False)  # fmt: skip
        self.score_cfg = _ix_program_config()
        # GLM53_PF_IX_BOUND=1: the per-chunk EXACT causal bound -- indexer_score_dsa reads the chunk start from the
        # device word ci.chunk_start_u (trace-safe) and scores only the pools this chip's rows can see (kv_len =
        # ceil((start + (rank + 1) C_l) / 128) tiles) instead of the whole captured window (cs_fixed); the columns
        # past the bound are left unwritten (stale) and masked by `vis` below, so the selection is unchanged
        self.bound = os.environ.get("GLM53_PF_IX_BOUND", "0") == "1"
        # GLM53_PF_IX_TOPK_BOUND=1 (with IX_BOUND; stage 1 of the visible-pool bound, coordinator 2026-10-23 23:29:42Z):
        # the score op already stamps the exact per-row causal -inf inside its written extent (the columns past it are
        # left stale), so the top-k runs on the score's own ROW_MAJOR plane with valid_length_tensor = the chip's word
        # ci.vis_cols_u: no TILE round trip, no P-wide lt / typecast / where. expand_rows reads only pools[r][0..nsel)
        self.topk_bound = self.bound and os.environ.get("GLM53_PF_IX_TOPK_BOUND", "0") == "1"
        # GLM53_PF_IX_TAP=1: keep the masked score plane of the last forward / replay per (rows, pools) in a
        # persistent buffer (taps[(C_l, P)]) for the bitwise A/B of the indexer levers (tests/s16/jobs/p3_long_check)
        self.taps = {} if os.environ.get("GLM53_PF_IX_TAP", "0") == "1" else None
        # GLM53_DSA_SHARD: the store is per chip (the slot's LOCAL pooled-key pages; the decode's stack_state)
        self.pages_per_slot = (tpp.max_seq_local if tpp.shard is not None else tpp.max_seq) // (KPOOL * POOL_PAGE)
        st = next((s for i, s in tpp.kv.extra.get("dsa", {}).items() if tpp.b.seq[i]["L"] == L), None)
        if st is None and tpp.shard is not None:
            raise ValueError(
                "GLM53_DSA_SHARD: the TP prefill writes the decode's LOCAL pooled-key store (GLM53_STACK_DSA=1)"
            )
        if st is not None:  # Stage 14 GLM53_STACK_DSA: the decode's per-layer store + ring (same pages)
            # (no early return: the sparse_sdpa latent store below is set up either way - 2026-10-22, int24 merge)
            # (slot_rows: the prefix-cache snapshot pool's virtual slots too -- StackBackend sizes the decode's store
            # with them; the prefix cache copies a slot's pages / ring tiles to and from them, StackBackend._dsa_ops)
            if st.get(
                "paged", False
            ):  # GLM53_DSA_PAGED_POOLS: pools [nb_local, 1, 32, 128] = one page per local latent
                # block (12's _paged_pool_state); the prefill writes / reads it through the latent tables (Stage 16)
                assert (
                    PAGED_POOLS and tpp.shard is not None
                ), "a paged pooled-key store needs GLM53_DSA_PAGED_POOLS + SHARD"
            else:
                ps = st.get("pool_slots", st["slots"])  # GLM53_DSA_STORE_SEATS: the store's seats
                assert st["tp"] // TILE == self.pages_per_slot and ps >= tpp.slot_rows, (st["tp"], ps, tpp.slot_rows)
            self.store, self.ring = st["pools"], st["ring"]
        else:
            # per-slot rows: every slot + the prefix-cache snapshot pool's virtual slots (tpp.slot_rows)
            self.store = ttnn.zeros((tpp.slot_rows * self.pages_per_slot, 1, POOL_PAGE, INDEX_DIM), dtype=BF16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)  # fmt: skip
            # the decode indexer's ring (mk_mla DsaIndexer.state, tests/s7b/golden_mla.dsa_state): row j = [k | g] of
            # position 4 (L // 4) + j if < L, else of the previous pool's position 4 (L // 4) - 4 + j, after L tokens
            self.ring = ttnn.zeros((tpp.slot_rows, 1, TILE, 2 * INDEX_DIM), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                   device=mesh, memory_config=DRAM)  # fmt: skip
        # GLM53_PF_SPARSE_ATTN=sdpa: the in-flight request's latents [1, 1, max_seq, 512] bf16 ROW_MAJOR (row = token
        # position; every chunk writes its rows, dense chunks included), the sparse_sdpa key store
        self.lat_rm = None
        if tpp.sparse_attn == "sdpa":
            self.lat_rm = ttnn.zeros((1, 1, tpp.max_seq, 512), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                     memory_config=DRAM)  # fmt: skip

    def write_latents(self, latent, ci: ChunkInputs):
        """the chunk's latent rows [1, 1, C, 512] bf16 (all rows) -> the ROW_MAJOR store at rows chunk_start + r"""
        rm = ttnn.to_layout(latent, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        rows_at(rm, ci.chunk_start, self.lat_rm)
        ttnn.deallocate(rm)

    def keys(self, h, ci: ChunkInputs):
        """h [1, 1, C, 4096] bf16 (all rows) -> the chunk's C / 4 pooled keys into the store (fp32 math, bf16 store):
        K_p[c] = sum_j softmax_j(g_{4p+j}[c] + ape_j[c]) k_{4p+j}[c], k = LayerNorm(x W_k), g = x G^T"""
        c = int(h.shape[-2])
        rm = self._kg_rm(h)
        self.write_ring(rm, ci, c)
        pooled = self._pooled(rm, c)
        ttnn.deallocate(rm)
        if self.tpp.shard is not None and PAGED_POOLS:  # each 64-token block's padded tile row -> its block's page
            src = paged_pool_src(pooled, c)
            tiles_to(src, ci.pool_idx, self.store)
            ttnn.deallocate(src)
        elif self.tpp.shard is not None:  # GLM53_DSA_SHARD: each unit's tile row to its OWNER chip's local page only
            tiles_to(pooled, ci.pool_idx, self.store)
        else:
            ttnn.experimental.paged_fill_cache(self.store, pooled, ci.pool_table, batch_idx=0)
        ttnn.deallocate(pooled)

    def keys_packed(self, h, pk: dict, k: int):
        """the PACKED prefill's pooled keys: h [1, 1, T, 4096] bf16 (all rows; segments on 128-row boundaries, so no pool
        straddles two segments) -> T / 4 pooled keys, every segment's pool tile rows to ITS slot's store pages
        (tiles_to by pk["pool_idx"], 0xFFFFFFFF = skip), and each segment's ring rows (pk["ring_rows"][b] /
        pk["ring_pages"][b]) -> ring[slot_b]"""
        t = int(h.shape[-2])
        rm = self._kg_rm(h)
        ext = self._ring_src(rm, t)
        for b in range(k):
            self._ring_to(ext, pk["ring_rows"][b], pk["ring_pages"][b])
        ttnn.deallocate(ext)
        pooled = self._pooled(rm, t)
        ttnn.deallocate(rm)
        if self.tpp.shard is not None and PAGED_POOLS:  # (segments on 128-row boundaries: whole 64-token blocks)
            src = paged_pool_src(pooled, t)
            tiles_to(src, pk["pool_idx"], self.store)
            ttnn.deallocate(src)
        else:
            tiles_to(pooled, pk["pool_idx"], self.store)
        ttnn.deallocate(pooled)

    def _kg_rm(self, h):
        """[k | g] of every row, ROW_MAJOR [1, 1, C, 256] fp32: k = LayerNorm(x W_k), g = x G^T"""
        c = int(h.shape[-2])
        kg = ttnn.linear(h, self.w_kg, dtype=FP32, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1,1,C,256]
        k = ttnn.layer_norm(ttnn.slice(kg, (0, 0, 0, 0), (1, 1, c, INDEX_DIM)), weight=self.k_w, bias=self.k_b,
                            epsilon=1e-6, compute_kernel_config=self.ckc)  # fmt: skip
        g = ttnn.slice(kg, (0, 0, 0, INDEX_DIM), (1, 1, c, 2 * INDEX_DIM))
        ttnn.deallocate(kg)
        kg = ttnn.concat([k, g], dim=-1)
        ttnn.deallocate(k)
        ttnn.deallocate(g)
        rm = ttnn.to_layout(kg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(kg)
        return rm

    def _pooled(self, rm, c: int):
        """the pooled keys [1, 1, C / 4, 128] bf16 from the [k | g] rows (rm stays the caller's)"""
        # a pool's 4 tokens side by side: [C, 256] -> [C / 4, 4 x 256] (a row-major view)
        grp = ttnn.to_layout(ttnn.reshape(rm, (1, 1, c // KPOOL, KPOOL * 2 * INDEX_DIM)), ttnn.TILE_LAYOUT,
                             memory_config=DRAM)  # fmt: skip
        n = c // KPOOL
        col = lambda a: ttnn.slice(grp, (0, 0, 0, a), (1, 1, n, a + INDEX_DIM))  # noqa: E731
        ks = [col(2 * INDEX_DIM * j) for j in range(KPOOL)]
        gs = [ttnn.add(col(2 * INDEX_DIM * j + INDEX_DIM), self.ape[j]) for j in range(KPOOL)]
        ttnn.deallocate(grp)
        m = ttnn.maximum(ttnn.maximum(gs[0], gs[1]), ttnn.maximum(gs[2], gs[3]))
        es = [ttnn.exp(ttnn.subtract(g_, m)) for g_ in gs]
        den = ttnn.add(ttnn.add(es[0], es[1]), ttnn.add(es[2], es[3]))
        num = ttnn.add(ttnn.add(ttnn.multiply(es[0], ks[0]), ttnn.multiply(es[1], ks[1])),
                       ttnn.add(ttnn.multiply(es[2], ks[2]), ttnn.multiply(es[3], ks[3])))  # fmt: skip
        pooled = ttnn.typecast(ttnn.divide(num, den), BF16)  # [1, 1, C / 4, 128]
        for t_ in ks + gs + es + [m, den, num]:
            ttnn.deallocate(t_)
        return pooled

    def write_ring(self, kg_rm, ci: ChunkInputs, c: int):
        """the slot's ring rows 0..3 from this chunk's [k | g] rows (ci.ring_rows; row C = zeros) -> ring[slot]"""
        ext = self._ring_src(kg_rm, c)
        self._ring_to(ext, ci.ring_rows, ci.ring_pages)
        ttnn.deallocate(ext)

    def _ring_src(self, kg_rm, c: int):
        """the [k | g] rows as bf16 ROW_MAJOR [C + 32, 256] (rows C.. = zeros: the ring's empty rows)"""
        return ttnn.to_layout(ttnn.pad(ttnn.typecast(ttnn.to_layout(ttnn.reshape(kg_rm, (c, 2 * INDEX_DIM)),
                                                                     ttnn.TILE_LAYOUT), BF16),
                                        ((0, TILE), (0, 0)), 0.0), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)  # fmt: skip

    def _ring_to(self, ext, ring_rows, ring_pages):
        rows = ttnn.reshape(ttnn.embedding(ring_rows, ext, layout=ttnn.TILE_LAYOUT, dtype=BF16),
                            (1, 1, TILE, 2 * INDEX_DIM))  # fmt: skip
        page_gather(rows, ring_pages, self.ring, RING_TILES, 1, ncores=RING_TILES, scatter=True)
        ttnn.deallocate(rows)

    def select(self, attn, h_local, ci: ChunkInputs, sp: "SparseCtx"):
        """the local rows' pool selection, gathered: [1, 1, C, P] bf16 1 = attend the pool's tokens (top-512 visible
        pools; every visible pool while fewer than 512; the open pool -- its causal tail is in sp.c_add)"""
        tpp = self.tpp
        cl = int(h_local.shape[-2])
        qk = pf_mm(attn.ds_qa_kva, h_local)
        q_r = ttnn.rms_norm(ttnn.slice(qk, (0, 0, 0, 0), (1, 1, cl, attn.q_lora)), weight=attn.qa_norm,
                            epsilon=attn.eps, compute_kernel_config=self.ckc)  # fmt: skip
        ttnn.deallocate(qk)
        qi = ttnn.linear(q_r, self.wq_b, dtype=BF16, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(q_r)
        qi, _, _ = ttnn.experimental.nlp_create_qkv_heads(qi, num_heads=INDEX_HEADS, num_kv_heads=0,
                                                          transpose_k_heads=False, memory_config=DRAM)  # fmt: skip
        w = ttnn.linear(h_local, self.w_p, dtype=FP32, compute_kernel_config=self.ckc, memory_config=DRAM)
        w = ttnn.typecast(ttnn.multiply(w, self.w_scale), BF16)  # [1, 1, C_l, 32]
        if sp.shard is not None and PAGED_POOLS:  # the block pages (the latent tables) -> pool_buf in global pool order
            sp.paged_pools(self.store, ci, sp.pool_buf)
        elif sp.shard is not None:  # the slot's LOCAL pooled-key pages of every chip -> pool_buf in global pool order
            sp._gather_local(self.store, ci.pool_user, sp.p_local // POOL_PAGE, sp.p // POOL_PAGE, INDEX_DIM // TILE,
                             sp.pool_g2l, (1, 1, sp.p_local, INDEX_DIM), sp.pool_buf)  # fmt: skip
        else:
            page_gather(self.store, ci.pool_user, sp.pool_buf, sp.p // POOL_PAGE, INDEX_DIM // TILE)
        s = ttnn.experimental.indexer_score_dsa(
            qi,
            sp.pool_buf,
            w,
            key_compression_ratio=KPOOL,
            compute_kernel_config=self.ckc_score,
            **({"chunk_start_idx_tensor": ci.chunk_start_u} if self.bound else {"chunk_start_idx": sp.cs_fixed}),
            **({"program_config": self.score_cfg} if getattr(self, "score_cfg", None) is not None else {}),
        )  # fmt: skip  # [1,1,C_l,P]
        ttnn.deallocate(qi)
        ttnn.deallocate(w)
        if getattr(self, "topk_bound", False) and sp.sdpa and ci.vis_cols_u is not None:
            s_rm = s if s.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
            if s_rm.dtype != BF16:
                s_rm = ttnn.typecast(s_rm, BF16)
            self.s_meta = dict(s_dtype=str(s.dtype), s_layout=str(s.layout), s_shape=list(s.shape), s_padded=list(s.padded_shape),
                               in_dtype=str(s_rm.dtype), in_layout=str(s_rm.layout), in_mem=str(s_rm.memory_config()))  # fmt: skip
            if self.taps is not None:  # the gate's probe: arm B's top-k input plane (a device copy in the trace)
                key = ("b", cl, sp.p)
                if key not in self.taps:
                    self.taps[key] = ttnn.zeros(tuple(s_rm.shape), dtype=s_rm.dtype, layout=s_rm.layout, device=tpp.mesh,
                                                memory_config=DRAM)  # fmt: skip
                ttnn.copy(s_rm, self.taps[key])
            pools = ttnn.experimental.topk_large_indices(s_rm, k=TOPK_POOLS, valid_length_tensor=ci.vis_cols_u)
            ttnn.deallocate(s_rm)
            if s_rm is not s:
                ttnn.deallocate(s)
            return self._expand_pools(pools, ci, sp)
        if s.layout != ttnn.TILE_LAYOUT:
            s = ttnn.to_layout(s, ttnn.TILE_LAYOUT)
        vis = ttnn.typecast(ttnn.lt(sp.pool_iota, sp.nvis), BF16)  # [1, 1, C_l, P] complete pools visible to the row
        if s.dtype != BF16:
            s = ttnn.typecast(s, BF16)
        sm = ttnn.where(vis, s, NEG)
        ttnn.deallocate(s)
        if self.taps is not None:  # the A/B tap (a device copy inside the trace)
            key = (cl, sp.p)
            if key not in self.taps:
                self.taps[key] = ttnn.zeros((1, 1, cl, sp.p), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=tpp.mesh,
                                            memory_config=DRAM)  # fmt: skip
            ttnn.copy(sm, self.taps[key])
        if sp.sdpa:  # top-512 pool ids per local row -> every row's (gathered) -> expanded token ids [1, 1, C, W]
            ttnn.deallocate(vis)
            pools = ttnn.experimental.topk_large_indices(ttnn.to_layout(sm, ttnn.ROW_MAJOR_LAYOUT), k=TOPK_POOLS)
            ttnn.deallocate(sm)
            return self._expand_pools(pools, ci, sp)
        srt, srt_i = ttnn.sort(sm, dim=-1, descending=True)
        ttnn.deallocate(srt_i)
        thr = ttnn.slice(srt, (0, 0, 0, TOPK_POOLS - 1), (1, 1, cl, TOPK_POOLS))  # rank-512 score (NEG: < 512 pools)
        ttnn.deallocate(srt)
        # unselected = visible and below the threshold; everything else (selected, open pool, future) -> 1
        sel = ttnn.rsub(ttnn.multiply(vis, ttnn.lt(sm, thr)), 1.0)
        for t_ in (vis, sm, thr):
            ttnn.deallocate(t_)
        return tpp.gather(sel, True) if tpp.ndev > 1 else sel

    def _expand_pools(self, pools, ci, sp: "SparseCtx"):
        """the top-512 pool ids per local row -> every row's (gathered) -> expanded token ids sp.ids (+ sp.counts)"""
        tpp = self.tpp
        pt = getattr(self, "pool_taps", None)
        if (
            pt is not None
        ):  # the stage-1 gate's probe: this select's local top-512 pool ids (a device copy in the trace)
            key = (int(pools.shape[-2]), int(sp.p), bool(getattr(self, "topk_bound", False)))
            if key not in pt:
                pt[key] = ttnn.zeros(tuple(pools.shape), dtype=pools.dtype, layout=pools.layout, device=tpp.mesh,
                                     memory_config=DRAM)  # fmt: skip
            ttnn.copy(pools, pt[key])
        if tpp.ndev > 1 and not sp.reshard:  # pool ids < 2^24: exact in fp32 through the (TILE) all-gather
            pf_ = ttnn.typecast(ttnn.to_layout(pools, ttnn.TILE_LAYOUT), FP32)
            ttnn.deallocate(pools)
            g = tpp.gather(pf_, True)
            ttnn.deallocate(pf_)
            pools = ttnn.to_layout(ttnn.typecast(g, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(g)
        if sp.bq:  # this chip's OWNED keys as local row ids (+ the per-row counts / valid flags)
            expand_rows(pools, ci.chunk_start, sp.ids, sp.row_base, shard=sp.shard_page, counts=sp.counts)
        else:
            expand_rows(pools, ci.chunk_start, sp.ids, sp.row_base)
        ttnn.deallocate(pools)
        return sp.ids

    def token_mask(self, sel, sp: "SparseCtx"):
        """[1, 1, C, P] pool selection -> additive [1, 1, C, T] bf16 over the pool-major keys"""
        pool_add = ttnn.multiply(ttnn.rsub(sel, 1.0), NEG)  # selected 0, else NEG
        ttnn.deallocate(sel)
        tok = ttnn.concat([pool_add] * KPOOL, dim=-1)
        ttnn.deallocate(pool_add)
        mask = ttnn.add(tok, sp.c_add)
        ttnn.deallocate(tok)
        return mask


def kda_snap_rows(snap_at, chunk_start: int, n_valid: int, c: int) -> tuple:
    """GLM53_PF_KDA_SNAPSHOT: (does this chunk carry the cut, the snapshot scalar). The cut snap_at (absolute) is
    carried by the chunk with chunk_start < cut <= chunk_start + n_valid; the scalar is then the chunk's rows before it,
    else 32 (the scan needs a nonempty interval: a one-chunk snapshot nobody reads). The rows land on the 32-row chunk
    grid (cuts are 128-aligned, chunk starts 64-aligned)."""
    carries = snap_at is not None and chunk_start < int(snap_at) <= chunk_start + n_valid
    rows = int(snap_at) - chunk_start if carries else 32
    if rows % 32 or not 0 < rows <= c:
        raise ValueError(
            f"KDA snapshot cut {snap_at} in the chunk [{chunk_start}, +{n_valid}) of {c} rows: {rows} rows"
        )
    return carries, rows


def ci_carry_missed(chunk_start: int, n_valid: int, snap_at: int) -> bool:
    """a requested KDA snapshot cut this chunk does not cover (the caller passed snap_at with another chunk)"""
    return not chunk_start < int(snap_at) <= chunk_start + n_valid


class TPPrefill:
    """Batched prefill of a StackBackend (TP mesh or one chip), traced per bucket. Usage:
    pf = TPPrefill(backend, kv); pf.prepare(buckets)        # compile (eager warm-up) every bucket, then capture
    tok = pf.prefill(tokens, slot=, chunk_start=, n_valid=)  # one chunk (host inputs + one replay + slot copies)
    """

    def __init__(self, backend, kv, *, buckets=(256, 512, 2048), source=None, moe_impl: Optional[str] = None):
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind

        self.b, self.kv = backend, kv
        self.mesh, self.cfg = backend.mesh, backend.cfg
        self.ndev = int(self.mesh.get_num_devices())
        self.source = source or FdTileCacheSource(CACHE_DIR, MANIFEST)
        self.moe_impl = moe_impl or os.environ.get("GLM53_PF_MOE", "ep")
        self.kda_scan = os.environ.get("GLM53_PF_KDA_SCAN", "op")
        # GLM53_PF_KDA_SNAPSHOT=1 (one-replay store A): every KDA layer also writes its state at the request's
        # checkpoint cut (an extra recurrent_chunk_scan with actual_end = the cut + the conv history there) into a
        # second scratch, inside the same replay; StackBackend.export_kda_state(source="snapshot") reads it
        # (kv.extra["kda_snap_stage"])
        self.kda_snapshot = os.environ.get("GLM53_PF_KDA_SNAPSHOT", "0") == "1"
        # residual / mHC layout: "comp" = flat [1, 1, T, 16384] + Stage 3's composite mHC; "sm" = stream-major
        # [1, 4, T, 4096] + StreamMajorHyperConnection (same fp32 math, fewer / larger ops)
        self.mhc_mode = os.environ.get("GLM53_PF_MHC", "comp")
        # comp: the post-mix (4 multiplies + 16 addcmuls + the concat per sublayer) as one program (mhc_post_prefill)
        self.mhc_post = os.environ.get("GLM53_PF_MHC_POST", "ops")
        # GLM53_PF_MHC_POST7A=1 (Stage 7a lever A, tt/mhc_pre_pf.mhc_post_prefill_7a): the fused post-mix with one SFPU
        # pass per output tile (same data movement and arithmetic, fp32 FMAs) for the fused-mHC layers
        self._post = mhc_post_prefill
        if os.environ.get("GLM53_PF_MHC_POST7A", "0") == "1":
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf import mhc_post_prefill_7a

            # up to GLM53_PF_MHC_POST7A_MAX_T local rows (card1 10-22 05:55Z, 1,000 replays: T 256 167.5 vs 268.2 us,
            # T 1,024 604 vs 827, T 2,048 1,121 vs 958: past ~1,024 rows its compute no longer hides under the reads)
            max_t = int(os.environ.get("GLM53_PF_MHC_POST7A_MAX_T", "1024"))
            self._post = lambda y, x, post, comb: (  # noqa: E731
                mhc_post_prefill_7a(y, x, post, comb)
                if int(x.shape[-2]) <= max_t
                else mhc_post_prefill(y, x, post, comb)
            )
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self.repl = ttnn.ReplicateTensorToMesh(self.mesh) if self.ndev > 1 else None
        self._shared_cache, self._ep = {}, None
        self.max_seq = int(kv.blocks_per_seq) * BLOCK
        self.dsa = os.environ.get("GLM53_PF_DSA", "0") == "1"  # P3 DSA indexer (off: the P2 dense-only path)
        # GLM53_DSA_SHARD (StackBackend.dsa_shard_owners: (C chips, local blocks per chip, blocks per 128-token unit);
        # tt/dsa_shard.py ShardMap): the latent cache and the pooled-key store are PER CHIP (unit u of a slot on chip
        # u % C at local block / page u // C); the prefill writes each unit to its owner only (ChunkInputs.lat_idx /
        # pool_idx) and reads the cached context through the owner-aware local tables (SparseCtx._gather_local)
        self.shard = getattr(kv, "extra", {}).get("dsa_shard")
        self.sm = None
        if self.shard is not None:
            self.sm = DSH.ShardMap(int(self.shard[0]), BLOCK)
            self.bps_local = int(kv.extra["bps_local"])
            self.max_seq_local = self.bps_local * BLOCK
        self.max_batch = int(kv.max_batch)
        # per-slot stores (DSA pooled keys / ring) also hold the prefix-cache snapshot pool's virtual slots
        self.slot_rows = self.max_batch + int(getattr(kv, "extra", {}).get("snapshot_entries", 0))  # exact entries
        lat = getattr(kv, "extra", {}).get("latent", {})
        self.latent_dtype = next(iter(lat.values())).dtype if lat else BF16
        # sparse_dsa chunks past position 2,050 (DSA top-512 selection): one trace per (bucket, key window T); the
        # windows to capture (tokens, multiples of 128, <= max_seq): GLM53_PF_SPARSE_T
        self.sparse_ts = tuple(int(v) for v in os.environ.get("GLM53_PF_SPARSE_T", "").split(",") if v.strip())
        # GLM53_PF_ROW_ALIGN (cf3h-k2 fix (a), coordinator 13:38:01Z): the cache row is aligned to the largest bucket
        # (generator.row_align); the TOP window is captured at that row in place of max_model_len, so a non-aligned
        # continuation chunk ending past max_model_len (up to a bucket - 1 tokens) still has a key
        # (14's review F1) only a top window within one 2,048 bucket of the row moves: a partial ladder (e.g. a 65,536
        # top at a 1M row) keeps its key; the unaligned row floors to the old top (no swap without the alignment)
        if self.sparse_ts:
            top_, row_ = max(self.sparse_ts), self.max_seq // 2048 * 2048
            if top_ < row_ <= top_ + 2048:
                self.sparse_ts = tuple(sorted((set(self.sparse_ts) - {top_}) | {row_}))
        self.sparse: dict = {}
        self._sparse_now = None
        # P3: the sparse_dsa attention of a chunk at position 0 in the non-absorbed (MHA) form (GLM53_PF_MLA_FORM=mha):
        # its own trace key (bucket, FRESH) for every bucket >= GLM53_PF_MHA_MIN; chunks with context stay absorbed
        self.mla_form = os.environ.get("GLM53_PF_MLA_FORM", "absorbed")
        # P3: sparse chunks' attention = "mask" (one flash SDPA over all T keys + the selection as an additive [C, T]
        # mask: O(C T)) or "sdpa" (ttnn.transformer.sparse_sdpa over each row's 2,051 selected ids: O(C) in T)
        # "gather" (DEFAULT since 2026-10-22, coordinator 14:48Z: Stage 9's int27 OOMed on mask mode's [C, T] tensors at
        # 32K) = sdpa with the key store gathered per layer from the paged cache (SparseCtx.latents_rm): no [C, T] tensor
        # and no per-layer max_seq store; dense chunks (<= 2,051) never reach this
        self.sparse_attn = os.environ.get("GLM53_PF_SPARSE_ATTN", "gather")
        assert self.sparse_attn in ("mask", "sdpa", "gather"), self.sparse_attn
        if self.shard is not None:
            assert self.sparse_attn == "gather", "GLM53_DSA_SHARD: GLM53_PF_SPARSE_ATTN=gather (the owner-aware reads)"
            # GLM53_DSA_BQ=1 (lever B-q, served_1m_memory_plan.md section 3B): the sparse attention is SPLIT-K across the
            # owner chips -- every chip attends only the selected keys it OWNS (expand_rows SHARD mode: local row ids) with
            # the q of ALL heads (all-gathered over the mesh), ttnn.transformer.sparse_sdpa_stats emits the partial's
            # (m, l) per (row, head), the owner of each head merges the C partials (M = max m_c, w_c = exp((m_c - M)
            # scale) l_c / sum, o = sum w_c o_c, fp32) through a stats all-gather + a head-dim reduce-scatter. The
            # selection is unchanged (the pooled keys still gather in global order: identical selections); only the
            # latent reads become local (no latent all-gather: 1 GiB per layer-chunk at 1M -> ~340 MiB of q + partials)
            if self.dsa and 2048 <= self.max_seq:  # a continuation chunk below the sparse bound (a prefix hit at the
                # 128 grid) reads its cached context through the sparse / gather path: its window must be captured
                self.sparse_ts = tuple(sorted(set(self.sparse_ts) | {2048}))
        # gather / sdpa with zero-padded heads (TP8): query rows per sparse_sdpa call (bounds the 4x padded copies)
        self.sparse_rows = int(os.environ.get("GLM53_PF_SPARSE_ROWS", "512"))
        # sparse_sdpa takes a token's heads as one tile: "pad" = the chip's heads zero-padded to 32 over all C rows
        # (TP8: 8 -> 32, 4x the attention work); "a2a" = heads -> sequence (two all_to_all_async_generic, mesh axis 0
        # then 1: every head of the chip's C / N rows, the deepseek_v3_d_p _sparse_mla reshard) and back
        self.sparse_reshard = os.environ.get("GLM53_PF_SPARSE_RESHARD", "pad")
        self.bq = self.shard is not None and os.environ.get("GLM53_DSA_BQ", "0") == "1"
        # GLM53_PMK_SPA=1 (Stage 16 SP-row attention phase, doc pmk/spaphase; coordinator 2026-10-25 09:20:12Z): on the
        # token-sharded cache every chip attends ITS OWN rows with all heads -- q all-to-all heads -> rows, the device-side
        # union of its rows' selected pools gathered from their owners over the fabric into a compact buffer
        # (tt/spa_union.py: bitmaps + exchange + owner push + id remap in one program), sparse_sdpa on the compact buffer,
        # o all-to-all back. Replaces B-q (no stats / owner merge / fp32 reduce-scatter); the selection keeps the a2a
        # reshard form (each chip's own rows' token ids). Default off.
        self.spa = os.environ.get("GLM53_PMK_SPA", "0") == "1"
        self._spa_cmp = None
        self._arena, self._arena_ids, self._spa_relay = {}, set(), None
        if self.spa:
            assert (
                self.shard is not None
            ), "GLM53_PMK_SPA=1: the SP-row phase reads the token-sharded cache (GLM53_DSA_SHARD=1)"
            assert self.ndev > 1, "GLM53_PMK_SPA=1: a TP mesh"
            self.bq = False
            self.sparse_reshard = "a2a"
        if self.bq:
            assert self.sparse_reshard != "a2a", "GLM53_DSA_BQ: every chip attends all rows (no a2a reshard)"
        # GLM53_PF_NOHEAD=1: a head-less trace per key as well, replayed for a request's NON-final chunks (the
        # generator's logits_rows="none"): no last row / head -> the decode rows stay untouched, so the backend skips
        # the per-chunk reads / waits / restores and the host prepares chunk k + 1 while chunk k runs
        self.nohead = os.environ.get("GLM53_PF_NOHEAD", "0") == "1"
        # GLM53_PF_NOHEAD_KEYS="2048:65536,..." (Stage 9 cf3g, coordinator 2026-10-23T19:07:21Z (3)): twins for the listed
        # (bucket, window) keys only -- the 1M line's trace region holds one ~10 MiB/bank twin, not one per key, so the
        # prompts <= 64K get the head-less chunk overlap and the longer windows keep head chunks. Unset: every key.
        nk = os.environ.get("GLM53_PF_NOHEAD_KEYS", "").strip()
        self.nohead_keys = {tuple(int(x) for x in v.split(":")) for v in nk.split(",") if v.strip()} if nk else None
        self.traces_nh: dict = {}
        self.fresh_min = int(os.environ.get("GLM53_PF_MHA_MIN", "1024"))
        # GLM53_PF_BATCH=nb (> 1): the batched admission trace (nb fresh requests of <= s tokens in one replay, s = the
        # largest bucket <= 2,048: dense attention), compiled / warmed / captured with the single-request keys
        # (capped at the backend's rows: the served env's PF_BATCH=4 on a B=1 backend would batch 4 slots of 1 row)
        nb_ = min(int(os.environ.get("GLM53_PF_BATCH", "0") or 0), self.max_batch)
        small = [c for c in self.buckets if c <= DENSE_EXACT]
        self.batch_keys = [(max(small), nb_)] if nb_ > 1 and small else []
        # GLM53_PF_PACKED_DEVMASK=1: the packed SDPA mask built on device from the per-row segment starts (256 KB of
        # host writes per admission instead of the 8-MB [T, T] mask; work_log 2026-10-22 06:01Z: host-bound packed wall)
        self.packed_devmask = os.environ.get("GLM53_PF_PACKED_DEVMASK", "0") == "1"
        # GLM53_PF_MOE_PADMASK=1 (single-request chunks): the bucket's padding rows leave the routed MoE -- their pairs
        # get the sentinel expert id E (local to no chip: the plans keep a pair iff id - lo < E_loc, so nothing is
        # dispatched) and weight 0 (the combines add 0 x a written row). 14b 06:09Z: at TP8 the first token is wrong
        # whenever the bucket is padded, right at full buckets; TP2 clean (coordinator 06:10:41Z, the fix candidate).
        self.moe_padmask = os.environ.get("GLM53_PF_MOE_PADMASK", "0") == "1"
        self._pad_in, self._pad_now = {}, None
        # GLM53_PF_REDUCE_FP32=1: the cross-chip sums of the attention o_proj / MoE partials (bf16 [1, 1, C, 4096] per
        # chip) run in fp32 -- typecast before the reduce-scatter / all-reduce, one bf16 rounding after it (TP8 sums 8
        # partials over two mesh axes in bf16 by default: up to 7 roundings of partial sums vs 1 at TP2; coordinator
        # 2026-10-22 07:23Z, the TP8-only numerics loss); =2: the fp32 sum is kept (the mHC post reads fp32 y; TP2 d0
        # 07:37Z: the fp32 -> bf16 cast after the sum rounded worse than the CCL's own bf16 add)
        # DEFAULT =2 at TP8 (coordinator 2026-10-22 15:50Z, coadmission_leak_analysis.md: the bf16 cross-chip partial sums
        # make a row's bits depend on its owner chip / bucket geometry; fp32 partials cost < 1 ms); TP2 unchanged (=0)
        self.reduce_fp32 = int(os.environ.get("GLM53_PF_REDUCE_FP32", "2" if self.ndev >= 8 else "0") or 0)
        # GLM53_PF_REDUCE_ROWS (default 2,048; 0 = off): a sequence-parallel reduce of more rows runs in row groups inside
        # the replay -- the reduce-scatter's Linear intermediate is 2x its input (fp32 [8192, 4096] -> 256 MiB at the 8K
        # bucket: Stage 9's 64K OOM, coordinator 19:39Z); with groups every bucket's peak is the 2,048 bucket's
        self.reduce_rows = int(os.environ.get("GLM53_PF_REDUCE_ROWS", "2048") or 0)
        # GLM53_PF_SLICE_RULE (default 1; 0 = off, the A/B control): Stage 16's slice-boundary rule (coordinator
        # 2026-10-22T23:13:59Z) -- a sequence-parallel chunk whose last valid row is the FIRST row of a chip's slice
        # ((L - 1) mod (C / N) == 0) gave a wrong first token at TP8 (golden[:97] in the 256 bucket): plan() moves such
        # a chunk to a larger bucket, slice_plan() tells the generator when to split it (tt/generator.py slice_plan)
        self.slice_rule = os.environ.get("GLM53_PF_SLICE_RULE", "1") == "1"
        # GLM53_PF_PROBE_LAYERS="0,1,4,43" (diagnosis): in-program probes -- inside the single-request traces each listed
        # layer's output residual [1, 1, C(_local), 16384] fp32 is copied into a persistent tensor (probe(c, L) reads it)
        self.probe_layers = {int(v) for v in os.environ.get("GLM53_PF_PROBE_LAYERS", "").split(",") if v.strip()}
        self._probe: dict = {}
        # PREFILL MEGAKERNEL integration (doc/ttnn/16-prefill-production/prefill_megakernel.md): {layer id: program}, a
        # callable (x, *, tpp, pl, ci, sp) -> the layer output residual [1, 1, C(_local), 16384] fp32 that runs the WHOLE
        # layer as one persistent program inside this bucket's trace (the op-graph layer otherwise; set per layer kind
        # as the PMK pieces land; the residual / KDA scratch / caches are the same tensors either way)
        self.pmk_layers: dict = {}
        # GLM53_PF_KDA_SCAN=pmk (Stage 16 PMK, 14:1xZ): the KDA core of every KDA layer (conv / gates + chunk prepare + scan
        # + gated norm = the kda piece's P4 / P6 bodies) as the PMK family program (tt/pmk_sparse_kda.kda_bodies), the
        # projections and o_proj still op-graph: {(layer index, bucket): (program, phase)} + ONE shared KDA store per
        # bucket (the programs bake its addresses; layers run in order)
        self.pmk_kda: dict = {}
        self.pmk_kda_store: dict = {}
        # PERSISTENT residuals (tt/pmk_stack.py: the PMK chain runs in place on a per-bucket tensor whose address the
        # programs bake in): ids of tensors layer() / forward() must never deallocate (keep_input is forced for them)
        self.persistent: set = set()
        # GLM53_PF_PACKED="T:KMAX" (e.g. 2048:4): the PACKED keys (T, K) for K = 2 .. KMAX -- K fresh requests of different
        # lengths back to back in one T-row replay (forward_packed) -- compiled / warmed / captured with the others
        # several packed buckets, comma-separated ("512:4,2048:4"): the generator packs each group into the smallest T that
        # holds it (one replay per total-token bucket)
        pk_ = os.environ.get("GLM53_PF_PACKED", "")
        self.packed_keys = []
        for spec_ in [v for v in pk_.split(",") if v.strip()]:
            t_, kmax_ = (int(v) for v in spec_.split(":"))
            assert t_ in self.buckets and t_ % TILE == 0, f"GLM53_PF_PACKED {pk_}: T must be a bucket"
            self.packed_keys += [(t_, k_) for k_ in range(2, kmax_ + 1)]
        self._fresh_now = False
        self._warmed = False  # warm() ran (every key's eager forward before any capture)
        self.layers = [PrefillLayer(self, e, layer_kind(self.cfg, e["L"])) for e in backend.seq]
        if self.kda_snapshot:  # the KDA export's view of the cut scratches (StackBackend._kda_state_src "snapshot"):
            # per KDA layer in the backend's _kda_layers order (S [1, H, 128, 128] pages from 0; conv carry in row 0)
            self.kv.extra["kda_snap_stage"] = [
                (pl.cut_scratch.S, 0, pl.cut_scratch.C, 0) for li, pl in enumerate(self.layers)
                if pl.kind != SPARSE_DSA and self.kv.states[0][li] is not None]  # fmt: skip
            self.kv.extra["kda_snap_at"] = None
            # the export at the cut goes through the KDA pack staging (+ the DSA rows): allocated here, before any
            # capture, so the snapshot never depends on a consumer having asked for it (the adapter asks with a
            # connector)
            ensure = getattr(backend, "ensure_state_export", None)
            if callable(ensure) and not ensure(self.kv):
                raise RuntimeError("GLM53_PF_KDA_SNAPSHOT=1 needs the KDA pack staging (GLM53_KDA_EXPORT_STAGE_LAYERS)")
        # packed segments start on 32-row tile rows; with the DSA indexer on 128-row boundaries (a pooled-key store page
        # = 32 pools = 128 tokens: one tile row per page, so tiles_to moves whole pages and no pool straddles segments)
        self.packed_align = KPOOL * POOL_PAGE if self.has_dsa() else TILE
        # GLM53_PF_PACKED_CONT=1: every packed key also gets a CONTINUATION-capable twin ("packedc", T, K): segments may
        # continue a slot's cached prefix (chunk start > 0: the MLA attends a prefix key buffer of
        # GLM53_PF_PACKED_PREFIX rows gathered from the slots' latent pages + the packed rows; the KDA starts from the
        # slot's state) and may be head-less (intermediate chunks: state only, no sample)
        self.packed_cont = os.environ.get("GLM53_PF_PACKED_CONT", "0") == "1"
        self.packed_prefix = int(os.environ.get("GLM53_PF_PACKED_PREFIX", "2048"))
        assert self.packed_prefix % BLOCK == 0, self.packed_prefix
        if self.shard is not None and any(pl.kind == SPARSE_DSA for pl in self.layers) and not self.dsa:
            raise ValueError(
                "GLM53_DSA_SHARD=1 with the TP prefill needs GLM53_PF_DSA=1 (the owner-only pooled-key writes)"
            )
        # a continuation chunk's start must be a multiple of every chunked-SDPA q chunk (the op divides the start tensor
        # by it on device: a 64-aligned start under q 128 (GLM53_PF_SDPA=128,...) would silently shift the causal mask)
        self.start_align = max([BLOCK] + [q for pl in self.layers if getattr(pl, "mla", None) is not None
                                          for q in (pl.mla.sdpa_prog.q_chunk_size, pl.mla.attn.prefill_program.q_chunk_size)])  # fmt: skip
        # the slot's KDA state <-> the scratch inside the trace (GLM53_PF_STATE_IO=host: the earlier SlotOps path)
        self.state_io = os.environ.get("GLM53_PF_STATE_IO", "trace")
        self.sio = (
            StateIO(self) if self.state_io == "trace" and any(pl.kind != SPARSE_DSA for pl in self.layers) else None
        )
        self.inputs: dict = {}
        self._cs_host: dict = {}  # bucket c -> the chunk start last written into its ChunkInputs (_check_window)
        self.traces: dict = {}
        self.counters = dict(replays=0, captures=0, eager=0, slot_ops=0)
        if (
            os.environ.get("GLM53_PF_SYNC_EVERY_LAUNCH", "0") == "1"
        ):  # the compile-pass launch fence (install_launch_fence)
            self.counters["launch_fence"] = install_launch_fence(self.mesh)
        # GLM53_PF_LAZY_KEYS (Stage 12, start <= 120 s): comma list of key classes whose trace is captured on FIRST USE
        # instead of at start -- bucket sizes ("8192": every key of that bucket), "sparse" (every t > 0 key), "batch",
        # "packed", "nohead" (the head-less twins). Every key is still compiled + warmed at start (programs precede the
        # first capture); the first request needing a deferred key pays its capture once. "" (default) = all at start.
        self.lazy = {v.strip() for v in os.environ.get("GLM53_PF_LAZY_KEYS", "").split(",") if v.strip()}
        self.deferred = set()
        # GLM53_MOCK_CAPTURE=1 (the image mock): the keys capture() would have captured; a replay of any other key
        # raises as on the device (coordinator 2026-10-23T15:50:53Z: a skipped capture must show on the mock)
        self.mock_captured = set()
        self.cost = []  # GLM53_PF_COST: one row per (phase, key) of compile / warm / capture: seconds, programs added

    # ------------------------------------------------------------------ construction helpers
    def arena(self, kind: str, key, make):
        """GLM53_PF_ARENA (default 1; 0 = a buffer per key, the A/B): ONE tensor per (kind, key) for every SparseCtx of
        this TPPrefill -- the per-key buffers that are per-replay scratch (written before they are read inside one replay)
        or constants of their shape (iotas, gather tables, row bases). Keys and W1 groups replay strictly in sequence on
        one CQ, so the traces' baked addresses may coincide. cf3h-j's DRAM census (14b, 2026-10-25): the per-key
        SparseCtx buffers grow ~linearly with the window (~118 MiB/bank over 9's 1M key set) and were pinned once per
        key. Allocated at the first key's build (before any capture)."""
        if os.environ.get("GLM53_PF_ARENA", "1") != "1":
            return make()
        k_ = (kind, key)
        if k_ not in self._arena:
            t_ = make()
            self._arena[k_] = t_
            self._arena_ids.add(id(t_))
        return self._arena[k_]

    def in_arena(self, t_) -> bool:
        return id(t_) in self._arena_ids

    def spa_relay(self, rows: int, t: int):
        """GLM53_PMK_SPA on the 2x4 (GLM53_PF_ARENA): the ONE relay-row store every key's SpaUnion shares, sized at the
        largest key's rcap = min(t / 128 x 32 / chips, line x rows x 513) over max_seq and the largest bucket's rows (the
        kernels bound it by each key's own rcap; it is written before it is read in every launch); None = a store per key
        """
        two = int(self.mesh.shape[0]) > 1 and int(self.mesh.shape[1]) > 1
        if not two or os.environ.get("GLM53_PF_ARENA", "1") != "1":
            return None
        from models.autoports.zai_org_glm_5_3_flash.tt.spa_union import UNIT  # noqa: PLC0415

        n, nall = int(self.mesh.shape[1]), int(self.mesh.shape[0]) * int(self.mesh.shape[1])
        rcap = lambda w_, r_: min((w_ // UNIT) * 32 // nall, n * r_ * 513)  # noqa: E731  (SpaUnion's default rcap)
        if self._spa_relay is None:
            rows_max = max(int(c_) for c_ in self.buckets) // self.ndev
            t_max = -(-self.max_seq // UNIT) * UNIT
            self._spa_relay = ttnn.zeros((1, 1, 4 * rcap(t_max, rows_max), 512), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=self.mesh, memory_config=DRAM)  # fmt: skip
            self._arena_ids.add(id(self._spa_relay))
        assert 4 * rcap(t, rows) <= int(self._spa_relay.shape[-2]), ("the shared relay store is too small", rows, t)
        return self._spa_relay

    def spa_syncs(self) -> tuple:
        """GLM53_PMK_SPA: the SpaUnion lane sync words (32 per lane, the only persistent L1 of the phase), ONE pair for every
        key: every chip runs the keys' launches in the same order, so the shared GEN / arrival counters stay in step"""
        if getattr(self, "_spa_syncs", None) is None:
            from models.autoports.zai_org_glm_5_3_flash.tt.spa_union import SpaUnion  # noqa: PLC0415

            two = int(self.mesh.shape[0]) > 1 and int(self.mesh.shape[1]) > 1
            self._spa_syncs = SpaUnion.make_syncs(self.mesh, two)
        return self._spa_syncs

    def spa_store(self, cap: int):
        """GLM53_PMK_SPA: the ONE compact key store every SparseCtx key shares (allocated once, before any capture, at
        GLM53_PMK_SPA_CAP pools, else the worst case of the largest bucket: min(max_seq / 4, rows per chip x 513))"""
        if self._spa_cmp is None:
            cap_all = int(os.environ.get("GLM53_PMK_SPA_CAP", "") or 0) or min(
                self.max_seq // KPOOL, (max(self.buckets) // self.ndev) * (TOPK_POOLS + 1))  # fmt: skip
            self._spa_cmp = ttnn.zeros((1, 1, 4 * cap_all, 512), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                       memory_config=DRAM)  # fmt: skip
        assert 4 * cap <= int(self._spa_cmp.shape[-2]), ("GLM53_PMK_SPA: the shared compact store is smaller than a key's cap",
                                                         cap, tuple(self._spa_cmp.shape))  # fmt: skip
        return self._spa_cmp

    def mode(self, c: int) -> str:
        # GLM53_PF_REP_BUCKETS (diagnosis): buckets forced replicated, e.g. the pair emulating TP8's 128-row bucket
        rep = {int(v) for v in os.environ.get("GLM53_PF_REP_BUCKETS", "").split(",") if v.strip()}
        return "sp" if self.ndev > 1 and c % (TILE * self.ndev) == 0 and c not in rep else "rep"

    def kda_scratch(self, kda):
        """per-layer slot-agnostic scratch state: S [1, H_r, 128, 128] fp32, C [1, 3, 1, 3 H_r 128] bf16, slot 0"""
        st = kda.allocate_state(1)
        ttnn.copy(ttnn.ones(tuple(st.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh),
                  st.slot_mask)  # fmt: skip
        return st

    def tp_shared(self, L: int, moe):
        """TP shard of layer L's shared expert (as Stage 14's GLM53_TP_SHARED): gate|up column slice / down row slice."""
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedSwiGLU
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
            OptimizedDecoderConfig,
            OptimizedSwiGLU,
            _upgrade,
        )
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8 import TPShards

        if self.ndev == 1:
            return moe.shared
        sh = TPShards(self.mesh, self.source, L, self.cfg)
        st = sh.shared_tensors()
        sh.drop()
        shared = FusedSwiGLU(self.mesh, st["gate"], st["up"], st["down"], limit=moe.shared.limit)
        ocfg = OptimizedDecoderConfig()
        shared.fcfg = getattr(moe.shared, "fcfg", None) or ocfg
        _upgrade(shared, OptimizedSwiGLU, ocfg, "shared")
        return shared

    def ep_bounds(self):
        """per-chip [lo, hi) of the chip's routed experts as fp32 [1, 1, 1, 1] mesh tensors (contiguous EP)"""
        if self._ep is None:
            e_all = int(self.cfg.n_routed_experts)
            e_loc = e_all // self.ndev
            mk = lambda vals: ttnn.from_torch(  # noqa: E731
                torch.tensor(vals, dtype=torch.float32).reshape(self.ndev, 1, 1, 1), dtype=FP32,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0) if self.ndev > 1 else None)  # fmt: skip
            self._ep = (mk([e_loc * r - 0.5 for r in range(self.ndev)]),
                        mk([e_loc * (r + 1) - 0.5 for r in range(self.ndev)]), e_loc)  # fmt: skip
        return self._ep

    def chunk_inputs(self, c: int) -> ChunkInputs:
        if c not in self.inputs:
            self.inputs[c] = self._new_chunk_inputs(c)
        self.pad_inputs(c, c)
        if self.probe_layers and not any(k[0] == c for k in self._probe):
            assert self.mhc_mode == "comp", "GLM53_PF_PROBE_LAYERS: the flat [1, 1, C, 16384] residual only"
            sp = self.mode(c) == "sp"
            mp = ttnn.ShardTensorToMesh(self.mesh, dim=2) if (sp and self.ndev > 1) else self.repl
            for pl in self.layers:
                if pl.L in self.probe_layers:
                    self._probe[(c, pl.L)] = ttnn.from_torch(torch.zeros(1, 1, c, NSTREAM * HIDDEN), dtype=FP32,
                                                             layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                                             memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
        return self.inputs[c]

    def probe(self, c: int, L: int) -> torch.Tensor:
        """GLM53_PF_PROBE_LAYERS: layer L's output residual from the last bucket-c forward / replay -> [c, NSTREAM,
        HIDDEN] fp32 on the host, every row in order (sp: the chips' row shards concatenated)"""
        t = self._probe[(c, L)]
        if self.ndev == 1:
            v = ttnn.to_torch(t)
        elif self.mode(c) == "sp":
            v = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=2))
        else:
            v = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        return v.float().reshape(c, NSTREAM, HIDDEN)

    def pad_inputs(self, key, rows: int):
        """GLM53_PF_MOE_PADMASK: the MoE routing mask of a program of `rows` rows -- keep [1, 1, rows, k] (1 on valid
        rows, 0 on padding rows) and pad_ids (the sentinel expert E on padding rows: local on no chip, weight 0); key =
        the bucket c, ("b", s, nb) (batched) or ("p", t, k) (packed). None when the knob is off."""
        if not self.moe_padmask:
            return None
        if key not in self._pad_in:
            k = int(self.cfg.num_experts_per_tok)
            tl = lambda v: ttnn.from_torch(v, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=self.mesh,  # noqa: E731
                                           memory_config=DRAM, mesh_mapper=self.repl)  # fmt: skip
            self._pad_in[key] = dict(key=str(key), rows=rows, keep=tl(torch.ones(1, 1, rows, k)),
                                     pad_ids=tl(torch.zeros(1, 1, rows, k)))  # fmt: skip
            self.counters.setdefault("padmask", {})[str(key)] = 0  # MoE layers masked per program build
        return self._pad_in[key]

    def write_pad(self, key, valid: torch.Tensor):
        """host side of pad_inputs: valid [rows] bool (the program's valid rows); no-op when the knob is off"""
        if not self.moe_padmask:
            return
        pin, k = self._pad_in[key], int(self.cfg.num_experts_per_tok)
        assert int(valid.numel()) == pin["rows"], (key, int(valid.numel()), pin["rows"])
        keep = valid.to(torch.float32).reshape(1, 1, -1, 1).expand(1, 1, pin["rows"], k).contiguous()
        pad_ids = (1.0 - keep) * float(self.cfg.n_routed_experts)
        for v, dst in ((keep, pin["keep"]), (pad_ids, pin["pad_ids"])):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=FP32, layout=ttnn.TILE_LAYOUT,
                                                            mesh_mapper=self.repl), dst)  # fmt: skip

    def seg_inputs(self, s: int, nb: int) -> list:
        """BATCHED prefill (P4 throughput): the per-request inputs of nb segments of bucket s in one chunk of nb x s
        rows -- segment 0 is chunk_inputs(s), segments 1.. get their own persistent tensors (one set per (s, b))"""
        extra = self.__dict__.setdefault("_seg_inputs", {})
        out = [self.chunk_inputs(s)]
        for b in range(1, nb):
            if (s, b) not in extra:
                extra[(s, b)] = self._new_chunk_inputs(s)
            out.append(extra[(s, b)])
        return out

    def _new_chunk_inputs(self, c: int) -> ChunkInputs:
        mesh, repl = self.mesh, self.repl
        sp = self.mode(c) == "sp"
        cl = c // self.ndev if sp else c
        shard = ttnn.ShardTensorToMesh(mesh, dim=-1) if self.ndev > 1 else None
        dshard = ttnn.ShardTensorToMesh(mesh, dim=2) if self.ndev > 1 else None
        mshard = ttnn.ShardTensorToMesh(mesh, dim=0) if self.ndev > 1 else None  # row r = chip r (rank order)
        rm = lambda v, dt, mp: ttnn.from_torch(v, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,  # noqa: E731
                                               memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
        ci = ChunkInputs(
            bucket=c,
            tokens=rm(torch.zeros(1, c, dtype=torch.int32), ttnn.uint32, shard if sp else repl),
            valid_end=rm(torch.tensor([c], dtype=torch.int32), ttnn.uint32, repl),
            valid_end_f=ttnn.from_torch(
                torch.full((1, 1, 1, 1), float(c)),
                dtype=FP32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=repl,
            ),  # fmt: skip
            chunk_table=rm(torch.zeros(1, c // BLOCK, dtype=torch.int32), ttnn.int32, repl),
            user_table=(
                rm(torch.zeros(self.ndev, self.bps_local, dtype=torch.int32), ttnn.int32, mshard)
                if self.shard is not None  # every chip: its LOCAL table of the slot
                else rm(torch.zeros(1, self.kv.blocks_per_seq, dtype=torch.int32), ttnn.int32, repl)
            ),
            chunk_start=rm(torch.zeros(1, dtype=torch.int32), ttnn.int32, repl),
            chunk_start_u=rm(torch.zeros(1, dtype=torch.int32), ttnn.uint32, repl),
            vis_cols_u=(
                rm(torch.zeros(self.ndev, 1, dtype=torch.int32), ttnn.uint32, mshard)
                if self.ndev > 1
                else rm(torch.zeros(1, dtype=torch.int32), ttnn.uint32, repl)
            ),  # fmt: skip
            pool_units_u=(
                rm(torch.zeros(self.ndev, 1, dtype=torch.int32), ttnn.uint32, mshard)
                if self.ndev > 1
                else rm(torch.zeros(1, dtype=torch.int32), ttnn.uint32, repl)
            ),  # fmt: skip
            snap_end_u=(rm(torch.full((1,), 32, dtype=torch.int32), ttnn.uint32, repl) if self.kda_snapshot else None),
            last_mask=ttnn.from_torch(
                torch.zeros(1, 1, cl * (self.ndev if sp else 1), 1),
                dtype=FP32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=dshard if sp else repl,
            ),  # fmt: skip
            chunk_start_f=ttnn.from_torch(
                torch.zeros(1, 1, 1, 1),
                dtype=FP32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=repl,
            ),  # fmt: skip
            pool_table=rm(torch.zeros(1, max(1, c // (KPOOL * POOL_PAGE)), dtype=torch.int32), ttnn.int32, repl),
            pool_user=rm(
                torch.zeros(
                    1,
                    (self.max_seq_local if self.shard is not None else self.max_seq) // (KPOOL * POOL_PAGE),
                    dtype=torch.int32,
                ),
                ttnn.int32,
                repl,
            ),  # fmt: skip
            ring_rows=rm(torch.full((1, TILE), c, dtype=torch.int32), ttnn.uint32, repl),
            ring_pages=rm(torch.arange(RING_TILES, dtype=torch.int32).reshape(1, -1), ttnn.int32, repl),
        )
        if self.shard is not None:  # per-chip scatter tables (SKIP everywhere until the host writes a chunk)
            skip = lambda n: torch.full((self.ndev, n), DSH.SKIP, dtype=torch.int64)  # noqa: E731
            ci.lat_idx = rm(skip(c // TILE * (512 // TILE)), ttnn.uint32, mshard)
            if dsa_dense_window():
                ci.win_idx = rm(
                    torch.full((1, c // TILE * (512 // TILE)), DSH.SKIP, dtype=torch.int64), ttnn.uint32, repl
                )
            ci.pool_idx = rm(skip(pool_idx_tiles(c, PAGED_POOLS)), ttnn.uint32, mshard)
        return ci

    # ------------------------------------------------------------------ collectives
    # 1-D fabric on a 2-D mesh: collectives run per mesh axis. Sequence rank r = row-major mesh index = R_cols * row +
    # col, so a gather runs along axis 1 (a row's columns, contiguous ranks) then axis 0; a reduce-scatter along axis 0
    # (halves by mesh row) then axis 1 (each row's half split over its columns).
    def _axes(self):
        r, c = int(self.mesh.shape[0]), int(self.mesh.shape[1])
        if r > 1 and c > 1:
            return [1, 0]
        return [None]

    def gather(self, x, sp: bool):
        if not sp:
            return x
        for ax in self._axes():
            y = ttnn.all_gather(x, dim=2, cluster_axis=ax, memory_config=DRAM)
            if ax != self._axes()[0]:
                ttnn.deallocate(x)
            x = y
        return x

    def gather_dim(self, x, dim: int):
        """all-gather along tensor dim `dim` over the mesh (axis 1 then 0: block index = rank, as gather())"""
        if self.ndev == 1:
            return x
        for ax in self._axes():
            y = ttnn.all_gather(x, dim=dim, cluster_axis=ax, memory_config=DRAM)
            if ax != self._axes()[0]:
                ttnn.deallocate(x)
            x = y
        return x

    def reduce_scatter_dim(self, x, dim: int):
        """reduce-scatter (sum) along tensor dim `dim` over the mesh (axis 0 then 1: chip rank r keeps block r of
        the summed tensor, as reduce() with sp); x is consumed"""
        if self.ndev == 1:
            return x
        for i, ax in enumerate(self._axes()[::-1]):
            y = ttnn.reduce_scatter(x, dim=dim, cluster_axis=ax, memory_config=DRAM)
            ttnn.deallocate(x)
            x = y
        return x

    def reduce(self, x, sp: bool):
        if self.ndev == 1:
            return x
        if sp and self.reduce_rows and int(x.shape[-2]) > self.reduce_rows:
            return self._reduce_groups(x, self.reduce_rows)
        return self._reduce(x, sp)

    def _reduce_groups(self, x, g: int):
        """reduce(x, sp=True) in row groups of <= g rows: group j stacks sub-block j of every chip's output block (chip p
        owns rows [p * blk, (p + 1) * blk) after the reduce-scatter), so each chip receives its own rows in order -- the
        same rows and the same per-element sums as one reduce; the caller's x is left alive (as in _reduce)"""
        n, rows, w = self.ndev, int(x.shape[-2]), int(x.shape[-1])
        blk = rows // n
        k = -(-rows // g)
        while k * TILE <= blk and blk % (k * TILE):
            k += 1
        s = blk // k
        assert blk * n == rows and s * k == blk and s % TILE == 0, ("reduce row groups", rows, n, g, k)
        ys = []
        for j in range(k):
            parts = [ttnn.slice(x, (0, 0, p * blk + j * s, 0), (1, 1, p * blk + (j + 1) * s, w)) for p in range(n)]
            xg = ttnn.concat(parts, dim=2)
            for t_ in parts:
                ttnn.deallocate(t_)
            ys.append(self._reduce(xg, True))
            ttnn.deallocate(xg)
        y = ttnn.concat(ys, dim=2)
        for t_ in ys:
            ttnn.deallocate(t_)
        return y

    def _reduce(self, x, sp: bool):
        axes = self._axes()[::-1]
        dt = x.dtype
        cast = self.reduce_fp32 > 0 and dt != FP32  # GLM53_PF_REDUCE_FP32: the partials summed in fp32
        if cast:
            x = ttnn.typecast(x, FP32)
        for i, ax in enumerate(axes):
            if sp:
                y = ttnn.reduce_scatter(x, dim=2, cluster_axis=ax, memory_config=DRAM)
            else:
                y = ttnn.all_reduce(x, cluster_axis=ax, memory_config=DRAM)
            if i or cast:
                ttnn.deallocate(x)
            x = y
        if cast and self.reduce_fp32 == 1:  # one rounding to the partials' dtype after the sum (=2: fp32 out)
            y = ttnn.typecast(x, dt)
            ttnn.deallocate(x)
            x = y
        return x

    # ------------------------------------------------------------------ the device forward (traced)
    def embed(self, ci: ChunkInputs, tokens=None):
        """tokens (default ci.tokens) -> X [1, 1, C_local, 16384] fp32 (the embedding in all 4 streams)"""
        tk = ci.tokens if tokens is None else tokens
        e = ttnn.embedding(tk, self.b.embed.table, layout=ttnn.TILE_LAYOUT, dtype=BF16)  # [1, C_l, 4096]
        e = ttnn.typecast(ttnn.reshape(e, (1, 1, int(e.shape[-2]), HIDDEN)), FP32)
        x = ttnn.concat([e] * NSTREAM, dim=1 if self.mhc_mode == "sm" else -1)
        ttnn.deallocate(e)
        return x

    def attention(self, pl: PrefillLayer, h, ci: ChunkInputs, li: int, h_local=None, sparse=None, scratch=None):
        if pl.kind == SPARSE_DSA:
            return pl.mla(h, self.kv.extra["latent"][li], ci, h_local, sparse, fresh=self._fresh_now,
                          window=self.kv.extra.get("dsa", {}).get(li, {}).get("window"))  # fmt: skip
        if self.kda_scan == "pmk" and scratch is None:
            # the PMK KDA scan writes no cut scratch: GLM53_PF_KDA_SNAPSHOT needs the op scan (review: reviews/
            # one_replay_store_A_scan_review.md finding 1)
            assert ci.snap_end_u is None, "GLM53_PF_KDA_SNAPSHOT with GLM53_PF_KDA_SCAN=pmk: no in-program cut state"
            return self._kda_pmk(pl, h, ci)
        snap = None
        if scratch is None and getattr(pl, "cut_scratch", None) is not None and ci.snap_end_u is not None:
            snap = (ci.snap_end_u, pl.cut_scratch)  # GLM53_PF_KDA_SNAPSHOT: the state at the cut, in the same replay
        return kda_prefill_tp(pl.kda, h, pl.scratch if scratch is None else scratch, ci.valid_end, ci.valid_end_f, scan=self.kda_scan,
                              row_ids=self.row_ids(int(h.shape[-2])), snap=snap)  # fmt: skip

    def _kda_pmk(self, pl, h, ci):
        """the KDA attention with its core as the PMK program (GLM53_PF_KDA_SCAN=pmk): the projections (qkv, low-rank,
        f_b / g_b ups) op-graph into the persistent KDA store (the H3 store: what the P3 / P4 stream phases will
        write), the kda piece's items + scan as the family layer program (conv / gates / prepare / scan / gated norm,
        S / C written into the layer's own state), then o_proj op-graph. Same numerics as pmk_int_kda_unit (on 0.99998,
        S 0.99997, C bitwise vs the op path)."""
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_sparse_kda as INT  # noqa: PLC0415

        kda, state = pl.kda, pl.scratch
        c = int(h.shape[-2])
        li = self.layers.index(pl)
        key = (li, c)
        if key not in self.pmk_kda:
            sp = self.mode(c) == "sp"
            prog = PMK.PmkSparseKdaLayer(self, li, c, sp, probe=False)
            phase = INT.kda_phase(prog, kda, nprod=1, h3_preraised=True)
            if c not in self.pmk_kda_store:
                self.pmk_kda_store[c] = phase.scratch()
            scr = self.pmk_kda_store[c]
            INT.register_kda(prog, INT.kda_bodies(phase, scr, state.S, state.C, ci.valid_end))
            self.pmk_kda[key] = (prog, phase)
        prog, phase = self.pmk_kda[key]
        scr = self.pmk_kda_store[c]
        h_, d_, w_ = kda.h, kda.d, kda.width
        # the projections written STRAIGHT into the store (14:3xZ: the four copies cost ~0.5 ms per KDA layer at TP2)
        pf_mm(kda.ds_qkv, h, out=scr["u"])
        lr = pf_mm(kda.ds_lowrank, h, out=scr["lr"])
        f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, c, d_))
        g_a = ttnn.slice(lr, (0, 0, 0, d_), (1, 1, c, 2 * d_))
        gate_out = kda._pf("ds_gb", g_a, kda.w_gb, kda.ckc_fp32, BF16)
        ttnn.linear(f_a, kda.w_fb, compute_kernel_config=kda.ckc_fp32, memory_config=DRAM, dtype=FP32,
                    optional_output_tensor=scr["f"])  # fmt: skip
        gt = ttnn.typecast(gate_out, scr["gate_out"].dtype)
        ttnn.copy(gt, scr["gate_out"])
        for t_ in (f_a, g_a, gate_out, gt):
            ttnn.deallocate(t_)
        prog(h, ci)  # the PMK KDA program: on -> scr["on"], S / C -> the layer's state (slot 0)
        return pf_mm(kda.ds_o, ttnn.reshape(scr["on"], (1, 1, c, w_)))

    def seg_scratch(self, pl: PrefillLayer, b: int):
        """segment b's KDA scratch of a KDA layer (batched prefill): b 0 = the layer's scratch, others allocated once"""
        if b == 0:
            return pl.scratch
        sc = pl.__dict__.setdefault("seg_scratches", {})
        if b not in sc:
            sc[b] = self.kda_scratch(pl.kda)
        return sc[b]

    def row_ids(self, c: int):
        """[1, 1, R >= c, 1] fp32 row positions 0 .. R - 1 (replicated), grown to the largest bucket seen (first use in
        an eager compile launch; a replaced table stays allocated: earlier traces read it)"""
        rid = getattr(self, "_row_ids", None)
        if rid is None or int(rid.shape[-2]) < c:
            if rid is not None:
                self.__dict__.setdefault("_row_ids_old", []).append(rid)
            n = max(c, 2048, *self.buckets)
            self._row_ids = ttnn.from_torch(torch.arange(n, dtype=torch.float32).reshape(1, 1, -1, 1), dtype=FP32,
                                            layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM,
                                            mesh_mapper=self.repl)  # fmt: skip
        return self._row_ids

    def moe_local(self, pl: PrefillLayer, hb, idx, scores):
        """moe_impl "sorted": routed experts of THIS chip for every routed row through the decoder's expert-sorted
        prefill with the other chips' pairs weighted 0 (every pair computed: the reference path; "ep" = EpMoe)"""
        lo, hi, _ = pl.ep
        moe = pl.moe
        ids_f = ttnn.typecast(idx, FP32)
        local = ttnn.multiply(ttnn.gt(ids_f, lo), ttnn.lt(ids_f, hi))  # [1, 1, C, k] 1.0 / 0.0
        return _routed_sorted_masked(moe, hb, idx, scores, local)

    def ffn(self, pl: PrefillLayer, h32_local, sp: bool):
        """h32_local [1, 1, C_l, 4096] fp32 (mhc_pre output) -> partial [1, 1, C, 4096] bf16"""
        if pl.kind == DENSE_KDA:
            hb = self.gather(pl.rms(h32_local, "ffn", BF16), sp)
            return swiglu_prefill(pl.mlp, hb)
        x32 = pl.rms(h32_local, "ffn", FP32)
        idx, scores = pl.moe.router(x32, decode=False)  # local rows (the router is row-independent)
        hb = self.gather(ttnn.typecast(x32, BF16), sp)
        if x32 is not h32_local:  # (fused7a: x32 IS the caller's h)
            ttnn.deallocate(x32)
        if self.moe_impl == "ep":
            # each pair's own (unbiased) score: ONE gather of [ids | selected scores] (fp32, 2k columns)
            k = int(idx.shape[-1])
            sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            pack = ttnn.concat([ttnn.typecast(idx, FP32), sel], dim=-1)
            ttnn.deallocate(scores)
            pack = self.gather(pack, sp)
            rows = int(pack.shape[-2])
            ids_f = ttnn.slice(pack, (0, 0, 0, 0), (1, 1, rows, k))
            sel = ttnn.slice(pack, (0, 0, 0, k), (1, 1, rows, 2 * k))
            ttnn.deallocate(pack)
            if self._pad_now is not None and rows == self._pad_now["rows"]:  # GLM53_PF_MOE_PADMASK
                pin = self._pad_now
                self.counters["padmask"][pin["key"]] += 1
                kept = ttnn.multiply(ids_f, pin["keep"])
                ttnn.deallocate(ids_f)
                ids_f = ttnn.add(kept, pin["pad_ids"])
                ttnn.deallocate(kept)
                sel_k = ttnn.multiply(sel, pin["keep"])
                ttnn.deallocate(sel)
                sel = sel_k
            idx = ttnn.typecast(ids_f, ttnn.uint16)
            ttnn.deallocate(ids_f)
            if getattr(pl, "epmoe", None) is None:
                pl.epmoe = EpMoe(self, pl.moe, xsl=getattr(pl, "xsl", None))
            routed = pl.epmoe(hb, idx, sel)
        else:
            if sp:
                idx = ttnn.typecast(self.gather(ttnn.typecast(idx, FP32), sp), ttnn.uint16)
                scores = self.gather(scores, sp)
            routed = self.moe_local(pl, hb, idx, scores)
        shared = swiglu_prefill(pl.shared, hb)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def layer(self, li: int, x, ci: ChunkInputs, sp: bool, keep_input: bool = False):
        if id(x) in self.persistent:  # a PMK chain's persistent residual: never freed by a layer
            keep_input = True
        pmk = self.pmk_layers.get(self.layers[li].L)
        if pmk is not None:  # the layer as ONE persistent program (PMK); an in-place program returns x itself
            out = pmk(x, tpp=self, pl=self.layers[li], ci=ci, sp=sp)
            if not keep_input and out is not x:
                ttnn.deallocate(x)
            return out
        if self.mhc_mode == "sm":
            return self._layer_sm(li, x, ci, sp, keep_input)
        pl = self.layers[li]
        if self.mhc_post == "fused":
            return self._layer_fused_mhc(pl, li, x, ci, sp, keep_input)
        h, post, comb, xs = pl.hc["attn"].pre_streams(x)
        if not keep_input:
            ttnn.deallocate(x)
        a = self._attn_sublayer(pl, h, ci, li, sp)
        s1 = pl.hc["attn"].post_streams(a, xs, post, comb)
        ttnn.deallocate(a)
        for t_ in xs:
            ttnn.deallocate(t_)
        x1 = ttnn.concat(s1, dim=-1)
        h, post, comb, _ = pl.hc["ffn"].pre_streams(x1, s1)
        ttnn.deallocate(x1)
        m = self.reduce(self.ffn(pl, h, sp), sp)
        ttnn.deallocate(h)
        s2 = pl.hc["ffn"].post_streams(m, s1, post, comb)
        ttnn.deallocate(m)
        for t_ in s1:
            ttnn.deallocate(t_)
        out = ttnn.concat(s2, dim=-1)
        for t_ in s2:
            ttnn.deallocate(t_)
        return out

    def _layer_fused_mhc(self, pl: PrefillLayer, li: int, x, ci: ChunkInputs, sp: bool, keep_input: bool):
        """layer() with the mHC mixes as programs on the flat residual (GLM53_PF_MHC_POST=fused): per sublayer the
        coefficients (fn projection, RMS scale, Sinkhorn), h = mhc_pre_prefill, the sublayer, x' = mhc_post_prefill --
        no stream slices, no 4 + 20 eltwise passes, no concat"""
        h, post, comb = self._mhc_pre(pl, "attn", x)
        a = self._attn_sublayer(pl, h, ci, li, sp)
        x1 = self._post(a, x, post, comb)
        for t_ in (a, post, comb):
            ttnn.deallocate(t_)
        if not keep_input:
            ttnn.deallocate(x)
        h, post, comb = self._mhc_pre(pl, "ffn", x1)
        m = self.reduce(self.ffn(pl, h, sp), sp)
        ttnn.deallocate(h)
        out = self._post(m, x1, post, comb)
        for t_ in (m, x1, post, comb):
            ttnn.deallocate(t_)
        return out

    def layer_batch(self, li: int, x, cis: list, sp: bool, keep_input: bool = False):
        """BATCHED prefill layer: x holds len(cis) segments of equal rows, one request each (fresh chunks: dense
        attention, chunk start 0). The row-wise work -- mHC, norms, gathers, the FFN / MoE (weights read ONCE for all
        segments), reduce-scatters -- runs over all rows; the attention runs per segment on its own inputs (latent /
        pooled-key pages, KDA scratch)."""
        assert self.mhc_mode == "comp" and self.mhc_post == "fused", "batched prefill: the fused-mHC layer only"
        assert self._sparse_now is None, "batched prefill: dense chunks only (<= 2,051 positions)"
        pl = self.layers[li]
        h, post, comb = self._mhc_pre(pl, "attn", x)
        a = self._attn_sublayer_batch(pl, h, cis, li, sp)
        x1 = self._post(a, x, post, comb)
        for t_ in (a, post, comb):
            ttnn.deallocate(t_)
        if not keep_input:
            ttnn.deallocate(x)
        h, post, comb = self._mhc_pre(pl, "ffn", x1)
        m = self.reduce(self.ffn(pl, h, sp), sp)
        ttnn.deallocate(h)
        out = self._post(m, x1, post, comb)
        for t_ in (m, x1, post, comb):
            ttnn.deallocate(t_)
        return out

    def _mhc_pre(self, pl: PrefillLayer, sub: str, x):
        """(h, post, comb) of sublayer `sub` on the flat residual: h = mhc_pre_prefill's fp32 output, or (fused7a) the
        normalised sublayer input (fp32 for a MoE router, else bf16), handed through the next pl.rms(h, sub)"""
        if pl.mpf is not None:
            hn, post, comb = pl.mpf[sub](x, hn_dtype=FP32 if (sub == "ffn" and pl.kind != DENSE_KDA) else BF16)
            pl._normed[sub] = hn
            return hn, post, comb
        pre, post, comb = mhc_coeffs(pl.hc[sub], x)
        h = mhc_pre_prefill(x, pre)
        ttnn.deallocate(pre)
        return h, post, comb

    def _attn_sublayer_batch(self, pl: PrefillLayer, h, cis: list, li: int, sp: bool):
        hn = pl.rms(h, "attn", BF16)
        if hn is not h:
            ttnn.deallocate(h)
        hb = self.gather(hn, sp)
        if hb is not hn:
            ttnn.deallocate(hn)
        rows, nb = int(hb.shape[-2]), len(cis)
        seg = rows // nb
        assert seg * nb == rows, (rows, nb)
        outs = []
        for b, ci in enumerate(cis):
            hb_b = ttnn.slice(hb, (0, 0, b * seg, 0), (1, 1, (b + 1) * seg, int(hb.shape[-1])))
            outs.append(self.attention(pl, hb_b, ci, li, scratch=self.seg_scratch(pl, b) if pl.kind != SPARSE_DSA
                                       else None))  # fmt: skip
            ttnn.deallocate(hb_b)
        ttnn.deallocate(hb)
        a = ttnn.concat(outs, dim=2) if nb > 1 else outs[0]
        if nb > 1:
            for t_ in outs:
                ttnn.deallocate(t_)
        return self.reduce(a, sp)

    def _attn_sublayer(self, pl: PrefillLayer, h, ci: ChunkInputs, li: int, sp: bool):
        """h (mhc_pre output, local rows) -> norm -> gather -> attention -> reduce(-scatter); the sparse_dsa
        selection also reads the local normed rows"""
        hn = pl.rms(h, "attn", BF16)
        if hn is not h:
            ttnn.deallocate(h)
        hb = self.gather(hn, sp)
        sparse = self._sparse_now if pl.kind == SPARSE_DSA else None
        if hb is not hn and sparse is None:
            ttnn.deallocate(hn)
        a = self.reduce(self.attention(pl, hb, ci, li, hn, sparse), sp)
        ttnn.deallocate(hb)
        if hb is not hn and sparse is not None:
            ttnn.deallocate(hn)
        return a

    def mhc_pre(self, pl: PrefillLayer, sub: str, x, xs=None):
        """(h [1, 1, T_l, 4096] fp32, ctx) of sublayer `sub` on the residual x (layer layout; comp: optional stream
        slices xs of x)"""
        if self.mhc_mode == "sm":
            h, post, comb = pl.hc[sub].pre(x)
            return h, (post, comb, x)
        h, post, comb, xs = pl.hc[sub].pre_streams(x, xs)
        return h, (post, comb, xs)

    def mhc_post_streams(self, pl: PrefillLayer, sub: str, y, ctx):
        """the new residual: sm -> [1, 4, T_l, 4096]; comp -> the 4 stream tensors (list)"""
        post, comb, xr = ctx
        if self.mhc_mode == "sm":
            return pl.hc[sub].post(y, xr, post, comb)
        return pl.hc[sub].post_streams(y, xr, post, comb)

    def _layer_sm(self, li: int, x, ci: ChunkInputs, sp: bool, keep_input: bool):
        """layer() on the stream-major residual x [1, 4, T_l, 4096] fp32"""
        pl = self.layers[li]
        h, post, comb = pl.hc["attn"].pre(x)
        a = self._attn_sublayer(pl, h, ci, li, sp)
        s1 = pl.hc["attn"].post(a, x, post, comb)
        ttnn.deallocate(a)
        if not keep_input:
            ttnn.deallocate(x)
        h, post, comb = pl.hc["ffn"].pre(s1)
        m = self.reduce(self.ffn(pl, h, sp), sp)
        ttnn.deallocate(h)
        s2 = pl.hc["ffn"].post(m, s1, post, comb)
        ttnn.deallocate(m)
        ttnn.deallocate(s1)
        return s2

    def streams_to_device(self, x: torch.Tensor, sp: bool):
        """host residual [T, 4, 4096] -> the layer layout on the mesh (sp: rows sharded)"""
        t_ = x.shape[0]
        xs = x.reshape(t_, NSTREAM, HIDDEN).float()
        host = (
            xs.permute(1, 0, 2).reshape(1, NSTREAM, t_, HIDDEN) if self.mhc_mode == "sm" else xs.reshape(1, 1, t_, -1)
        )
        mp = (ttnn.ShardTensorToMesh(self.mesh, dim=2) if sp else self.repl) if self.ndev > 1 else None
        return ttnn.from_torch(host.contiguous(), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=DRAM, mesh_mapper=mp)  # fmt: skip

    def streams_from_device(self, t, sp: bool) -> torch.Tensor:
        """the layer layout -> host [T, 4, 4096]"""
        parts = ttnn.get_device_tensors(t) if self.ndev > 1 else [t]
        hs = [ttnn.to_torch(p).float() for p in parts]
        y = torch.cat(hs, dim=2) if sp else hs[0]
        if self.mhc_mode == "sm":
            return y[0].permute(1, 0, 2).contiguous()
        return y.reshape(y.shape[-2], NSTREAM, HIDDEN)

    def last_row(self, x, ci: ChunkInputs, sp: bool):
        """the final residual row of the last valid token on every chip: [1, 1, 1, 16384] fp32 (masked sum of the local
        rows: exactly one chip holds a 1 in last_mask; an all-reduce sums the chips)"""
        sel = ttnn.where(ci.last_mask, x, 0.0)
        row = ttnn.sum(sel, dim=2, keepdim=True)  # [1, 1, 1, 16384] ("sm": [1, 4, 1, 4096])
        ttnn.deallocate(sel)
        if sp:
            row = self.reduce(row, False)
        if self.mhc_mode == "sm":
            flat = ttnn.reshape(ttnn.reshape(row, (1, 1, NSTREAM, HIDDEN)), (1, 1, 1, NSTREAM * HIDDEN))
            ttnn.deallocate(row)
            row = flat
        return row

    def sparse_ctx(self, c: int, t: int) -> "SparseCtx":
        if (c, t) not in self.sparse:
            self.sparse[(c, t)] = SparseCtx(self, c, t)
        return self.sparse[(c, t)]

    def forward(self, c: int, *, head: bool = True, t: int = 0):
        """the whole bucket on device: embed -> every layer -> the last row into the decode residuals -> MkHead. Writes
        only persistent tensors (row_buf, the decode residuals, the head's outputs, KDA scratch, latent caches, pooled
        indexer keys). t > 0: the sparse_dsa layers select (DSA top-512) over the slot's first t tokens."""
        ci = self.chunk_inputs(c)
        sp = self.mode(c) == "sp"
        self._sparse_now = self.sparse_ctx(c, t) if t > 0 else None
        self._fresh_now = t == FRESH
        self._pad_now = self.pad_inputs(c, c)
        if self.sio is not None:
            self.sio.state_in()  # the slot's KDA state (or zeros: fresh request) -> every layer's scratch
        if self._sparse_now is not None:
            self._sparse_now.begin(ci)
        x = self.embed(ci)
        for li in range(len(self.layers)):
            x = self.layer(li, x, ci, sp)
            pb = self._probe.get((c, self.layers[li].L))
            if pb is not None:  # GLM53_PF_PROBE_LAYERS: this layer's output residual, copied inside the trace
                ttnn.copy(x, pb)
        if self._sparse_now is not None:
            self._sparse_now.end()
            self._sparse_now = None
        self._fresh_now = False
        self._pad_now = None
        row = self.last_row(x, ci, sp) if head else None  # head=False: a non-final chunk (no last row, no head)
        if id(x) not in self.persistent:
            ttnn.deallocate(x)
        if self.sio is not None:
            self.sio.state_out()  # every layer's scratch -> the slot (overlaps the head below on other cores)
        if head:
            ttnn.copy(row, self.row_buf())
            stk = ttnn.reshape(row, (1, 1, NSTREAM, HIDDEN))
            for res in self.kv.res:  # every decode row of every chunk (the head reads the slot's; the others are
                # don't-care between steps: the next decode step's embed rewrites the residuals)
                if int(res.shape[-1]) != HIDDEN:  # native-t flat residual [1, 1, B, 16384]
                    src = ttnn.repeat(row, (1, 1, int(res.shape[-2]), 1))
                elif int(res.shape[0]) > 1:
                    src = ttnn.repeat(stk, (int(res.shape[0]), 1, 1, 1))
                else:
                    src = None
                ttnn.copy(stk if src is None else src, res)
                if src is not None:
                    ttnn.deallocate(src)
            self.b.head(self.kv.res, self.b.pos, pos_copies=self.b.pos_c)
            ttnn.deallocate(row)

    # ------------------------------------------------------------------ batched prefill (P4 throughput)
    def batch_inputs(self, s: int, nb: int) -> dict:
        """the batch's own persistent tensors: tokens [1, nb s] (sp: sharded), one last-row mask per segment
        [1, 1, nb s, 1] (one-hot on its owner chip), one StateIO per segment (its scratches <-> its slot)"""
        cache = self.__dict__.setdefault("_batch_in", {})
        if (s, nb) not in cache:
            mesh, repl, r = self.mesh, self.repl, nb * s
            sp = self.mode(r) == "sp"
            shard = ttnn.ShardTensorToMesh(mesh, dim=-1) if self.ndev > 1 else None
            dshard = ttnn.ShardTensorToMesh(mesh, dim=2) if self.ndev > 1 else None
            tokens = ttnn.from_torch(torch.zeros(1, r, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
                                     mesh_mapper=shard if sp else repl)  # fmt: skip
            masks = [ttnn.from_torch(torch.zeros(1, 1, r, 1), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     memory_config=DRAM, mesh_mapper=dshard if sp else repl) for _ in range(nb)]  # fmt: skip
            sios = [StateIO(self, seg=b) for b in range(nb)] if self.sio is not None else None
            # the head rows: segment b's last row -> decode row slot_b (rows_to by this table; staging [B, 16384] fp32)
            slot_idx = ttnn.from_torch(torch.arange(nb, dtype=torch.int32).reshape(1, nb), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
                                       mesh_mapper=repl)  # fmt: skip
            nrows = int(self.b.chunk) * len(self.kv.res)
            dec = ttnn.from_torch(torch.zeros(1, 1, nrows, NSTREAM * HIDDEN), dtype=FP32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
            cache[(s, nb)] = dict(tokens=tokens, masks=masks, sios=sios, sp=sp, slot_idx=slot_idx, dec=dec,
                                  pad=self.pad_inputs(("b", s, nb), r))  # fmt: skip
        return cache[(s, nb)]

    def write_batch(self, s: int, reqs: list):
        """host inputs of a batched replay: reqs = [(tokens, slot, n_valid)] fresh requests (chunk start 0, n_valid <=
        s, distinct slots); segment b's first token is sampled in decode row `slot`"""
        nb = len(reqs)
        cis, bi = self.seg_inputs(s, nb), self.batch_inputs(s, nb)
        r = nb * s
        put = lambda v, dt, dst, mp, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=mp), dst)  # fmt: skip
        tok = torch.zeros(1, r, dtype=torch.int32)
        sp = bi["sp"]
        slots = [int(slot) for _, slot, _ in reqs]
        nrows = int(self.b.chunk) * len(self.kv.res)
        assert len(set(slots)) == nb and all(0 <= v < nrows for v in slots), (slots, nrows)
        put(torch.tensor(slots, dtype=torch.int32).reshape(1, nb), ttnn.uint32, bi["slot_idx"], self.repl)
        valid = torch.zeros(r, dtype=torch.bool)
        for b, (tk, slot, nv) in enumerate(reqs):
            assert 0 < nv <= s, (b, slot, nv, s)
            valid[b * s : b * s + nv] = True
            self.write_inputs(s, tk, slot=slot, chunk_start=0, n_valid=nv, ci=cis[b])
            tok[0, b * s : b * s + nv] = tk.reshape(-1)[:nv].to(torch.int32)
            m = torch.zeros(1, 1, r, 1)
            m[0, 0, b * s + nv - 1, 0] = 1.0
            put(m, FP32, bi["masks"][b], ttnn.ShardTensorToMesh(self.mesh, dim=2) if sp else self.repl,
                ttnn.TILE_LAYOUT)  # fmt: skip
            if bi["sios"] is not None:
                bi["sios"][b].write(slot, fresh=True)
        put(tok, ttnn.uint32, bi["tokens"], ttnn.ShardTensorToMesh(self.mesh, dim=-1) if sp else self.repl)
        self.write_pad(("b", s, nb), valid)

    def forward_batch(self, s: int, nb: int, *, head: bool = True):
        """nb fresh requests of <= s tokens in ONE program of nb x s rows: every segment's KDA state in, the batch
        embedding, layer_batch per layer, the states out, each segment's last valid row into decode row b, the head"""
        cis, bi = self.seg_inputs(s, nb), self.batch_inputs(s, nb)
        sp = bi["sp"]
        self._sparse_now = None
        # the single-request path's choice for a fresh chunk (fresh_keys: the MHA form of the dense attention)
        self._fresh_now = bool(self.fresh_keys((s,)))
        self._pad_now = bi["pad"]
        if bi["sios"] is not None:
            for sio in bi["sios"]:
                sio.state_in()
        x = self.embed(cis[0], tokens=bi["tokens"])
        for li in range(len(self.layers)):
            x = self.layer_batch(li, x, cis, sp)
        self._fresh_now = False
        self._pad_now = None
        if bi["sios"] is not None:
            for sio in bi["sios"]:
                sio.state_out()
        if head:
            rows = []
            for m in bi["masks"]:  # segment b's last valid row (masked sum: exact fp32; all-reduced over the chips)
                sel = ttnn.where(m, x, 0.0)
                rw = ttnn.sum(sel, dim=2, keepdim=True)
                ttnn.deallocate(sel)
                rows.append(self.reduce(rw, False) if sp else rw)
            stk = ttnn.concat(rows, dim=2) if nb > 1 else rows[0]  # [1, 1, nb, 16384]
            for rw in rows if nb > 1 else []:
                ttnn.deallocate(rw)
            stk_rm = ttnn.to_layout(stk, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(stk)
            rows_to(stk_rm, bi["slot_idx"], bi["dec"])  # row b -> staging row slot_b (other rows: don't-care)
            ttnn.deallocate(stk_rm)
            dec = ttnn.to_layout(bi["dec"], ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 1, B, 16384]
            ch = int(self.b.chunk)
            for k, res in enumerate(self.kv.res):  # every decode row (the single path writes all of them too)
                part = ttnn.slice(dec, (0, 0, k * ch, 0), (1, 1, (k + 1) * ch, NSTREAM * HIDDEN))
                src = part if int(res.shape[-1]) != HIDDEN else ttnn.reshape(part, (ch, 1, NSTREAM, HIDDEN))
                ttnn.copy(src, res)
                for t_ in {id(v): v for v in (part, src)}.values():
                    ttnn.deallocate(t_)
            ttnn.deallocate(dec)
            self.b.head(self.kv.res, self.b.pos, pos_copies=self.b.pos_c)
        ttnn.deallocate(x)

    def capture_batch(self, s: int, nb: int, *, head: bool = True):
        """the batched trace of (s, nb) (its programs must have compiled -- forward_batch eagerly -- before ANY capture
        of the process: compile-after-capture keeps buffers alive in earlier traces)"""
        key = ("batch", s, nb)
        if os.environ.get("GLM53_MOCK") == "1" or key in self.traces:
            return
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.forward_batch(s, nb, head=head)
        except BaseException:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.release_trace(self.mesh, tid)
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[key] = tid
        self.counters["captures"] += 1
        ttnn.synchronize_device(self.mesh)
        gc.collect()

    def prefill_batch(self, s: int, reqs: list, *, head: bool = True, blocking: bool = False):
        """nb fresh requests [(tokens, slot, n_valid)] in ONE replay of the batched trace (a lazy key is captured at
        its first use; an uncaptured key raises); the caller owns cur_pos / the sampler rows (StackBackend.prefill_batch_tp)
        """
        nb = len(reqs)
        self.write_batch(s, reqs)
        key = ("batch", s, nb)
        if os.environ.get("GLM53_MOCK") == "1":  # the image mock runs no traces
            self._mock_check(key)
            self.forward_batch(s, nb, head=head)
            self.counters["eager"] += 1
            return
        if key not in self.traces and key in self.deferred:
            self._lazy_capture(key, head)
        if key not in self.traces:  # no eager fallback: the served path runs captured traces only
            raise RuntimeError(f"prefill batch key {key} was not captured (captured: {sorted(self.traces, key=str)})")
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=blocking)
        self.counters["replays"] += 1

    # ------------------------------------------------------------------ PACKED prefill (mixed ISL)
    # Coordinator 2026-10-22 03:15Z: vLLM's step is a flat token stream holding several requests of different lengths.
    # A packed key (T, K) = ONE trace of T rows holding K fresh requests back to back (segment k = rows [o_k, o_k + n_k),
    # o_k on 32-row boundaries); every boundary is a device tensor written per replay (write_packed), so one trace serves
    # any lengths summing (32-rounded) to <= T. Row-wise work (mHC, norms, gathers, matmuls, MoE, reduce-scatters) runs
    # over all T rows; MLA attends per segment through a block-diagonal causal mask, KDA runs its conv / scan per segment
    # (kda_prefill_packed), the latent rows go to each segment's slot pages (tiles_to), the KDA states to each slot
    # (StateIO per segment), and each segment's last row to its decode row for the head (rows_to). Phase 1: fresh
    # requests only (chunk start 0), no DSA indexer (the service knobs).
    PACKED_ALIGN = TILE

    def packed_inputs(self, t: int, k: int, cont: bool = False) -> dict:
        cache = self.__dict__.setdefault("_packed_in", {})
        if (t, k, cont) not in cache:
            mesh, repl = self.mesh, self.repl
            sp = self.mode(t) == "sp"
            shard = ttnn.ShardTensorToMesh(mesh, dim=-1) if self.ndev > 1 else None
            dshard = ttnn.ShardTensorToMesh(mesh, dim=2) if self.ndev > 1 else None
            rm = lambda v, dt, mp: ttnn.from_torch(v, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,  # noqa: E731
                                                   memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
            tl = lambda v, dt, mp: ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                   memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
            nrows = int(self.b.chunk) * len(self.kv.res)
            kp = self.packed_prefix if cont else 0  # prefix key rows (continuation twin)
            cache[(t, k, cont)] = dict(
                sp=sp,
                cont=cont,
                tokens=rm(torch.zeros(1, t, dtype=torch.int32), ttnn.uint32, shard if sp else repl),
                seg_masks=[tl(torch.zeros(1, 1, t, 1), FP32, repl) for _ in range(k)],
                seg_ends=[rm(torch.zeros(1, dtype=torch.int32), ttnn.uint32, repl) for _ in range(k)],
                attn_mask=tl(torch.zeros(1, 1, t, kp + t), BF16, repl),
                pre_table=rm(torch.zeros(1, max(1, kp // BLOCK), dtype=torch.int32), ttnn.int32, repl),
                carry_idx=[rm(torch.zeros(1, CONV_ROWS, dtype=torch.int32), ttnn.uint32, repl) for _ in range(k)]
                if cont
                else None,
                starts=tl(torch.zeros(1, 1, t, TILE), FP32, repl),  # GLM53_PF_PACKED_DEVMASK: row i's segment start
                lat_idx=(
                    rm(
                        torch.full((self.ndev, t // TILE * (512 // TILE)), DSH.SKIP, dtype=torch.int64),
                        ttnn.uint32,
                        ttnn.ShardTensorToMesh(mesh, dim=0),
                    )
                    if self.shard is not None  # per chip (its units)
                    else rm(torch.zeros(1, t // TILE * (512 // TILE), dtype=torch.int32), ttnn.uint32, repl)
                ),
                win_idx=(
                    rm(torch.full((1, t // TILE * (512 // TILE)), DSH.SKIP, dtype=torch.int64), ttnn.uint32, repl)
                    if self.shard is not None and dsa_dense_window()
                    else None
                ),
                masks=[tl(torch.zeros(1, 1, t, 1), FP32, dshard if sp else repl) for _ in range(k)],
                slot_idx=rm(torch.arange(k, dtype=torch.int32).reshape(1, k), ttnn.uint32, repl),
                dec=ttnn.from_torch(
                    torch.zeros(1, 1, nrows, NSTREAM * HIDDEN),
                    dtype=FP32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=repl,
                ),  # fmt: skip
                sios=[StateIO(self, seg=b) for b in range(k)] if self.sio is not None else None,
                pad=self.pad_inputs(("p", t, k), t),
            )
            if self.has_dsa():  # the indexer's per-segment inputs (DsaIndexer.keys_packed)
                cache[(t, k, cont)].update(
                    ring_rows=[rm(torch.full((1, TILE), t, dtype=torch.int32), ttnn.uint32, repl) for _ in range(k)],
                    ring_pages=[rm(torch.arange(RING_TILES, dtype=torch.int32).reshape(1, -1), ttnn.int32, repl)
                                for _ in range(k)],  # fmt: skip
                    pool_idx=(rm(torch.full((self.ndev, pool_idx_tiles(t, PAGED_POOLS)), DSH.SKIP,
                                            dtype=torch.int64), ttnn.uint32, ttnn.ShardTensorToMesh(mesh, dim=0))
                              if self.shard is not None
                              else rm(torch.zeros(1, t // (KPOOL * POOL_PAGE) * (INDEX_DIM // TILE), dtype=torch.int32),
                                      ttnn.uint32, repl)),  # fmt: skip
                )
        return cache[(t, k, cont)]

    def packed_layout(self, t: int, lens: list) -> list:
        """segment offsets o_k (32-row aligned, back to back) for lengths lens in a T-row packed chunk"""
        offs, o = [], 0
        for n in lens:
            offs.append(o)
            o += -(-int(n) // self.packed_align) * self.packed_align
        assert o <= t, f"packed: {lens} need {o} rows > {t}"
        return offs

    @staticmethod
    def packed_reqs(reqs: list) -> list:
        """[(tokens, slot, n_valid[, chunk_start[, head]])] -> 5-tuples (chunk start 0, head True by default)"""
        out = []
        for r in reqs:
            tk, slot, nv = r[0], int(r[1]), int(r[2])
            out.append((tk, slot, nv, int(r[3]) if len(r) > 3 else 0, bool(r[4]) if len(r) > 4 else True))
        return out

    def write_packed(self, t: int, reqs: list, cont: bool = False):
        """host inputs of a packed replay: reqs = [(tokens, slot, n_valid[, chunk_start[, head]])] (distinct slots).
        cont: the continuation twin's inputs (prefix pages + the [T, prefix + T] mask); a segment with chunk_start > 0
        needs it (its rows attend the slot's cached rows < chunk_start, its KDA starts from the slot's state)."""
        full = self.packed_reqs(reqs)
        reqs = [(tk, slot, nv) for tk, slot, nv, _, _ in full]
        starts = [st for *_, st, _ in full]
        assert cont or not any(starts), "continuation segments need the continuation twin (GLM53_PF_PACKED_CONT=1)"
        for _, _, nv, st, _ in full:
            assert st % max(BLOCK, self.start_align) == 0, f"segment start {st}: not a multiple of {self.start_align}"
            assert st + nv <= DENSE_EXACT or not self.has_dsa(), f"segment [{st}, {st + nv}) past {DENSE_EXACT}: sparse"
        k = len(reqs)
        pk = self.packed_inputs(t, k, cont)
        sp = pk["sp"]
        offs = self.packed_layout(t, [nv for _, _, nv in reqs])
        put = lambda v, dt, dst, mp, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=mp), dst)  # fmt: skip
        dshard = ttnn.ShardTensorToMesh(self.mesh, dim=2) if self.ndev > 1 else None
        tok = torch.zeros(1, t, dtype=torch.int32)
        devmask = self.packed_devmask and not cont
        kp = self.packed_prefix if cont else 0
        if devmask:  # row i sees keys start(i) .. i: its segment's start, a padding row only itself (start = i)
            starts_dev = torch.arange(t, dtype=torch.float32)
        else:  # keys = [prefix rows (cont) | packed rows]
            mask = torch.full((t, kp + t), NEG)
            mask[torch.arange(t), kp + torch.arange(t)] = 0.0  # padding rows attend to themselves only (never read)
            pre_pages = []
        wt = 512 // TILE
        lat = torch.full((t // TILE * wt,), 0xFFFFFFFF, dtype=torch.int64)
        slots = [int(slot) for _, slot, _ in reqs]
        assert len(set(slots)) == k, slots
        put(torch.tensor(slots, dtype=torch.int32).reshape(1, k), ttnn.uint32, pk["slot_idx"], self.repl)
        valid = torch.zeros(t, dtype=torch.bool)
        for b, ((tk, slot, nv), o) in enumerate(zip(reqs, offs)):
            nv, st = int(nv), starts[b]
            tok[0, o : o + nv] = tk.reshape(-1)[:nv].to(torch.int32)
            valid[o : o + nv] = True
            if devmask:
                starts_dev[o : o + nv] = float(o)
            else:
                mask[o : o + nv, kp + o : kp + o + nv] = torch.where(torch.ones(nv, nv).tril() > 0, 0.0, NEG)
                if st > 0:  # the slot's cached rows < st, gathered into the prefix keys at page q
                    row_ = self.page_row(slot)
                    q = len(pre_pages) * BLOCK
                    pre_pages += [int(row_[p]) for p in range(st // BLOCK)]
                    assert (
                        len(pre_pages) * BLOCK <= kp
                    ), f"prefixes {len(pre_pages) * BLOCK} rows > {kp} (PACKED_PREFIX)"
                    mask[o : o + nv, q : q + st] = 0.0
            m = torch.zeros(1, 1, t, 1)
            m[0, 0, o : o + nv, 0] = 1.0
            put(m, FP32, pk["seg_masks"][b], self.repl, ttnn.TILE_LAYOUT)
            put(torch.tensor([o + nv], dtype=torch.int32), ttnn.uint32, pk["seg_ends"][b], self.repl)
            last = torch.zeros(1, 1, t, 1)
            last[0, 0, o + nv - 1, 0] = 1.0
            put(last, FP32, pk["masks"][b], dshard if sp else self.repl, ttnn.TILE_LAYOUT)
            row = self.page_row(slot)
            for p in range(
                -(-nv // TILE) if self.shard is None else 0
            ):  # the segment's tile rows -> its slot's pages at
                # positions st + 32 p (the sharded cache: packed_lat_idx below)
                pos = st + p * TILE
                page = int(row[pos // BLOCK])
                dst = page * (BLOCK // TILE) + (pos % BLOCK) // TILE
                src = o // TILE + p
                lat[src * wt : (src + 1) * wt] = torch.arange(dst * wt, (dst + 1) * wt)
            if pk["sios"] is not None:
                pk["sios"][b].write(slot, fresh=st == 0)  # a continuation starts from the slot's KDA state
            if cont:  # its conv carry rows land just before o (o = 0: the conv's predecessor_carry)
                ci_ = torch.full((1, CONV_ROWS), 0xFFFFFFFF, dtype=torch.int64)
                if st > 0 and o >= CONV_ROWS:
                    ci_[0] = torch.arange(o - CONV_ROWS, o)
                put(ci_, ttnn.uint32, pk["carry_idx"][b], self.repl)
        put(tok, ttnn.uint32, pk["tokens"], ttnn.ShardTensorToMesh(self.mesh, dim=-1) if sp else self.repl)
        if devmask:  # 256 KB instead of the 8-MB [T, T] mask (built on device by _packed_mask, inside the trace)
            col = torch.zeros(1, 1, t, TILE)
            col[0, 0, :, 0] = starts_dev
            put(col, FP32, pk["starts"], self.repl, ttnn.TILE_LAYOUT)
        else:
            put(mask.reshape(1, 1, t, kp + t), BF16, pk["attn_mask"], self.repl, ttnn.TILE_LAYOUT)
            if cont:  # unused prefix pages -> the null block (masked)
                pt_ = torch.full((1, kp // BLOCK), NULL_BLOCK, dtype=torch.int32)
                pt_[0, : len(pre_pages)] = torch.tensor(pre_pages, dtype=torch.int32)
                put(pt_, ttnn.int32, pk["pre_table"], self.repl)
        mshard = ttnn.ShardTensorToMesh(self.mesh, dim=0) if self.ndev > 1 else None
        if self.shard is not None:  # every segment's tile rows to their OWNER chips' local blocks (per-chip tables)
            # (packed_lat_idx places a segment from position 0: no continuation segments on the sharded cache)
            assert not any(starts), "GLM53_DSA_SHARD packed continuation: sm.packed_lat_idx has no segment start"
            segs = [(o, int(nv), self.page_row(int(slot))) for (_, slot, nv), o in zip(reqs, offs)]
            put(self.sm.packed_lat_idx(t, segs), ttnn.uint32, pk["lat_idx"], mshard)
            if pk.get("win_idx") is not None:  # the dense window: every (fresh) segment's rows < 2,112 of its slot
                put(window_idx(t, [(o, 0, int(nv), int(slot)) for (_, slot, nv), o in zip(reqs, offs)]), ttnn.uint32,
                    pk["win_idx"], self.repl)  # fmt: skip
        else:
            put(lat.reshape(1, -1), ttnn.uint32, pk["lat_idx"], self.repl)  # int64 0xFFFFFFFF -> uint32 skip
        if (
            self.has_dsa()
        ):  # each segment's pool pages -> its slot's store pages; its ring rows (single path: write_inputs)
            pps, page = self.max_seq // (KPOOL * POOL_PAGE), KPOOL * POOL_PAGE
            wt = INDEX_DIM // TILE
            pidx = torch.full((t // page * wt,), 0xFFFFFFFF, dtype=torch.int64)
            for b, ((_, slot, nv), o) in enumerate(zip(reqs, offs)):
                nv, st = int(nv), starts[b]
                assert o % page == 0 and st % page == 0, (o, st, page)
                # pools of positions st + 128 p .. -> the slot's store page st / 128 + p (the sharded store: owner-only
                # writes on device, no host page list)
                for p in range(-(-nv // page) if self.shard is None else 0):
                    src, dst = o // page + p, int(slot) * pps + st // page + p
                    pidx[src * wt : (src + 1) * wt] = torch.arange(dst * wt, (dst + 1) * wt)
                rr = torch.full((1, TILE), t, dtype=torch.int32)
                end = st + nv  # the ring after this segment (write_inputs: golden_mla.dsa_state at p = end)
                base = KPOOL * (end // KPOOL)
                for j in range(KPOOL):
                    pos = base + j if base + j < end else base - KPOOL + j
                    if 0 <= pos - st < nv:
                        rr[0, j] = o + pos - st
                put(rr, ttnn.uint32, pk["ring_rows"][b], self.repl)
                put((torch.arange(RING_TILES) + int(slot) * RING_TILES).to(torch.int32).reshape(1, -1), ttnn.int32,
                    pk["ring_pages"][b], self.repl)  # fmt: skip
            if self.shard is not None and PAGED_POOLS:  # the segments' padded block tile rows -> their pages
                put(self.sm.packed_pool_idx_paged(t, segs), ttnn.uint32, pk["pool_idx"], mshard)
            elif self.shard is not None:
                pps_l = self.max_seq_local // (KPOOL * POOL_PAGE)
                put(self.sm.packed_pool_idx(t, [(o, int(nv), int(slot)) for (_, slot, nv), o in zip(reqs, offs)], pps_l),
                    ttnn.uint32, pk["pool_idx"], mshard)  # fmt: skip
            else:
                put(pidx.reshape(1, -1), ttnn.uint32, pk["pool_idx"], self.repl)
        self.write_pad(("p", t, k), valid)
        return offs

    def _packed_mask(self, t: int, pk: dict):
        """the packed SDPA's additive mask [1, 1, T, T] bf16 on device: 0 where start(i) <= j <= i, else NEG -- the
        row starts (pk["starts"], column 0) broadcast over the columns by a K = 32 fp32 matmul against a one-hot row,
        compared with the column index; the causal half is a resident constant. Equals write_packed's host mask."""
        cst = self.__dict__.setdefault("_packed_const", {})
        if t not in cst:
            tl = lambda v, dt: ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,  # noqa: E731
                                               memory_config=DRAM, mesh_mapper=self.repl)  # fmt: skip
            e_row = torch.zeros(1, 1, TILE, t)
            e_row[0, 0, 0, :] = 1.0
            cst[t] = dict(e_row=tl(e_row, FP32), jidx=tl(torch.arange(t, dtype=torch.float32).expand(t, t).reshape(1, 1, t, t), FP32),
                          causal=tl(torch.where(torch.ones(t, t).tril() > 0, 0.0, NEG).reshape(1, 1, t, t), BF16),
                          ckc=_ckc_fp32(self.mesh))  # fmt: skip
        c = cst[t]
        s_full = ttnn.matmul(pk["starts"], c["e_row"], dtype=FP32, compute_kernel_config=c["ckc"], memory_config=DRAM)
        ge = ttnn.ge(c["jidx"], s_full)
        ttnn.deallocate(s_full)
        cond = ttnn.typecast(ge, BF16)  # where(TTS) takes one dtype (the causal constant's)
        ttnn.deallocate(ge)
        m = ttnn.where(cond, c["causal"], NEG)
        ttnn.deallocate(cond)
        if m.dtype != BF16:
            m2 = ttnn.typecast(m, BF16)
            ttnn.deallocate(m)
            m = m2
        return m

    def _attn_sublayer_packed(self, pl: PrefillLayer, h, pk: dict, li: int, sp: bool, k: int):
        hn = pl.rms(h, "attn", BF16)
        if hn is not h:
            ttnn.deallocate(h)
        hb = self.gather(hn, sp)
        if hb is not hn:
            ttnn.deallocate(hn)
        if pl.kind == SPARSE_DSA:
            m = pk.get("mask_now")
            a = pl.mla.packed(hb, self.kv.extra["latent"][li], pk["lat_idx"], m if m is not None else pk["attn_mask"],
                              pk=pk, k=k, window=self.kv.extra.get("dsa", {}).get(li, {}).get("window"))  # fmt: skip
        else:
            a = kda_prefill_packed(pl.kda, hb, [self.seg_scratch(pl, b) for b in range(k)], pk["seg_masks"],
                                   pk["seg_ends"], scan=self.kda_scan, carry_idx=pk.get("carry_idx"))  # fmt: skip
        ttnn.deallocate(hb)
        return self.reduce(a, sp)

    def layer_packed(self, li: int, x, pk: dict, sp: bool, k: int):
        assert self.mhc_mode == "comp" and self.mhc_post == "fused", "packed prefill: the fused-mHC layer only"
        pl = self.layers[li]
        h, post, comb = self._mhc_pre(pl, "attn", x)
        a = self._attn_sublayer_packed(pl, h, pk, li, sp, k)
        x1 = self._post(a, x, post, comb)
        for t_ in (a, post, comb, x):
            ttnn.deallocate(t_)
        h, post, comb = self._mhc_pre(pl, "ffn", x1)
        m = self.reduce(self.ffn(pl, h, sp), sp)
        ttnn.deallocate(h)
        out = self._post(m, x1, post, comb)
        for t_ in (m, x1, post, comb):
            ttnn.deallocate(t_)
        return out

    def forward_packed(self, t: int, k: int, *, head: bool = True, cont: bool = False):
        """K fresh requests packed in ONE program of T rows (the host wrote write_packed): every segment's KDA state in,
        the embedding, layer_packed per layer, the states out, each segment's last row into its decode row, the head"""
        pk = self.packed_inputs(t, k, cont)
        sp = pk["sp"]
        self._sparse_now = None
        if pk["sios"] is not None:
            for sio in pk["sios"]:
                sio.state_in()
        x = self.embed(self.chunk_inputs(t), tokens=pk["tokens"])
        if self.packed_devmask and not cont and any(pl.kind == SPARSE_DSA for pl in self.layers):
            pk["mask_now"] = self._packed_mask(t, pk)
        self._pad_now = pk["pad"]
        for li in range(len(self.layers)):
            x = self.layer_packed(li, x, pk, sp, k)
        self._pad_now = None
        if pk.get("mask_now") is not None:
            ttnn.deallocate(pk.pop("mask_now"))
        if pk["sios"] is not None:
            for sio in pk["sios"]:
                sio.state_out()
        if head:
            rows = []
            for m in pk["masks"]:  # segment k's last valid row (masked sum: exact fp32; all-reduced over the chips)
                sel = ttnn.where(m, x, 0.0)
                rw = ttnn.sum(sel, dim=2, keepdim=True)
                ttnn.deallocate(sel)
                rows.append(self.reduce(rw, False) if sp else rw)
            stk = ttnn.concat(rows, dim=2) if k > 1 else rows[0]
            for rw in rows if k > 1 else []:
                ttnn.deallocate(rw)
            stk_rm = ttnn.to_layout(stk, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(stk)
            rows_to(stk_rm, pk["slot_idx"], pk["dec"])
            ttnn.deallocate(stk_rm)
            dec = ttnn.to_layout(pk["dec"], ttnn.TILE_LAYOUT, memory_config=DRAM)
            ch = int(self.b.chunk)
            for kk, res in enumerate(self.kv.res):
                part = ttnn.slice(dec, (0, 0, kk * ch, 0), (1, 1, (kk + 1) * ch, NSTREAM * HIDDEN))
                src = part if int(res.shape[-1]) != HIDDEN else ttnn.reshape(part, (ch, 1, NSTREAM, HIDDEN))
                ttnn.copy(src, res)
                for t_ in {id(v): v for v in (part, src)}.values():
                    ttnn.deallocate(t_)
            ttnn.deallocate(dec)
            self.b.head(self.kv.res, self.b.pos, pos_copies=self.b.pos_c)
        ttnn.deallocate(x)

    def capture_packed(self, t: int, k: int, *, head: bool = True, cont: bool = False):
        key = ("packedc" if cont else "packed", t, k)
        if os.environ.get("GLM53_MOCK") == "1" or key in self.traces:
            return
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.forward_packed(t, k, head=head, cont=cont)
        except BaseException:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.release_trace(self.mesh, tid)
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[key] = tid
        self.counters["captures"] += 1
        ttnn.synchronize_device(self.mesh)
        gc.collect()

    def prefill_packed(self, t: int, reqs: list, *, head: bool = True, blocking: bool = False):
        """K requests [(tokens, slot, n_valid[, chunk_start[, head]])] packed in ONE replay (an uncaptured key raises) ->
        segment offsets; a continuation segment (chunk start > 0) takes the continuation twin ("packedc", t, K), which
        also serves fresh-only steps when only the twin is captured"""
        k = len(reqs)
        cont = any(st > 0 for *_, st, _ in self.packed_reqs(reqs))
        if not cont and ("packed", t, k) not in self.traces and ("packedc", t, k) in self.traces:
            cont = True
        offs = self.write_packed(t, reqs, cont=cont)
        key = ("packedc" if cont else "packed", t, k)
        if os.environ.get("GLM53_MOCK") == "1":  # the image mock runs no traces
            self._mock_check(key)
            self.forward_packed(t, k, head=head, cont=cont)
            self.counters["eager"] += 1
            return offs
        if key not in self.traces and key in self.deferred:
            self._lazy_capture(key, head)
        if key not in self.traces:  # no eager fallback: the served path runs captured traces only
            raise RuntimeError(f"prefill packed key {key} was not captured (captured: {sorted(self.traces, key=str)})")
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=blocking)
        self.counters["replays"] += 1
        return offs

    def row_buf(self):
        if getattr(self, "_row_buf", None) is None:
            self._row_buf = ttnn.from_torch(torch.zeros(1, 1, 1, NSTREAM * HIDDEN), dtype=FP32, layout=ttnn.TILE_LAYOUT,
                                            device=self.mesh, memory_config=DRAM, mesh_mapper=self.repl)  # fmt: skip
        return self._row_buf

    # ------------------------------------------------------------------ traces
    def bucket_for(self, n: int, *, sparse: bool = False) -> int:
        for b in self.buckets:
            if n <= b and (not sparse or self.ndev == 1 or self.mode(b) == "sp"):
                return b
        raise ValueError(f"chunk of {n} tokens (sparse={sparse}) fits no bucket of {self.buckets}")

    def has_dsa(self) -> bool:
        return self.dsa and any(pl.kind == SPARSE_DSA for pl in self.layers)

    def fresh_keys(self, buckets=None):
        """(bucket, FRESH) of every MHA-form trace (GLM53_PF_MLA_FORM=mha: dense chunks at position 0)"""
        if self.mla_form != "mha" or not any(pl.kind == SPARSE_DSA for pl in self.layers):
            return []
        return [(c, FRESH) for c in tuple(buckets or self.buckets) if c >= self.fresh_min]

    def slice_rows(self, c: int) -> int:
        """rows per chip of a c-row chunk (bucket, packed T, batched nb x s) when it runs sequence-parallel; 0 when
        replicated or on one chip -- the slice-boundary rule's geometry (tt/generator.py slice_boundary)"""
        return c // self.ndev if self.ndev > 1 and self.mode(c) == "sp" else 0

    def _sparse(self, chunk_start: int, n_valid: int) -> bool:
        # (the sharded cache, GLM53_DSA_SHARD: a continuation chunk reads its cached context only through the sparse /
        # gather path -- the dense chunked SDPA reads a replicated cache -- so every chunk past position 0 is a sparse
        # key there; below the bound every visible pool is selected, so it is the same exact attention)
        return self.has_dsa() and (
            chunk_start + n_valid > DENSE_EXACT or (getattr(self, "shard", None) is not None and chunk_start > 0)
        )

    def slice_choice(self, chunk_start: int, n_valid: int) -> tuple:
        """(natural bucket, the bucket the chunk runs in or None) under the slice-boundary rule: the natural bucket
        when its geometry is safe, else the smallest larger bucket with a captured key for the chunk (sequence-parallel
        with a captured window >= chunk_start + c in the sparse regime) inside the slot's cache row (a bucket running
        past max_seq would send pad pages to the null block, chunk_pages) whose geometry is safe; None when none is.
        GLM53_PF_SLICE_RULE=0: (natural, natural)."""
        sparse = self._sparse(chunk_start, n_valid)
        nat = self.bucket_for(n_valid, sparse=sparse)
        if not self.slice_rule:
            return nat, nat

        def runs(c: int) -> bool:
            need = chunk_start + c
            if need > self.max_seq:
                return False
            if not sparse:
                return True  # (c, 0) / (c, FRESH) of every bucket are captured
            if self.ndev > 1 and self.mode(c) != "sp":
                return False
            return any(t >= need for t in self.sparse_ts) if self.sparse_ts else -(-need // 2048) * 2048 <= self.max_seq

        return nat, slice_safe_bucket(n_valid, nat, self.buckets, self.slice_rows, runs)

    def _plan_ok(self, chunk_start: int, n_valid: int) -> bool:
        """plan(chunk_start, n_valid) has a key: a bucket for the chunk and, in the sparse regime, a key window inside
        the cache (plan() raises ValueError otherwise)"""
        try:
            self.plan(chunk_start, n_valid)
        except ValueError:
            return False
        return True

    def clamp_lengths(self, chunk_start: int, n_valid: int) -> list:
        """the max_seq clamp (coordinator 2026-10-23 03:30:50Z: n29h5's 64K prompt died in plan() -- 'chunk [59392,
        67584) past the cache (66048 tokens)': the generator's 8,192-token chunk grid gave the prompt's LAST chunk the
        8,192 bucket although its replay ran past the cache / the captured window). A chunk whose replay would not fit
        runs as pieces: each the LARGEST bucket < the tokens left whose replay fits (a full bucket of tokens, so every
        next start stays on the bucket's 128-row grid), then the rest in its own bucket -- e.g. 4,608 tokens at 59,392
        under a 65,536 window: 2,048 + 2,048 + 512. [n_valid] when the chunk fits (every chunk but a long prompt's
        last) or when no piece does (plan() then raises exactly as before)."""
        al = int(getattr(self, "start_align", TILE))
        lens, s, left = [], int(chunk_start), int(n_valid)
        while not self._plan_ok(s, left):
            fit = [c for c in self.buckets if c < left and c % al == 0 and self._plan_ok(s, c)]
            if not fit:
                break
            c = max(fit)
            lens.append(c)
            s, left = s + c, left - c
        return lens + [left]

    def slice_plan(self, chunk_start: int, n_valid: int, front: int) -> tuple:
        """the generator's per-chunk rule (tt/generator.py slice_plan over slice_choice) -> (chunk lengths, branch):
        [n_valid] ("" / "larger": plan() picks the bucket at replay; "unavoidable") or [front, n_valid - front]
        ("split"); "row0" marks a 1-token chunk of a sequence-parallel bucket (its row 0 cannot move). First the
        max_seq clamp (clamp_lengths): a chunk whose replay would run past the cache runs as fitting pieces, each piece
        then under the slice rule (branch "clamp", or "clamp+<the pieces' slice branch>")."""
        pieces = self.clamp_lengths(int(chunk_start), int(n_valid))
        if len(pieces) > 1:
            lens, fired, s = [], [], int(chunk_start)
            for n in pieces:
                l_, b_ = self._slice_plan_one(s, n, front)
                lens += l_
                fired += [b_] if b_ else []
                s += n
            return lens, "+".join(["clamp"] + fired)
        return self._slice_plan_one(int(chunk_start), int(n_valid), front)

    def _slice_plan_one(self, chunk_start: int, n_valid: int, front: int) -> tuple:
        if not self.slice_rule:
            return [int(n_valid)], ""
        lens, branch = _slice_plan(int(chunk_start), int(n_valid), int(front), self.slice_choice)
        if n_valid == 1 and self.slice_rows(self.slice_choice(chunk_start, 1)[0]):
            branch = "row0"
        return lens, branch

    def plan(self, chunk_start: int, n_valid: int):
        """(bucket, key window t) of a chunk: t = 0 (dense: no query past position 2,050, or no sparse_dsa layer) or the
        smallest captured window >= chunk_start + bucket (GLM53_PF_SPARSE_T; else the next multiple of 2,048). The
        bucket is the slice-boundary rule's (slice_choice): the natural one unless its geometry puts the last valid row
        on a chip slice's first row and a larger captured bucket does not."""
        sparse = self._sparse(chunk_start, n_valid)
        nat, safe = self.slice_choice(chunk_start, n_valid)
        c = nat if safe is None else safe
        if not sparse:
            return (c, FRESH) if chunk_start == 0 and (c, FRESH) in self.fresh_keys((c,)) else (c, 0)
        need = chunk_start + c
        for t in sorted(self.sparse_ts):
            if t >= need:
                return c, t
        t = -(-need // 2048) * 2048
        if t > self.max_seq:
            raise ValueError(f"chunk [{chunk_start}, {chunk_start + c}) past the cache ({self.max_seq} tokens)")
        return c, t

    def sparse_keys(self, buckets=None):
        """(bucket, t) of every sparse trace: each sequence-parallel bucket x each window t >= bucket, t > 2,051"""
        if not self.has_dsa():
            return []
        buckets = tuple(buckets or self.buckets)
        return [(c, t) for c in buckets for t in self.sparse_ts
                if (self.ndev == 1 or self.mode(c) == "sp") and t >= c and (t > DENSE_EXACT or self.shard is not None)]  # fmt: skip

    def prepare(self, buckets=None, *, capture: bool = True, head: bool = True):
        """compile every bucket (one eager forward each: the first launch of every program is its compile) BEFORE any
        capture, then capture one trace per bucket. Overwrites slot 0's KDA scratch / latent pages and the head rows
        (call at setup; the generator's prepare restores the request state)."""
        self.compile(buckets, head=head)
        if capture:
            self.capture(buckets, head=head)

    # ------------------------------------------------------------------ start-time attribution (Stage 12)
    @contextlib.contextmanager
    def _cost(self, phase: str, key):
        """GLM53_PF_COST (default 1): the device-synchronized seconds and the program-cache entries a (phase, key) of
        compile / warm / capture adds -> self.cost and one summary line per phase (_cost_report): which keys the
        served start pays for (gx2c6: warmup_prefill 67 s + capture_prefill 54 s)."""
        if os.environ.get("GLM53_PF_COST", "1") != "1":
            yield
            return
        p0 = int(self.mesh.num_program_cache_entries())
        t0 = time.perf_counter()
        try:
            yield
        except BaseException:  # e.g. the trace region overflows: which keys of the phase fit before this one
            TPPrefill._cost_report(self, phase, note=f"RAISED at {key} after")  # never masks the error
            raise
        ttnn.synchronize_device(self.mesh)
        self.cost.append(dict(phase=phase, key=str(key), s=round(time.perf_counter() - t0, 3),
                              programs=int(self.mesh.num_program_cache_entries()) - p0))  # fmt: skip

    def _cost_report(self, phase: str, note: str = "") -> None:
        rows = [r for r in self.cost if r["phase"] == phase]
        if not rows and not note:
            return
        total = sum(r["s"] for r in rows)
        keys = "; ".join(f"{r['key']} {r['s']:.2f}s +{r['programs']}" for r in rows)
        print(f"TPPrefill {phase}{' ' + note if note else ''}: {len(rows)} keys {total:.1f} s, "
              f"programs +{sum(r['programs'] for r in rows)}: {keys}", flush=True)  # fmt: skip

    def compile(self, buckets=None, *, head: bool = True):
        buckets = tuple(buckets or self.buckets)
        for c in buckets:
            with self._cost("compile", (c, 0)):
                self.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
                self.forward(c, head=head)
            self.counters["eager"] += 1
        for c, t in self.fresh_keys(buckets):
            with self._cost("compile", (c, t)):
                self.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
                self.forward(c, head=head, t=t)
            self.counters["eager"] += 1
        for c, t in self.sparse_keys(buckets):
            with self._cost("compile", (c, t)):
                self.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=t - c, n_valid=c)
                self.forward(c, head=head, t=t)
            self.counters["eager"] += 1
        for s_, nb in self.batch_keys:  # batched admission (slots 0 .. nb-1 overwritten: setup only)
            with self._cost("compile", ("batch", s_, nb)):
                self.write_batch(s_, [(torch.zeros(s_, dtype=torch.int64), b, s_) for b in range(nb)])
                self.forward_batch(s_, nb, head=head)
            self.counters["eager"] += 1
        for t_, k_ in self.packed_keys:  # packed admission (slots 0 .. K-1 overwritten: setup only)
            n_ = (
                t_ // k_ // self.packed_align * self.packed_align
            )  # 14b int26: the DSA's 128-row segments (K = 3: 3 x 768 > 2,048)
            with self._cost("compile", ("packed", t_, k_)):
                self.write_packed(t_, [(torch.zeros(n_, dtype=torch.int64), b, n_) for b in range(k_)])
                self.forward_packed(t_, k_, head=head)
            self.counters["eager"] += 1
            if self.packed_cont:  # the continuation twin (fresh warm-up segments: every prefix page masked)
                self.write_packed(t_, [(torch.zeros(n_, dtype=torch.int64), b, n_) for b in range(k_)], cont=True)
                self.forward_packed(t_, k_, head=head, cont=True)
                self.counters["eager"] += 1
        self.row_buf()
        ttnn.synchronize_device(self.mesh)
        self._cost_report("compile")

    def warm(self, buckets=None, *, head: bool = True):
        """one eager forward per trace key (every bucket, fresh and sparse key) under the device state the captures will
        see, BEFORE any trace capture (StackBackend.capture_decode calls it ahead of the decode trace): programs that
        (re)compile here -- and any buffer such a program allocates for itself -- precede every trace; capture() then
        records without per-key warm forwards. Overwrites slot 0's scratch / pages and the head rows (warm-up only)."""
        # the image mock (GLM53_MOCK=1) skips this unless GLM53_MOCK_PF_WARM=1: then its eager forwards run on the mock
        # cluster for their ALLOCATIONS (no kernel runs), so a warm-up transient that does not fit device DRAM (int27
        # 14:41Z: the 256 MiB fp32 causal mask of the 32K sparse key) fails the rehearsal instead of the device open
        mock_warm = os.environ.get("GLM53_MOCK") == "1" and os.environ.get("GLM53_MOCK_PF_WARM") == "1"
        if (os.environ.get("GLM53_MOCK") == "1" and not mock_warm) or os.environ.get(
            "GLM53_PF_WARM_CAPTURE", "1"
        ) != "1":
            return
        buckets = tuple(buckets or self.buckets)
        for key in [(c, 0) for c in buckets] + self.fresh_keys(buckets) + self.sparse_keys(buckets):
            with self._cost("warm", key):
                self._key_inputs(key, full=True)
                self.forward(key[0], head=head, t=key[1])
            self.counters["eager"] += 1
        for s_, nb in self.batch_keys:
            with self._cost("warm", ("batch", s_, nb)):
                self.forward_batch(s_, nb, head=head)
            self.counters["eager"] += 1
        for t_, k_ in self.packed_keys:
            with self._cost("warm", ("packed", t_, k_)):
                self.forward_packed(t_, k_, head=head)
            self.counters["eager"] += 1
            if self.packed_cont:
                self.forward_packed(t_, k_, head=head, cont=True)
                self.counters["eager"] += 1
        self._cost_report("warm")
        if self.sio is not None and self.slot_rows > self.max_batch:  # the prefix cache's copy program (plain DMA)
            self.sio.copy(self.max_batch, self.max_batch)  # entry 0 <- entry 0: compiles it before any capture
        ttnn.synchronize_device(self.mesh)
        gc.collect()
        self._warmed = True

    def capture(self, buckets=None, *, head: bool = True):
        buckets = tuple(buckets or self.buckets)
        if os.environ.get("GLM53_MOCK") == "1":
            if os.environ.get("GLM53_MOCK_CAPTURE", "0") == "1":  # each captured forward once, for its allocations
                for key in [(c, 0) for c in buckets] + self.fresh_keys(buckets) + self.sparse_keys(buckets):
                    self._key_inputs(key, full=not self._warmed)
                    self.forward(key[0], head=head, t=key[1])
                    if self._twin(key):
                        self.forward(key[0], head=False, t=key[1])
                    (self.deferred if self._lazy(key) else self.mock_captured).add(key)
                for s_, nb in self.batch_keys:
                    self.forward_batch(s_, nb, head=head)
                    (self.deferred if self._lazy(("batch", s_, nb)) else self.mock_captured).add(("batch", s_, nb))
                for t_, k_ in self.packed_keys:
                    (self.deferred if self._lazy(("packed", t_, k_)) else self.mock_captured).add(("packed", t_, k_))
                    if self.packed_cont:
                        self.forward_packed(t_, k_, head=head, cont=True)
                        (self.deferred if self._lazy(("packed", t_, k_)) else self.mock_captured).add(
                            ("packedc", t_, k_)
                        )
                gc.collect()
            return

        leak_dbg = os.environ.get("GLM53_TRACE_LEAK_DEBUG") == "1"
        report = _TraceReport(self.mesh)  # GLM53_PF_CAPTURE_REPORT=1: a per-key size table
        for key in [(c, 0) for c in buckets] + self.fresh_keys(buckets) + self.sparse_keys(buckets):
            if key in self.traces:
                continue
            if self._lazy(key):  # GLM53_PF_LAZY_KEYS: captured on first use (replay)
                self.deferred.add(key)
                continue
            with self._cost("capture", key):
                self._key_inputs(key, full=not self._warmed)  # (warmed: the capture runs nothing, only the words)
                self._capture_key(key, head, leak_dbg, twin=self._twin(key) and not self._lazy(("nohead", key)))
            report(f"prefill {key} head{' + twin' if self._twin(key) and not self._lazy(('nohead', key)) else ''}")
            if self._twin(key) and self._lazy(("nohead", key)):
                self.deferred.add(("nohead", key))
        for s_, nb in self.batch_keys:
            if self._lazy(("batch", s_, nb)):
                self.deferred.add(("batch", s_, nb))
                continue
            with self._cost("capture", ("batch", s_, nb)):
                self.capture_batch(s_, nb, head=head)
            report(f"prefill batch {(s_, nb)}")
        for t_, k_ in self.packed_keys:
            if self._lazy(("packed", t_, k_)):
                self.deferred.add(("packed", t_, k_))
                continue
            with self._cost("capture", ("packed", t_, k_)):
                self.capture_packed(t_, k_, head=head)
                if self.packed_cont:  # the continuation twin ("packedc", T, K)
                    self.capture_packed(t_, k_, head=head, cont=True)
            report(f"prefill packed {(t_, k_)}")
        if self.deferred:
            print(f"TPPrefill capture: {len(self.deferred)} key(s) deferred to first use (GLM53_PF_LAZY_KEYS="
                  f"{os.environ.get('GLM53_PF_LAZY_KEYS', '')}): {sorted(self.deferred, key=str)}", flush=True)  # fmt: skip
        ttnn.synchronize_device(self.mesh)
        self._cost_report("capture")
        # temporaries of the captured forwards held only by reference cycles would sit in the traces' memory at replay
        # (TT_METAL_TRACE_ALLOC_TRACKING: 'device buffer(s) still alive before trace replay'): free them now
        gc.collect()

    def _mock_check(self, key) -> None:
        """GLM53_MOCK_CAPTURE=1: the device's 'an uncaptured key raises' on the mock (a deferred key is captured at its
        first use, as _lazy_capture does on the device)"""
        if os.environ.get("GLM53_MOCK_CAPTURE", "0") != "1":
            return
        captured = getattr(self, "mock_captured", set())
        if key in captured:
            return
        if key in self.deferred or ("nohead", key) in self.deferred:
            captured.add(key)
            self.deferred.discard(key)
            self.deferred.discard(("nohead", key))
            return
        raise RuntimeError(
            f"prefill key {key} was not captured (mock capture flow; captured: {sorted(captured, key=str)})"
        )

    def _key_inputs(self, key, full: bool) -> None:
        """rebind the bucket's chunk inputs to key (c, t) before its warm forward / capture: the ChunkInputs are one per
        BUCKET, shared by every window t of it, and hold whatever was written last (14b realgate7, 2026-10-25 12:01Z:
        after compile() the inputs held the (2048, 32768) key's chunk start 30,720; the (2048, 2048) key's eager warm
        forward then fed SpaUnion ids far past its 2,048 window and spa_um hung on an out-of-bounds L1 remap). full:
        compile()'s exact write_inputs (an eager forward follows); else only the chunk-start words (a capture runs
        nothing; its replays get the request's inputs from prefill())."""
        c, t = key
        cs = t - c if t > 0 else 0  # compile(): bucket / fresh keys at position 0, a sparse key's chunk ends at t
        if full:
            self.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=cs, n_valid=c)
            return
        ci = self.chunk_inputs(c)
        put = lambda v, dt, dst, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=self.repl), dst)  # fmt: skip
        put(torch.tensor([cs], dtype=torch.int32), ttnn.int32, ci.chunk_start)
        put(torch.tensor([cs], dtype=torch.int32), ttnn.uint32, ci.chunk_start_u)
        put(torch.full((1, 1, 1, 1), float(cs)), FP32, ci.chunk_start_f, ttnn.TILE_LAYOUT)
        self._cs_host[c] = cs

    def _check_window(self, c: int, t: int, where: str) -> None:
        """the bucket's last-written chunk start fits key (c, t): a sparse key's window t must cover the chunk
        (chunk_start + c <= t) -- the SPA union reads ids made from the device chunk start (a bad one: ids past the
        window; spa_um drops them, counted in SpaUnion.status(), but the attention is then wrong)"""
        cs = self._cs_host.get(c)
        if t <= 0 or cs is None:  # dense / fresh keys: no window; a bucket never written: compile writes it first
            return
        assert (
            0 <= cs and cs + c <= t
        ), f"{where}: key {(c, t)} with chunk start {cs} (rows {cs}..{cs + c - 1} past window {t})"

    def _lazy(self, key) -> bool:
        """key's capture is deferred to first use (GLM53_PF_LAZY_KEYS)"""
        if not self.lazy:
            return False
        if key[0] == "nohead":
            return "nohead" in self.lazy
        if key[0] in ("batch", "packed"):
            return key[0] in self.lazy
        c, t = key
        return str(c) in self.lazy or ("sparse" in self.lazy and t > 0)

    def _lazy_capture(self, key, head: bool = True) -> None:
        """the first use of a deferred key: capture it now (no new program compiles: warm() ran for every key at start)
        between steps, the head trace + its head-less twin; one log line with the seconds the request paid"""
        t0 = time.perf_counter()
        ttnn.synchronize_device(self.mesh)
        with self._cost("lazy", key):
            if key[0] == "batch":
                self.capture_batch(key[1], key[2], head=head)
            elif key[0] == "packed":
                self.capture_packed(key[1], key[2], head=head)
            elif key not in self.traces:
                self._capture_key(key, head, twin=self._twin(key))
            elif self._twin(key) and key not in self.traces_nh:  # only the twin was deferred
                self._capture_key_twin(key)
        self.deferred.discard(key)
        self.deferred.discard(("nohead", key))
        self.counters["lazy_captures"] = self.counters.get("lazy_captures", 0) + 1
        gc.collect()
        print(f"TPPrefill lazy capture {key}: {time.perf_counter() - t0:.2f} s (first use)", flush=True)

    def _capture_key_twin(self, key) -> None:
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.forward(key[0], head=False, t=key[1])
        except BaseException:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.release_trace(self.mesh, tid)
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces_nh[key] = tid
        self.counters["captures"] += 1

    def _capture_key(self, key, head: bool, leak_dbg: bool = False, twin: bool = True):
        """capture() of one (c, t) key: its warm forward if warm() did not run, the head trace, and (GLM53_PF_NOHEAD)
        the head-less twin unless twin=False (deferred separately). Also the lazy first-use capture (GLM53_PF_LAZY_KEYS).
        """
        self._check_window(key[0], key[1], "capture")
        if os.environ.get("GLM53_PF_WARM_CAPTURE", "1") == "1" and not self._warmed:
            # one eager forward of this key right before its capture: the programs are (re)loaded under the device
            # state the capture sees (20:59Z TP8: after the decode warmup a prefill program compiled in
            # prepare_prefill missed the program cache inside the capture -> "Cannot load new binaries during trace
            # capture"). Prefer warm() BEFORE any trace capture: a program (re)compiled here may allocate buffers
            # of its own that live with its cache entry and sit in every earlier trace's memory (02:00Z pair: the
            # decode trace's replay found one such buffer from MlaPrefill._mha's warm forward)
            self.quiesce()  # an eager pass that may follow trace replays (the decode warmup's, earlier keys')
            self.forward(key[0], head=head, t=key[1])
            self.counters["eager"] += 1
            ttnn.synchronize_device(self.mesh)
            gc.collect()
        before = _live_device_tensors() if leak_dbg else None
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.forward(key[0], head=head, t=key[1])
        except BaseException:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.release_trace(self.mesh, tid)
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        if before is not None:  # tensors created inside the capture and still referenced after it
            gc.collect()
            for t_ in _live_device_tensors().values():
                if id(t_) not in before:
                    holders = [type(r).__name__ + (f"[{list(r.keys())[:6]}]" if isinstance(r, dict) else "")
                               for r in gc.get_referrers(t_) if not isinstance(r, list) or len(r) < 64][:4]  # fmt: skip
                    print(f"TRACE_LEAK prefill {key} {tuple(t_.shape)} {t_.dtype} held by {holders}", flush=True)
        self.traces[key] = tid
        self.counters["captures"] += 1
        if twin and self.nohead and key not in self.traces_nh:  # the same programs minus the head: nothing new compiles
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self.forward(key[0], head=False, t=key[1])
            except BaseException:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                ttnn.release_trace(self.mesh, tid)
                raise
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces_nh[key] = tid
            self.counters["captures"] += 1

    def replay(self, c: int, blocking: bool = False, t: int = 0, head: bool = True):
        traces = self.traces if head else self.traces_nh
        if os.environ.get("GLM53_MOCK") == "1":  # the image mock runs no traces
            self._mock_check((c, t))
            self.forward(c, t=t, head=head)
            self.counters["eager"] += 1
            return
        if (c, t) not in traces and ((c, t) in self.deferred or ("nohead", (c, t)) in self.deferred):
            self._lazy_capture((c, t), head)
        if (c, t) not in traces:  # no eager fallback: the served path runs captured traces only
            raise RuntimeError(f"prefill key {(c, t)} (head={head}) was not captured (captured: {sorted(traces)})")
        ttnn.execute_trace(self.mesh, traces[(c, t)], cq_id=0, blocking=blocking)
        self.counters["replays"] += 1

    def release(self):
        for traces in (self.traces, self.traces_nh):
            for key, tid in list(traces.items()):
                ttnn.release_trace(self.mesh, tid)
                traces.pop(key)

    def quiesce(self) -> None:
        """mesh.quiesce_devices() before an EAGER pass that may follow trace replays (the TP8 composite hang, 2026-10-23,
        fda3938f3b5: the watcher showed the dependent axis-0 gather running while its axis-1 stage was mid-kernel). A
        replay re-arms the host's expected worker count to the recorded one while the device's count can sit above it
        (the traced mux CCLs: per-chip worker sets evened out by dummy go messages), so later go-waits pass early and
        programs overlap; ttnn.synchronize_device does not reset this, quiesce zeroes both sides (14's review 19:01Z).
        A no-op on the mock (its dispatch reset segfaults there) and with GLM53_PF_QUIESCE=0. Re-arms the
        GLM53_PF_SYNC_EVERY_LAUNCH fence: a quiesce starts a compile pass."""
        if _LAUNCH_FENCE is not None:
            _LAUNCH_FENCE["on"] = True
        if os.environ.get("GLM53_MOCK") == "1" or os.environ.get("GLM53_PF_QUIESCE", "1") != "1":
            return
        self.mesh.quiesce_devices()
        self.counters["quiesces"] = self.counters.get("quiesces", 0) + 1

    def nohead_ready(self, chunk_start: int, n_valid: int) -> bool:
        """a head-less trace exists for this chunk's key (GLM53_PF_NOHEAD=1 and captured)"""
        key = self.plan(chunk_start, n_valid)
        return self._twin(key) and (key in self.traces_nh or key in self.deferred or ("nohead", key) in self.deferred)

    def _twin(self, key) -> bool:
        """key has a head-less twin: GLM53_PF_NOHEAD=1 and (GLM53_PF_NOHEAD_KEYS unset or listing it)"""
        return self.nohead and (self.nohead_keys is None or tuple(key) in self.nohead_keys)

    def prefill(self, tokens: torch.Tensor, *, slot: int, chunk_start: int, n_valid: int, head: bool = True,
                state_src: Optional[int] = None, state_snap: Optional[int] = None,
                snap_at: Optional[int] = None):  # fmt: skip
        """one chunk of a request: the slot's KDA state into the scratch, the chunk inputs, ONE replay (all layers +
        the head on the last valid row), the scratch back into the slot. The caller owns cur_pos / the sampler rows
        (StackBackend.prefill_chunk). head=False (a non-final chunk): the head-less trace when captured
        (GLM53_PF_NOHEAD, see nohead_ready), else the full one.
        Prefix cache: `state_src` = the snapshot pool entry whose state starts the chunk (instead of the slot's: a
        fresh request attaching at chunk_start > 0), `state_snap` = the entry that also receives the state after the
        chunk; both are virtual slots max_batch + entry of the same state_io / SlotOps programs (no extra copy inside
        the trace: state_io reads the source and writes both destinations)."""
        assert chunk_start % self.start_align == 0, f"chunk start {chunk_start} not a multiple of {self.start_align}"
        src = None if state_src is None else self.max_batch + int(state_src)
        snap = None if state_snap is None else self.max_batch + int(state_snap)
        for v in (src, snap):
            assert v is None or self.max_batch <= v < self.slot_rows, (v, self.max_batch, self.slot_rows)
        assert src is None or chunk_start > 0, "an attach continues a cached prefix (chunk_start > 0)"
        c, t = self.plan(chunk_start, n_valid)
        if c != self.bucket_for(n_valid, sparse=self._sparse(chunk_start, n_valid)):  # the slice-boundary rule moved it
            self.counters["slice_larger"] = self.counters.get("slice_larger", 0) + 1
        head = head or not (self.nohead and (c, t) in self.traces_nh)
        t0 = time.perf_counter()
        if self.sio is None:
            self.b.slot_ops.run(self.state_ops(slot if src is None else src, into_scratch=True))
        # the cached prefix's DSA indexer state (pooled keys of positions < chunk_start + the ring) moves with an
        # attach and into a snapshot entry: state_io carries the KDA state only (SlotOps page copies, before / after
        # the replay on the same CQ; 2026-10-22: without them a sparse-regime attach read the slot's stale pools)
        copy = getattr(self, "prefix_dsa_copy", True)  # False: the pre-fix behaviour (a unit's control variant)
        dsa_in = [] if src is None or not copy else self.b._dsa_ops(self.kv, slot, src, upto=chunk_start)
        if dsa_in:
            self.b.slot_ops.run(dsa_in)
            self.counters["slot_ops"] += 1
        carries = self.write_inputs(c, tokens, slot=slot, chunk_start=chunk_start, n_valid=n_valid, state_src=src,
                                    state_snap=snap, snap_at=snap_at)  # fmt: skip
        t1 = time.perf_counter()
        self._check_window(c, t, "prefill")
        self.replay(c, t=t, head=head)
        if carries:  # the cut scratches hold the slot's state at snap_at (export_kda_state(source="snapshot"))
            self.kv.extra["kda_snap_at"] = (int(slot), int(snap_at))
            self.counters["kda_snapshots"] = self.counters.get("kda_snapshots", 0) + 1
        elif snap_at is not None and ci_carry_missed(chunk_start, n_valid, snap_at):
            self.counters["kda_snapshot_outside"] = self.counters.get("kda_snapshot_outside", 0) + 1
        dsa_out = [] if snap is None or not copy else self.b._dsa_ops(self.kv, snap, slot, upto=chunk_start + n_valid)
        if dsa_out:
            self.b.slot_ops.run(dsa_out)
            self.counters["slot_ops"] += 1
        if os.environ.get("GLM53_PF_HOST_TIMING") == "1":
            acc = self.counters.setdefault("host_ms", {})
            acc["write_inputs"] = acc.get("write_inputs", 0.0) + (t1 - t0) * 1e3
            acc["replay_enqueue"] = acc.get("replay_enqueue", 0.0) + (time.perf_counter() - t1) * 1e3
        if self.sio is None:
            self.b.slot_ops.run(self.state_ops(slot, into_scratch=False)
                                + ([] if snap is None else self.state_ops(snap, into_scratch=False)))  # fmt: skip
            self.counters["slot_ops"] += 2
        return c

    # ------------------------------------------------------------------ host side
    def write_inputs(self, c: int, tokens: torch.Tensor, *, slot: int, chunk_start: int, n_valid: int, ci=None,
                     state_src: Optional[int] = None, state_snap: Optional[int] = None,
                     snap_at: Optional[int] = None):  # fmt: skip
        """ci: a batched segment's inputs (seg_inputs) instead of the bucket's; its slot's KDA state I/O is the caller's.
        state_src / state_snap: VIRTUAL slots (max_batch + entry) of the prefix-cache snapshot pool, see prefill()
        snap_at (GLM53_PF_KDA_SNAPSHOT): the request's checkpoint cut (absolute position); when this chunk covers it
        the snapshot scalar = cut - chunk_start (else 32: a no-op snapshot). Returns True when this chunk carries the
        cut."""
        seg = ci is not None
        ci = ci or self.chunk_inputs(c)
        carries = False
        if ci.snap_end_u is not None and not seg:
            carries, rows = kda_snap_rows(snap_at, chunk_start, n_valid, c)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([rows], dtype=torch.int32), dtype=ttnn.uint32,
                                                            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.repl),
                                            ci.snap_end_u)  # fmt: skip
        if self.sio is not None and not seg:
            self.sio.write(slot, fresh=chunk_start == 0 and state_src is None, src=state_src, snap=state_snap)
        sp = self.mode(c) == "sp"
        put = lambda v, dt, dst, mp=None, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=mp), dst)  # fmt: skip
        tok = torch.zeros(1, c, dtype=torch.int32)
        tok[0, :n_valid] = tokens.reshape(-1)[:n_valid].to(torch.int32)
        # GLM53_PF_PAD_TOKENS (A/B, 14b 2026-10-22 TP8 padded-bucket triage): the padding rows' token ids -- "zero"
        # (default), "repeat" (the valid ids cycled: spreads the padding rows' routing over the experts the request uses),
        # or "random:<seed>". Valid rows never read padding rows (causal attention / KDA identity steps), unless an op
        # mixes rows (e.g. a per-expert capacity in the MoE): then the valid rows change with the padding content.
        pad_mode = os.environ.get("GLM53_PF_PAD_TOKENS", "zero")
        if n_valid < c and pad_mode != "zero":
            if pad_mode == "repeat":
                tok[0, n_valid:] = tok[0, torch.arange(c - n_valid) % n_valid]
            elif pad_mode.startswith("random"):
                g = torch.Generator().manual_seed(int(pad_mode.split(":")[1]) if ":" in pad_mode else 0)
                tok[0, n_valid:] = torch.randint(0, 150000, (c - n_valid,), generator=g, dtype=torch.int32)
        put(tok, ttnn.uint32, ci.tokens, ttnn.ShardTensorToMesh(self.mesh, dim=-1) if sp else self.repl)
        put(torch.tensor([n_valid], dtype=torch.int32), ttnn.uint32, ci.valid_end, self.repl)
        if not seg:  # rows >= n_valid: sentinel expert E, weight 0 (GLM53_PF_MOE_PADMASK; batched: write_batch)
            self.write_pad(c, torch.arange(c) < n_valid)
        put(torch.full((1, 1, 1, 1), float(n_valid)), FP32, ci.valid_end_f, self.repl, ttnn.TILE_LAYOUT)
        row = self.page_row(slot)
        mshard = ttnn.ShardTensorToMesh(self.mesh, dim=0) if self.ndev > 1 else None
        if self.shard is not None:  # the PHYSICAL row (owner-aware, tt/owner_blocks.py) -> per-chip local tables + the
            # chunk's latent tiles -> their owner chips' local blocks (prefill_lat_idx: valid tile rows only)
            sm = self.sm
            blocks = row.to(torch.int64)[chunk_start // BLOCK : (chunk_start + c) // BLOCK]
            put(sm.prefill_lat_idx(blocks, n_valid), ttnn.uint32, ci.lat_idx, mshard)
            if ci.win_idx is not None:  # the dense window: this chunk's valid rows < 2,112 of the slot
                put(window_idx(c, [(0, chunk_start, n_valid, slot)]), ttnn.uint32, ci.win_idx, self.repl)
            put(sm.page_tables_local(row.reshape(1, -1), self.bps_local).reshape(self.ndev, -1), ttnn.int32,
                ci.user_table, mshard)  # fmt: skip
        else:
            blocks = chunk_pages(row, chunk_start, n_valid, c)
            put(blocks.reshape(1, -1).to(torch.int32), ttnn.int32, ci.chunk_table, self.repl)
            put(row.reshape(1, -1).to(torch.int32).clamp(min=0), ttnn.int32, ci.user_table, self.repl)
        put(torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32, ci.chunk_start, self.repl)
        put(torch.tensor([chunk_start], dtype=torch.int32), ttnn.uint32, ci.chunk_start_u, self.repl)
        if (
            ci.vis_cols_u is not None
        ):  # GLM53_PF_IX_TOPK_BOUND: each chip's top-k column bound (the score's written extent)
            cl_ = c // self.ndev if (sp and self.ndev > 1) else c
            vc = [-(-(chunk_start + (r + 1) * cl_) // 128) * 32 for r in range(self.ndev if self.ndev > 1 else 1)]
            if self.ndev > 1:
                put(torch.tensor(vc, dtype=torch.int32).reshape(self.ndev, 1), ttnn.uint32, ci.vis_cols_u,
                    ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip
            else:
                put(torch.tensor(vc, dtype=torch.int32), ttnn.uint32, ci.vis_cols_u, self.repl)
        if ci.pool_units_u is not None and self.ndev > 1:  # GLM53_PF_IX_GATHER_BOUND: each chip's visible local units
            um = -(-(chunk_start + c) // DSH.UNIT)
            pu = [max(0, -(-(um - r) // self.ndev)) for r in range(self.ndev)]
            put(torch.tensor(pu, dtype=torch.int32).reshape(self.ndev, 1), ttnn.uint32, ci.pool_units_u,
                ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip
        put(torch.full((1, 1, 1, 1), float(chunk_start)), FP32, ci.chunk_start_f, self.repl, ttnn.TILE_LAYOUT)
        if not seg:
            self._cs_host[c] = int(chunk_start)
        if self.has_dsa():  # the slot's pooled-key store pages (128 tokens each): the chunk's, and all of the slot's
            assert chunk_start % (KPOOL * POOL_PAGE) == 0, f"sparse_dsa prefill chunks start at multiples of 128"
            if self.shard is not None and PAGED_POOLS:  # each block's padded tile row -> its page on its owner chip
                pps = self.max_seq_local // (KPOOL * POOL_PAGE)  # (pool_user below: unused by the paged read)
                pidx = sm.prefill_pool_idx_paged(blocks, n_valid)  # [C, len(blocks) x 4]: pad to the table's width
                wfull = pool_idx_tiles(c, True)
                if int(pidx.shape[1]) < wfull:
                    pidx = torch.cat([pidx, torch.full((int(pidx.shape[0]), wfull - int(pidx.shape[1])), DSH.SKIP,
                                                       dtype=pidx.dtype)], dim=1)  # fmt: skip
                put(pidx, ttnn.uint32, ci.pool_idx, mshard)
            elif self.shard is not None:  # each unit's pooled keys to its owner's local page; the slot's local pages
                pps = self.max_seq_local // (KPOOL * POOL_PAGE)
                put(sm.prefill_pool_idx(c, chunk_start, n_valid, slot, pps), ttnn.uint32, ci.pool_idx, mshard)
            else:
                pps = self.max_seq // (KPOOL * POOL_PAGE)
                p0 = chunk_start // (KPOOL * POOL_PAGE)
                pt = (torch.arange(p0, p0 + int(ci.pool_table.shape[-1])).clamp(max=pps - 1) + slot * pps).to(
                    torch.int32
                )
                put(pt.reshape(1, -1), ttnn.int32, ci.pool_table, self.repl)
            put((torch.arange(pps) + slot * pps).to(torch.int32).reshape(1, -1), ttnn.int32, ci.pool_user, self.repl)
            end = chunk_start + n_valid  # the ring after this chunk (golden_mla.dsa_state at p = end)
            rr = torch.full((1, TILE), c, dtype=torch.int32)
            base = KPOOL * (end // KPOOL)
            for j in range(KPOOL):
                pos = base + j if base + j < end else base - KPOOL + j
                if 0 <= pos - chunk_start < n_valid:
                    rr[0, j] = pos - chunk_start
            put(rr, ttnn.uint32, ci.ring_rows, self.repl)
            put((torch.arange(RING_TILES) + slot * RING_TILES).to(torch.int32).reshape(1, -1), ttnn.int32,
                ci.ring_pages, self.repl)  # fmt: skip
        m = torch.zeros(1, 1, c, 1)
        m[0, 0, n_valid - 1, 0] = 1.0
        put(m, FP32, ci.last_mask, ttnn.ShardTensorToMesh(self.mesh, dim=2) if sp else self.repl, ttnn.TILE_LAYOUT)
        return carries

    def page_row(self, slot: int) -> torch.Tensor:
        pt = getattr(self.b, "_pt_host", None)
        if pt is None and self.shard is not None:  # no table published yet (compile / warm-up): the owners' null blocks
            return self.sm.null_row(self.kv.blocks_per_seq)
        if pt is None:  # the identity mapping of allocate_decode_io
            bps = self.kv.blocks_per_seq
            return (torch.arange(slot * bps, (slot + 1) * bps) % self.kv.num_blocks).to(torch.int64)
        return pt[slot].to(torch.int64)

    def state_ops(self, slot: int, *, into_scratch: bool) -> list:
        """SlotOps copying slot `slot`'s KDA S rows / conv rows into (or out of) every layer's scratch"""
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _pages, _rows

        cd, rd = divmod(slot, self.b.chunk)
        ops = []
        for li, pl in enumerate(self.layers):
            if pl.kind == SPARSE_DSA:
                continue
            S, C = self.kv.states[cd][li]
            n_s = _pages(S) // _rows(S)  # per-user pages (the pool's last chunk may be shorter)
            sc = pl.scratch
            assert _pages(sc.S) == n_s and _pages(sc.C) == _pages(C), (li, _pages(sc.S), n_s)
            if into_scratch:
                ops.append((1, S.buffer_address(), rd * n_s, sc.S.buffer_address(), 0, n_s, 4096, 0))
                ops.append((2, C.buffer_address(), 0, sc.C.buffer_address(), 0, _pages(C), 2048, rd << 16))
            else:
                ops.append((1, sc.S.buffer_address(), 0, S.buffer_address(), rd * n_s, n_s, 4096, 0))
                ops.append((2, sc.C.buffer_address(), 0, C.buffer_address(), 0, _pages(C), 2048, rd))
        return ops


KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
# kda ops' sequence_parallel_axis sentinel (kda/factory/chronology_binding.hpp KDA_NO_SEQUENCE_PARALLEL): the chunk is
# NOT split over the mesh -- every chip is rank 0 of 1 (TP heads on the 2x4 mesh, whose axis 0 has extent 2)
KDA_NO_SP = 0xFFFF
# kda_scan="grouped": chunk groups per head scanned in parallel (GLM53_PF_KDA_GROUPS; must divide C / 32)
KDA_GROUPS = int(os.environ.get("GLM53_PF_KDA_GROUPS", "8"))


# Static CB bytes per core of the prefill's own programs. In a serving process the decode stack keeps persistent L1
# tensors (its intermediates pools, doc/ttnn/14b-persistent-stack/l1_ledger.md): tt-metal refuses a program whose CB
# region (base 308,352 at the stack's worker_l1_size 1,264,512) ends above the lowest of them (measured 1,355,392 in the
# TF flow at L0-L4: ~1.04 MB of CB room). Stage 3's prefill matmul sizes against a 900 KB ESTIMATE (o_proj measured
# 1,048,576 B -> the 18:33Z mock clash); the TP prefill bounds the estimate (x ~1.17 measured) and moe_fused_swiglu's.
PF_MM_EST = int(os.environ.get("GLM53_PF_MM_KB", "640")) * 1024
PF_MOE_CB = int(os.environ.get("GLM53_PF_MOE_KB", "960")) * 1024


def pf_mm(ds, x, dtype=BF16, memory_config=DRAM, out=None):
    """DSLinear.prefill (2D multicast program over the DRAM width-sharded weight, out_block_w == per_core_N) with the
    program's CB estimate bounded by PF_MM_EST (programs cached on the DSLinear per (M tiles, budget))."""
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import prefill_program

    m_tiles = -(-int(x.shape[-2]) // TILE) * (int(x.shape[-3]) if len(x.shape) > 2 else 1)
    nq = int(os.environ.get("GLM53_PF_MM_NQ", "4"))  # per-core N block quantum (prefill_program n_quantum; P3)
    est = int(os.environ.get("GLM53_PF_MM_KB", str(PF_MM_EST // 1024))) * 1024  # per call: long-lived servers
    # GLM53_PF_MM_F32ACC=0 (P3): bf16 destination accumulation for the prefill projections -> 8-tile output sub-blocks
    # instead of fp32's 4 (the decode ckc accumulates in fp32)
    f32acc = os.environ.get("GLM53_PF_MM_F32ACC", "1") == "1"
    key = ("s16", m_tiles, est, nq, f32acc)
    prog = ds._prefill_programs.get(key)
    if prog is None:
        wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(ds.weight.dtype, 2048)
        prog = prefill_program(ds.grid, m_tiles, ds.k // TILE, ds.n_pad // TILE, weight_tile_bytes=wtb,
                               m_round=ds.ocfg.prefill_m_block_round, max_block_w=ds.ocfg.prefill_k_block,
                               l1_budget=est, n_quantum=nq, max_subblock=4 if f32acc else 8)  # fmt: skip
        ds._prefill_programs[key] = prog
    ckc = ds.ckc
    fid = os.environ.get("GLM53_PF_MM_FID", "")  # prefill-only math fidelity of the block-float weights (P3; "" = the
    blockfloat = ds.weight.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)  # decode's ds.ckc)
    if (fid and blockfloat) or not f32acc:
        cache = ds.__dict__.setdefault("_pf_ckc", {})
        ck = (fid if blockfloat else "", f32acc)
        if ck not in cache:
            c0 = ds.ckc
            mf = getattr(ttnn.MathFidelity, ck[0]) if ck[0] else c0.math_fidelity
            cache[ck] = ttnn.init_device_compute_kernel_config(
                x.device().arch(), math_fidelity=mf, math_approx_mode=c0.math_approx_mode,
                fp32_dest_acc_en=c0.fp32_dest_acc_en and f32acc, packer_l1_acc=c0.packer_l1_acc)  # fmt: skip
        ckc = cache[ck]
    if out is not None and ds.n_pad == ds.n:  # straight into a persistent tensor (the PMK KDA store: no copy)
        return ttnn.linear(x, ds.weight, program_config=prog, memory_config=memory_config, compute_kernel_config=ckc,
                           dtype=dtype, optional_output_tensor=out)  # fmt: skip
    y = ttnn.linear(x, ds.weight, program_config=prog, memory_config=memory_config, compute_kernel_config=ckc,
                    dtype=dtype)  # fmt: skip
    if out is not None:  # (padded N: the sliced result is copied)
        y_ = ttnn.slice(y, [0] * len(y.shape), list(y.shape)[:-1] + [ds.n])
        ttnn.copy(y_, out)
        ttnn.deallocate(y)
        ttnn.deallocate(y_)
        return out
    if ds.n_pad != ds.n:
        y = ttnn.slice(y, [0] * len(y.shape), list(y.shape)[:-1] + [ds.n])
    return y


def swiglu_prefill(mlp, x):
    """OptimizedSwiGLU._prefill on pf_mm (column gate|up shard -> clamped SiLU-GLU -> row down shard)."""
    i = mlp.intermediate
    gu = pf_mm(mlp.ds_gate_up, x)
    shape = list(gu.shape)
    gate = ttnn.slice(gu, [0, 0, 0, 0], shape[:-1] + [i])
    up = ttnn.slice(gu, [0, 0, 0, i], shape)
    ttnn.deallocate(gu)
    act = ttnn.clamped_silu_glu(gate, up, mlp.limit)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    out = pf_mm(mlp.ds_down, act)
    ttnn.deallocate(act)
    return out


def kda_prefill_tp(kda, x, state, actual_end, valid_end_f, *, scan: str = "op", row_ids=None, snap=None):
    """OptimizedKDA.prefill (Stage 1 / 3's exact chunked path through slot 0 of `state`) for the TP prefill: the same
    resident weights (the chip's head shard), every KDA op told the chunk is not sequence-split (KDA_NO_SP), and the
    chunk scan as ONE recurrent_chunk_scan program (scan="op"; the per-chunk state decay = the exact fp32
    exp(sum g) of Stage 1, not the op's biased final_decay) or Stage 1's per-chunk op loop (scan="loop").
    x [1, 1, C, 4096] bf16 -> o_proj partial [1, 1, C, 4096] bf16. row_ids: [1, 1, >= C, 1] fp32 row positions (the
    KDA object's own table holds 2,048 rows: buckets past 2,048 pass TPPrefill.row_ids).
    snap = (snap_end [1] uint32 device scalar, cut scratch) (GLM53_PF_KDA_SNAPSHOT, one-replay store A): the state after
    the first snap_end rows -- one extra recurrent_chunk_scan with actual_end = snap_end over the SAME prepared inputs
    (its state == a prefill that ended at the cut, bitwise: doc/ttnn/12-fast-start/one_replay, d0 scan_cut_unit; its y
    rows past the cut are unspecified, so its y is dropped) and the conv history at snap_end (chronological_selections)
    -- written into the cut scratch; the served scan and every output are unchanged."""
    from models.autoports.zai_org_glm_5_3_flash.tt.kda import CHUNK, chunk_live_masks, keep_if_dead
    from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections

    c = int(x.shape[-2])
    h, d, width = kda.h, kda.d, kda.width
    qkv = pf_mm(kda.ds_qkv, x)
    lr = pf_mm(kda.ds_lowrank, x)
    f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, c, d))
    g_a = ttnn.slice(lr, (0, 0, 0, d), (1, 1, c, 2 * d))
    b = ttnn.slice(lr, (0, 0, 0, 2 * d), (1, 1, c, 2 * d + h))
    ttnn.deallocate(lr)
    gate_out = kda._pf("ds_gb", g_a, kda.w_gb, kda.ckc_fp32, BF16)
    g32 = kda._decay(f_a)
    g = ttnn.typecast(g32, BF16)
    beta = ttnn.sigmoid(ttnn.typecast(b, FP32))
    carry, s0 = kda.read_slot(state, 0)
    selections = ChronologicalSelections(
        ttnn.experimental.kda.chronological_selections(kda.zero_u32, KDA_NO_SP, c, h, d, d, actual_end=actual_end)
    )
    qkv_rm = ttnn.to_layout(ttnn.reshape(qkv, (1, c, 3 * width)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(qkv)
    new_carry = selections.select_local_final_history(qkv_rm, carry)
    carry_cut = None
    if snap is not None:  # the conv history (pre-conv rows) ending at the cut
        carry_cut = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(kda.zero_u32, KDA_NO_SP, c, h, d, d, actual_end=snap[0])
        ).select_local_final_history(qkv_rm, carry)
    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        qkv_rm, carry, *kda.taps_prefill, width, width, width, program_config=kda.conv_program,
        actual_start=kda.zero_u32, sequence_parallel_axis=KDA_NO_SP, predecessor_carry=carry, memory_config=DRAM)  # fmt: skip
    ttnn.deallocate(qkv_rm)
    n_chunks = c // CHUNK
    beta_by_chunk = ttnn.reshape(ttnn.permute(ttnn.reshape(beta, (1, c, h)), (0, 2, 1)), (h, n_chunks, CHUNK, 1))
    v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = ttnn.experimental.kda.prepare_chunk_recurrence(
        q, k, v, ttnn.reshape(g, (1, c, width)), beta_by_chunk, h, memory_config=DRAM,
        compute_kernel_config=kda.ckc_kda, output_bf16_mask=0, actual_start=kda.zero_u32, actual_end=actual_end,
        sequence_parallel_axis=KDA_NO_SP)  # fmt: skip
    ttnn.deallocate(final_decay)  # recomputed in fp32 (Stage 1: the op's value is biased for slow channels)
    for t_ in (q, k, v):
        ttnn.deallocate(t_)
    rows = ttnn.slice(kda.row_ids if row_ids is None else row_ids, (0, 0, 0, 0), (1, 1, c, 1))
    g_valid = ttnn.multiply(g32, ttnn.lt(rows, valid_end_f))  # padded rows are identity steps
    g_sum = ttnn.sum(ttnn.reshape(g_valid, (n_chunks, CHUNK, h * d)), dim=1, keepdim=True)  # [N, 1, HD]
    chunk_decay = ttnn.exp(ttnn.reshape(ttnn.permute(ttnn.reshape(g_sum, (n_chunks, h, d)), (1, 0, 2)),
                                        (h, n_chunks, d, 1)))  # fmt: skip
    state0 = ttnn.reshape(s0, (h, d, d))
    if snap is not None:  # the state at the cut: the plain scan over the same prepared inputs, stopped at snap_end
        y_cut, s_cut = ttnn.experimental.kda.recurrent_chunk_scan(
            v_beta, kd, q_decay, intra, k_dec_t, chunk_decay, t_inv, state0, actual_start=kda.zero_u32,
            tail_entry_states=state0, groups_per_head=1, memory_config=DRAM, compute_kernel_config=kda.ckc_fp32,
            sequence_parallel_axis=KDA_NO_SP, actual_end=snap[0])  # fmt: skip
        ttnn.deallocate(y_cut)  # rows past the cut unspecified; rows before it == the served scan's
        kda.write_slot(snap[1], 0, carry_cut, ttnn.reshape(s_cut, (1, h, d, d)))
    groups = 1
    if scan == "grouped":  # the largest G <= KDA_GROUPS dividing the chunk count with one summary owner per core
        cores = x.device().compute_with_storage_grid_size()
        cap = cores.x * cores.y
        groups = max(g for g in range(1, KDA_GROUPS + 1) if n_chunks % g == 0 and h * g <= cap)
    if groups > 1:
        # G groups of consecutive chunks per head scanned in parallel (the deepseek_v3_d_p grouped recurrence): each
        # group's affine summary (state -> a state + b), the exclusive scan of the group entry states, then ONE
        # recurrent_chunk_scan over H * G owners (8 heads per chip at TP8: 8x the parallel work of the plain scan)
        gc = n_chunks // groups
        hg = h * groups
        rs = lambda t_, a_, b_: ttnn.reshape(t_, (hg, gc, a_, b_))  # noqa: E731
        grouped = (rs(v_beta, CHUNK, d), rs(kd, CHUNK, d), rs(q_decay, CHUNK, d), rs(intra, CHUNK, CHUNK),
                   rs(k_dec_t, d, CHUNK), rs(chunk_decay, d, 1), rs(t_inv, CHUNK, CHUNK))  # fmt: skip
        grid = x.device().compute_with_storage_grid_size()
        owners = ttnn.num_cores_to_corerangeset(hg, grid, row_wise=True)
        # the affine summaries / entry states in DRAM by default: L1-sharded summaries (128 KB per owner core) put an
        # L1 buffer under summarize's CBs with the full stack's L1 floor (00:04Z pair mock of the L0-L4 TTFT job: CB end 1,283,296 > buffer at 1,243,648)
        l1 = os.environ.get("GLM53_PF_KDA_GROUPED_MEM", "dram") == "l1"
        summary_mem = ttnn.create_sharded_memory_config((hg, d, d), core_grid=owners,
                                                        strategy=ttnn.ShardStrategy.HEIGHT,
                                                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                        use_height_and_width_as_shard_shape=True) if l1 else DRAM  # fmt: skip
        a_, b_, ta, tb = ttnn.experimental.kda.summarize_chunk_recurrence(
            *grouped, groups_per_head=groups, memory_config=summary_mem, compute_kernel_config=kda.ckc_fp32,
            actual_start=kda.zero_u32, actual_end=actual_end, sequence_parallel_axis=KDA_NO_SP)  # fmt: skip
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        entry = ttnn.experimental.kda.affine_exclusive_scan(
            a_, b_, state0, groups, memory_config=ttnn.L1_MEMORY_CONFIG if l1 else DRAM, compute_kernel_config=kda.ckc_fp32,
            actual_start=kda.zero_u32, actual_end=actual_end, tail_a=a_, tail_b=b_, tail_entry_states=state0,
            local_rows=c, sequence_parallel_axis=KDA_NO_SP)  # fmt: skip
        ttnn.deallocate(a_)
        ttnn.deallocate(b_)
        y, s_all = ttnn.experimental.kda.recurrent_chunk_scan(
            *grouped, entry, actual_start=kda.zero_u32, tail_entry_states=state0, groups_per_head=groups,
            memory_config=DRAM, compute_kernel_config=kda.ckc_fp32, sequence_parallel_axis=KDA_NO_SP,
            actual_end=actual_end)  # fmt: skip
        ttnn.deallocate(entry)
        y = ttnn.reshape(y, (h, c, d))
        s_all = ttnn.reshape(s_all, (h, groups, d, d))
        final_state = ttnn.reshape(ttnn.slice(s_all, (0, groups - 1, 0, 0), (h, groups, d, d)), (1, h, d, d))
    elif scan in ("op", "grouped"):
        y, s_fin = ttnn.experimental.kda.recurrent_chunk_scan(
            v_beta, kd, q_decay, intra, k_dec_t, chunk_decay, t_inv, state0, actual_start=kda.zero_u32,
            tail_entry_states=state0, groups_per_head=1, memory_config=DRAM, compute_kernel_config=kda.ckc_fp32,
            sequence_parallel_axis=KDA_NO_SP, actual_end=actual_end)  # fmt: skip
        y = ttnn.reshape(y, (h, c, d))
        final_state = ttnn.reshape(s_fin, (1, h, d, d))
    else:
        mm = kda._mm
        live = chunk_live_masks(rows, valid_end_f, n_chunks)
        st = state0
        outs = []
        for i in range(n_chunks):
            sl = lambda t_, r_, cols: ttnn.reshape(
                ttnn.slice(t_, (0, i, 0, 0), (h, i + 1, r_, cols)), (h, r_, cols)
            )  # noqa: E731
            u = mm(sl(t_inv, CHUNK, CHUNK), ttnn.subtract(sl(v_beta, CHUNK, d), mm(sl(kd, CHUNK, d), st)))
            outs.append(ttnn.add(mm(sl(q_decay, CHUNK, d), st), mm(sl(intra, CHUNK, CHUNK), u)))
            new_state = ttnn.add(ttnn.multiply(st, sl(chunk_decay, d, 1)), mm(sl(k_dec_t, d, CHUNK), u))
            st = keep_if_dead(live[i], new_state, st)
        y = outs[0] if n_chunks == 1 else ttnn.concat(outs, dim=1)
        final_state = ttnn.reshape(st, (1, h, d, d))
    for t_ in (v_beta, kd, q_decay, intra, k_dec_t, t_inv, chunk_decay):
        ttnn.deallocate(t_)
    o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        y, ttnn.reshape(gate_out, (1, c, width)), kda.o_norm, h, epsilon=kda.eps, memory_config=DRAM,
        compute_kernel_config=kda.ckc_kda, output_dtype=BF16)  # fmt: skip
    out = pf_mm(kda.ds_o, ttnn.reshape(o, (1, 1, c, width)))
    if os.environ.get("GLM53_PF_KDA_PADMASK", "0") == "1":
        # Stage 12 (the cf3h-g 129-token defect, 2026-10-25): recurrent_chunk_scan writes y only up to actual_end, so the
        # padding rows' output is stale trace memory -- layout-dependent, non-finite in TP8 probes (padding rows of every
        # layer from L0) -- and any row-mixing step downstream can carry it into a valid row. A SELECT zeroes them
        # (x 0 would keep NaN); valid rows are bitwise unchanged.
        # the comparison in fp32 (row positions up to 1M), the 0 / 1 result cast to the output's dtype (exact; where()
        # refuses an fp32 condition with a bf16 tensor)
        keep = ttnn.repeat(
            ttnn.typecast(ttnn.lt(rows, valid_end_f), out.dtype), ttnn.Shape((1, 1, 1, int(out.shape[-1])))
        )
        masked = ttnn.where(keep, out, 0.0)
        ttnn.deallocate(keep)
        ttnn.deallocate(out)
        out = masked
    kda.write_slot(state, 0, new_carry, final_state)
    return out


def kda_prefill_packed(kda, x, states, masks, ends, *, scan: str = "op", carry_idx=None):
    """PACKED prefill (mixed ISL, coordinator 2026-10-22 03:15Z): x [1, 1, T, 4096] bf16 holds K FRESH requests back to
    back (segment k = rows [o_k, o_k + n_k), boundaries only in the device tensors masks[k] ([1, 1, T, 1] fp32 row mask)
    and ends[k] (uint32 [1] = o_k + n_k), written per replay); every other row is padding. The projections, gates and
    the output norm / o_proj run once over all rows; the causal conv, the chunk preparation and the chunk scan run once
    PER SEGMENT on its masked rows (other segments' qkv rows zeroed, beta 0 and log-decay 0 outside the segment: identity
    steps), from the segment scratch's state (zeroed by StateIO for a fresh request) -> the segment's rows of y, its final
    state and conv carry (the 3 rows ending at ends[k]) into states[k]. -> o_proj partial [1, 1, T, 4096] bf16.
    carry_idx (the continuation twin): per segment [1, 3] uint32 packed rows o_k - 3 .. o_k - 1 that receive its conv
    carry (a CONTINUATION segment's causal-conv history = its slot's last 3 rows, not the zeroed rows before o_k;
    0xFFFFFFFF = skip: fresh segments, and o_k = 0 where predecessor_carry already supplies it)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.kda import CHUNK
    from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections

    assert scan in ("op", "grouped"), f"packed KDA prefill: scan {scan} (op / grouped run as one recurrent_chunk_scan)"
    c = int(x.shape[-2])
    h, d, width = kda.h, kda.d, kda.width
    n_chunks = c // CHUNK
    qkv = pf_mm(kda.ds_qkv, x)
    lr = pf_mm(kda.ds_lowrank, x)
    f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, c, d))
    g_a = ttnn.slice(lr, (0, 0, 0, d), (1, 1, c, 2 * d))
    b = ttnn.slice(lr, (0, 0, 0, 2 * d), (1, 1, c, 2 * d + h))
    ttnn.deallocate(lr)
    gate_out = kda._pf("ds_gb", g_a, kda.w_gb, kda.ckc_fp32, BF16)
    g32 = kda._decay(f_a)
    beta = ttnn.sigmoid(ttnn.typecast(b, FP32))
    y_acc = None
    for si, (state, m, end) in enumerate(zip(states, masks, ends)):
        carry, s0 = kda.read_slot(state, 0)
        selections = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(kda.zero_u32, KDA_NO_SP, c, h, d, d, actual_end=end)
        )
        qkv_k = ttnn.multiply(qkv, ttnn.typecast(m, qkv.dtype))  # this segment's rows only
        qkv_rm = ttnn.to_layout(ttnn.reshape(qkv_k, (1, c, 3 * width)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(qkv_k)
        if carry_idx is not None:  # a continuation segment's conv history: its carry rows just before o_k
            rows_to(carry, carry_idx[si], qkv_rm)
        new_carry = selections.select_local_final_history(qkv_rm, carry)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_rm, carry, *kda.taps_prefill, width, width, width, program_config=kda.conv_program,
            actual_start=kda.zero_u32, sequence_parallel_axis=KDA_NO_SP, predecessor_carry=carry, memory_config=DRAM)  # fmt: skip
        ttnn.deallocate(qkv_rm)
        g32_k = ttnn.multiply(g32, m)  # identity (log-decay 0) outside the segment
        beta_k = ttnn.multiply(beta, m)  # no update outside the segment
        beta_by_chunk = ttnn.reshape(ttnn.permute(ttnn.reshape(beta_k, (1, c, h)), (0, 2, 1)), (h, n_chunks, CHUNK, 1))
        v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = ttnn.experimental.kda.prepare_chunk_recurrence(
            q, k, v, ttnn.reshape(ttnn.typecast(g32_k, BF16), (1, c, width)), beta_by_chunk, h, memory_config=DRAM,
            compute_kernel_config=kda.ckc_kda, output_bf16_mask=0, actual_start=kda.zero_u32, actual_end=end,
            sequence_parallel_axis=KDA_NO_SP)  # fmt: skip
        ttnn.deallocate(final_decay)
        for t_ in (q, k, v, beta_k, beta_by_chunk):
            ttnn.deallocate(t_)
        g_sum = ttnn.sum(ttnn.reshape(g32_k, (n_chunks, CHUNK, h * d)), dim=1, keepdim=True)
        ttnn.deallocate(g32_k)
        chunk_decay = ttnn.exp(ttnn.reshape(ttnn.permute(ttnn.reshape(g_sum, (n_chunks, h, d)), (1, 0, 2)),
                                            (h, n_chunks, d, 1)))  # fmt: skip
        ttnn.deallocate(g_sum)
        state0 = ttnn.reshape(s0, (h, d, d))
        y, s_fin = ttnn.experimental.kda.recurrent_chunk_scan(
            v_beta, kd, q_decay, intra, k_dec_t, chunk_decay, t_inv, state0, actual_start=kda.zero_u32,
            tail_entry_states=state0, groups_per_head=1, memory_config=DRAM, compute_kernel_config=kda.ckc_fp32,
            sequence_parallel_axis=KDA_NO_SP, actual_end=end)  # fmt: skip
        for t_ in (v_beta, kd, q_decay, intra, k_dec_t, t_inv, chunk_decay):
            ttnn.deallocate(t_)
        # the segment's rows of y by SELECTION (where), never by multiplying: the scan leaves the rows past actual_end
        # unwritten (any value, NaN included -- NaN x 0 = NaN corrupted every later segment: 04:12Z d0 row)
        y = ttnn.reshape(y, (h, c, d))
        m_y = ttnn.typecast(ttnn.reshape(m, (1, c, 1)), y.dtype)  # where: one dtype for the condition and values
        y_new = ttnn.where(m_y, y, 0.0 if y_acc is None else y_acc)
        ttnn.deallocate(m_y)
        ttnn.deallocate(y)
        if y_acc is not None:
            ttnn.deallocate(y_acc)
        y_acc = y_new
        kda.write_slot(state, 0, new_carry, ttnn.reshape(s_fin, (1, h, d, d)))
    for t_ in (qkv, g32, beta):
        ttnn.deallocate(t_)
    o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        y_acc, ttnn.reshape(gate_out, (1, c, width)), kda.o_norm, h, epsilon=kda.eps, memory_config=DRAM,
        compute_kernel_config=kda.ckc_kda, output_dtype=BF16)  # fmt: skip
    ttnn.deallocate(y_acc)
    return pf_mm(kda.ds_o, ttnn.reshape(o, (1, 1, c, width)))


def _mhc_mix(x, coef_a, coef_b=None, y=None):
    """kernels/prefill/mhc_post_pf_* over the flat residual x [1, 1, T, 16384] fp32 (stream i = columns [4096 i, ..)):
    y given -> post-mix (coef_a = post [T, 4], coef_b = comb [T, 16]) -> [1, 1, T, 16384] fp32; else pre-mix
    (coef_a = pre [T, 4]) -> [1, 1, T, 4096] fp32. Exact fp32 SFPU arithmetic (the composite ops')."""
    pre = y is None
    t = int(x.shape[-2])
    ct = HIDDEN // TILE
    units = (t // TILE) * ct
    dev = x.device()
    grid = dev.compute_with_storage_grid_size()
    ncores = max(1, min(units, grid.x * grid.y))
    per = -(-units // ncores)
    cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(a, b)) for a, b in cores])
    out = ttnn.empty((1, 1, t, HIDDEN if pre else NSTREAM * HIDDEN), FP32, ttnn.TILE_LAYOUT, dev, DRAM)
    ysrc = x if pre else y  # pre: y unused (any valid buffer)
    ypage = 2048 if ysrc.dtype == BF16 else 4096
    cb_b = coef_a if coef_b is None else coef_b
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (a, b) in enumerate(cores):
        u0 = min(i * per, units)
        n = min(per, units - u0)
        rr[a][b] = [ysrc.buffer_address(), x.buffer_address(), coef_a.buffer_address(), cb_b.buffer_address(), u0, n]
        rc[a][b] = [u0, n]
        rw[a][b] = [out.buffer_address(), u0, n]
    defs = [("MPP_PRE", "1")] if pre else []
    src = lambda name: os.path.join(KDIR, "prefill", f"mhc_post_pf_{name}.cpp")  # noqa: E731
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True)  # fmt: skip
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb_ in (1, 2) + ((0,) if ysrc.dtype == FP32 else ()):  # CB_X, CB_COEF (+ an fp32 y): no tf32 truncation
        modes[cb_] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct_, runtime_args=rt, defines=defs, config=cfg)  # fmt: skip
    kernels = [kd("reader", [ct, ypage], rr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
               kd("compute", [ct], rc, cc),
               kd("writer", [ct], rw, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
    cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[  # noqa: E731
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
    ncoef, nout = (4, 1) if pre else (20, 4)
    cbs = [cb(0, ysrc.dtype, ypage, 2), cb(1, FP32, 4096, 8), cb(2, FP32, 4096, ncoef), cb(3, FP32, 4096, 2),
           cb(4, FP32, 4096, 2 * nout)]  # fmt: skip
    io = [x, coef_a, out] if pre else [y, x, coef_a, coef_b, out]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def mhc_pre_prefill(x, pre):
    """h = sum_i pre_i * X_i of the flat residual x [1, 1, T, 16384] fp32 -> [1, 1, T, 4096] fp32 (one program)"""
    return _mhc_mix(x, pre)


def mhc_post_prefill(y, x, post, comb):
    """the composite post_streams + the stream concat as ONE program: y [1, 1, T, 4096] (bf16 / fp32), the flat residual
    x [1, 1, T, 16384] fp32, post [1, 1, T, 4] / comb [1, 1, T, 16] fp32 -> the new flat residual [1, 1, T, 16384] fp32
    = post_s * y + sum_i C[i][s] X_i (exact fp32 SFPU)"""
    return _mhc_mix(x, post, comb, y)


def mhc_coeffs(hc, x):
    """OptimizedCompositeHyperConnection.pre_streams' coefficients on the flat residual x [1, 1, T, 16384] fp32 (the fn
    projection, the RMS scale, the Sinkhorn split) -> (pre [1, 1, T, 4], post [1, 1, T, 4], comb [1, 1, T, 16]) fp32"""
    t = int(x.shape[-2])
    mixes = hc._project(x)
    sq = ttnn.square(x) if hc.ocfg is not None and hc.ocfg.mhc_prefill_ss == "square" else ttnn.multiply(x, x)
    ss = ttnn.sum(sq, dim=-1, keepdim=True)
    ttnn.deallocate(sq)
    inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / x.shape[-1]), hc.norm_eps))
    mixes = ttnn.multiply(mixes, inv)
    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes, hc.consts, hc.n, hc.iters,
                                                                            hc.hc_eps)  # fmt: skip
    r4 = lambda u: ttnn.reshape(u, (1, 1, t, u.shape[-1]))  # noqa: E731
    return r4(pre), r4(post), r4(comb)


class StateIO:
    """The request slot's KDA state <-> the per-layer prefill scratch, as two programs INSIDE the prefill trace
    (kernels/prefill/state_io.cpp: one core per KDA layer; the slot and the fresh-request flag come from `ctl`, a device
    tensor the host writes before the replay, so the trace is slot-agnostic and no host-dispatched state op remains on
    the request path)."""

    def __init__(self, tpp: "TPPrefill", seg: int = 0):
        """seg: a batched segment's KDA scratches (TPPrefill.seg_scratch) instead of the layers' own"""
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _pages, _rows

        self.tpp, self.mesh = tpp, tpp.mesh
        kv, b = tpp.kv, tpp.b
        self.layers = [li for li, pl in enumerate(tpp.layers) if pl.kind != SPARSE_DSA]
        # every state chunk, the prefix-cache snapshot pool's chunks included (virtual slots max_batch + k)
        self.chunk, self.n_chunks = int(b.chunk), len(kv.states)
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))][: len(self.layers)]
        assert len(cores) == len(self.layers), "one core per KDA layer"
        self.ctl = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                   mesh_mapper=tpp.repl)  # fmt: skip
        scr = lambda li: tpp.seg_scratch(tpp.layers[li], seg)  # noqa: E731
        self.n_s, self.n_c = _pages(scr(self.layers[0]).S), _pages(scr(self.layers[0]).C)
        self.rt = {}
        for (x, y), li in zip(cores, self.layers):
            sc = scr(li)
            assert (_pages(sc.S), _pages(sc.C)) == (self.n_s, self.n_c)
            row = [self.ctl.buffer_address(), sc.S.buffer_address(), sc.C.buffer_address()]
            for c in range(self.n_chunks):
                S, C = kv.states[c][li]
                assert _pages(S) == self.n_s * _rows(S) and _pages(C) == self.n_c, (li, c)
                row += [S.buffer_address(), C.buffer_address()]
            self.rt[(x, y)] = row
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
        self.io = [self.ctl]  # every state tensor is reached by address (persistent: stable across replays)

    def _run(self, mode: str):
        rt = ttnn.RuntimeArgs()
        for (x, y), row in self.rt.items():
            rt[x][y] = list(row)
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "prefill", "state_io.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
            compile_time_args=[self.n_s, self.n_c, self.chunk, self.n_chunks], runtime_args=rt,
            defines=[("SIO_CB", "0"), ({"in": "STATE_IN", "out": "STATE_OUT", "copy": "STATE_COPY"}[mode], "1")],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=2 * 4096 + 2 * 2048, core_ranges=self.cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=2 * 4096 + 2 * 2048)])  # fmt: skip
        ttnn.generic_op([self.ctl, self.ctl], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))

    def state_in(self):
        self._run("in")

    def state_out(self):
        self._run("out")

    def copy(self, src: int, dst: int):
        """slot dst <- slot src (either may be a snapshot pool entry's virtual slot), every KDA layer in one launch
        of one core per layer, OUTSIDE the trace (the prefix cache's trailing snapshots between decode steps and the
        decode-step prefill's attach). Plain DMA; compiled by warm() before any capture."""
        self.write(dst, False, src=src)
        self._run("copy")

    NONE = 0xFFFFFFFF

    def write(self, slot: int, fresh: bool, src: Optional[int] = None, snap: Optional[int] = None):
        """ctl: word 0 the slot (STATE_OUT writes it), 1 fresh (STATE_IN zeroes the scratch), 2 the slot STATE_IN reads
        (NONE: word 0; a prefix-cache attach reads the snapshot pool's virtual slot), 3 a second slot STATE_OUT writes
        (NONE: none; a prefix-cache snapshot entry's virtual slot)"""
        v = torch.zeros(1, 32, dtype=torch.int32)  # NONE = -1 here = 0xFFFFFFFF in the uint32 device tensor
        v[0, 0], v[0, 1] = int(slot), int(bool(fresh))
        v[0, 2] = -1 if src is None else int(src)
        v[0, 3] = -1 if snap is None else int(snap)
        for w in (0, 2, 3):  # the real rows: every live slot + the pool's exact entries (its last chunk may be short)
            assert int(v[0, w]) == -1 or 0 <= int(v[0, w]) < self.tpp.slot_rows, (w, int(v[0, w]), self.tpp.slot_rows)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=self.tpp.repl), self.ctl)  # fmt: skip


# moe_fused_swiglu's tested worker grid (its unit / perf tests pin 11x8): kgroups = 8 divides M_BLOCK * hn_pad at hidden
# 2048 (8 * 6); the device's full grid does not (the 17:52Z watcher assert: cb_h_slice stepped by 48 // 10 = 4 pages on
# a 6-page ring)
MFS_GRID = ttnn.CoreCoord(11, 8)


class EpMoe:
    """The chip's routed experts over a prefill chunk, EP (expert-parallel over the mesh), weights read in place:
      1. moe_plan (kernels/prefill/moe_plan.cpp, one core): per-chip exact routing plan -- counts / tile-aligned region
         starts of the chip's E_loc experts, dispatch row -> token, pair -> dispatch row;
      2. dispatch gather: x rows into the expert-sorted dispatch buffer (ttnn.embedding, multi-core);
      3. moe_fused_swiglu (one program, every local expert, device-resident counts: only the routed rows run, each
         expert's weights read once) on the decode stack's EP weights (bfp4 gate|up tile-interleaved, ND-sharded;
         gate_up_interleaved) with GLM's clamped SwiGLU (ClampedSiluGlu, limit 10);
      4. combine: each pair's output row gathered back, routing weight (0 for another chip's pair), summed over top-k.
    The partial (this chip's experts only) is summed over the mesh by the FFN reduce-scatter / all-reduce.
    xsl = (tp, gate_up, down) of the served stack's batched expert TP (every expert on every chip, chip k = slice k of
    its intermediate): all E experts are local (e_lo 0 on every chip), each at width I / TP, and the reduce-scatter
    sums the slices' partials instead of the experts'."""

    def __init__(self, tpp: "TPPrefill", moe, xsl=None):
        self.tpp, self.moe, self.mesh = tpp, moe, tpp.mesh
        ndev = tpp.ndev
        self.e_all = int(moe.num_experts)
        self.xsl = xsl is not None
        if self.xsl:
            assert int(xsl[0]) == ndev, ("XSL slices per chip", xsl[0], ndev)
            self.gate_up, self.down = list(xsl[1]), list(xsl[2])
            self.e_loc = self.e_all
        else:
            self.e_loc = self.e_all // ndev
            gu, dn = moe.gate_up, moe.down
            slots = getattr(gu, "slots", None)
            if slots is not None:  # EpList: slot i holds expert e_loc r + i on chip r
                self.gate_up, self.down = list(gu.slots), list(dn.slots)
            else:  # replicated experts (one chip): every expert
                self.gate_up, self.down = list(gu), list(dn)
        assert len(self.gate_up) == self.e_loc and self.gate_up[0] is not None, (len(self.gate_up), self.e_loc)
        self.ep = -(-self.e_loc // 32) * 32
        # moe_fused_swiglu splits each expert's width (down's rows: I, or I / TP for XSL) over the grid's columns, at
        # least one tile each: XSL at TP8 = 256 = 8 tiles -> 8 x 8 (03:37Z 2x4 mock: 8 tiles cannot split over 11)
        hid_t = int(self.down[0].shape[-2]) // TILE
        self.mfs_grid = ttnn.CoreCoord(min(MFS_GRID.x, hid_t), MFS_GRID.y)
        repl = tpp.repl
        rm = lambda v, mp: ttnn.from_torch(v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,  # noqa: E731
                                           memory_config=DRAM, mesh_mapper=mp)  # fmt: skip
        lo = torch.zeros(ndev, 32, dtype=torch.int32)
        lo[:, 0] = 0 if self.xsl else torch.arange(ndev, dtype=torch.int32) * self.e_loc
        self.lo = rm(lo, ttnn.ShardTensorToMesh(self.mesh, dim=0) if ndev > 1 else None)
        self.idx_table = rm(torch.arange(self.ep, dtype=torch.int32).reshape(1, -1), repl)
        # GLM53_PF_MOE_FID (P3, default HiFi2 = the plan's prefill policy): the bfloat4_b expert weights carry 3
        # mantissa bits, all of which LoFi's one fidelity phase already consumes (decode runs the experts at LoFi)
        fid = getattr(ttnn.MathFidelity, os.environ.get("GLM53_PF_MOE_FID", "") or "HiFi2")
        self.ckc = ttnn.init_device_compute_kernel_config(self.mesh.arch(), math_fidelity=fid, math_approx_mode=False,
                                                          fp32_dest_acc_en=False, packer_l1_acc=False)  # fmt: skip
        self.bufs = {}
        # v2 (GLM53_PF_MOE_V2=1): dispatch bounded by the rows the plan uses (kernels/prefill/moe_dispatch.cpp) and the
        # combine bounded by the local pairs (kernels/prefill/moe_combine_*), instead of O(C k) / O(CAP) row passes
        v2 = os.environ.get("GLM53_PF_MOE_V2", "0") == "1"
        self.v2_dispatch = v2 or os.environ.get("GLM53_PF_MOE_V2_DISPATCH", "0") == "1"
        self.v2_combine = v2 or os.environ.get("GLM53_PF_MOE_V2_COMBINE", "0") == "1"
        # XSL: every pair is this chip's -> the v2 combine (64-B face-row reads per pair, built to skip EP's 7/8 remote
        # pairs) costs 12.2 ms per TP8-chip chunk vs 2.97 ms for v1 (untilize + 8-KB row gather + weighted sum;
        # moe_xsl_bench 05:03Z); GLM53_PF_MOE_XSL_V2C=1 keeps v2
        if self.xsl and os.environ.get("GLM53_PF_MOE_XSL_V2C", "0") != "1":
            self.v2_combine = False
        self.v2 = self.v2_dispatch  # the plan with the rows-used word (moe_plan_v2.cpp)
        # GLM53_PF_MOE_XGATHER=1: no dispatch gather -- moe_fused_swiglu reads each dispatch row's activation from
        # the undispatched token rows through the plan's disp table (x_row_index; XSL: 16,384+ rows x 8 KB per chip)
        self.xgather = os.environ.get("GLM53_PF_MOE_XGATHER", "0") == "1"
        # GLM53_PF_MOE_COMBINE=rm (when the v2 combine is off): the gather + weighted top-k sum in ONE program over the
        # untilized expert outputs (kernels/prefill/moe_combine_rm_*: 8-KB row reads by dest, fp32 L1 accumulation,
        # tilized y) instead of v1's embedding copy [C k, H] + multiply + reduce passes; GLM53_PF_MOE_CRM_SPLIT = column
        # parts per 32-token chunk (1 | 2 | 4)
        # default rm under XSL (served-config TF checked, work_log 2026-10-21), v1 otherwise
        self.combine = os.environ.get("GLM53_PF_MOE_COMBINE", "") or ("rm" if self.xsl else "v1")
        # GLM53_PF_MOE_EP_V2C=0 (EP layers, e.g. the served sparse_dsa layers: no XSL slices): the rm combine instead of
        # v2's local-pair combine (TP2 L3 n=512, real FFN input: 9.00 vs 10.01 ms per MoE, PCC 0.999996; v1 dispatch +
        # rm 9.46, v2 dispatch + v1 combine 9.53 -- moe_ep_combine_ab); default 1 = v2's combine
        if not self.xsl and self.v2_combine and os.environ.get("GLM53_PF_MOE_EP_V2C", "1") == "0":
            self.v2_combine = False
            self.combine = os.environ.get("GLM53_PF_MOE_COMBINE", "") or "rm"
        self.crm_split = int(os.environ.get("GLM53_PF_MOE_CRM_SPLIT", "2"))
        # GLM53_PF_MOE_PLAN=mc: the routing plan on P cores in 3 programs (kernels/prefill/moe_plan_mc_*: chunk
        # histograms -> region layout + dest per chunk -> disp per dispatch-row range; the same tables as moe_plan_v2's
        # one core, whose CB also stops fitting L1 at 8 x 2,048 rows); GLM53_PF_MOE_PLAN_CORES = P (64)
        # GLM53_PF_MOE_FFN=xffn (XSL only): the routed experts one per core at a time (kernels/prefill/moe_xffn_*: gate|up
        # over K chunks, clamped SwiGLU, down untilized straight into the ROW_MAJOR expert output) instead of
        # moe_fused_swiglu's whole-grid split per expert (sync-bound for XSL's 8-tile-wide experts); implies the rm
        # combine (no untilize pass). GLM53_PF_MOE_XFFN_MG = tile rows per M group (4)
        # default: xffn for XSL slices <= 8 tiles wide (the TP8 geometry: MoE -46 % per 2,048-row chunk, PCC 0.99998;
        # TP2 served-config TF checked with xffn forced); wider slices (TP2 / TP4) re-read weights per tile row -> mfs
        narrow = self.xsl and int(self.down[0].shape[-2]) <= 8 * TILE
        self.ffn = os.environ.get("GLM53_PF_MOE_FFN", "") or ("xffn" if narrow else "mfs")
        self.xffn_mg = int(os.environ.get("GLM53_PF_MOE_XFFN_MG", "4"))
        self.xffn_xsplit = os.environ.get("GLM53_PF_MOE_XFFN_XSPLIT", "0") == "1"
        self.xffn_snake = os.environ.get("GLM53_PF_MOE_XFFN_SNAKE", "0") == "1"
        self.xffn_valid = os.environ.get("GLM53_PF_MOE_XFFN_VALID", "0") == "1"
        if self.ffn == "xffn":
            if self.xsl:
                self.combine = "rm"
            elif (
                os.environ.get("GLM53_PF_MOE_XFFN_EP", "0") == "1"
            ):  # the L3 accuracy fix candidate (2026-10-24): EP through
                # xffn too -- moe_fused_swiglu inflates every expert row by ~1.04; other-chip pairs weigh 0 in the rm
                # combine (w x local) and the expert rows are zero-filled first (no unwritten row can carry a NaN)
                self.combine = "rm"
            else:  # EP's other-chip pairs point at row 0 (unwritten by xffn): such layers keep moe_fused_swiglu
                self.ffn = "mfs"
        # GLM53_PF_MOE_RM_SKIP=1 (W8 unit 1, coordinator 2026-10-25 13:44:19Z): the rm combine reads no DRAM row for a
        # weight-0 pair (EP's other-chip pairs, all pointing at dispatch row 0: one bank) and EP's xffn output is not
        # zero-filled first (no unwritten row is read). Bitwise. W6 TP8 DSA layers: combine 2.03 ms vs 0.42 on XSL, fill
        # 0.46 ms
        self.rm_skip = os.environ.get("GLM53_PF_MOE_RM_SKIP", "0") == "1"
        self.plan_mode = os.environ.get(
            "GLM53_PF_MOE_PLAN", "mc2"
        )  # mc / mc2: the same tables as v2 (XSL and EP), on P cores
        self.plan_cores = int(os.environ.get("GLM53_PF_MOE_PLAN_CORES", "64"))
        # GLM53_PF_MOE_GROUP_ROWS (default 2,048; 0 = off): more rows than this run the routed MoE in row groups of that
        # many rows INSIDE the same replay (coordinator 2026-10-22 18:39Z: 9's 8,192 bucket OOMed on the [cap, 4096]
        # dispatch / fused-output buffers, 546 MB each; per group they are the 2,048 bucket's). Per-row math unchanged.
        self.group_rows = int(os.environ.get("GLM53_PF_MOE_GROUP_ROWS", "2048") or 0)

    def _bufs(self, c: int):
        if c not in self.bufs:
            k = int(self.moe.top_k)
            n = c * k
            cap = n + 32 * self.e_loc
            cap = -(-cap // 32) * 32
            z = lambda w: ttnn.from_torch(torch.zeros(1, w, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                          layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                          mesh_mapper=self.tpp.repl)  # fmt: skip
            self.bufs[c] = dict(n=n, cap=cap, disp=z(cap), dest=z(n), cnt=z(self.ep), start=z(self.ep))
        return self.bufs[c]

    def plan(self, ids_rm, c: int):
        if self.plan_mode in ("mc", "mc2"):
            return self._plan_mc(ids_rm, c)
        b = self._bufs(c)
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [ids_rm.buffer_address(), self.lo.buffer_address(), b["disp"].buffer_address(),
                    b["dest"].buffer_address(), b["cnt"].buffer_address(), b["start"].buffer_address()]  # fmt: skip
        if self.v2:
            if "meta" not in b:
                b["meta"] = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                            mesh_mapper=self.tpp.repl)  # fmt: skip
            rt[0][0] = list(rt[0][0]) + [b["meta"].buffer_address()]
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "prefill", "moe_plan_v2.cpp" if self.v2 else "moe_plan.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core,
            compile_time_args=[b["n"], c, self.e_loc, b["cap"], self.ep], runtime_args=rt,
            defines=[("PLAN_CB", "0")],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        words = 2 * b["n"] + b["cap"] + 3 * self.ep + 32
        cb = ttnn.CBDescriptor(total_size=words * 4, core_ranges=core, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=words * 4)])  # fmt: skip
        io = [ids_rm, self.lo, b["disp"], b["dest"], b["cnt"], b["start"]] + ([b["meta"]] if self.v2 else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))
        return b

    def _plan_mc(self, ids_rm, c: int):
        """the routing plan on P cores, 3 programs (kernels/prefill/moe_plan_mc_{count,dest,disp}.cpp): the same
        disp / dest / cnt / start / meta tables as moe_plan_v2.cpp. plan_mode "mc2": the dest program also stages each
        chunk's expert-sorted tokens and the disp table is assembled from those runs (moe_plan_mc_runs.cpp: O(P E / P)
        per core) instead of every core scanning all N pairs"""
        b = self._bufs(c)
        n, cap, ep = b["n"], b["cap"], self.ep
        cores, crs, _ = self._cores(self.plan_cores)
        P = len(cores)
        r16 = lambda v: -(-v // 16) * 16  # noqa: E731
        chunk, dchunk = r16(-(-n // P)), r16(-(-cap // P))
        blk = 1024 if n % 1024 == 0 else 256
        assert n % 256 == 0 and ep % 16 == 0, (n, ep)
        mk = lambda shape: ttnn.from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                           mesh_mapper=self.tpp.repl)  # fmt: skip
        if "meta" not in b:
            b["meta"] = mk((1, 32))
        if b.get("hist") is None or int(b["hist"].shape[0]) != P:
            b["hist"] = mk((P, ep))
        dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                               noc=ttnn.NOC.RISCV_0_default)  # fmt: skip

        runs = self.plan_mode == "mc2"
        if runs and (b.get("stage") is None or int(b["stage"].shape[0]) != P):
            b["stage"] = mk((P, chunk))

        def prog(name, ct, rts, words, io, defines=()):
            rt = ttnn.RuntimeArgs()
            for k_, (a, b_) in enumerate(cores):
                rt[a][b_] = rts(k_)
            kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "prefill", f"moe_plan_mc_{name}.cpp"),
                                       source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
                                       compile_time_args=ct, runtime_args=rt,
                                       defines=[("PLAN_CB", "0")] + list(defines), config=dm)  # fmt: skip
            cb = ttnn.CBDescriptor(total_size=words * 4, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=words * 4)])  # fmt: skip
            ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))

        ids, lo, hist = ids_rm.buffer_address(), self.lo.buffer_address(), b["hist"].buffer_address()
        prog("count", [n, self.e_loc, ep, chunk], lambda k_: [ids, lo, hist, k_], chunk + ep + 32,
             [ids_rm, self.lo, b["hist"]])  # fmt: skip
        stage = b["stage"].buffer_address() if runs else 0
        prog("dest", [n, self.e_loc, ep, chunk, P, c],
             lambda k_: [ids, lo, hist, b["dest"].buffer_address(), b["cnt"].buffer_address(),
                         b["start"].buffer_address(), b["meta"].buffer_address(), k_, stage],
             P * ep + 3 * chunk + 4 * ep + 32,
             [ids_rm, self.lo, b["hist"], b["dest"], b["cnt"], b["start"], b["meta"]] + ([b["stage"]] if runs else []),
             defines=[("PLAN_SORTED", "1")] if runs else [])  # fmt: skip
        if runs:
            epc = -(-self.e_loc // P)
            rmax = r16(epc * c)
            # GLM53_PF_MOE_PLAN_WIN (words; Stage 14 W1's op arm at n x 2,048 rows): the windowed runs form of
            # moe_plan_mc_runs.cpp (RMAX < EPC C) -- the plan's L1 stops growing with c (32,768 rows: 2 x 163,840 words)
            win = int(os.environ.get("GLM53_PF_MOE_PLAN_WIN", "0") or 0)
            if win > 0 and rmax > r16(win):
                rmax = max(r16(win), chunk)
            prog("runs", [self.e_loc, ep, chunk, P, c, cap, epc, rmax],
                 lambda k_: [hist, b["cnt"].buffer_address(), b["start"].buffer_address(), b["meta"].buffer_address(),
                             b["stage"].buffer_address(), b["disp"].buffer_address(), k_],
                 P * ep + 2 * ep + 32 + 2 * rmax + 32 * P + 3 * P,
                 [b["hist"], b["cnt"], b["start"], b["meta"], b["stage"], b["disp"]])  # fmt: skip
        else:
            prog("disp", [n, c, self.e_loc, cap, dchunk, blk],
                 lambda k_: [ids, lo, b["dest"].buffer_address(), b["disp"].buffer_address(), k_],
                 dchunk + 4 * blk + 32, [ids_rm, self.lo, b["dest"], b["disp"]])  # fmt: skip
        return b

    def _xffn_full_height_shards(self) -> bool:
        """every expert weight ND-sharded as [full height, 2 tiles] shards (fused_decoder._nd_dram(rows, 64)): a shard's
        pages are then contiguous in one bank in (k, n % 2) order (moe_fused_swiglu_bank_runs.hpp)"""
        if os.environ.get("GLM53_PF_MOE_XFFN_BASES", "1") != "1":
            return False
        try:
            for t_, rows in (
                (self.gate_up[0], int(self.gate_up[0].shape[-2])),
                (self.down[0], int(self.down[0].shape[-2])),
            ):
                spec = t_.memory_config().nd_shard_spec
                if spec is None or list(spec.shard_shape)[-2:] != [rows, 2 * TILE]:
                    return False
            return True
        except Exception:  # noqa: BLE001  (no ND spec API / interleaved weights: the TensorAccessor path)
            return False

    def _xffn(self, x_ext, b, out_rm, c: int):
        """XSL expert FFN, one expert per core at a time (kernels/prefill/moe_xffn_*): out_rm [CAP, H] bf16 ROW_MAJOR
        rows of every expert region (valid rows only) from x_ext [C + 32, H] bf16 ROW_MAJOR through the plan's disp"""
        h = int(self.moe.hidden)
        h_t, w_t = h // TILE, int(self.down[0].shape[-2]) // TILE
        subw, wtb = 4, 576

        # GLM53_PF_MOE_XFFN_WD_SLOTS (default 2; the EP path -- GLM53_PF_MOE_XFFN_EP, full-width experts -- 1): the down
        # weight ring's slots; one slot frees W x NC bfp4 tiles (147 KB at W = 64, NC = 4) so xffn-EP fits the served
        # L1 (14b 21:31:08Z: +110 KB over with two); the kernels touch CB_WD through the CB API only
        wd_slots = int(os.environ.get("GLM53_PF_MOE_XFFN_WD_SLOTS", "") or (2 if self.xsl else 1))
        self.xffn_wd_slots = wd_slots

        def l1(mg_, kc_, nc_):  # the CB bytes of one core
            return (2 * kc_ * 2 * w_t * wtb + 3 * mg_ * kc_ * 2048 + wd_slots * w_t * nc_ * wtb + mg_ * 2 * w_t * 4096
                    + mg_ * w_t * 2048 + 2 * nc_ * 2048 + 16384)  # fmt: skip

        # the widest blocks within ~900 KB of CBs (the CB region then ends <= 308,352 + 921,600 = 1,229,952, under the
        # prefill's L1 ceiling; TP8 XSL, W = 8 tiles: 4 / 8 / 8 = 794,624 B; the pair's W = 32: 1 / 4 / 4)
        mg, kc, nc = next(((m_, k_, n_) for m_, k_, n_ in ((4, 8, 8), (2, 8, 8), (2, 4, 8), (1, 4, 8), (1, 4, 4),
                                                         (1, 2, 4), (1, 1, 4))
                           if m_ <= self.xffn_mg and h_t % k_ == 0 and h_t % n_ == 0 and l1(m_, k_, n_) <= 900 * 1024),
                          (1, 1, 4))  # fmt: skip
        # GLM53_PF_MOE_XFFN_BLOCKS="mg,kc,nc": explicit blocks (A/B of the transaction sizes: the x gather moves KC 64-B
        # row segments per read, the output rows NC 64-B segments per write), within the same 900 KB CB budget
        # The explicit blocks apply where they fit that budget: the XSL layers' narrow experts (W 256 at TP8, W8 lever:
        # 4,8,16 = 901 KB); an EP layer's full-width experts (W 2,048) keep the default selection (cf3h-j 2026-10-25:
        # 4,8,16 there = 4.67 MB) -- recorded in self.xffn_blocks_skip
        blk = os.environ.get("GLM53_PF_MOE_XFFN_BLOCKS", "")
        if blk:
            bm, bk, bn = (int(v) for v in blk.split(","))
            if l1(bm, bk, bn) <= 900 * 1024 and bn % 4 == 0 and h_t % bk == 0 and h_t % bn == 0:
                mg, kc, nc = bm, bk, bn
            else:
                self.xffn_blocks_skip = (blk, l1(bm, bk, bn), (mg, kc, nc))
        assert int(self.gate_up[0].shape[-1]) == 2 * w_t * TILE and h_t % kc == 0 and h_t % nc == 0, (h_t, w_t)
        self.xffn_cfg = (mg, kc, nc)
        if getattr(self, "xffn_tbl", None) is None:  # the experts' weight buffers are static: one address table
            addrs = [
                int(a_) for g, d in zip(self.gate_up, self.down) for a_ in (g.buffer_address(), d.buffer_address())
            ]
            assert max(addrs) < 2**32, max(addrs)
            tbl = torch.tensor([a_ - 2**32 if a_ >= 2**31 else a_ for a_ in addrs], dtype=torch.int32).reshape(
                1, -1
            )
            self.xffn_tbl = ttnn.from_torch(tbl, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                            memory_config=DRAM, mesh_mapper=self.tpp.repl)  # fmt: skip
        grid = self.mesh.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        P = gx * gy
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        acc = lambda t_: list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())  # noqa: E731
        ct_r = [h_t, w_t, kc, nc, mg, self.e_loc, self.ep, P, wtb] + acc(self.gate_up[0]) + acc(self.down[0])
        # GLM53_PF_MOE_XFFN_PROBE=1: in-program probe counters (kernels/prefill/moe_xffn_probe.hpp) into
        # self.xffn_probe [2 P, 16] uint32 per chip (read with xffn_probe_read); the probe args are always passed
        # (address 0 without the probe): a fixed runtime-arg count
        probe = os.environ.get("GLM53_PF_MOE_XFFN_PROBE", "0") == "1"
        if probe and getattr(self, "xffn_probe", None) is None:
            self.xffn_probe = ttnn.from_torch(torch.zeros(2 * P, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
                                              mesh_mapper=self.tpp.repl)  # fmt: skip
        pa = self.xffn_probe.buffer_address() if probe else 0
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for y_ in range(gy):
            for x_ in range(gx):
                rr[x_][y_] = [x_ext.buffer_address(), b["disp"].buffer_address(), b["cnt"].buffer_address(),
                              b["start"].buffer_address(), self.xffn_tbl.buffer_address(), y_ * gx + x_, b["cap"], pa]  # fmt: skip
                rw[x_][y_] = [out_rm.buffer_address(), x_ext.buffer_address(), b["disp"].buffer_address(), b["cap"],
                              pa, P, y_ * gx + x_]  # fmt: skip
        src = lambda name: os.path.join(KDIR, "prefill", f"moe_xffn_{name}.cpp")  # noqa: E731
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        fid = getattr(ttnn.MathFidelity, os.environ.get("GLM53_PF_MOE_FID", "") or "HiFi2")
        # GLM53_PF_MOE_XFFN_FULLSYNC=1: full-sync DEST (8 fp32 tiles) -> gate|up subblocks of 8 (half the calls)
        full = os.environ.get("GLM53_PF_MOE_XFFN_FULLSYNC", "0") == "1"
        suba = 8 if full and (2 * w_t) % 8 == 0 else subw
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                          dst_full_sync_en=full)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[16] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_GUP: the fp32 partials copied exactly
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        # GLM53_PF_MOE_XFFN_DIAG (timing diagnostics, wrong outputs): "nomath" = data movement only, "nodown" = no
        # down phase (gate|up + act), "nomath_nodown" = gate|up data movement only
        diag = os.environ.get("GLM53_PF_MOE_XFFN_DIAG", "")
        defs = ([("XFFN_NO_MATH", "1")] if "nomath" in diag else []) + (
            [("XFFN_NO_DOWN", "1")] if "nodown" in diag else []
        )
        if self._xffn_full_height_shards():  # per-shard base addresses instead of TensorAccessor math per read
            defs.append(("XFFN_SHARD_BASES", "1"))
            if "coalesce" in diag:  # one read per shard chunk into a shard-major CB: data-movement timing only
                assert "nomath" in diag, "XFFN_COALESCE has no compute indexing: diagnostics with nomath only"
                defs.append(("XFFN_COALESCE", "1"))
        # GLM53_PF_MOE_XFFN_XNOC1=1: the x-row gather on the writer RISC (NoC1), the weights on the reader (NoC0)
        xnoc1 = os.environ.get("GLM53_PF_MOE_XFFN_XNOC1", "1") == "1"
        if xnoc1:
            defs.append(("XFFN_X_ON_NOC1", "1"))
        # GLM53_PF_MOE_XFFN_XSPLIT=1 (with the NoC1 gather): the x gather of the even K chunks on the reader (NoC0, CB_XRM)
        # and of the odd ones on the writer (NoC1, CB_XRM1) -- the gather is the gate|up phase's critical path (probe
        # 2026-10-22); the two CBs share the double buffer's bytes. GLM53_PF_MOE_XFFN_SNAKE=1: units dealt by cost class
        # over the cores (moe_xffn_reader.cpp). GLM53_PF_MOE_XFFN_VALID=1: only valid rows gathered, x_ext unpadded
        xsplit = xnoc1 and self.xffn_xsplit
        if xsplit:
            defs.append(("XFFN_X_SPLIT", "1"))
        if self.xffn_snake:
            defs.append(("XFFN_SNAKE", "1"))
        if self.xffn_valid:
            defs.append(("XFFN_VALID_ROWS", "1"))
        if probe:
            defs.append(("XFFN_PROBE", "1"))
        kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct_, runtime_args=rt, defines=defs, config=cfg)  # fmt: skip
        empty_rt = ttnn.RuntimeArgs()
        for y_ in range(gy):
            for x_ in range(gx):
                empty_rt[x_][y_] = []
        kernels = [kd("reader", ct_r, rr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
                   kd("compute", [h_t, w_t, kc, nc, mg, subw, suba], empty_rt, cc),
                   kd("writer", [h_t, nc, kc, mg], rw, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
        cb = lambda i, dt, page, n_: ttnn.CBDescriptor(total_size=page * n_, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        bfp4 = ttnn.bfloat4_b
        tab_words = 2 * self.ep + (2 * self.e_loc + 15) // 16 * 16 + mg * 32 + 16 + 2 * (w_t + h_t // 2)
        cbs = [cb(0, ttnn.uint32, 32, 2), cb(1, bfp4, wtb, 2 * kc * 2 * w_t), cb(2, BF16, 2048, (1 if xsplit else 2) * mg * kc),
               cb(3, BF16, 2048, mg * kc), cb(4, bfp4, wtb, self.xffn_wd_slots * w_t * nc), cb(5, ttnn.uint32, 32, 2),
               cb(6, ttnn.uint32, tab_words * 4, 1), cb(16, FP32, 4096, mg * 2 * w_t), cb(17, BF16, 2048, mg * w_t),
               cb(18, BF16, 2048, 2 * nc), cb(7, ttnn.uint32, mg * 32 * 4, 1)]  # fmt: skip
        if xsplit:
            cbs.append(cb(8, BF16, 2048, mg * kc))
        io = [x_ext, b["disp"], b["cnt"], b["start"], self.xffn_tbl, out_rm] + list(self.gate_up) + list(self.down)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def xffn_probe_read(self) -> dict:
        """the xffn probe counters of the last run (GLM53_PF_MOE_XFFN_PROBE=1), per chip: {"reader": [P, 16],
        "writer": [P, 16]} int64 cycles (kernels/prefill/moe_xffn_probe.hpp slots)"""
        t = ttnn.to_torch(self.xffn_probe, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh, 0)) if self.tpp.ndev > 1 \
            else ttnn.to_torch(self.xffn_probe)  # fmt: skip
        t = t.to(torch.int64) & 0xFFFFFFFF
        p2 = int(self.xffn_probe.shape[0])
        return [dict(reader=t[i * p2 : i * p2 + p2 // 2], writer=t[i * p2 + p2 // 2 : (i + 1) * p2])
                for i in range(t.shape[0] // p2)]  # fmt: skip

    def _cores(self, n_units: int, cap: int = 0):
        grid = self.mesh.compute_with_storage_grid_size()
        ncores = max(1, min(n_units, cap or grid.x * grid.y, grid.x * grid.y))
        per = -(-n_units // ncores)
        cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b_), ttnn.CoreCoord(a, b_)) for a, b_ in cores])
        return cores, crs, per

    def _dispatch(self, x_ext, b, xd, row_bytes: int):
        """xd[d] = x_ext[disp[d]] for d < rows used (kernels/prefill/moe_dispatch.cpp)"""
        cap, depth = b["cap"], 4
        cores, crs, per = self._cores(cap, 64)
        rt = ttnn.RuntimeArgs()
        for i, (a, b_) in enumerate(cores):
            r0 = min(i * per, cap)
            rt[a][b_] = [x_ext.buffer_address(), b["disp"].buffer_address(), b["meta"].buffer_address(),
                         xd.buffer_address(), r0, min(per, cap - r0)]  # fmt: skip
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "prefill", "moe_dispatch.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=[row_bytes, cap, depth], runtime_args=rt, defines=[("DISP_CB", "0")],
            config=ttnn.ReaderConfigDescriptor())  # fmt: skip
        size = (cap * 4 + 63) // 64 * 64 + 64 + depth * row_bytes
        cb = ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])  # fmt: skip
        ttnn.generic_op(
            [x_ext, b["disp"], b["meta"], xd], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])
        )

    def _combine(self, out, dest, wt, y, c: int, k: int, h: int):
        """y[t] = sum_j wt[j C + t] * out[dest[j C + t]] (kernels/prefill/moe_combine_*): local pairs only"""
        ctc = h // TILE
        units = (c // TILE) * ctc
        cores, crs, per = self._cores(units)
        rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (a, b_) in enumerate(cores):
            u0 = min(i * per, units)
            nn = min(per, units - u0)
            rr[a][b_] = [out.buffer_address(), dest.buffer_address(), wt.buffer_address(), u0, nn]
            rc[a][b_] = [u0, nn]
            rw[a][b_] = [y.buffer_address(), u0, nn]
        src = lambda name: os.path.join(KDIR, "prefill", f"moe_combine_{name}.cpp")  # noqa: E731
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_W: fp32 weights (no tf32 truncation)
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct_, runtime_args=rt, config=cfg)  # fmt: skip
        kernels = [kd("reader", [c, k, ctc], rr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
                   kd("compute", [k, ctc], rc, cc),
                   kd("writer", [ctc], rw, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
        cb = lambda i, dt, page, n_: ttnn.CBDescriptor(total_size=page * n_, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        idx_page = 2 * k * 32 * 4 + 64 * 64
        cbs = [cb(0, BF16, 2048, 2 * k), cb(1, FP32, 4096, k), cb(2, ttnn.uint32, idx_page, 1), cb(3, BF16, 2048, 2)]
        ttnn.generic_op([out, dest, wt, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def _combine_rm(self, out_rm, dest, wt, y, c: int, k: int, h: int):
        """y[t] = sum_j wt[t, j] * out_rm[dest[j C + t]] (kernels/prefill/moe_combine_rm_*): out_rm [CAP, H] bf16
        ROW_MAJOR, wt [1, 1, C, k] fp32 TILE (0 for another chip's pair; read in place, one tile per 32 tokens), y
        [1, 1, C, H] bf16 TILE"""
        hs = self.crm_split
        part = h * 2 // hs
        pt = part // 2048
        assert pt * 2048 == part and 1 <= pt <= 4 and c % TILE == 0, (h, hs, c)
        units = (c // TILE) * hs
        cores, crs, per = self._cores(units)
        rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (a, b_) in enumerate(cores):
            u0 = min(i * per, units)
            nn = min(per, units - u0)
            rr[a][b_] = [out_rm.buffer_address(), dest.buffer_address(), wt.buffer_address(), u0, nn]
            rc[a][b_] = [nn]
            rw[a][b_] = [y.buffer_address(), u0, nn]
        src = lambda name: os.path.join(KDIR, "prefill", f"moe_combine_rm_{name}.cpp")  # noqa: E731
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        kd = lambda name, ct_, rt, cfg, defs=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct_, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
        skip = bool(getattr(self, "rm_skip", False))  # GLM53_PF_MOE_RM_SKIP: weight-0 pairs read no DRAM row
        kernels = [kd("reader", [c, k, h * 2, part, pt, hs], rr,
                      dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
                      [("RM_SKIP_ZERO_W", "1")] if skip else ()),
                   kd("compute", [k, pt], rc, cc),
                   kd("writer", [pt, h // TILE, hs], rw,
                      dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
        cb = lambda i, dt, page, n_: ttnn.CBDescriptor(total_size=page * n_, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        cbs = [cb(0, BF16, 2048, 2 * k * pt), cb(1, FP32, 4096, 2 * k), cb(2, ttnn.uint32, k * 32 * 4 + 4096, 1),
               cb(16, BF16, 2048, 32 * pt), cb(17, FP32, 4096, 32 * pt)]  # fmt: skip
        if skip:  # the zero part (moe_combine_rm_reader.cpp RM_SKIP_ZERO_W)
            cbs.append(cb(3, BF16, 2048, pt))
        ttnn.generic_op([out_rm, dest, wt, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def __call__(self, x, idx, sel, upto=None):
        """x [1, 1, C, 4096] bf16, idx [1, 1, C, k] uint16, sel [1, 1, C, k] fp32 -> this chip's routed partial [1, 1, C,
        4096] bf16; C > GLM53_PF_MOE_GROUP_ROWS (a multiple of it): row groups of that size, each the whole routed MoE
        (_call_rows: plan -> dispatch -> SwiGLU -> combine) on its rows, concatenated -- the peak buffers are one group's
        """
        c = int(x.shape[-2])
        gr = self.group_rows
        if upto is not None or gr <= 0 or c <= gr or c % gr:
            return self._call_rows(x, idx, sel, upto)
        h, k = int(x.shape[-1]), int(idx.shape[-1])
        ys = []
        for r0 in range(0, c, gr):
            parts = [ttnn.slice(t_, (0, 0, r0, 0), (1, 1, r0 + gr, int(t_.shape[-1]))) for t_ in (x, idx, sel)]
            ys.append(self._call_rows(*parts))
            for t_ in parts:
                ttnn.deallocate(t_)
        y = ttnn.concat(ys, dim=2, memory_config=DRAM)
        for t_ in ys:
            ttnn.deallocate(t_)
        assert tuple(y.shape)[-2:] == (c, h), (tuple(y.shape), c, h, k)
        return y

    def _call_rows(self, x, idx, sel, upto=None):
        """x [1, 1, C, 4096] bf16 (every row), idx [1, 1, C, k] uint16 (global ids), sel [1, 1, C, k] fp32 (each pair's
        unbiased router score) -> this chip's routed partial [1, 1, C, 4096] bf16. v1: the embedding dispatch over CAP
        rows + the C k-row combine; v2 halves (each selectable, for the hang diagnosis): the dispatch bounded by the rows
        the plan uses (moe_dispatch.cpp), the combine bounded by the local pairs (moe_combine_*). upto ("plan" |
        "dispatch" | "swiglu"): stop after that phase and return None (moe_xsl_bench's per-phase timing)."""
        moe = self.moe
        c, h, k = int(x.shape[-2]), int(moe.hidden), int(moe.top_k)
        n = c * k
        # pair order p = j * C + t (slot-major): ids [1, n] uint32 ROW_MAJOR
        ids_t = ttnn.transpose(ttnn.typecast(idx, ttnn.uint32), -2, -1)  # [1, 1, k, C]
        ids_rm = ttnn.to_layout(ttnn.reshape(ids_t, (1, n)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(ids_t)
        b = self.plan(ids_rm, c)
        ttnn.deallocate(ids_rm)
        if upto == "plan":
            return None
        # routing weights per pair (fused_experts' normalisation x routed_scaling_factor) and the local mask
        w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
        w = ttnn.multiply(w, moe.scaling)
        # EP: a pair is this chip's if its expert is in the chip's range; XSL: every pair is (each chip holds a slice of
        # every expert -- an EP mask here dropped the other ranges' slices: L4 PCC 0.997435 vs EP 0.999813, 03:42Z)
        if self.xsl:
            ids_f = local = None
        else:
            lo, hi, _ = self.tpp.ep_bounds()
            ids_f = ttnn.typecast(idx, FP32)
            local = ttnn.multiply(ttnn.gt(ids_f, lo), ttnn.lt(ids_f, hi))
        # dispatch: x (+ 32 zero rows: row C is the padding rows' source) -> [1, 1, cap, 4096] bf16 ROW_MAJOR
        if self.ffn == "xffn" and (self.xffn_valid or self.xffn_xsplit):  # valid rows only: no zero row, one copy
            x_ext = ttnn.to_layout(ttnn.reshape(x, (c, h)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        else:
            x_ext = ttnn.to_layout(ttnn.pad(ttnn.reshape(x, (c, h)), ((0, TILE), (0, 0)), 0.0), ttnn.ROW_MAJOR_LAYOUT,
                                   memory_config=DRAM)  # fmt: skip
        gather_kw = {}
        if self.ffn == "xffn":  # one expert per core; output rows ROW_MAJOR in dispatch order -> the rm combine
            if self.xsl:
                out_rm = ttnn.empty((b["cap"], h), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                    memory_config=DRAM)  # fmt: skip
            elif getattr(self, "rm_skip", False):  # GLM53_PF_MOE_RM_SKIP: the combine never reads an unwritten row
                out_rm = ttnn.empty((b["cap"], h), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                    memory_config=DRAM)  # fmt: skip
            else:  # GLM53_PF_MOE_XFFN_EP: rows of other chips' experts stay unwritten -> zeros (capture-legal device fill)
                out_rm = ttnn.moreh_full([b["cap"], h], 0.0, self.mesh, dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         memory_config=DRAM)  # fmt: skip
            self._xffn(x_ext, b, out_rm, c)
            ttnn.deallocate(x_ext)
            if getattr(self, "tap_on", False):  # diagnosis (acc_l3_taps): the expert rows + the pair -> row map
                self.tap = dict(
                    out=ttnn.clone(out_rm, memory_config=DRAM), dest=ttnn.clone(b["dest"], memory_config=DRAM)
                )
            if upto in ("dispatch", "swiglu", "untilize"):
                ttnn.deallocate(out_rm)
                return None
            wt = w if local is None else ttnn.multiply(w, local)  # XSL: every pair is this chip's; EP: 0 = not local
            if wt is not w:
                ttnn.deallocate(w)
            for t_ in {id(v): v for v in (ids_f, local) if v is not None}.values():
                ttnn.deallocate(t_)
            y = ttnn.empty((1, 1, c, h), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
            self._combine_rm(out_rm, b["dest"], wt, y, c, k, h)
            ttnn.deallocate(out_rm)
            ttnn.deallocate(wt)
            return y
        if self.xgather:  # the op gathers: x_ext as input (C + 32 rows, row C = zeros), the disp table as its index
            xd = ttnn.reshape(x_ext, (1, 1, c + TILE, h))
            if xd.buffer_address() != x_ext.buffer_address():  # a copy, not a view: free the source now
                ttnn.deallocate(x_ext)
            # an expert's count <= C (top-k experts are distinct per token): m_t <= ceil(C / 32), a C x 4 B scratch
            gather_kw = dict(x_row_index=b["disp"], input_m_tiles=-(-c // TILE))
        elif self.v2_dispatch:
            mk = ttnn.zeros if os.environ.get("GLM53_PF_MOE_V2_XDZERO") == "1" else ttnn.empty
            xd = mk((1, 1, b["cap"], h), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                    memory_config=DRAM)  # fmt: skip
            self._dispatch(x_ext, b, xd, h * 2)
        else:
            xd = ttnn.reshape(ttnn.embedding(b["disp"], x_ext, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=BF16),
                              (1, 1, b["cap"], h))  # fmt: skip
        if not self.xgather:
            ttnn.deallocate(x_ext)
        if upto == "dispatch":
            ttnn.deallocate(xd)
            return None
        out = ttnn.empty((1, 1, b["cap"], h), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                         memory_config=DRAM)  # fmt: skip
        mod = ttnn.experimental.deepseek_prefill
        mod.moe_fused_swiglu(xd, self.gate_up, self.gate_up, self.down, b["cnt"], self.idx_table,
                             dtype=BF16, compute_kernel_config=self.ckc, output=out,
                             core_grid=None if os.environ.get("GLM53_MFS_GRID") == "device" else self.mfs_grid,
                             expert_region_offsets=b["start"], read_x_at_offset=True,
                             activation=(ttnn.RoutedExpertActivation.Silu if os.environ.get("GLM53_PF_MFS_ACT", "") == "silu"
                                         else ttnn.RoutedExpertActivation.ClampedSiluGlu),  # A/B (L3 accuracy hunt)
                             gate_up_interleaved=True,
                             l1_budget_bytes=PF_MOE_CB, **gather_kw)  # fmt: skip
        ttnn.deallocate(xd)
        if getattr(self, "tap_on", False):  # diagnosis (Stage 16 accuracy, tests/s16/jobs/acc_l3_taps.py): the expert
            # rows before the combine + the pair -> dispatch-row map, copied inside the unit's capture
            self.tap = dict(out=ttnn.clone(out, memory_config=DRAM), dest=ttnn.clone(b["dest"], memory_config=DRAM))
        if upto == "swiglu":
            ttnn.deallocate(out)
            return None
        if self.v2_combine:  # local pairs only, fp32 accumulation over the k slots
            wl = w if local is None else ttnn.multiply(w, local)  # [1, 1, C, k], 0: not local
            wt = ttnn.to_layout(
                ttnn.reshape(ttnn.transpose(wl, -2, -1), (1, n)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM
            )  # fmt: skip  # [1, k C] fp32, pair p = j C + t
            for t_ in {id(v): v for v in (w, ids_f, local, wl) if v is not None}.values():
                ttnn.deallocate(t_)
            y = ttnn.empty((1, 1, c, h), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
            self._combine(out, b["dest"], wt, y, c, k, h)
            ttnn.deallocate(out)
            ttnn.deallocate(wt)
            return y
        if self.combine == "rm":  # gather + weight + top-k sum in one program over the untilized outputs
            wt = w if local is None else ttnn.multiply(w, local)  # [1, 1, C, k] fp32 TILE, 0: not local
            for t_ in {id(v): v for v in (w, ids_f, local) if v is not None and v is not wt}.values():
                ttnn.deallocate(t_)
            out_rm = ttnn.to_layout(ttnn.reshape(out, (b["cap"], h)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(out)
            if upto == "untilize":
                ttnn.deallocate(out_rm)
                ttnn.deallocate(wt)
                return None
            y = ttnn.empty((1, 1, c, h), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
            self._combine_rm(out_rm, b["dest"], wt, y, c, k, h)
            ttnn.deallocate(out_rm)
            ttnn.deallocate(wt)
            return y
        col = lambda t_: ttnn.reshape(ttnn.transpose(t_, -2, -1), (1, 1, n, 1))  # noqa: E731  [1,1,C,k] -> [1,1,n,1]
        w_col, local_col = col(w), None if local is None else col(local)
        # combine: pair p's output row (dest[p]) -> where(local) * weight -> sum over the k slots
        out_rm = ttnn.to_layout(ttnn.reshape(out, (b["cap"], h)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(out)
        if upto == "untilize":
            for t_ in {id(v): v for v in (w_col, local_col, out_rm) if v is not None}.values():
                ttnn.deallocate(t_)
            return None
        yp = ttnn.reshape(ttnn.embedding(b["dest"], out_rm, layout=ttnn.TILE_LAYOUT, dtype=BF16), (1, 1, n, h))
        ttnn.deallocate(out_rm)
        if local_col is not None:
            yp = ttnn.where(ttnn.typecast(local_col, BF16), yp, 0.0)
        yp = ttnn.multiply(yp, ttnn.typecast(w_col, BF16))
        y = ttnn.experimental.fast_reduce_nc(ttnn.reshape(yp, (1, k, c, h)), dims=[1])
        ttnn.deallocate(yp)
        return ttnn.reshape(y, (1, 1, c, h))


def _routed_sorted_masked(moe, x, idx, scores, local):
    """OptimizedMoE.routed_sorted with a per-pair weight mask (`local` [1, 1, P, k] 0/1): pairs routed to another chip's
    experts get weight 0 (their fused_experts output -- the same slot index of THIS chip's EP weights -- is finite and
    multiplied by 0). Same expert math as the token path for the local pairs."""
    TILE_ = TILE
    rows, h, k = x.shape[-2], moe.hidden, moe.top_k
    n = rows * k
    tok_table, ones = moe._sorted_consts(rows)
    sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
    w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
    w = ttnn.multiply(ttnn.multiply(w, moe.scaling), local)
    flat = lambda t: ttnn.reshape(ttnn.transpose(t, -2, -1), (1, 1, 1, n))  # noqa: E731
    ids_f = flat(ttnn.typecast(idx, FP32))
    w_f = flat(w)
    sorted_ids, perm = ttnn.sort(ids_f, dim=-1)
    col = lambda t: ttnn.transpose(t, -2, -1)  # noqa: E731
    tok_sorted = ttnn.gather(tok_table, dim=-1, index=perm)
    w_sorted = ttnn.gather(w_f, dim=-1, index=perm)
    w_col = ttnn.typecast(col(w_sorted), BF16)
    ids_col = ttnn.typecast(col(sorted_ids), ttnn.uint16)
    tok_rm = ttnn.to_layout(ttnn.typecast(ttnn.reshape(tok_sorted, (1, n)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.to_layout(ttnn.reshape(x, (rows, h)), ttnn.ROW_MAJOR_LAYOUT)
    xs = ttnn.reshape(ttnn.embedding(tok_rm, x_rm, layout=ttnn.TILE_LAYOUT), (1, 1, n, h))
    ttnn.deallocate(x_rm)
    fid = moe.phase_fidelity()
    outs = []
    for c0 in range(0, n, TILE_):
        sl = lambda t, w_: ttnn.slice(t, (0, 0, c0, 0), (1, 1, c0 + TILE_, w_))  # noqa: E731
        y = moe._fused_experts(sl(xs, h), routing_indices=sl(ids_col, 1), routing_scores=ones,
                               num_experts=min(moe.num_experts, TILE_), intermediate_size=moe.intermediate,
                               swiglu_limit=moe.limit, top_k=1, routed_scaling_factor=1.0, routing_eps=0.0,
                               experts_block_size=1, math_fidelity=fid)  # fmt: skip
        outs.append(ttnn.reshape(y, (1, 1, TILE_, h)))
    ttnn.deallocate(xs)
    ys = ttnn.concat(outs, dim=2)
    for o in outs:
        ttnn.deallocate(o)
    ys = ttnn.multiply(ys, w_col)
    _, inv = ttnn.sort(ttnn.typecast(perm, FP32), dim=-1)
    inv_rm = ttnn.to_layout(ttnn.typecast(ttnn.reshape(inv, (1, n)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    ys_rm = ttnn.to_layout(ttnn.reshape(ys, (n, h)), ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(ys)
    yp = ttnn.embedding(inv_rm, ys_rm, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(ys_rm)
    return ttnn.slice(ttnn.experimental.fast_reduce_nc(ttnn.reshape(yp, (1, k, rows, h)), dims=[1]), (0, 0, 0, 0),
                      (1, 1, rows, h))  # fmt: skip
