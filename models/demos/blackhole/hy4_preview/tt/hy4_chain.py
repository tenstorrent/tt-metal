# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview layer chain on the 4x8 mesh (rows = users, columns = TP8 / expert groups): embedding -> iHC streams (hc_mult 4, fp32) -> per layer
[iHC pre/collapse + norm, attention (Hy4Attention incl. indexer), iHC post, iHC pre/collapse + norm, dense / MoE MLP, iHC post] -> hc_head -> norm -> lm_head.

Everything here is assembly of the earlier tracks (hy4_attention, hy4_dense, hy4_moe + hy4_prefill_moe); the only new maths is the generic-T iHC (plain ttnn ops, the DSV4 mHC
kernels only serve T <= 32 rows) and the head. Layouts: stream x4 = [1, 4, n, 6144] fp32 TILE per device (rows hold different users, replicated over the 8 columns).

Prefill (one chunk of n = 512 rows per mesh row): the SAME code path a server prefill would use (chunked, cache append, indexer, unified dispatch/combine MoE).
Env: HY4_ATTN_BFP4=gate,q_a,q_b (attention tensors stored bfp4, default off; the rest bfp8 with HY4_ATTN_WDT=bfp8), HY4_IDX_ROPE_FIRST (0 = oracle default LAST 64 dims).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import DENSE, FULL, Hy4Config
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload

HID = 6144
NSTREAM = 4
CHUNK = 512


def dram_view(md):
    """allocator report of the mesh DRAM (identical on every chip): bytes per bank."""
    try:
        v = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
        return dict(
            banks=v.num_banks,
            total_per_bank=v.total_bytes_per_bank,
            alloc_per_bank=v.total_bytes_allocated_per_bank,
            free_per_bank=v.total_bytes_free_per_bank,
            largest_free_per_bank=v.largest_contiguous_bytes_free_per_bank,
        )
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e), alloc_per_bank=-1)


# ------------------------------------------------------------------ host loading
class CkptReader:
    """safetensors index + threaded per-layer reads (>= 8 parallel readers, the NFS gives ~300 MB/s per stream)."""

    def __init__(self, ckpt, threads=16):
        self.ckpt = ckpt
        self.wm = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.bytes_read = 0

    def _get(self, key):
        from safetensors import safe_open

        with safe_open(f"{self.ckpt}/{self.wm[key]}", "pt") as fh:
            t = fh.get_tensor(key)
        self.bytes_read += t.numel() * t.element_size()
        return t

    def get(self, key):
        return self._get(key)

    def layer_sd(self, L, skip=("mlp.experts.",)):
        """tensors of layer L keyed by the path after ``model.layers.<L>.`` (experts skipped: loaded by load_moe_layer)."""
        pre = f"model.layers.{L}." if L >= 0 else "model.mtp_layers.0."
        keys = [k for k in self.wm if k.startswith(pre) and not any(k[len(pre) :].startswith(s) for s in skip)]
        vals = list(self.pool.map(self._get, keys))
        return {k[len(pre) :]: v for k, v in zip(keys, vals)}


def wdtypes_from_env():
    """{host-weight key: dtype} override of the attention weights (D9-1: bfp4 for gate, q_a, q_b; everything else bfp8)."""
    names = [s for s in os.environ.get("HY4_ATTN_BFP4", "").split(",") if s]
    return {n: ttnn.bfloat4_b for n in names} or None


# ------------------------------------------------------------------ iHC (generic T)
class Hy4IHCGen:
    """One iHC site (attn or mlp of a layer): m = (x_flat * rsqrt(mean(x_flat^2) + eps)) @ fn^T ; pre = sigmoid(m[:4] s0 + b[:4]) + hc_eps ; post = 2 sigmoid(m[4:] s1 + b[4:]).
    x4 [1,4,n,D] fp32. ``pre_post(x4)`` -> (pre [1,4,n,1], post [1,4,n,1]); collapse = sum_i pre_i x_i; expand = x4 + post * y.
    """

    def __init__(self, md, fn8, base8, scale2, norm_eps, hc_eps, magnitude=2.0):
        self.md, self.norm_eps = md, norm_eps
        fn = fn8.float().reshape(8, NSTREAM, HID).permute(1, 2, 0).contiguous()  # [stream, D, 8]
        self.w = upload(md, fn.reshape(1, NSTREAM, HID, 8), ttnn.float32)
        sc = torch.tensor([float(scale2[0])] * 4 + [float(scale2[1])] * 4).reshape(1, 1, 1, 8)
        self.scale = upload(md, sc, ttnn.float32)
        self.base = upload(md, base8.float().reshape(1, 1, 1, 8), ttnn.float32)
        self.mul = upload(md, torch.tensor([1.0] * 4 + [magnitude] * 4).reshape(1, 1, 1, 8), ttnn.float32)
        self.add = upload(md, torch.tensor([hc_eps] * 4 + [0.0] * 4).reshape(1, 1, 1, 8), ttnn.float32)
        self.ckc = compute_cfg(md, "hifi4", True)

    def pre_post(self, x4):
        n = x4.shape[2]
        sq = ttnn.multiply(x4, x4)
        ss = ttnn.sum(sq, dim=3, keepdim=True)  # [1,4,n,1]
        ttnn.deallocate(sq)
        ss = ttnn.sum(ss, dim=1, keepdim=True)  # [1,1,n,1]
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / (NSTREAM * HID)), self.norm_eps))
        ttnn.deallocate(ss)
        m = ttnn.matmul(x4, self.w, compute_kernel_config=self.ckc, dtype=ttnn.float32)  # [1,4,n,8]
        m = ttnn.sum(m, dim=1, keepdim=True)  # [1,1,n,8]
        m = ttnn.multiply(m, r)
        ttnn.deallocate(r)
        a = ttnn.sigmoid(ttnn.add(ttnn.multiply(m, self.scale), self.base))
        a = ttnn.add(ttnn.multiply(a, self.mul), self.add)  # [1,1,n,8]: pre (+eps) | 2 * post
        ttnn.deallocate(m)
        a = ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.permute(a, (0, 3, 2, 1))  # [1,8,n,1]
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
        pre = ttnn.slice(a, [0, 0, 0, 0], [1, 4, n, 1])
        post = ttnn.slice(a, [0, 4, 0, 0], [1, 8, n, 1])
        ttnn.deallocate(a)
        return pre, post

    @staticmethod
    def collapse(x4, pre):
        w = ttnn.multiply(x4, pre)
        h = ttnn.sum(w, dim=1, keepdim=True)  # [1,1,n,D] fp32
        ttnn.deallocate(w)
        return h

    @staticmethod
    def expand(x4, y, post):
        """y [1,1,n,D] (bf16 or fp32) -> x4 + post * y (fp32); frees x4 and post."""
        yf = ttnn.typecast(y, ttnn.float32) if y.dtype != ttnn.float32 else y
        add = ttnn.multiply(post, yf)
        out = ttnn.add(x4, add)
        for t in (add, x4, post):
            ttnn.deallocate(t)
        if yf is not y:
            ttnn.deallocate(yf)
        return out


# ------------------------------------------------------------------ one layer
class Hy4ChainLayer:
    def __init__(self, ch, L, sd):
        self.ch, self.L = ch, L
        md, cfg = ch.md, ch.cfg
        self.spec = cfg.layer(L)
        t0 = time.time()
        rms = lambda k: upload(md, sd[k].float().reshape(1, 1, -1, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.w_in, self.w_post = rms("input_layernorm.weight"), rms("post_attention_layernorm.weight")
        hc = lambda s: Hy4IHCGen(
            md,
            sd[f"hc_{s}_layer.hc_pre.hc_fn"],
            sd[f"hc_{s}_layer.hc_pre.hc_base"],
            sd[f"hc_{s}_layer.hc_pre.hc_scale"],
            cfg.rms_norm_eps,
            cfg.hc_eps,
            cfg.hc_magnitude,
        )
        self.mtp = L >= cfg.num_hidden_layers
        if not self.mtp:
            self.ihc_a, self.ihc_m = hc("attn"), hc("mlp")
        self.attn = Hy4Attention(
            md,
            ch.mc,
            ch.ccl,
            cfg,
            {k: v for k, v in sd.items() if k.startswith("self_attn.")},
            L,
            kv_fmt=ch.kv_fmt,
            rope_interleave=True,
            idx_rope_interleave=True,
            wdtype=ch.wdt,
            wdtypes=ch.wdtypes,
        )
        if self.attn.indexer is not None:
            self.attn.indexer.idx_rope_first = ch.idx_rope_first
        self.ckc_norm = compute_cfg(md, "hifi4", False)
        self.t_attn = time.time() - t0
        t0 = time.time()
        if self.spec.mlp == DENSE:
            t = lambda k: sd[k].T.contiguous().to(torch.bfloat16)[None, None]
            self.dense = D.make_dense(
                md, t("mlp.gate_proj.weight"), t("mlp.up_proj.weight"), t("mlp.down_proj.weight"), decode=False
            )
        else:
            from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
            from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
            from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import Hy4UnifiedMoE

            W = load_moe_layer(L, W=ch.W, prefix="mtp.0." if self.mtp else None)
            self.t_read = time.time() - t0
            self.blk = Hy4MoEBlock(
                md,
                W,
                batch_per_device=1,
                buffers=ch.moe_buffers if os.environ.get("HY4_SHARE_MOE_BUF", "1") == "1" else None,
            )
            if ch.moe_buffers is None:
                ch.moe_buffers = (
                    self.blk.decode.buffers
                )  # dispatch/combine scratch shared by all layers (as the DSV4 model does)
            self.blk.warmup()
            self.um = Hy4UnifiedMoE(
                md, L, ring=(self.blk.decode.expert_state.tt_w0_w1, self.blk.decode.expert_state.tt_w2)
            )
            sw = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
            self.shared = D.make_shared(
                md, *sw, limit=D.NO_CLAMP, decode=False
            )  # coordinator 11-13: the shared expert is NOT clamped (CE 1.77 vs 0.79)
            del W
        if (
            self.mtp
        ):  # MTP layer: loaded, unused until the spec-mtp device stage (no iHC; eh_proj [6144, 12288], enorm, hnorm, final_layernorm)
            self.eh_proj = upload(
                md,
                sd["eh_proj.weight"].T.contiguous().reshape(1, 1, 2 * HID, HID).to(torch.bfloat16),
                ttnn.bfloat8_b,
                shard_dim=3,
            )
            self.mtp_norms = {
                k: upload(md, sd[k + ".weight"].float().reshape(1, 1, -1, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                for k in ("enorm", "hnorm", "final_layernorm")
            }
        self.t_mlp = time.time() - t0

    # --- helpers
    def _norm(self, h32, w):
        h = ttnn.typecast(h32, ttnn.bfloat16)
        out = ttnn.rms_norm(h, weight=w, epsilon=self.ch.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(h)
        return out

    def _mlp_prefill(self, h):
        """h [1,1,n,D] bf16 (normed) -> y [1,1,n,D] bf16 replicated over the columns."""
        ch = self.ch
        if self.spec.mlp == DENSE:
            return D.allreduce_cols(self.dense.forward(h))
        from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import COLS, moe_cols

        n = h.shape[2]
        G = n // 256
        h4 = ttnn.reshape(h, [G, COLS, 32, HID])
        own = ttnn.mesh_partition(h4, dim=1, cluster_axis=1)  # [G,1,32,D]
        own = ttnn.reshape(own, [1, 1, G * 32, HID])
        routed_own = moe_cols(self.um, self.blk.gate, own, ch.mc, ch.ccl)  # [1,1,n/8,D]
        ttnn.deallocate(own)
        g = ch.mc.allgather(routed_own, ch.ccl, axis=1, dim=2)  # [1,1,n,D] rows ordered [c][g][32]
        ttnn.deallocate(routed_own)
        g = ttnn.reshape(g, [COLS, G, 32, HID])
        g = ttnn.permute(g, (1, 0, 2, 3))
        routed = ttnn.reshape(g, [1, 1, n, HID])
        sh = D.allreduce_cols(self.shared.forward(h))
        y = ttnn.add(routed, sh)
        ttnn.deallocate(routed)
        ttnn.deallocate(sh)
        return y

    def prefill_chunk(self, x4, start, ids_by_layer, taps=None):
        """x4 [1,4,n,D] fp32 -> x4' (same layout). ids_by_layer: layer -> top-k ids of the full-indexer layers (IndexCache carry)."""
        pre, post = self.ihc_a.pre_post(x4)
        raw = self.ihc_a.collapse(x4, pre)
        ttnn.deallocate(pre)
        h = self._norm(raw, self.w_in)
        ttnn.deallocate(raw)
        src = self.spec.indexer_source
        ids_in = None if self.spec.indexer == FULL else ids_by_layer[src]
        a, ids = self.attn.prefill_chunk(h, start, ids_in=ids_in)
        ttnn.deallocate(h)
        if self.spec.indexer == FULL:
            if self.L in ids_by_layer:
                ttnn.deallocate(ids_by_layer[self.L])
            ids_by_layer[self.L] = ids
        x4 = self.ihc_a.expand(x4, a, post)
        ttnn.deallocate(a)
        if taps is not None:
            taps["after_attn"] = x4
        pre, post = self.ihc_m.pre_post(x4)
        raw = self.ihc_m.collapse(x4, pre)
        ttnn.deallocate(pre)
        h = self._norm(raw, self.w_post)
        ttnn.deallocate(raw)
        y = self._mlp_prefill(h)
        ttnn.deallocate(h)
        x4 = self.ihc_m.expand(x4, y, post)
        ttnn.deallocate(y)
        return x4


# ------------------------------------------------------------------ model
class Hy4Chain:
    def __init__(
        self, md, mc, ccl, ckpt="/ckpt", kv_fmt="bf16", wdt=None, wdtypes=None, idx_rope_first=None, weights=None
    ):
        from models.demos.blackhole.hy4_preview.weights import Hy4Weights

        self.md, self.mc, self.ccl, self.kv_fmt = md, mc, ccl, kv_fmt
        self.cfg = Hy4Config.from_checkpoint(ckpt)
        self.rd = CkptReader(ckpt)
        self.W = weights or Hy4Weights(ckpt)
        self.wdt = wdt or {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("HY4_ATTN_WDT", "bfp8")]
        self.wdtypes = wdtypes if wdtypes is not None else wdtypes_from_env()
        self.idx_rope_first = (
            idx_rope_first if idx_rope_first is not None else os.environ.get("HY4_IDX_ROPE_FIRST", "0") == "1"
        )
        self.layers = []
        self.mtp_layers = []  # built, unused until the spec-mtp device stage
        self.moe_buffers = None
        self.load_log = []
        self._head = None

    # --- weights
    def add_layers(self, lo, hi, log=print):
        for L in range(lo, hi):
            t0, b0 = time.time(), self.rd.bytes_read
            sd = self.rd.layer_sd(L if L < self.cfg.num_hidden_layers else -1)
            t_sd = time.time() - t0
            ly = Hy4ChainLayer(self, L, sd)
            (self.mtp_layers if L >= self.cfg.num_hidden_layers else self.layers).append(ly)
            rec = dict(
                layer=L,
                sec=round(time.time() - t0, 1),
                sd_read_s=round(t_sd, 1),
                non_expert_bytes=self.rd.bytes_read - b0,
                attn_s=round(ly.t_attn, 1),
                mlp_s=round(ly.t_mlp, 1),
                expert_read_s=round(getattr(ly, "t_read", 0.0), 1),
                dram_alloc_per_bank=dram_view(self.md)["alloc_per_bank"],
            )
            self.load_log.append(rec)
            log("LAYER_LOAD " + json.dumps(rec))

    def build_head(self):
        md, cfg = self.md, self.cfg
        g = lambda k: self.rd.get(k)
        emb = g("model.embed_tokens.weight")  # [V, D]
        self.embed = upload(md, emb.reshape(1, 1, *emb.shape), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, shard_dim=3)
        lm = g("lm_head.weight")  # [V, D]
        self.lm_head = upload(
            md, lm.T.contiguous().reshape(1, 1, HID, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3
        )
        del emb, lm
        fn = g("model.hc_head.hc_head_fn").float()  # [4, 24576]
        self.hh_w = upload(
            md, fn.reshape(NSTREAM, NSTREAM, HID).permute(1, 2, 0).reshape(1, NSTREAM, HID, NSTREAM), ttnn.float32
        )
        self.hh_scale = float(g("model.hc_head.hc_head_scale").float().reshape(-1)[0])
        self.hh_base = upload(md, g("model.hc_head.hc_head_base").float().reshape(1, 1, 1, NSTREAM), ttnn.float32)
        self.final_norm = upload(
            md, g("model.norm.weight").float().reshape(1, 1, -1, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        self.ckc_head = compute_cfg(md, "hifi4", True)
        self.ckc_norm = compute_cfg(md, "hifi4", False)

    # --- embedding / head
    def embed_ids(self, ids):
        """ids torch [4, n] int (one prompt per mesh row) -> x4 [1,4,n,D] fp32."""
        n = ids.shape[1]
        t = upload(self.md, ids.reshape(4, 1, n).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, row_dim=0)
        t = ttnn.reshape(t, [1, n])
        e = ttnn.embedding(t, self.embed, layout=ttnn.TILE_LAYOUT)  # [1,n,768]
        e = ttnn.reshape(e, [1, 1, n, HID // 8])
        e = self.mc.allgather(e, self.ccl, axis=1, dim=3)  # [1,1,n,D]
        e32 = ttnn.typecast(e, ttnn.float32)
        ttnn.deallocate(e)
        x4 = ttnn.concat([e32] * NSTREAM, dim=1)
        ttnn.deallocate(e32)
        return x4

    def head(self, x4):
        """x4 [1,4,n,D] -> (final normed hidden [1,1,n,D] bf16, logits [1,1,n,V/8 per column] bf16)."""
        m = ttnn.matmul(x4, self.hh_w, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)  # [1,4,n,4]
        ss = ttnn.sum(ttnn.sum(ttnn.multiply(x4, x4), dim=3, keepdim=True), dim=1, keepdim=True)
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / (NSTREAM * HID)), self.cfg.rms_norm_eps))
        m = ttnn.sum(m, dim=1, keepdim=True)  # [1,1,n,4]
        m = ttnn.multiply(m, r)
        w = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(m, self.hh_scale), self.hh_base)), self.cfg.hc_eps)
        w = ttnn.permute(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (0, 3, 2, 1))  # [1,4,n,1]
        w = ttnn.to_layout(w, ttnn.TILE_LAYOUT)
        h = ttnn.sum(ttnn.multiply(x4, w), dim=1, keepdim=True)  # [1,1,n,D] fp32
        hb = ttnn.typecast(h, ttnn.bfloat16)
        hn = ttnn.rms_norm(
            hb, weight=self.final_norm, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm
        )
        logits = ttnn.linear(hn, self.lm_head, compute_kernel_config=self.ckc_head)
        return hn, logits

    # --- prefill
    def prefill(self, ids, start=0, ids_by_layer=None, taps=None, layers=None):
        """one chunk: ids [4, 512] -> x4 after the layers (list taps[L] = x4 after layer L when taps is a dict)."""
        x4 = self.embed_ids(ids)
        ids_by_layer = {} if ids_by_layer is None else ids_by_layer
        for ly in layers or self.layers:
            x4 = ly.prefill_chunk(x4, start, ids_by_layer)
            if taps is not None:
                tr = int(
                    os.environ.get("HY4_TAP_ROWS", "0")
                )  # >0: keep only the first rows (the short dump prompts) of every layer's stream
                taps[ly.L] = ttnn.slice(x4, [0, 0, 0, 0], [1, NSTREAM, tr, HID]) if tr else ttnn.clone(x4)
        return x4, ids_by_layer

    def reset(self):
        for ly in self.layers:
            ly.attn.reset()
