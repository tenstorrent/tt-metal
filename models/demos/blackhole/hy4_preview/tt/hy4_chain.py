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


def rms_exact(x32, w32, eps):
    """fp32 rms norm of the 6144-wide rows of x32 [1,1,n,D] fp32 with weight w32 [1,1,1,D] fp32 -> bf16. ``ttnn.rms_norm`` at D = 6144 does not fit L1 with fp32 accumulation, and without
    it the sum of squares carries 3-4 % systematic error (dump-diff of layer 0: h rel L2 0.04); this composite equals the oracle to bf16 output rounding.
    """
    sq = ttnn.multiply(x32, x32)
    ss = ttnn.sum(sq, dim=3, keepdim=True)
    ttnn.deallocate(sq)
    r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / x32.shape[3]), eps))
    ttnn.deallocate(ss)
    y = ttnn.multiply(x32, r)
    ttnn.deallocate(r)
    z = ttnn.multiply(y, w32, dtype=ttnn.bfloat16)
    ttnn.deallocate(y)
    return z


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
        if os.environ.get("HY4_B49_IHC", "0") != "0" and os.environ.get("HY4_B49_COLLAPSE", "1") != "0":
            # B49: sum_i pre_i x_i as 4 slice * bcast-multiply + 3 adds (the stock ttnn.sum over the stream dim 1 of [1,4,n,D] fp32 costs 5.1 ms at n=512, this 0.82 ms; fp32 reorder only)
            n, Dd = x4.shape[2], x4.shape[3]
            acc = None
            for i in range(NSTREAM):
                xi = ttnn.slice(x4, [0, i, 0, 0], [1, i + 1, n, Dd])
                pi = ttnn.slice(pre, [0, i, 0, 0], [1, i + 1, n, 1])
                t = ttnn.multiply(xi, pi)  # (xi / pi are NOT deallocated: ttnn.slice may return a view of x4 / pre)
                if acc is None:
                    acc = t
                else:
                    a2 = ttnn.add(acc, t)
                    ttnn.deallocate(acc)
                    ttnn.deallocate(t)
                    acc = a2
            return acc
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
        f32row = lambda k: upload(md, sd[k].float().reshape(1, 1, 1, HID), ttnn.float32)
        self.w_in32, self.w_post32 = f32row("input_layernorm.weight"), f32row("post_attention_layernorm.weight")
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
            self.ihcpf_a = self.ihcpf_m = None
            if (
                os.environ.get("HY4_B49_IHC", "0") == "fused"
            ):  # B49: prefill iHC as 5 JIT programs per site (tt/hy4_ihc_pf.py), shares the hc_fn tensor of the stock iHC
                from models.demos.blackhole.hy4_preview.tt.hy4_ihc_pf import Hy4IHCPF

                mkpf = lambda s_, g_, nw: Hy4IHCPF(
                    md,
                    g_.w,
                    sd[f"hc_{s_}_layer.hc_pre.hc_base"],
                    sd[f"hc_{s_}_layer.hc_pre.hc_scale"],
                    sd[nw],
                    cfg.hc_eps,
                    cfg.rms_norm_eps,
                    cfg.hc_magnitude,
                    dim=HID,
                    n=CHUNK,
                )
                self.ihcpf_a = mkpf("attn", self.ihc_a, "input_layernorm.weight")
                self.ihcpf_m = mkpf("mlp", self.ihc_m, "post_attention_layernorm.weight")
            if (
                os.environ.get("HY4_IHC_FUSED", "0") == "1"
            ):  # fused iHC kernels (ihc-fuse e037d61bccb): packed [1,1,4T,D] streams, decode only
                from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused

                mk = lambda s_: Hy4IHCFused(
                    md,
                    sd[f"hc_{s_}_layer.hc_pre.hc_fn"].float(),
                    sd[f"hc_{s_}_layer.hc_pre.hc_base"].float(),
                    sd[f"hc_{s_}_layer.hc_pre.hc_scale"].float(),
                    dim=HID,
                    hc_eps=cfg.hc_eps,
                    norm_eps=cfg.rms_norm_eps,
                )
                self.ihcf_a, self.ihcf_m = mk("attn"), mk("mlp")
                self.hw_in, self.hw_post = (
                    sd["input_layernorm.weight"].float(),
                    sd["post_attention_layernorm.weight"].float(),
                )
        if (
            os.environ.get("HY4_POOL", "0") == "1" and os.environ.get("HY4_POOL_PF", "0") == "1"
        ):  # pool-prefill: DIRECT-TO-POOL chunked prefill (no replicated cache, no hand-off); needs the pool (Hy4PoolDecoder) before the first prefill
            from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF as _Attn
        elif (
            os.environ.get("HY4_POOL", "0") == "1"
        ):  # integrated long-context decode: Hy4AttentionPool (superset of Hy4AttentionOpt / Hy4Attention; prefill unchanged)
            from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import Hy4AttentionPool as _Attn
        else:
            _Attn = Hy4Attention
        self.attn = _Attn(
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
        elif os.environ.get("HY4_MTP_HOSTMOE", "0") == "1" and self.mtp:
            # mtp-device 1-slot tests: the MoE decode block needs the 4-row mesh (cluster_axis dim 4); on a 1x8 slot only the shared expert is built, the routed part runs on the host oracle
            from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer

            W = load_moe_layer(L, W=ch.W, prefix="mtp.0.", load_experts=False)
            sw = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
            self.shared = D.make_shared(md, *sw, limit=D.NO_CLAMP, decode=False)
            self.shared_dec = None
            self.router_w = (W["gate_weight"], W["gate_bias"])
            self.blk = None
            del W
        elif os.environ.get("HY4_NO_MOE", "0") == "1":
            pass  # 1-slot (1x8) wiring tests only: the MoE block needs the 4x8 mesh; _mlp_prefill returns a scaled copy of its input (identical stand-in on every path under test)
        else:
            from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
            from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
            from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import Hy4UnifiedMoE

            tp32_only = (
                os.environ.get("HY4_TP32_ONLY", "0") == "1" and not self.mtp
            )  # product3: the MTP layer (one layer) keeps the ring experts (verify rows + prompt KV use its Hy4MoEBlock); tp32-single: TP32 is the ONLY routed-expert copy (no ring weights, no prefill um)
            if tp32_only:
                assert os.environ.get("HY4_MOE_TP32", "0") == "1", "HY4_TP32_ONLY needs HY4_MOE_TP32=1"
            W = load_moe_layer(L, W=ch.W, prefix="mtp.0." if self.mtp else None, load_experts=not tp32_only)
            self.t_read = time.time() - t0
            self.blk = Hy4MoEBlock(
                md,
                W,
                batch_per_device=1,
                buffers=(
                    ch.moe_buffers
                    if os.environ.get("HY4_SHARE_MOE_BUF", "1") == "1"
                    and not (
                        self.mtp and os.environ.get("HY4_TP32_ONLY", "0") == "1"
                    )  # the TP32-only layers share weightless scratch: the MTP ring layer allocates its own
                    else None
                ),
                tp32_only=tp32_only,
            )
            if ch.moe_buffers is None:
                ch.moe_buffers = (
                    self.blk.decode.buffers
                )  # dispatch/combine scratch shared by all layers (as the DSV4 model does)
            if not tp32_only:
                self.blk.warmup()
            self.um = (
                None
                if tp32_only
                else Hy4UnifiedMoE(
                    md, L, ring=(self.blk.decode.expert_state.tt_w0_w1, self.blk.decode.expert_state.tt_w2)
                )
            )
            sw = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
            self.shared = D.make_shared(
                md, *sw, limit=D.NO_CLAMP, decode=False
            )  # coordinator 11-13: the shared expert is NOT clamped (CE 1.77 vs 0.79)
            self.shared_dec = None
            if "shared" in os.environ.get("HY4_SKINNY", "").replace("all", "shared,router,oproj,hproj").split(
                ","
            ):  # HY4_SKINNY=shared: bank-local decode kernel (second weight copy, 5 MB/layer/chip)
                self.shared_dec = D.make_shared(md, *sw, limit=D.NO_CLAMP, decode=True)
            self.blk_tp = None
            if os.environ.get("HY4_MOE_TP32", "0") == "1" and (
                not self.mtp or os.environ.get("HY4_B42_MTP_TP24", "0") == "1"
            ):  # B42: the MTP layer gets a tp24 copy too (HY4_B42_MTP_TP24=1, decode rows only). kernel-track: TP32 routed experts for decode (second weight copy; prefill keeps the ring layout)
                from models.demos.blackhole.hy4_preview.tt import hy4_tp32_cache as _TC
                from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts
                from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block

                _root = _TC.cache_root()
                Wf = None
                if tp32_only:
                    assert (
                        _root and _TC.check_layer(_root, L) is not None
                    ), f"TP32-only needs the TP32 cache for layer {L}"
                if not (
                    _root and _TC.check_layer(_root, L) is not None
                ):  # no TP32 dump for this layer: pack from the checkpoint
                    Wf = load_moe_layer(L, W=ch.W, experts=list(range(256)), prefix="mtp.0." if self.mtp else None)
                if (
                    os.environ.get("HY4_MOE_TP32_IMPL", "24") == "24"
                ):  # tp24: 24-core variant (T up to 16), W2 layout v1
                    from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp24 import (  # noqa: F811
                        Tp24Experts as Tp32Experts,
                    )

                tpe = (
                    Tp32Experts(md, layer=L)
                    if Wf is None
                    else Tp32Experts(md, Wf["w0"][0], Wf["w1"][0], Wf["w2"][0], layer=L)
                )
                del Wf
                self.blk_tp = Hy4MoeTp32Block(
                    md, ch.ccl, self.blk.gate, tpe, int(os.environ.get("HY4_MOE_TP32_T", "1"))
                )
                if os.environ.get("HY4_MOE_FUSE", "0") == "1" and hasattr(tpe, "shared_cores"):
                    # B19: shared expert beside the tp24 kernel in one program: w2 scaled by 1/4 (all 4 mesh rows compute all Tt tokens, the row reduction sums them back to 1x)
                    from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyShared

                    self.blk_tp.shared_fuse = SkinnyShared(md, sw[0], sw[1], sw[2] * 0.25, limit=D.NO_CLAMP)
                self.blk_pf = None
                if (
                    os.environ.get("HY4_TP32_PF", "0") == "1" and not self.mtp
                ):  # tp32-prefill2: grouped-GEMM prefill on the SAME weights (no second copy)
                    from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_prefill_block import Hy4MoeTp32PrefillBlock

                    self.blk_pf = Hy4MoeTp32PrefillBlock(md, ch.ccl, self.blk.gate, tpe, CHUNK)
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
        if os.environ.get("HY4_NORM", "exact") == "exact":
            return rms_exact(h32, self.w_in32 if w is self.w_in else self.w_post32, self.ch.cfg.rms_norm_eps)
        h = ttnn.typecast(h32, ttnn.bfloat16)
        out = ttnn.rms_norm(h, weight=w, epsilon=self.ch.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(h)
        return out

    def _mlp_prefill(self, h):
        """h [1,1,n,D] bf16 (normed) -> y [1,1,n,D] bf16 replicated over the columns."""
        ch = self.ch
        if self.spec.mlp == DENSE:
            return D.allreduce_cols(self.dense.forward(h))
        if os.environ.get("HY4_NO_MOE", "0") == "1":
            return ttnn.multiply(h, 0.05)
        from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import COLS, moe_cols

        if getattr(self, "blk_pf", None) is not None:  # TP32 grouped-GEMM prefill (single weight copy)
            y = self.blk_pf.forward(h)  # [1,1,n,D/8]: routed partial summed over all 32 chips, own rows
            g = ttnn.all_gather(
                y,
                dim=3,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1,1,n,D] replicated over the columns
            ttnn.deallocate(y)
            sh = D.allreduce_cols(self.shared.forward(h))
            out = ttnn.add(g, sh)
            ttnn.deallocate(g)
            ttnn.deallocate(sh)
            return out
        assert (
            self.um is not None
        ), "HY4_TP32_ONLY: no ring experts, the MoE prefill needs HY4_TP32_PF=1 (tp32-prefill2 grouped-GEMM kernels)"
        n = h.shape[2]
        G = n // 256
        h4 = ttnn.reshape(h, [G, COLS, 32, HID])
        own = ttnn.mesh_partition(h4, dim=1, cluster_axis=1)  # [G,1,32,D]
        own = ttnn.reshape(own, [1, 1, G * 32, HID])
        from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt

        pfo = (
            PfOpt.from_env()
        )  # HY4_PF_OPT=shfold: the shared expert joins the routed partial before the column reduce-scatter (no shared all-reduce)
        routed_own = moe_cols(
            self.um, self.blk.gate, own, ch.mc, ch.ccl, shared=self.shared if pfo.shfold else None, pf=pfo
        )  # [1,1,n/8,D]
        ttnn.deallocate(own)
        g = ch.mc.allgather(routed_own, ch.ccl, axis=1, dim=2)  # [1,1,n,D] rows ordered [c][g][32]
        ttnn.deallocate(routed_own)
        g = ttnn.reshape(g, [COLS, G, 32, HID])
        g = ttnn.permute(g, (1, 0, 2, 3))
        routed = ttnn.reshape(g, [1, 1, n, HID])
        if pfo.shfold:
            return routed
        sh = D.allreduce_cols(self.shared.forward(h))
        y = ttnn.add(routed, sh)
        ttnn.deallocate(routed)
        ttnn.deallocate(sh)
        return y

    def _dbg(self, taps, name, t):
        """debug taps: host copy (row r = device (r, 0)) of an intermediate, first 32 token rows."""
        if taps is None:
            return
        c = int(self.ch.md.shape[1])
        devs = ttnn.get_device_tensors(t)
        taps[name] = [ttnn.to_torch(devs[r * c]).float() for r in range(int(self.ch.md.shape[0]))]

    def prefill_chunk(self, x4, start, ids_by_layer, taps=None):
        """x4 [1,4,n,D] fp32 -> x4' (same layout). ids_by_layer: layer -> top-k ids of the full-indexer layers (IndexCache carry).
        taps (dict, debugging): host copies of pre_a, post_a, h_a, a, x4a, pre_m, post_m, h_m, y."""
        if getattr(self, "ihcpf_a", None) is not None and taps is None and x4.shape[2] == self.ihcpf_a.n:
            return self._prefill_chunk_pf(x4, start, ids_by_layer)
        pre, post = self.ihc_a.pre_post(x4)
        self._dbg(taps, "pre_a", pre)
        self._dbg(taps, "post_a", post)
        raw = self.ihc_a.collapse(x4, pre)
        ttnn.deallocate(pre)
        h = self._norm(raw, self.w_in)
        ttnn.deallocate(raw)
        self._dbg(taps, "h_a", h)
        src = self.spec.indexer_source
        ids_in = None if self.spec.indexer == FULL else ids_by_layer[src]
        a, ids = self.attn.prefill_chunk(h, start, ids_in=ids_in)
        ttnn.deallocate(h)
        self._dbg(taps, "a", a)
        if self.spec.indexer == FULL:
            if self.L in ids_by_layer:
                ttnn.deallocate(ids_by_layer[self.L])
            ids_by_layer[self.L] = ids
        x4 = self.ihc_a.expand(x4, a, post)
        ttnn.deallocate(a)
        self._dbg(taps, "x4a", x4)
        pre, post = self.ihc_m.pre_post(x4)
        self._dbg(taps, "pre_m", pre)
        self._dbg(taps, "post_m", post)
        raw = self.ihc_m.collapse(x4, pre)
        ttnn.deallocate(pre)
        h = self._norm(raw, self.w_post)
        ttnn.deallocate(raw)
        self._dbg(taps, "h_m", h)
        y = self._mlp_prefill(h)
        ttnn.deallocate(h)
        self._dbg(taps, "y", y)
        x4 = self.ihc_m.expand(x4, y, post)
        ttnn.deallocate(y)
        return x4

    def _prefill_chunk_pf(self, x4, start, ids_by_layer):
        """B49 fused-iHC variant of prefill_chunk (same data flow; the iHC of both sites is 5 programs each)."""
        bf = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
        h, st = self.ihcpf_a.pre(x4)
        src = self.spec.indexer_source
        ids_in = None if self.spec.indexer == FULL else ids_by_layer[src]
        a, ids = self.attn.prefill_chunk(h, start, ids_in=ids_in)
        ttnn.deallocate(h)
        if self.spec.indexer == FULL:
            if self.L in ids_by_layer:
                ttnn.deallocate(ids_by_layer[self.L])
            ids_by_layer[self.L] = ids
        ab = bf(a)
        x4a = self.ihcpf_a.expand(x4, ab, st)
        for t in (x4, st) + ((a, ab) if ab is not a else (a,)):
            ttnn.deallocate(t)
        h, st = self.ihcpf_m.pre(x4a)
        y = self._mlp_prefill(h)
        ttnn.deallocate(h)
        yb = bf(y)
        x4n = self.ihcpf_m.expand(x4a, yb, st)
        for t in (x4a, st) + ((y, yb) if yb is not y else (y,)):
            ttnn.deallocate(t)
        return x4n


# ------------------------------------------------------------------ model
class Hy4Chain:
    def __init__(
        self, md, mc, ccl, ckpt="/ckpt", kv_fmt=None, wdt=None, wdtypes=None, idx_rope_first=None, weights=None
    ):
        kv_fmt = kv_fmt or os.environ.get(
            "HY4_KV_FMT", "bf16"
        )  # "scaled_fp8" for the pool decode path (prefill then writes SCALED_FP8 rows)
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
        self.final_norm32 = upload(md, g("model.norm.weight").float().reshape(1, 1, 1, HID), ttnn.float32)
        self.ckc_head = compute_cfg(md, "hifi4", True)
        self.ckc_norm = compute_cfg(md, "hifi4", False)

    # --- embedding / head
    def embed_ids(self, ids):
        """ids torch [4, n] int (one prompt per mesh row) -> x4 [1,4,n,D] fp32."""
        n = ids.shape[1]
        t = upload(
            self.md,
            ids.reshape(ids.shape[0], 1, n).to(torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            row_dim=0 if int(self.md.shape[0]) > 1 else None,
        )
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
        if (
            os.environ.get("HY4_B49_IHC", "0") != "0"
            and os.environ.get("HY4_B49_COLLAPSE", "1") != "0"
            and os.environ.get("HY4_B49_HEAD", "1")
            != "0"  # HY4_B49_HEAD=0: stock sum in the head (ihc-repair bisect; costs the 402 MB transient)
        ):
            # B49 on the HEAD too: the stock ttnn.sum over the stream dim allocates the tile-padded 402 MB temporary (512x32x6144x4 B) that OOMed mega0's first request with 125 MB/bank free
            h = Hy4IHCGen.collapse(x4, w)  # sum_i w_i x_i as 4 slice*mul + 3 adds, w [1,4,n,1]
        else:
            h = ttnn.sum(ttnn.multiply(x4, w), dim=1, keepdim=True)  # [1,1,n,D] fp32
        if os.environ.get("HY4_NORM", "exact") == "exact":
            hn = rms_exact(h, self.final_norm32, self.cfg.rms_norm_eps)
        else:
            hb = ttnn.typecast(h, ttnn.bfloat16)
            hn = ttnn.rms_norm(
                hb, weight=self.final_norm, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm
            )
        # HY4_HEAD_FP32=1 (device loop, D12-1): fp32 logits out of the lm_head (bf16 weights, fp32 accumulation), as the checkpoint's enable_lm_head_fp32
        logits = ttnn.linear(
            hn,
            self.lm_head,
            compute_kernel_config=self.ckc_head,
            dtype=ttnn.float32 if os.environ.get("HY4_HEAD_FP32", "0") == "1" else None,
        )
        return hn, logits

    def head_packed(self, x):
        """HY4_HEAD_FAST=1 (verify-opt track): ``head`` on the PACKED streams x [1,1,4n,D] fp32 TILE (row 4t + i = stream i of token t) without the permute / layout churn, and with the
        D-long reductions split 8 ways over a batch dim (the stock sum / matmul on 4n <= 16 rows run on ONE core per 192 K tiles: 55-200 us each). Same math as ``head`` (fp32): the
        4-stream mix is a [n, 4n] x [4n, D] matmul, the mixes / sum of squares come from the split-K partials. Returns (hn bf16 [1,1,n,D], logits).
        """
        n = int(x.shape[2]) // 4
        NC_ = 8
        DC = HID // NC_
        key = ("hf", n)
        if not hasattr(self, "_hf"):
            self._hf = {}
        if key not in self._hf:
            import torch as _t

            md = self.md
            up = lambda t: upload(md, t.contiguous(), ttnn.float32)
            hw = (
                ttnn.to_torch(ttnn.get_device_tensors(self.hh_w)[0]).float().reshape(NSTREAM, HID, NSTREAM)
            )  # [i, d, o]
            hhc = hw.permute(1, 0, 2).reshape(HID, NSTREAM * NSTREAM)  # [d, 4i+o]
            hhb = hhc.reshape(NC_, DC, NSTREAM * NSTREAM).reshape(1, NC_, DC, NSTREAM * NSTREAM)
            r4 = _t.arange(4 * n)
            G = (r4[None, :] // 4 == _t.arange(n)[:, None]).float()  # [n, 4n]
            Mk = (r4[:, None] % 4 == _t.arange(16)[None, :] // 4).float()  # [4n, 16]
            Fm = (_t.arange(16)[:, None] % 4 == _t.arange(4)[None, :]).float()  # [16, 4]
            E = (_t.arange(4 * n)[None, :] % 4 == _t.arange(4)[:, None]).float()  # [4, 4n]
            self._hf[key] = (up(hhb),) + tuple(up(t.reshape(1, 1, *t.shape)) for t in (G, Mk, Fm, E))
        hhb, G, Mk, Fm, E = self._hf[key]
        xb = ttnn.concat(
            [ttnn.slice(x, [0, 0, 0, c * DC], [1, 1, 4 * n, (c + 1) * DC]) for c in range(NC_)], dim=1
        )  # [1,8,4n,768]
        m8 = ttnn.matmul(xb, hhb, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)  # [1,8,4n,16]
        mm = ttnn.multiply(ttnn.sum(m8, dim=1, keepdim=True), Mk)  # [1,1,4n,16]: only the own-stream block of each row
        ttnn.deallocate(m8)
        m = ttnn.matmul(
            G,
            ttnn.matmul(mm, Fm, compute_kernel_config=self.ckc_head, dtype=ttnn.float32),
            compute_kernel_config=self.ckc_head,
            dtype=ttnn.float32,
        )  # [1,1,n,4]
        sq = ttnn.multiply(xb, xb)
        ttnn.deallocate(xb)
        ss = ttnn.matmul(
            G,
            ttnn.sum(ttnn.sum(sq, dim=3, keepdim=True), dim=1, keepdim=True),
            compute_kernel_config=self.ckc_head,
            dtype=ttnn.float32,
        )  # [1,1,n,1]
        ttnn.deallocate(sq)
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / (NSTREAM * HID)), self.cfg.rms_norm_eps))
        m = ttnn.multiply(m, r)
        w = ttnn.add(
            ttnn.sigmoid(ttnn.add(ttnn.multiply(m, self.hh_scale), self.hh_base)), self.cfg.hc_eps
        )  # [1,1,n,4]
        W = ttnn.multiply(
            ttnn.matmul(w, E, compute_kernel_config=self.ckc_head, dtype=ttnn.float32), G
        )  # [1,1,n,4n]: W[t, 4t+i] = w[t,i]
        h = ttnn.matmul(W, x, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)  # [1,1,n,D] fp32
        hn = rms_exact(h, self.final_norm32, self.cfg.rms_norm_eps)
        logits = ttnn.linear(
            hn,
            self.lm_head,
            compute_kernel_config=self.ckc_head,
            dtype=ttnn.float32 if os.environ.get("HY4_HEAD_FP32", "0") == "1" else None,
        )
        return hn, logits

    # --- prefill
    def prefill(self, ids, start=0, ids_by_layer=None, taps=None, layers=None):
        """one chunk: ids [4, 512] -> x4 after the layers (list taps[L] = x4 after layer L when taps is a dict)."""
        x4 = self.embed_ids(ids)
        ids_by_layer = {} if ids_by_layer is None else ids_by_layer
        from models.demos.blackhole.hy4_preview.tt.hy4_attention import clear_rope_cache

        for ly in layers or self.layers:
            x4 = ly.prefill_chunk(x4, start, ids_by_layer)
            if taps is not None:
                tr = int(
                    os.environ.get("HY4_TAP_ROWS", "0")
                )  # >0: keep only the first rows (the short dump prompts) of every layer's stream
                taps[ly.L] = ttnn.slice(x4, [0, 0, 0, 0], [1, NSTREAM, tr, HID]) if tr else ttnn.clone(x4)
        clear_rope_cache()
        return x4, ids_by_layer

    def reset(self):
        for ly in self.layers:
            ly.attn.reset()
