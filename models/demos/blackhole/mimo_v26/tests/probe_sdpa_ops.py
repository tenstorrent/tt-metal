# SPDX-License-Identifier: Apache-2.0
"""Phase A op probe for MiMo-V2.6-Flash attention (single Blackhole chip, per-chip TP=8 shapes: 8 Q heads, 1 KV head, QK 192, V 128).

    PROBE=<group[,group..]> PROBE_OUT=<jsonl> python probe_sdpa_ops.py
groups: dec_corr dec_gqa dec_ring dec_perf pre_corr pre_gqa pre_perf

Every case is wrapped in try/except and appends one JSON line {case, ...}. Reference: torch fp32, HF semantics:
scores = q.k * 192**-0.5 (+ causal / window mask), SWA adds the RAW sink logit as an extra softmax column (after scaling), window 128 includes self.
Device sink input is tested in two forms: sink/scale (gpt-oss convention) and raw sink.
"""
import json
import os
import time
import traceback

import torch

import ttnn

DQK, DV = 192, 128
SCALE = DQK**-0.5
WINDOW = 128
OUT = os.environ.get("PROBE_OUT", "/tmp/probe_sdpa.jsonl")
BS = 64  # paged block size
SINK_MULT = float(
    os.environ.get("PROBE_SINK_MULT", "1")
)  # study: real L1 sinks in [0.56,1.14]; >1 makes the sink convention testable


def rec(case, **kw):
    kw["case"] = case
    line = json.dumps(kw, default=str)
    print("PROBE_RESULT " + line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if a.numel() != b.numel():
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def err(e):
    s = str(e).strip().splitlines()
    msg = " | ".join(x.strip() for x in s if x.strip())[:400]
    return f"{type(e).__name__}: {msg}"


def tt(dev, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, device=dev, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pad_last(t, n):
    return torch.nn.functional.pad(t, (0, n - t.shape[-1]))


def make_paged(x, page_perm_seed=0):
    """x [B, nkv, L, D] -> (paged [B*nb, nkv, BS, D], page_table [B, nb] int32), shuffled physical blocks."""
    B, nkv, L, D = x.shape
    nb = L // BS
    paged = x.reshape(B, nkv, nb, BS, D).transpose(1, 2).reshape(B * nb, nkv, BS, D)
    g = torch.Generator().manual_seed(page_perm_seed)
    perm = torch.randperm(B * nb, generator=g)
    inv = torch.argsort(perm)
    return paged[perm].contiguous(), inv.reshape(B, nb).to(torch.int32)


def ref_decode(q, K, V, pos, sink=None, window=None, nkv=1, key_pos=None):
    """q [B,nh,DQK]; K [B,nkv,L,DQK]; V [B,nkv,L,dv]; pos [B]. key_pos [B,L] absolute position of each cache slot (default arange).
    returns [B,nh,dv] fp32."""
    B, nh, _ = q.shape
    out = torch.zeros(B, nh, V.shape[-1])
    for b in range(B):
        p = int(pos[b])
        kp = torch.arange(K.shape[2]) if key_pos is None else key_pos[b]
        vis = kp <= p
        if window is not None:
            vis &= kp > p - window
        for h in range(nh):
            kv = h // (nh // nkv)
            s = (K[b, kv][vis].float() @ q[b, h].float()) * SCALE
            if sink is not None:
                s = torch.cat([s, sink[h].reshape(1).float()])
            w = torch.softmax(s, 0)
            if sink is not None:
                w = w[:-1]
            out[b, h] = w @ V[b, kv][vis].float()
    return out


def ref_prefill(q, k, v, q_start=0, sink=None, window=None):
    """q [1,nh,Sq,DQK] at absolute positions q_start..; k [1,nkv,Sk,DQK], v [1,nkv,Sk,dv] positions 0..Sk-1. returns [1,nh,Sq,dv]."""
    _, nh, Sq, _ = q.shape
    nkv = k.shape[1]
    Sk = k.shape[2]
    qi = torch.arange(Sq)[:, None] + q_start
    kj = torch.arange(Sk)[None, :]
    mask = kj <= qi
    if window is not None:
        mask &= kj > qi - window
    out = torch.zeros(1, nh, Sq, v.shape[-1])
    for h in range(nh):
        kv = h // (nh // nkv)
        s = (q[0, h].float() @ k[0, kv].float().T) * SCALE
        s = s.masked_fill(~mask, float("-inf"))
        if sink is not None:
            s = torch.cat([s, sink[h].reshape(1, 1).expand(Sq, 1).float()], 1)
        w = torch.softmax(s, 1)
        if sink is not None:
            w = w[:, :-1]
        out[0, h] = w @ v[0, kv].float()
    return out


def bench(dev, fn, n=20, warm=3):
    for _ in range(warm):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n * 1e3


def dec_cfg(dev, kchunk=128, mcores=None):
    kw = dict(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=kchunk,
        exp_approx_mode=False,
    )
    if mcores:
        kw["max_cores_per_head_batch"] = mcores
    return ttnn.SDPAProgramConfig(**kw)


def pre_cfg(dev, qc=128, kc=128):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=False,
    )


HIFI4 = lambda: ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
HIFI2 = lambda: ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)


class DecodeCase:
    """Builds paged K (192) / V (128 and padded 192) caches + q + sink and exposes op closures."""

    def __init__(self, dev, B, nh, nkv, Lc, pos, swa, sink_div="scale", seed=0, ring_mod=None, V_real=True):
        g = torch.Generator().manual_seed(seed)
        self.dev, self.B, self.nh, self.nkv, self.Lc, self.swa = dev, B, nh, nkv, Lc, swa
        self.pos = torch.tensor(pos, dtype=torch.int32)
        self.q = (torch.randn(B, nh, DQK, generator=g) * 1.0).bfloat16()
        self.K = (torch.randn(B, nkv, Lc, DQK, generator=g) * 1.0).bfloat16()
        self.V = (torch.randn(B, nkv, Lc, DV, generator=g) * 0.7).bfloat16()
        self.sink = (
            ((torch.rand(nh, generator=g) * 0.58 + 0.56) * SINK_MULT).bfloat16() if swa else None
        )  # study: L1 sinks in [0.56, 1.14]
        pk, self.pt = make_paged(self.K)
        pv, _ = make_paged(self.V)
        pv192, _ = make_paged(pad_last(self.V, DQK))
        self.tk, self.tv, self.tv192 = tt(dev, pk), tt(dev, pv), tt(dev, pv192)
        self.tpt = tt(dev, self.pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.tpos = tt(dev, self.pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.tq = tt(dev, self.q.reshape(1, B, nh, DQK))
        self.set_sink(sink_div)
        self.ring_mod = ring_mod

    def set_sink(self, div):
        if self.sink is None:
            self.tsink = None
            return
        nhp = ((self.nh + 31) // 32) * 32
        s = torch.zeros(nhp, 32)
        s[: self.nh, 0] = self.sink.float() / (SCALE if div == "scale" else 1.0)
        self.tsink = tt(self.dev, s.bfloat16())

    def kw(self, kchunk=128, mcores=None):
        d = dict(
            cur_pos_tensor=self.tpos,
            scale=SCALE,
            program_config=dec_cfg(self.dev, kchunk, mcores),
            compute_kernel_config=HIFI4(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.swa:
            d["sliding_window_size"] = WINDOW
            d["attention_sink"] = self.tsink
        return d

    def op_pad(self, **a):
        extra = {}
        if self.ring_mod:
            extra["cache_position_modulo"] = self.ring_mod
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.tq, self.tk, self.tv192, page_table_tensor=self.tpt, **self.kw(**a), **extra
        )

    def op_mla(self, **a):
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            self.tq, self.tk, self.tv, head_dim_v=DV, page_table_tensor=self.tpt, **self.kw(**a)
        )

    def op_mla_konly(self, **a):  # timing lower bound only: V = prefix of K (NOT MiMo semantics)
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            self.tq, self.tk, None, head_dim_v=DV, page_table_tensor=self.tpt, **self.kw(**a)
        )

    def ref(self):
        return ref_decode(
            self.q, self.K, self.V, self.pos, sink=self.sink, window=WINDOW if self.swa else None, nkv=self.nkv
        )

    def to_host(self, o):
        t = ttnn.to_torch(o)  # [1,B,nhp,d]
        return t[0, :, : self.nh, :DV].float()


def run_case(name, fn, **meta):
    try:
        r = fn()
        rec(name, ok=True, **meta, **(r or {}))
    except Exception as e:  # noqa
        rec(name, ok=False, error=err(e), **meta)
        traceback.print_exc()


# ----------------------------------------------------------------------------- decode
def g_dec_corr(dev):
    Lc = 2048
    for swa in (False, True):
        for B in (1, 32):
            if swa:
                pos = [5, 127, 128, 129, 1000, 2047, 300, 64] * 4
                pos = pos[:B] if B > 1 else [1000]
            else:
                pos = [0, 31, 63, 64, 1000, 2047, 1500, 777] * 4
                pos = pos[:B] if B > 1 else [2047]
            for sd in ["scale", "raw"] if swa else [None]:

                def f():
                    c = DecodeCase(dev, B, 8, 1, Lc, pos, swa, sink_div=sd or "scale")
                    ref = c.ref()
                    res = {}
                    for opn in ("pad192", "mla128"):
                        try:
                            o = c.op_pad() if opn == "pad192" else c.op_mla()
                            res[f"pcc_{opn}"] = round(pcc(ref, c.to_host(o)), 6)
                            # per-user min pcc
                            h = c.to_host(o)
                            res[f"minuser_{opn}"] = round(min(pcc(ref[b], h[b]) for b in range(B)), 6)
                        except Exception as e:  # noqa
                            res[f"err_{opn}"] = err(e)
                    return res

                run_case("dec_corr", f, swa=swa, B=B, nh=8, nkv=1, Lc=Lc, sink_div=sd, pos=pos[:8])


def g_dec_gqa(dev):
    # acceptance with un-sharded head counts (what a naive non-TP mapping would hand the op)
    for swa, nkv in ((True, 8), (False, 4)):

        def f():
            c = DecodeCase(dev, 4, 64, nkv, 1024, [1023, 500, 130, 5], swa, sink_div="scale")
            ref = c.ref()
            res = {}
            for opn in ("pad192", "mla128"):
                try:
                    o = c.op_pad() if opn == "pad192" else c.op_mla()
                    res[f"pcc_{opn}"] = round(pcc(ref, c.to_host(o)), 6)
                except Exception as e:  # noqa
                    res[f"err_{opn}"] = err(e)
            return res

        run_case("dec_gqa", f, swa=swa, nh=64, nkv=nkv, B=4)


def g_dec_ring(dev):
    # bounded ring cache: 256 slots, positions far past the window, non-MLA op with cache_position_modulo
    ring = 256
    for B, p in ((1, 5000), (4, 5000), (4, 255), (4, 256)):

        def f():
            g = torch.Generator().manual_seed(1)
            nh = 8
            hist = 6000
            Kh = (torch.randn(B, 1, hist, DQK, generator=g)).bfloat16()
            Vh = (torch.randn(B, 1, hist, DV, generator=g) * 0.7).bfloat16()
            posl = [p - i * 3 for i in range(B)]
            # ring slot j holds latest position <= max pos with pos % ring == j
            Kr = torch.zeros(B, 1, ring, DQK, dtype=torch.bfloat16)
            Vr = torch.zeros(B, 1, ring, DV, dtype=torch.bfloat16)
            kpos = torch.full((B, ring), -(10**9), dtype=torch.long)
            for b in range(B):
                for t in range(posl[b] + 1):
                    Kr[b, 0, t % ring] = Kh[b, 0, t]
                    Vr[b, 0, t % ring] = Vh[b, 0, t]
                    kpos[b, t % ring] = t
            q = torch.randn(B, nh, DQK, generator=g).bfloat16()
            sink = ((torch.rand(nh, generator=g) * 0.58 + 0.56) * SINK_MULT).bfloat16()
            ref = ref_decode(q, Kr, Vr, torch.tensor(posl), sink=sink, window=WINDOW, nkv=1, key_pos=kpos)
            pk, pt = make_paged(Kr)
            pv, _ = make_paged(pad_last(Vr, DQK))
            s = torch.zeros(32, 32)
            s[:nh, 0] = sink.float() / SCALE
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt(dev, q.reshape(1, B, nh, DQK)),
                tt(dev, pk),
                tt(dev, pv),
                page_table_tensor=tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                cur_pos_tensor=tt(dev, torch.tensor(posl, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                scale=SCALE,
                sliding_window_size=WINDOW,
                attention_sink=tt(dev, s.bfloat16()),
                cache_position_modulo=ring,
                program_config=dec_cfg(dev),
                compute_kernel_config=HIFI4(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            h = ttnn.to_torch(o)[0, :, :nh, :DV].float()
            return {"pcc": round(pcc(ref, h), 6), "minuser": round(min(pcc(ref[b], h[b]) for b in range(B)), 6)}

        run_case("dec_ring", f, B=B, pos=p, ring=ring)


def g_dec_perf(dev):
    n = int(os.environ.get("PROBE_N", "20"))
    # GA: 64k cache, no sink/window; compare pad192 vs mla128 (and K-only lower bound), k_chunk variants
    for B in (1, 8, 32):
        for pos in (65535, 32767):

            def f():
                c = DecodeCase(dev, B, 8, 1, 65536, [pos] * B, False)
                res = {}
                for kc in (128, 256, 512):
                    res[f"pad192_k{kc}_ms"] = round(bench(dev, lambda: c.op_pad(kchunk=kc), n), 4)
                    res[f"mla128_k{kc}_ms"] = round(bench(dev, lambda: c.op_mla(kchunk=kc), n), 4)
                    res[f"mlaKonly_k{kc}_ms"] = round(bench(dev, lambda: c.op_mla_konly(kchunk=kc), n), 4)
                return res

            run_case("dec_perf_ga", f, B=B, pos=pos, Lc=65536, nh=8, nkv=1)
            ttnn.synchronize_device(dev)
    # SWA: window 128 + sink; cache 2048 (no ring) paired pad vs mla; and ring (256 slots, modulo) for pad at pos 65535
    for B in (1, 8, 32):

        def f():
            c = DecodeCase(dev, B, 8, 1, 2048, [2047] * B, True)
            res = {}
            for kc in (128, 256):
                res[f"pad192_k{kc}_ms"] = round(bench(dev, lambda: c.op_pad(kchunk=kc), n), 4)
                res[f"mla128_k{kc}_ms"] = round(bench(dev, lambda: c.op_mla(kchunk=kc), n), 4)
            return res

        run_case("dec_perf_swa", f, B=B, pos=2047, Lc=2048, nh=8, nkv=1)

        def f2():
            c = DecodeCase(dev, B, 8, 1, 256, [65535] * B, True, ring_mod=256)
            return {"pad192_ring256_k128_ms": round(bench(dev, lambda: c.op_pad(kchunk=128), n), 4)}

        run_case("dec_perf_swa_ring", f2, B=B, pos=65535, ring=256)


# ----------------------------------------------------------------------------- prefill
def pf_inputs(S, nh, nkv, seed=0, Sk=None):
    g = torch.Generator().manual_seed(seed)
    Sk = Sk or S
    q = torch.randn(1, nh, S, DQK, generator=g).bfloat16()
    k = torch.randn(1, nkv, Sk, DQK, generator=g).bfloat16()
    v = (torch.randn(1, nkv, Sk, DV, generator=g) * 0.7).bfloat16()
    sink = ((torch.rand(nh, generator=g) * 0.58 + 0.56) * SINK_MULT).bfloat16()
    return q, k, v, sink


def sink_tt(dev, sink, div):
    nh = sink.numel()
    s = (sink.float() / (SCALE if div == "scale" else 1.0)).reshape(1, nh, 1, 1)
    return tt(dev, s.bfloat16())


def g_pre_corr(dev):
    S = 2048
    nh, nkv = 8, 1
    q, k, v, sink = pf_inputs(S, nh, nkv)
    tq, tk, tv, tv192 = tt(dev, q), tt(dev, k), tt(dev, v), tt(dev, pad_last(v, DQK))
    # SWA self-attention: sink convention + window
    for div in ("scale", "raw"):

        def f():
            ref = ref_prefill(q, k, v, sink=sink, window=WINDOW)
            o = ttnn.transformer.scaled_dot_product_attention(
                tq,
                tk,
                tv192,
                is_causal=True,
                scale=SCALE,
                sliding_window_size=WINDOW,
                attention_sink=sink_tt(dev, sink, div),
                program_config=pre_cfg(dev),
                compute_kernel_config=HIFI2(),
            )
            h = ttnn.to_torch(o)[..., :DV].float()
            return {"pcc": round(pcc(ref, h), 6)}

        run_case("pre_swa_self_pad192", f, S=S, sink_div=div)

    # window w/o sink and sink w/o window to isolate
    def f():
        ref = ref_prefill(q, k, v, window=WINDOW)
        o = ttnn.transformer.scaled_dot_product_attention(
            tq,
            tk,
            tv192,
            is_causal=True,
            scale=SCALE,
            sliding_window_size=WINDOW,
            program_config=pre_cfg(dev),
            compute_kernel_config=HIFI2(),
        )
        return {"pcc": round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)}

    run_case("pre_swa_self_pad192_window_only", f, S=S)

    # GA self attention: pad192 vs MLA explicit V
    def f():
        ref = ref_prefill(q, k, v)
        res = {}
        o = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv192, is_causal=True, scale=SCALE, program_config=pre_cfg(dev), compute_kernel_config=HIFI2()
        )
        res["pcc_pad192"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
        try:
            o2 = ttnn.transformer.flash_mla_prefill(
                tq, tk, tv, scale=SCALE, is_causal=True, program_config=pre_cfg(dev), compute_kernel_config=HIFI2()
            )
            h = ttnn.to_torch(o2)
            res["mla_out_shape"] = list(h.shape)
            res["pcc_mla_explicitV"] = round(pcc(ref, h[..., :DV].float()), 6)
        except Exception as e:  # noqa
            res["err_mla_explicitV"] = err(e)
        return res

    run_case("pre_ga_self", f, S=S, nh=nh, nkv=nkv)

    # chunked with paged cache (cache read): chunk 2048 at start 4096 of an 8192 cache
    Lc, start = 8192, 4096
    for swa in (True, False):

        def f():
            qc, kc_, vc, sk = pf_inputs(S, nh, nkv, seed=3, Sk=Lc)
            ref = ref_prefill(
                qc,
                kc_[:, :, : start + S],
                vc[:, :, : start + S],
                q_start=start,
                sink=sk if swa else None,
                window=WINDOW if swa else None,
            )
            pk, pt = make_paged(kc_)
            pv192, _ = make_paged(pad_last(vc, DQK))
            pv128, _ = make_paged(vc)
            args = dict(
                program_config=pre_cfg(dev), compute_kernel_config=HIFI2()
            )  # scale omitted: noconvert float binding rejects python float; default = DH**-0.5 = 192**-0.5
            if swa:
                args.update(sliding_window_size=WINDOW, attention_sink=sink_tt(dev, sk, "scale"))
            tpt = tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            res = {}
            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                tt(dev, qc), tt(dev, pk), tt(dev, pv192), tpt, start, **args
            )
            res["pcc_pad192"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
            try:
                o = ttnn.transformer.chunked_scaled_dot_product_attention(
                    tt(dev, qc), tt(dev, pk), tt(dev, pv128), tpt, start, **args
                )
                res["pcc_v128_native"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
            except Exception as e:  # noqa
                res["err_v128_native"] = err(e)
            if swa:
                # sink convention check on the chunked path too
                try:
                    a2 = dict(args)
                    a2["attention_sink"] = sink_tt(dev, sk, "raw")
                    o = ttnn.transformer.chunked_scaled_dot_product_attention(
                        tt(dev, qc), tt(dev, pk), tt(dev, pv192), tpt, start, **a2
                    )
                    res["pcc_pad192_rawsink"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
                except Exception as e:  # noqa
                    res["err_rawsink"] = err(e)
            try:  # chunked MLA: V is implied as K prefix -> not MiMo semantics; record acceptance only
                o = ttnn.transformer.chunked_flash_mla_prefill(
                    tt(dev, qc),
                    tt(dev, pk),
                    DV,
                    tpt,
                    start,
                    program_config=pre_cfg(dev),
                    compute_kernel_config=HIFI2(),
                )
                res["chunked_flash_mla_accepts"] = True
            except Exception as e:  # noqa
                res["err_chunked_flash_mla"] = err(e)
            return res

        run_case("pre_chunked", f, S=S, Lc=Lc, start=start, swa=swa)


def g_pre_gqa(dev):
    S = 512
    for swa, nkv in ((True, 8), (False, 4)):

        def f():
            q, k, v, sink = pf_inputs(S, 64, nkv)
            tq, tk, tv, tv192 = tt(dev, q), tt(dev, k), tt(dev, v), tt(dev, pad_last(v, DQK))
            ref = ref_prefill(q, k, v, sink=sink if swa else None, window=WINDOW if swa else None)
            res = {}
            a = dict(is_causal=True, scale=SCALE, program_config=pre_cfg(dev), compute_kernel_config=HIFI2())
            if swa:
                a.update(sliding_window_size=WINDOW, attention_sink=sink_tt(dev, sink, "scale"))
            try:
                o = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv192, **a)
                res["pcc_pad192"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
            except Exception as e:  # noqa
                res["err_pad192"] = err(e)
            if not swa:
                try:
                    o = ttnn.transformer.flash_mla_prefill(
                        tq,
                        tk,
                        tv,
                        scale=SCALE,
                        is_causal=True,
                        program_config=pre_cfg(dev),
                        compute_kernel_config=HIFI2(),
                    )
                    res["pcc_mla_explicitV"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
                except Exception as e:  # noqa
                    res["err_mla_explicitV"] = err(e)
            return res

        run_case("pre_gqa", f, S=S, nh=64, nkv=nkv, swa=swa)


def g_pre_perf(dev):
    n = int(os.environ.get("PROBE_N", "10"))
    nh, nkv = 8, 1
    for S in (2048, 8192):

        def f():
            q, k, v, sink = pf_inputs(S, nh, nkv)
            tq, tk, tv, tv192 = tt(dev, q), tt(dev, k), tt(dev, v), tt(dev, pad_last(v, DQK))
            res = {}
            for qc, kc in ((128, 128), (256, 256), (128, 256)):
                cfg = pre_cfg(dev, qc, kc)
                try:
                    res[f"ga_pad192_q{qc}k{kc}_ms"] = round(
                        bench(
                            dev,
                            lambda: ttnn.transformer.scaled_dot_product_attention(
                                tq,
                                tk,
                                tv192,
                                is_causal=True,
                                scale=SCALE,
                                program_config=cfg,
                                compute_kernel_config=HIFI2(),
                            ),
                            n,
                        ),
                        3,
                    )
                    res[f"ga_mla128_q{qc}k{kc}_ms"] = round(
                        bench(
                            dev,
                            lambda: ttnn.transformer.flash_mla_prefill(
                                tq,
                                tk,
                                tv,
                                scale=SCALE,
                                is_causal=True,
                                program_config=cfg,
                                compute_kernel_config=HIFI2(),
                            ),
                            n,
                        ),
                        3,
                    )
                    res[f"swa_pad192_q{qc}k{kc}_ms"] = round(
                        bench(
                            dev,
                            lambda: ttnn.transformer.scaled_dot_product_attention(
                                tq,
                                tk,
                                tv192,
                                is_causal=True,
                                scale=SCALE,
                                sliding_window_size=WINDOW,
                                attention_sink=sink_tt(dev, sink, "scale"),
                                program_config=cfg,
                                compute_kernel_config=HIFI2(),
                            ),
                            n,
                        ),
                        3,
                    )
                except Exception as e:  # noqa
                    res[f"err_q{qc}k{kc}"] = err(e)
            return res

        run_case("pre_perf_self", f, S=S, nh=nh, nkv=nkv)
    # chunked with 64k cache: chunk 2048 at start 61440 (GA) and SWA at start 61440 / 2048
    Lc = 65536
    q, k, v, sink = pf_inputs(2048, nh, nkv, Sk=Lc)
    pk, pt = make_paged(k)
    pv192, _ = make_paged(pad_last(v, DQK))
    tq, tk, tv, tpt = tt(dev, q), tt(dev, pk), tt(dev, pv192), tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    ts = sink_tt(dev, sink, "scale")
    for start in (61440, 30720, 2048):

        def f():
            res = {}
            for qc, kc in ((128, 128), (256, 256)):
                cfg = pre_cfg(dev, qc, kc)
                res[f"ga_pad192_q{qc}k{kc}_ms"] = round(
                    bench(
                        dev,
                        lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                            tq, tk, tv, tpt, start, program_config=cfg, compute_kernel_config=HIFI2()
                        ),
                        n,
                    ),
                    3,
                )
                res[f"swa_pad192_q{qc}k{kc}_ms"] = round(
                    bench(
                        dev,
                        lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                            tq,
                            tk,
                            tv,
                            tpt,
                            start,
                            program_config=cfg,
                            compute_kernel_config=HIFI2(),
                            sliding_window_size=WINDOW,
                            attention_sink=ts,
                        ),
                        n,
                    ),
                    3,
                )
            return res

        run_case("pre_perf_chunked_64kcache", f, S=2048, start=start, Lc=Lc)


def g_edge(dev):
    """Sliding-window boundary and determinism, with a needle: exactly one key (position j0) is attractive to q and carries a huge V.
    Rows/users whose window contains j0 produce a large output, others a small one. Window must contain exactly 128 keys incl self.
    """
    nh = 8
    g = torch.Generator().manual_seed(7)
    c = torch.randn(DQK, generator=g)
    c = c / c.norm() * 60.0  # q.k = 60*|k|/... large positive for the needle
    j0 = 1000

    def build(L):
        K = (torch.randn(1, 1, L, DQK, generator=g) * 0.2).bfloat16()
        V = (torch.randn(1, 1, L, DV, generator=g) * 0.2).bfloat16()
        K[0, 0, j0] = (c / 5.0).bfloat16()
        V[0, 0, j0] = 50.0
        return K, V

    # decode: users with pos = j0 + d, d in [-1, 0, 126, 127, 128, 129]
    ds = [-1, 0, 126, 127, 128, 129]
    L = 2048
    K, V = build(L)
    B = len(ds)
    Kb, Vb = K.expand(B, 1, L, DQK).contiguous(), V.expand(B, 1, L, DV).contiguous()
    q = c.reshape(1, 1, DQK).expand(B, nh, DQK).contiguous().bfloat16()
    pos = [j0 + d for d in ds]
    pk, pt = make_paged(Kb)
    pv, _ = make_paged(Vb)
    pv192, _ = make_paged(pad_last(Vb, DQK))
    tq = tt(dev, q.reshape(1, B, nh, DQK))
    tpt = tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    tpos = tt(dev, torch.tensor(pos, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    sink = torch.full((nh,), 1.0).bfloat16()
    s = torch.zeros(32, 32)
    s[:nh, 0] = sink.float() / SCALE
    ts = tt(dev, s.bfloat16())
    ref = ref_decode(q, Kb, Vb, torch.tensor(pos), sink=sink, window=WINDOW, nkv=1)
    ref_mag = ref.abs().amax(dim=(1, 2))

    def f():
        res = {"d": ds, "ref_absmax": [round(float(x), 2) for x in ref_mag]}
        kw = dict(
            cur_pos_tensor=tpos,
            scale=SCALE,
            sliding_window_size=WINDOW,
            attention_sink=ts,
            program_config=dec_cfg(dev),
            compute_kernel_config=HIFI4(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tq, tt(dev, pk), tt(dev, pv192), page_table_tensor=tpt, **kw
        )
        h = ttnn.to_torch(o)[0, :, :nh, :DV].float()
        res["dev_absmax_pad192"] = [round(float(x), 2) for x in h.abs().amax(dim=(1, 2))]
        res["pcc_pad192"] = round(pcc(ref, h), 5)
        o = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tq, tt(dev, pk), tt(dev, pv), head_dim_v=DV, page_table_tensor=tpt, **kw
        )
        h = ttnn.to_torch(o)[0, :, :nh, :DV].float()
        res["dev_absmax_mla128"] = [round(float(x), 2) for x in h.abs().amax(dim=(1, 2))]
        res["pcc_mla128"] = round(pcc(ref, h), 5)
        return res

    run_case("edge_decode_window", f)

    # prefill self + chunked: rows i = j0 + d
    S = 2048
    Kp, Vp = build(S)
    qp = c.reshape(1, 1, 1, DQK).expand(1, nh, S, DQK).contiguous().bfloat16()
    sinkp = torch.full((nh,), 1.0).bfloat16()
    rows = [j0 - 1, j0, j0 + 126, j0 + 127, j0 + 128, j0 + 129]

    def f2():
        res = {"rows": rows}
        refp = ref_prefill(qp[:, :, :1400], Kp[:, :, :1400], Vp[:, :, :1400], sink=sinkp, window=WINDOW)
        res["ref_absmax"] = [round(float(refp[0, :, r].abs().max()), 2) for r in rows]
        o = ttnn.transformer.scaled_dot_product_attention(
            tt(dev, qp),
            tt(dev, Kp),
            tt(dev, pad_last(Vp, DQK)),
            is_causal=True,
            scale=SCALE,
            sliding_window_size=WINDOW,
            attention_sink=sink_tt(dev, sinkp, "scale"),
            program_config=pre_cfg(dev),
            compute_kernel_config=HIFI2(),
        )
        h = ttnn.to_torch(o)[..., :DV].float()
        res["dev_self_absmax"] = [round(float(h[0, :, r].abs().max()), 2) for r in rows]
        res["pcc_self_rows<1400"] = round(pcc(refp, h[:, :, :1400]), 5)
        return res

    run_case("edge_prefill_self_window", f2)

    def f3():
        # chunked: cache 8192, q chunk 2048 at start 4096, needle at 4096+200
        Lc, start = 8192, 4096
        jn = start + 200
        Kc = (torch.randn(1, 1, Lc, DQK, generator=g) * 0.2).bfloat16()
        Vc = (torch.randn(1, 1, Lc, DV, generator=g) * 0.2).bfloat16()
        Kc[0, 0, jn] = (c / 5.0).bfloat16()
        Vc[0, 0, jn] = 50.0
        pk, pt = make_paged(Kc)
        pv192, _ = make_paged(pad_last(Vc, DQK))
        rr = [jn - 1, jn, jn + 127, jn + 128]
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt(dev, qp),
            tt(dev, pk),
            tt(dev, pv192),
            tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            start,
            program_config=pre_cfg(dev),
            compute_kernel_config=HIFI2(),
            sliding_window_size=WINDOW,
            attention_sink=sink_tt(dev, sinkp, "scale"),
        )
        h = ttnn.to_torch(o)[..., :DV].float()
        refc = ref_prefill(
            qp[:, :, : (jn + 140 - start)],
            Kc[:, :, : jn + 140],
            Vc[:, :, : jn + 140],
            q_start=start,
            sink=sinkp,
            window=WINDOW,
        )
        return {
            "rows_abs": rr,
            "dev_absmax": [round(float(h[0, :, r - start].abs().max()), 2) for r in rr],
            "ref_absmax": [round(float(refc[0, :, r - start].abs().max()), 2) for r in rr],
        }

    run_case("edge_prefill_chunked_window", f3)

    # determinism: 5 repeats bit-identical
    def f4():
        c2 = DecodeCase(dev, 32, 8, 1, 2048, [5, 127, 128, 129, 1000, 2047, 300, 64] * 4, True)
        outs = {"pad192": [], "mla128": []}
        for _ in range(5):
            outs["pad192"].append(ttnn.to_torch(c2.op_pad()))
            outs["mla128"].append(ttnn.to_torch(c2.op_mla()))
        return {k: all(torch.equal(v[0], x) for x in v[1:]) for k, v in outs.items()}

    run_case("determinism_decode_5x", f4)

    def f5():
        q_, k_, v_, sk_ = pf_inputs(2048, 8, 1, seed=11, Sk=8192)
        pk, pt = make_paged(k_)
        pv192, _ = make_paged(pad_last(v_, DQK))
        a = [tt(dev, q_), tt(dev, pk), tt(dev, pv192), tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)]
        ts_ = sink_tt(dev, sk_, "scale")
        outs = [
            ttnn.to_torch(
                ttnn.transformer.chunked_scaled_dot_product_attention(
                    *a,
                    4096,
                    program_config=pre_cfg(dev),
                    compute_kernel_config=HIFI2(),
                    sliding_window_size=WINDOW,
                    attention_sink=ts_,
                )
            )
            for _ in range(5)
        ]
        ga = [
            ttnn.to_torch(
                ttnn.transformer.chunked_scaled_dot_product_attention(
                    *a, 4096, program_config=pre_cfg(dev), compute_kernel_config=HIFI2()
                )
            )
            for _ in range(5)
        ]
        return {"swa": all(torch.equal(outs[0], x) for x in outs[1:]), "ga": all(torch.equal(ga[0], x) for x in ga[1:])}

    run_case("determinism_chunked_5x", f5)


def g_pre_long(dev):
    """chunked prefill correctness and tile-size perf at a 64k cache, chunk 2048 at start 61440"""
    Lc, start, S = 65536, 61440, 2048
    q, k, v, sk = pf_inputs(S, 8, 1, seed=5, Sk=Lc)
    pk, pt = make_paged(k)
    pv192, _ = make_paged(pad_last(v, DQK))
    tq, tk, tv, tpt = tt(dev, q), tt(dev, pk), tt(dev, pv192), tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    ts = sink_tt(dev, sk, "scale")
    for swa in (False, True):

        def f():
            ref = ref_prefill(
                q,
                k[:, :, : start + S],
                v[:, :, : start + S],
                q_start=start,
                sink=sk if swa else None,
                window=WINDOW if swa else None,
            )
            kw = dict(sliding_window_size=WINDOW, attention_sink=ts) if swa else {}
            res = {}
            for qc, kc in ((256, 256), (512, 512)):
                try:
                    cfg = pre_cfg(dev, qc, kc)
                    o = ttnn.transformer.chunked_scaled_dot_product_attention(
                        tq, tk, tv, tpt, start, program_config=cfg, compute_kernel_config=HIFI2(), **kw
                    )
                    res[f"pcc_q{qc}k{kc}"] = round(pcc(ref, ttnn.to_torch(o)[..., :DV].float()), 6)
                    res[f"ms_q{qc}k{kc}"] = round(
                        bench(
                            dev,
                            lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                                tq, tk, tv, tpt, start, program_config=cfg, compute_kernel_config=HIFI2(), **kw
                            ),
                            10,
                        ),
                        3,
                    )
                except Exception as e:  # noqa
                    res[f"err_q{qc}k{kc}"] = err(e)
            return res

        run_case("pre_long_chunked", f, swa=swa, Lc=Lc, start=start)


def g_pre_dbg(dev):
    """bisect which kwarg the chunked op binding rejects"""
    Lc, start, S = 8192, 4096, 2048
    qc, kc_, vc, sk = pf_inputs(S, 8, 1, seed=3, Sk=Lc)
    pk, pt = make_paged(kc_)
    pv192, _ = make_paged(pad_last(vc, DQK))
    tq, tk, tv, tpt = tt(dev, qc), tt(dev, pk), tt(dev, pv192), tt(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    variants = {
        "min": dict(),
        "pcfg": dict(program_config=pre_cfg(dev)),
        "ckc": dict(compute_kernel_config=HIFI2()),
        "scale": dict(scale=SCALE),
        "window": dict(sliding_window_size=WINDOW),
        "sink": dict(attention_sink=sink_tt(dev, sk, "scale")),
        "kw_start": dict(),
    }
    for name, kw in variants.items():

        def f():
            if name == "kw_start":
                o = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=tq,
                    input_tensor_k=tk,
                    input_tensor_v=tv,
                    page_table_tensor=tpt,
                    chunk_start_idx=start,
                    **kw,
                )
            else:
                o = ttnn.transformer.chunked_scaled_dot_product_attention(tq, tk, tv, tpt, start, **kw)
            return {"shape": list(o.shape)}

        run_case("pre_dbg", f, variant=name)

    edge = (g_edge,)
    pre_long = (g_pre_long,)


GROUPS = dict(
    edge=g_edge,
    pre_long=g_pre_long,
    pre_dbg=g_pre_dbg,
    dec_corr=g_dec_corr,
    dec_gqa=g_dec_gqa,
    dec_ring=g_dec_ring,
    dec_perf=g_dec_perf,
    pre_corr=g_pre_corr,
    pre_gqa=g_pre_gqa,
    pre_perf=g_pre_perf,
)

if __name__ == "__main__":
    groups = os.environ.get("PROBE", "dec_corr").split(",")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        rec(
            "env",
            groups=groups,
            sink_mult=SINK_MULT,
            grid=str(dev.compute_with_storage_grid_size()),
            arch=str(dev.arch()) if hasattr(dev, "arch") else "?",
        )
        for gname in groups:
            t0 = time.time()
            try:
                GROUPS[gname](dev)
            except Exception as e:  # noqa
                rec("group_fail", group=gname, error=err(e))
                traceback.print_exc()
            rec("group_done", group=gname, secs=round(time.time() - t0, 1))
    finally:
        ttnn.close_mesh_device(dev)
