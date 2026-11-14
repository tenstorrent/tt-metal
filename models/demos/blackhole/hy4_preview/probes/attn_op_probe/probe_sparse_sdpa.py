"""attn-op-probe: stock ttnn.transformer.sparse_sdpa on Hy4 MLA+DSA shapes, one chip. Run inside the DSV4 image (see run_probe.sh).
Env: CASES=comma list of case names (default all), NREP (timing reps, default 30).
Prints one line per case: RES {json}.
Reference = torch fp32 sparse MLA with sink (logit domain, exp(sink) added to the softmax denominator), K=576, V=K[:, :512].
"""
import json
import os
import time

import torch

import ttnn

H_REAL = 64
KD, VD, TOPK = 576, 512, 2048
SCALE = 256**-0.5  # Hy4: qk head dim 192 nope + 64 rope = 256 (U: no yarn mscale)
SENT = 0xFFFFFFFF
NREP = int(os.environ.get("NREP", "30"))


def ref_sparse(q, kv, idx, sink_logit):
    """q [H,S,576] bf16, kv [T,576] bf16, idx [S,K] int64 (-1 = masked), sink_logit [H] fp32 (logit domain). -> [H,S,512] fp32"""
    H, S, _ = q.shape
    out = torch.zeros(H, S, VD)
    kvf = kv.float()
    for s in range(S):
        ix = idx[s]
        ix = ix[ix >= 0]
        k = kvf[ix]  # [n,576]
        sc = (q[:, s].float() @ k.T) * SCALE  # [H,n]
        m = torch.maximum(sc.max(dim=1).values, sink_logit)
        p = torch.exp(sc - m[:, None])
        den = p.sum(dim=1) + torch.exp(sink_logit - m)
        out[:, s] = (p @ k[:, :VD]) / den[:, None]
    return out


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def pow2_scales_pack(kv):
    """host pack of [T,576] bf16 -> packed uint8 [T,656] + dequantized bf16 [T,576]. scale per 128 block = 2^ceil(log2(amax/448))."""
    T = kv.shape[0]
    lat = kv[:, :VD].float().reshape(T, VD // 128, 128)
    amax = lat.abs().amax(dim=-1).clamp(min=1e-12)
    sc = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q8 = (lat / sc[..., None]).to(torch.float8_e4m3fn)
    deq = (q8.float() * sc[..., None]).reshape(T, VD)
    packed = torch.empty(T, 656, dtype=torch.uint8)
    packed[:, :VD] = q8.reshape(T, VD).view(torch.uint8)
    packed[:, VD : VD + 16] = sc.contiguous().view(torch.uint8).reshape(T, 16)
    packed[:, VD + 16 :] = kv[:, VD:].contiguous().view(torch.uint8).reshape(T, 128)
    return packed, torch.cat([deq.to(torch.bfloat16), kv[:, VD:]], dim=-1)


def to_dev(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_idx(S, topk, regions, valid):
    """regions: per query row (lo, hi) key range; valid: number of valid entries (rest sentinel tail). returns [S,topk] int64 (-1 = masked)"""
    idx = torch.full((S, topk), -1, dtype=torch.int64)
    for s in range(S):
        lo, hi = regions[s]
        n = min(valid, hi - lo)
        idx[s, :n] = lo + torch.randperm(hi - lo)[:n]
    return idx


def run_case(md, name, H, Hreal, S, T, fmt, valid=TOPK, kchunk=128, fid="hifi4", users_regions=True, nrep=NREP):
    torch.manual_seed(0)
    # kv pool: random data, a few hot rows
    kv = (torch.randn(T, KD) * 1.0).to(torch.bfloat16)
    per = T // S if users_regions else T
    regions = [((s * per), ((s + 1) * per)) if users_regions else (0, T) for s in range(S)]
    # real context length per user caps the region used for random picks (PCC cases keep T small)
    idx = make_idx(S, TOPK, regions, valid)
    q = (torch.randn(Hreal, S, KD)).to(torch.bfloat16)
    sink_logit = torch.linspace(-2.0, 2.0, Hreal)
    qpad = torch.zeros(H, S, KD, dtype=torch.bfloat16)
    qpad[:Hreal] = q
    sink_pad = torch.zeros(H)
    sink_pad[:Hreal] = sink_logit
    if fmt == "scaled_fp8":
        dp = ttnn.experimental.deepseek_prefill
        t_lat = to_dev(md, kv[:, :VD].reshape(1, 1, T, VD), ttnn.bfloat16)
        t_rope = to_dev(md, kv[:, VD:].reshape(1, 1, T, KD - VD), ttnn.bfloat16)
        t_fp8, t_sc = dp.per_token_cast_to_fp8(t_lat, round_scale_to_power_of_two=True)
        kv_dev = dp.pack_scaled_fp8_kv_cache(t_fp8, t_sc, t_rope)
        rec = ttnn.to_torch(dp.per_token_cast_back(t_fp8, t_sc, output_dtype=ttnn.bfloat16)).reshape(T, VD)
        kv_ref = torch.cat([rec.to(torch.bfloat16), kv[:, VD:]], dim=-1)
        for t in (t_lat, t_rope, t_fp8, t_sc):
            ttnn.deallocate(t)
        kvf = ttnn.transformer.SparseKVFormat.SCALED_FP8
    else:
        kv_ref = kv
        kv_dev = to_dev(md, kv.reshape(1, 1, T, KD), ttnn.bfloat16)
        kvf = ttnn.transformer.SparseKVFormat.BF16
    q_dev = to_dev(md, qpad.reshape(1, H, S, KD), ttnn.bfloat16)
    ix = idx.clone()
    ix[ix < 0] = SENT & 0xFFFFFFFF
    # uint32 via int64->int32 reinterpretation
    ix32 = torch.where(idx < 0, torch.full_like(idx, -1), idx).to(torch.int32)
    idx_dev = to_dev(md, ix32.reshape(1, 1, S, TOPK), ttnn.uint32)
    # sink: model sink in logit domain -> pass sink/scale (op multiplies by scale)
    sink_dev = to_dev(md, (sink_pad / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16)
    arch_cfg = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}[fid],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def call():
        return ttnn.transformer.sparse_sdpa(
            q_dev,
            kv_dev,
            idx_dev,
            VD,
            kv_format=kvf,
            scale=SCALE,
            k_chunk_size=kchunk,
            compute_kernel_config=arch_cfg,
            attention_sink=sink_dev,
        )

    res = {
        "case": name,
        "H": H,
        "Hreal": Hreal,
        "S": S,
        "T": T,
        "fmt": fmt,
        "valid": valid,
        "kchunk": kchunk,
        "fid": fid,
    }
    t0 = time.time()
    out = call()
    ttnn.synchronize_device(md)
    res["first_call_s"] = round(time.time() - t0, 2)
    o = ttnn.to_torch(out)[0, :Hreal].float()  # [Hreal,S,512]
    ref = ref_sparse(q, kv_ref, idx, sink_logit)
    res["pcc_vs_dequant_ref"] = round(pcc(o, ref), 6)
    if fmt == "scaled_fp8":
        ref0 = ref_sparse(q, kv, idx, sink_logit)
        res["pcc_vs_bf16_ref"] = round(pcc(o, ref0), 6)
    res["max_abs_err"] = round((o - ref).abs().max().item(), 5)
    ttnn.deallocate(out)
    for _ in range(3):
        ttnn.deallocate(call())
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(nrep):
        ttnn.deallocate(call())
    ttnn.synchronize_device(md)
    res["us_per_call"] = round((time.perf_counter() - t0) / nrep * 1e6, 1)
    # traced replay (removes host dispatch)
    try:
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(10):
            ttnn.deallocate(call()) if False else call()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        res["us_per_call_traced"] = round((time.perf_counter() - t0) / 30 * 1e6, 1)
        ttnn.release_trace(md, tid)
    except Exception as e:  # noqa
        res["trace_err"] = str(e)[:200]
    print("RES " + json.dumps(res), flush=True)
    for t in (q_dev, kv_dev, idx_dev, sink_dev):
        ttnn.deallocate(t)


CASES = {
    # name: kwargs. Reference cost grows with S, so PCC cases use moderate T.
    # decode-shaped: S = users on this chip, H = 32 padded (8 real), flat pool, per-user region
    "dec_h32r8_s1_bf16": dict(H=32, Hreal=8, S=1, T=65536, fmt="bf16"),
    "dec_h32r8_s4_bf16": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16"),
    "dec_h32r8_s4_fp8": dict(H=32, Hreal=8, S=4, T=65536, fmt="scaled_fp8"),
    "dec_h32r8_s1_fp8": dict(H=32, Hreal=8, S=1, T=65536, fmt="scaled_fp8"),
    "dec_h32r8_s16_bf16": dict(H=32, Hreal=8, S=16, T=131072, fmt="bf16"),
    "dec_h32r8_s16_fp8": dict(H=32, Hreal=8, S=16, T=131072, fmt="scaled_fp8"),
    "dec_h64_s1_bf16": dict(H=64, Hreal=64, S=1, T=65536, fmt="bf16"),
    "dec_h64_s1_fp8": dict(H=64, Hreal=64, S=1, T=65536, fmt="scaled_fp8"),
    "dec_h32full_s1_bf16": dict(H=32, Hreal=32, S=1, T=65536, fmt="bf16"),
    "dec_h32full_s1_fp8": dict(H=32, Hreal=32, S=1, T=65536, fmt="scaled_fp8"),
    "dec_h8_s1_bf16": dict(H=8, Hreal=8, S=1, T=65536, fmt="bf16"),  # expected FAIL: H%32
    "dec_h32r8_s4_bf16_valid512": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", valid=512),
    "dec_h32r8_s4_bf16_valid1500": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", valid=1500),
    "dec_h32r8_s4_bf16_kc256": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", kchunk=256),
    "dec_h32r8_s4_bf16_kc64": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", kchunk=64),
    "dec_h32r8_s4_bf16_hifi2": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", fid="hifi2"),
    # prefill chunk 512 rows, causal-ish random picks from a pool (single user region)
    "pf_h32r8_s512_bf16": dict(H=32, Hreal=8, S=512, T=16384, fmt="bf16", users_regions=False, nrep=10),
    "pf_h32r8_s512_fp8": dict(H=32, Hreal=8, S=512, T=16384, fmt="scaled_fp8", users_regions=False, nrep=10),
    "pf_h64_s64_bf16": dict(H=64, Hreal=64, S=64, T=16384, fmt="bf16", users_regions=False, nrep=10),
    "pf_h64_s64_fp8": dict(H=64, Hreal=64, S=64, T=16384, fmt="scaled_fp8", users_regions=False, nrep=10),
    "pf_h64_s512_bf16": dict(H=64, Hreal=64, S=512, T=16384, fmt="bf16", users_regions=False, nrep=5),
    "pf_h64_s512_fp8": dict(H=64, Hreal=64, S=512, T=16384, fmt="scaled_fp8", users_regions=False, nrep=5),
    "pf_h32full_s64_bf16": dict(H=32, Hreal=32, S=64, T=16384, fmt="bf16", users_regions=False, nrep=10),
    # latency vs context pool size (1M-row bf16 pool = 1.2 GB; fp8 = 0.7 GB)
    "dec_h32r8_s4_bf16_T1M": dict(H=32, Hreal=8, S=4, T=1048576, fmt="bf16"),
    "dec_h32r8_s4_fp8_T1M": dict(H=32, Hreal=8, S=4, T=1048576, fmt="scaled_fp8"),
    # round 2: scaling in rows, fp8 tuning
    "dec_h32r8_s2_bf16": dict(H=32, Hreal=8, S=2, T=65536, fmt="bf16"),
    "dec_h32r8_s8_bf16": dict(H=32, Hreal=8, S=8, T=131072, fmt="bf16"),
    "dec_h32r8_s32_bf16": dict(H=32, Hreal=8, S=32, T=262144, fmt="bf16"),
    "dec_h32r8_s64_bf16": dict(H=32, Hreal=8, S=64, T=262144, fmt="bf16"),
    "dec_h32r8_s120_bf16": dict(H=32, Hreal=8, S=120, T=524288, fmt="bf16"),
    "dec_h32r8_s130_bf16": dict(H=32, Hreal=8, S=130, T=524288, fmt="bf16"),
    "dec_h32r8_s4_fp8_kc256": dict(H=32, Hreal=8, S=4, T=65536, fmt="scaled_fp8", kchunk=256),
    "dec_h32r8_s4_fp8_hifi2": dict(H=32, Hreal=8, S=4, T=65536, fmt="scaled_fp8", fid="hifi2"),
    "dec_h32r8_s4_bf16_kc256_hifi2": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", kchunk=256, fid="hifi2"),
    "dec_h32r8_s4_fp8_kc256_hifi2": dict(H=32, Hreal=8, S=4, T=65536, fmt="scaled_fp8", kchunk=256, fid="hifi2"),
    "dec_h32r8_s4_bf16_kc512_hifi2": dict(H=32, Hreal=8, S=4, T=65536, fmt="bf16", kchunk=512, fid="hifi2"),
    "dec_h64_s4_bf16": dict(H=64, Hreal=64, S=4, T=65536, fmt="bf16"),
    "dec_h64_s8_bf16": dict(H=64, Hreal=64, S=8, T=131072, fmt="bf16"),
    "dec_h16_s1_bf16": dict(H=16, Hreal=16, S=1, T=65536, fmt="bf16"),
}


def main():
    sel = os.environ.get("CASES")
    names = sel.split(",") if sel else list(CASES)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        for n in names:
            kw = dict(CASES[n])
            try:
                run_case(md, n, **kw)
            except Exception as e:  # noqa
                print(
                    "RES "
                    + json.dumps(
                        {"case": n, "error": str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e)}
                    ),
                    flush=True,
                )
                print("ERRDETAIL", n, str(e)[:1500].replace("\n", " | "), flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
