# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""math_thread_row_sums A/B: traced us/call and PCC, flag off vs on in the same binary.

  python bench_row_sums.py <arms.json>

Each arm is a dict of overrides on DEFAULTS. Prints one RES line per arm.
"""
import json
import statistics
import sys
import time

import torch

import ttnn

DEFAULTS = dict(
    name="?",
    B=8,
    nqh=32,
    nkh=8,
    s=512,
    d=128,
    qc=128,
    kc=512,
    grid=[12, 8],
    fid="LoFi",
    exp_approx=True,
    placement="kvl1",
    reuse_kv=True,
    concat=True,
    rounds=5,
    n=4,
    dtype="bfp8",
    pack_gqa=False,
    skv=None,  # separate K/V sequence length (cross attention); default = s
)

FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
DT = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def traced(D, fn, n=1):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(D)
    tid = ttnn.begin_trace_capture(D, cq_id=0)
    outs = [fn() for _ in range(n)]
    ttnn.end_trace_capture(D, tid, cq_id=0)
    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(8):
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / 8 / n * 1e6
    ttnn.release_trace(D, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def ref_attention(q, k, v, scale):
    # q [B,nqh,s,d], k/v [B,nkh,skv,d]; GQA broadcast
    B, nqh, s, d = q.shape
    nkh = k.shape[1]
    k = k.repeat_interleave(nqh // nkh, dim=1)
    v = v.repeat_interleave(nqh // nkh, dim=1)
    w = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
    return w @ v


def run_arm(D, a):
    B, nqh, nkh, s, d = a["B"], a["nqh"], a["nkh"], a["s"], a["d"]
    skv = a["skv"] or s
    dt = DT[a["dtype"]]
    scale = d**-0.5
    CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FID[a["fid"]], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*a["grid"]),
        q_chunk_size=a["qc"],
        k_chunk_size=a["kc"],
        exp_approx_mode=a["exp_approx"],
    )
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    qm, kvm, om = {
        "dram": (DRAM, DRAM, DRAM),
        "kvl1": (DRAM, L1, DRAM),
        "l1": (L1, L1, L1),
    }[a["placement"]]

    torch.manual_seed(0)
    q_h = torch.randn(B, nqh, s, d)
    k_h = torch.randn(B, nkh, skv, d)
    v_h = torch.randn(B, nkh, skv, d)

    q = ttnn.from_torch(q_h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=qm)
    k = ttnn.from_torch(k_h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)
    v = ttnn.from_torch(v_h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=kvm)

    def mk(flag):
        return lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=scale,
            program_config=cfg,
            compute_kernel_config=CK,
            memory_config=om,
            output_concat_heads=a["concat"],
            pack_gqa_heads=a["pack_gqa"],
            reuse_kv=a["reuse_kv"],
            math_thread_row_sums=flag,
        )

    # Accuracy + "did the flag change anything" check.
    outs = {}
    for flag in (False, True):
        t = mk(flag)()
        o = ttnn.to_torch(t)
        ttnn.deallocate(t)
        outs[flag] = o.reshape(B, s, nqh, d).permute(0, 2, 1, 3) if a["concat"] else o
    ref = ref_attention(q_h, k_h, v_h, scale)
    pcc_off, pcc_on = pcc(outs[False], ref), pcc(outs[True], ref)
    identical = bool(torch.equal(outs[False], outs[True]))

    # Latency: alternate off/on each round so clock drift cancels.
    off, on = [], []
    for r in range(a["rounds"]):
        order = (False, True) if r % 2 == 0 else (True, False)
        for flag in order:
            (off if not flag else on).append(traced(D, mk(flag), n=a["n"]))
    for t in (q, k, v):
        ttnn.deallocate(t)

    mo, mn = statistics.median(off), statistics.median(on)
    return dict(
        name=a["name"],
        us_off=mo,
        us_on=mn,
        delta_pct=(mn - mo) / mo * 100.0,
        pcc_off=pcc_off,
        pcc_on=pcc_on,
        identical=identical,
        off_all=[round(x, 1) for x in off],
        on_all=[round(x, 1) for x in on],
    )


def main():
    arms = json.load(open(sys.argv[1]))
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200 * 1024 * 1024)
    results = []
    try:
        for spec in arms:
            a = dict(DEFAULTS)
            a.update(spec)
            try:
                r = run_arm(D, a)
            except Exception as e:
                import traceback

                traceback.print_exc()
                msg = (str(e) or repr(e)).splitlines()
                r = dict(name=a["name"], error=(msg[0] if msg else repr(e))[:300])
            results.append(r)
            print("RES " + json.dumps(r), flush=True)
    finally:
        ttnn.close_device(D)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out:
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
