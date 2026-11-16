"""Sweep DRAM-sharded decode matmul configs for Llama-3.3-70B TP8 per-chip shapes on one Wormhole chip.

Each config is timed by replaying a trace of ITERS back-to-back matmuls, so host dispatch is excluded.
"""

import itertools
import json
import math
import sys
import time

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc

sys.path.insert(0, "tests/ttnn/unit_tests/operations/matmul")
from test_matmul_dram_sharded_multicore import _matmul, _operands, _reference  # noqa: E402

ITERS = 200
DRAM_GBPS = 288.0
BYTES_PER_ELEM = {"bfp8": 1088 / 1024, "bfp4": 576 / 1024}

# name: (K, N, weight, activation, output, activation grid, activation shard width (tiles), per_core_N,
#        fidelity, fp32 Dest, stock in0_block_w)
SHAPES = {
    "qkv": (8192, 1280, "bfp8", "bf16", "bf16", ("rect", 8, 4), 8, 2, "HiFi2", True, 16),
    "wo": (1024, 8192, "bfp8", "bf16", "bf16", ("rect", 8, 1), 4, 32, "HiFi2", True, 4),
    "w1w3": (8192, 3584, "bfp4", "bf16", "bf16", ("rect", 8, 2), 16, 7, "LoFi", False, 8),
    "w2": (3584, 8192, "bfp8", "bfp8", "bf16", ("rect", 8, 2), 7, 16, "HiFi2", False, 7),
}


def time_config(device, x, w, kw):
    out = _matmul(x, w, **kw)
    ttnn.synchronize_device(device)
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(ITERS):
        o = _matmul(x, w, **kw)
        o.deallocate(True)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / ITERS
    ttnn.release_trace(device, tid)
    return dt * 1e6


def main():
    names = sys.argv[1:] or list(SHAPES)
    device = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
    results = []
    try:
        for name in names:
            k, n, wd, xd, od, xl, xsw, pcn, fid, fp32, stock_ibw = SHAPES[name]
            x, w = _operands(device, k, n, wd, xd, xl, xsw)
            ref = _reference(x, w, n)
            bound_us = k * n * BYTES_PER_ELEM[wd] / (DRAM_GBPS * 1e3)
            k_tiles = k // 32
            ibws = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0 and d <= 64]
            for ibw, cpb, fp in itertools.product(ibws, [0, 2, 3, 4], sorted({fp32, False})):
                kw = dict(
                    out_mc=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    out_dtype=od,
                    per_core_n=pcn,
                    in0_block_w=ibw,
                    cores_per_bank=cpb,
                    fidelity=fid,
                    fp32=fp,
                )
                rec = dict(shape=name, in0_block_w=ibw, cores_per_bank=cpb, fp32=fp)
                try:
                    out = _matmul(x, w, **kw)
                    ok, pcc = comp_pcc(ref, ttnn.to_torch(out).double().reshape(ref.shape[0], -1)[:, :n], 0.999)
                    out.deallocate(True)
                    us = time_config(device, x, w, kw)
                    rec.update(us=round(us, 2), pcc=str(pcc)[:8], pcc_ok=bool(ok), bound_us=round(bound_us, 1))
                    rec["stock"] = ibw == stock_ibw and cpb == 0 and fp == fp32
                except Exception as e:
                    rec.update(error=str(e).splitlines()[0][:160])
                results.append(rec)
                print(json.dumps(rec), flush=True)
            x.deallocate(True)
            w.deallocate(True)
    finally:
        ttnn.close_device(device)
    with open(f"/home/ubuntu/teja/sweep/results_{'_'.join(names)}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
