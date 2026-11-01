# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""b) The real Bfp2_b weight read: can the STOCK matmul consume a 2-bit weight?

This is the measurement the PQ2_0 branch hinges on. The floors fit predicts
1.50-1.57x over bfp4 at bfp2's byte count (27.85 MB against 50.14 MB on
ffn_down), and the only way to know is to read actual Bfp2_b tiles.

Why this may need no kernel at all: `DataType::BFLOAT2_B` now exists in the tree
and `datatype_to_dataformat_converter` maps it to `DataFormat::Bfp2_b`, which the
LLK already unpacks (cunpack_common.h treats Bfp2_b as a B-side block float that
expands to BF16). So `ttnn.linear(x, w)` should pick up the format from the
tensor's dtype and configure itself, exactly as it does for bfloat4_b. If that
works, (b) is a measurement, not a factory fork -- which is worth knowing before
writing any kernel.

The bench is staged so a failure localises itself:

  1. **Pack** -- build a bfloat2_b tensor from ternary codes and read it back.
     Checks the codes round-trip through the host packer before any matmul, so a
     packing bug is not mistaken for a matmul bug. The authoritative encoding is
     established elsewhere (01=+1, 11=-1, 00=0; 320 B tiles; 64 B exponents then
     256 B of row-major-quadrant data), so this asserts against it.
  2. **Correct** -- one ttnn.linear against a float64 reference on the same codes.
     Requires codes_exact and PCC >= the gate.
  3. **Fast** -- device-kernel-duration timing against bfp4 on the same shape and
     config, which is the 1.4x question.

Construction of the bfloat2_b tensor is attempted by several routes because which
ones exist depends on how far bfloat2 support reaches into the host tensor path;
the first that works is reported and used.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2 --shape ffn_down
    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2 --stage pack
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from .a1_calib import device_kernel_us
from .placement import act_mem, describe, out_mem
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

#: (name, K, N, group, in0_block_w, cores) -- the bfp4-best config for each shape.
SHAPES = [
    ("ffn_down", 17408, 5120, 128, 4, 110),
    ("ffn_gate", 5120, 17408, 128, 4, 110),
    ("gdn_qkvz", 5120, 16384, 128, 4, 110),
    ("attn_q_gate", 5120, 12288, 128, 4, 110),
]


def weight_mb(n: int, k: int, fmt: str) -> float:
    if fmt == "bfp8":
        return (n * k + n * k // 16) / 1e6
    if fmt == "bfp4":
        return (n * k // 2 + n * k // 16) / 1e6
    if fmt == "bfp2":
        return (n * k // 4 + n * k // 16) / 1e6
    raise ValueError(fmt)


def pcc(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def make_bfp2(ttnn, device, codes_f32: np.ndarray, mem=None):
    """Ternary codes [N, K] -> a device bfloat2_b tensor [K, N] for linear's in1.

    Routes are tried in order because how far bfloat2 reaches into the host tensor
    path is not something this bench can assume. The first that produces a tensor
    is returned with its route name.
    """
    mem = mem or ttnn.DRAM_MEMORY_CONFIG
    # ttnn.linear(x, w) computes x @ w (verified: see bench/a1_orient.py), so the
    # weight must be stored [K, N] with N the output width, i.e. the plain
    # transpose of the [N, K] codes -- and the reference is x @ w, NOT x @ w.T.
    wt = np.ascontiguousarray(codes_f32.T)  # [K, N]
    attempts = []

    def route(name, fn):
        try:
            return name, fn()
        except Exception as e:  # noqa: BLE001
            attempts.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            return None, None

    def via_from_torch():
        return ttnn.from_torch(
            torch.from_numpy(wt.copy()), dtype=ttnn.DataType.BFLOAT2_B,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    def via_float_host_tensor():
        t = ttnn.from_torch(
            torch.from_numpy(wt.copy()), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, memory_config=mem)
        t = t.to(device)
        return ttnn.to_dtype(t, ttnn.DataType.BFLOAT2_B)

    def via_typecast_from_bf4():
        # bfloat4_b already packs -1/0/+1 exactly, so a 4->2 bit typecast should be
        # lossless for ternary data and exercises a path that definitely exists.
        t = ttnn.from_torch(
            torch.from_numpy(wt.copy()), dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        return ttnn.to_dtype(t, ttnn.DataType.BFLOAT2_B)

    for name, fn in (("from_torch(bfloat2_b)", via_from_torch),
                     ("float32->to_dtype", via_float_host_tensor),
                     ("bfloat4_b->to_dtype", via_typecast_from_bf4)):
        got_name, t = route(name, fn)
        if t is not None:
            return got_name, t, attempts
    return None, None, attempts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_down")
    ap.add_argument("--stage", default="pack,correct,fast",
                    help="comma list of pack,correct,fast")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--pcc", type=float, default=0.9999)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_bfp2.jsonl"))
    args = ap.parse_args()

    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn

    want = set(args.stage.split(","))
    shapes = [s for s in SHAPES if s[0] == args.shape]
    if not shapes:
        print(f"unknown shape {args.shape}; known: {[s[0] for s in SHAPES]}")
        return 2

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    rc = 0
    try:
        for name, k, n, group, bw, cores in shapes:
            k_tiles, n_tiles = k // 32, n // 32
            print(f"\n=== {name} K={k} N={n}, {describe(name)}, "
                  f"bfp2 {weight_mb(n, k, 'bfp2'):.2f} MB vs bfp4 {weight_mb(n, k, 'bfp4'):.2f} MB",
                  flush=True)
            rng = np.random.default_rng(0)
            codes = rng.integers(-1, 2, size=(n, k)).astype(np.float32)
            scales = rng.uniform(1e-3, 2e-2, size=(n, k // group)).astype(np.float16)

            # ---------- 1. pack ----------
            route_name, w2, attempts = make_bfp2(ttnn, device, codes)
            if w2 is None:
                print("  PACK FAILED: no construction route worked:")
                for a in attempts:
                    print(f"    {a}")
                rc = 1
                continue
            print(f"  pack route: {route_name}", flush=True)
            for a in attempts:
                print(f"    (tried, failed) {a}")

            if "pack" in want:
                try:
                    back = ttnn.to_torch(w2).float().numpy().T  # [N, K]
                    # Round-trip must reproduce the codes exactly: ternary values are
                    # exactly representable in a 2-bit block float.
                    exact = int((back == codes).sum())
                    total = codes.size
                    print(f"  pack round-trip: {exact}/{total} exact "
                          f"({exact / total * 100:.4f}%), max|err| {np.abs(back - codes).max():.3f}",
                          flush=True)
                    if exact != total:
                        print("    NOT exact -- the codes are not surviving bfloat2_b;"
                              " check the value encoding (01=+1, 11=-1, 00=0)")
                        rc = 1
                except Exception as e:  # noqa: BLE001
                    print(f"  pack readback FAILED: {type(e).__name__}: {str(e)[:200]}")
                    rc = 1

            # ---------- 2. correct ----------
            if "correct" in want:
                x_np = rng.standard_normal((1, k)).astype(np.float32)
                x = ttnn.from_torch(
                    torch.from_numpy(x_np).to(torch.bfloat16).reshape(1, 1, 1, k),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=act_mem(ttnn, name))
                pcN = math.ceil(n_tiles / cores)
                cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
                subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                progcfg = None
                try:
                    progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(cols, rows),
                        in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                        per_core_M=1, per_core_N=pcN, fuse_batch=True,
                        fused_activation=None, mcast_in0=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  progcfg build failed ({str(e)[:80]}); using auto config")
                try:
                    y = ttnn.linear(x, w2, compute_kernel_config=ckc,
                                    program_config=progcfg,
                                    memory_config=out_mem(ttnn, name))
                    got = ttnn.to_torch(y).float().numpy().reshape(1, n)
                    ttnn.deallocate(y)
                    sfull = np.repeat(scales.astype(np.float64), group, axis=1)
                    wt_full = (codes.astype(np.float64) * sfull).T  # [K, N]
                    ref = x_np.astype(np.float64) @ wt_full           # x @ w
                    c = pcc(got, ref)
                    ok = c >= args.pcc
                    print(f"  matmul PCC {c:.7f} vs float64 {'PASS' if ok else 'FAIL'}"
                          f" (gate {args.pcc})", flush=True)
                    if not ok:
                        rc = 1
                except Exception as e:  # noqa: BLE001
                    print(f"  MATMUL FAILED: {type(e).__name__}: {str(e)[:400]}")
                    print("    -> the stock matmul does not accept a bfloat2_b in1;"
                          " this is the case that needs the factory/kernel work")
                    rc = 1
                finally:
                    ttnn.deallocate(x)

            # ---------- 3. fast ----------
            if "fast" in want:
                x = ttnn.from_torch(
                    torch.randn((1, 1, 1, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem(ttnn, name))
                pcN = math.ceil(n_tiles / cores)
                cols, rows = min(11, cores), math.ceil(cores / min(11, cores))
                subW = max([i for i in range(1, 5) if pcN % i == 0] or [1])
                try:
                    progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(cols, rows),
                        in0_block_w=bw, out_subblock_h=1, out_subblock_w=subW,
                        per_core_M=1, per_core_N=pcN, fuse_batch=True,
                        fused_activation=None, mcast_in0=True)
                except Exception:  # noqa: BLE001
                    progcfg = None

                results = {}
                for fmt, wt in (("bfp2", w2), ("bfp4", None), ("bfp8", None)):
                    if wt is None:
                        wt = ttnn.from_torch(
                            torch.from_numpy(codes.T.copy()),
                            dtype=ttnn.bfloat4_b if fmt == "bfp4" else ttnn.bfloat8_b,
                            layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

                    def once(wt=wt):
                        y = ttnn.linear(x, wt, compute_kernel_config=ckc,
                                        program_config=progcfg,
                                        memory_config=out_mem(ttnn, name))
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                    try:
                        med, samples = timer.best(once)
                        once()
                        dev = device_kernel_us(ttnn, device) or med * 1e6
                        nb = weight_mb(n, k, fmt) * 1e6
                        results[fmt] = dev
                        gbs = nb / (dev * 1e-6) / 1e9
                        rec = new_record(f"bfp2_{name}", K=k, N=n, fmt=fmt, cores=cores,
                                         in0_block_w=bw, placement=describe(name))
                        rec.samples = samples
                        append_jsonl(args.out, rec, int(nb))
                        print(f"  {fmt:5s} {nb / 1e6:7.2f} MB  {dev:8.1f} us  "
                              f"{gbs:7.1f} GB/s ({gbs / DRAM_PEAK_GBS * 100:4.1f}% pk)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {fmt:5s} timing FAILED: {type(e).__name__}: {str(e)[:150]}")
                    finally:
                        if fmt != "bfp2":
                            ttnn.deallocate(wt)
                if "bfp2" in results and "bfp4" in results:
                    ratio = results["bfp4"] / results["bfp2"]
                    verdict = ("PASSES the 1.4x gate" if ratio >= 1.4
                               else "below the 1.4x gate")
                    print(f"  bfp2 vs bfp4: {ratio:.2f}x  -> {verdict}", flush=True)
                ttnn.deallocate(x)
            ttnn.deallocate(w2)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
