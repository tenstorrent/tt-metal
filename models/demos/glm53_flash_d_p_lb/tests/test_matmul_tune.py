# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Matmul schedule sweep for every model matmul above ~1 ms per chunk (chunk 5120 on the 2x4 mesh, per-chip shapes),
one chip, random data with the model's shapes, dtypes and fidelity. Per shape:
  - ttnn.linear, auto config (packer L1 acc on: the model today)
  - ttnn.linear with explicit 2D-multicast configs (MatmulMultiCoreReuseMultiCastProgramConfig): M over grid rows, N
    over grid columns (or transposed), in0_block_w / out subblock grid; identical numerics to the auto config
  - ttnn.experimental.minimal_matmul, auto and a blocking grid (bf16-output shapes only: its fp32 output is 0.03% low)
Prints, per shape, the best few: mean device-bound time per call (ITERS calls queued between two syncs), math
utilization (achieved FLOP/s over the fidelity peak, tt-perf-report's Blackhole model: cores x 4096 FLOP/cycle x
1.35 GHz / fidelity phases), DRAM share (in0 + weight + output bytes over 512 GB/s) and rel L2 vs fp32 torch math.
GLM_MMT_SHAPES (comma list) selects shapes, GLM_MMT_ITERS (default 20). GLM_MMT_TRUTH=1: only the auto configs with
packer_l1_acc off / on, scored against fp32 torch (rel L2 and scale)."""

import itertools
import math
import os
import time

import pytest
import torch

import ttnn

ITERS = int(os.environ.get("GLM_MMT_ITERS", "20"))
TRUTH = os.environ.get("GLM_MMT_TRUTH") == "1"
LOFI, HIFI2, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
PHASES = {LOFI: 1, HIFI2: 2, HIFI4: 4}
BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE_BYTES = {BF16: 2048, FP32: 4096, ttnn.bfloat8_b: 1088}
# name: (M, K, N, in0 dtype, weight dtype, out dtype, fidelity, calls per chunk on the 45-layer model)
SHAPES = {
    "kda_in": (2560, 4096, 8352, BF16, BF16, BF16, HIFI4, 34),
    "mla_o": (640, 16384, 4096, BF16, BF16, BF16, HIFI2, 11),
    "se_down": (5120, 256, 4096, FP32, BF16, FP32, HIFI4, 42),
    "kda_o": (2560, 2048, 4096, FP32, BF16, FP32, HIFI4, 34),
    "router": (2560, 4096, 288, FP32, FP32, FP32, HIFI4, 42),
    "se_gate": (5120, 4096, 256, BF16, BF16, FP32, HIFI4, 84),
    "se_gate_up": (5120, 4096, 512, BF16, BF16, FP32, HIFI4, 42),
    "mla_qb": (640, 1536, 16384, BF16, BF16, BF16, HIFI2, 11),
    "mla_kva": (5120, 4096, 512, BF16, BF16, FP32, HIFI2, 11),
    "idx_k": (5120, 4096, 128, BF16, BF16, FP32, HIFI2, 22),
    "q_a": (640, 4096, 1536, BF16, BF16, FP32, HIFI2, 11),
    "idx_q": (640, 1536, 4096, BF16, BF16, BF16, HIFI2, 11),
    "mlp_gate": (5120, 4096, 1536, BF16, BF16, FP32, HIFI4, 6),
    "mlp_down": (5120, 1536, 4096, FP32, BF16, FP32, HIFI4, 3),
}
SEL = os.environ.get("GLM_MMT_SHAPES", ",".join(SHAPES)).split(",")


def _ckc(fid, l1acc=True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1acc
    )


def _timed(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(ITERS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    return (time.time() - t0) / ITERS * 1e3, out


def _divisors(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def mm2d_configs(grid, M, K, N):
    """2D-multicast candidates: (gx, gy) grids from the full grid down, in0_block_w dividing K, the largest out
    subblock (h * w <= 4, fp32 DEST) dividing the per-core block; transpose_mcast puts M on columns."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    out = []
    for transpose in (False, True):
        gm_max, gn_max = (grid.x, grid.y) if transpose else (grid.y, grid.x)
        for gm, gn in {(gm_max, gn_max), (gm_max, max(1, gn_max // 2)), (max(1, gm_max // 2), gn_max)}:
            pm, pn = math.ceil(Mt / gm), math.ceil(Nt / gn)
            gm2, gn2 = math.ceil(Mt / pm), math.ceil(Nt / pn)
            sw = max(_divisors(pn, 4))
            sh = max(_divisors(pm, max(1, 4 // sw)))
            for bw in (1, 2, 4, 8, 16):
                if Kt % bw:
                    continue
                gx, gy = (gm2, gn2) if transpose else (gn2, gm2)
                out.append(
                    (
                        f"2d {gx}x{gy}{' T' if transpose else ''} pm{pm} pn{pn} bw{bw} sb{sh}x{sw}",
                        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                            in0_block_w=bw,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            out_block_h=pm,
                            out_block_w=pn,
                            per_core_M=pm,
                            per_core_N=pn,
                            transpose_mcast=transpose,
                            fused_activation=None,
                            fuse_batch=True,
                        ),
                    )
                )
    return out


def minimal_configs(grid, M, N):
    mt, nt = M // 32, N // 32
    out = [("minimal auto", None)]
    for mb, kb, nb, (sh, sw) in itertools.product((2, 4, 8), (4, 8, 16), (2, 4, 8), ((1, 4), (4, 1), (2, 2))):
        if mb % sh or nb % sw or mb > mt or nb > nt:
            continue
        out.append(
            (
                f"minimal M{mb} K{kb} N{nb} sb{sh}x{sw}",
                ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=grid,
                ),
            )
        )
    return out


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_matmul_tune(device):
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    torch.manual_seed(0)
    mc = ttnn.DRAM_MEMORY_CONFIG
    summary = []
    for name in SEL:
        M, K, N, adt, wdt, odt, fid, calls = SHAPES[name]
        a = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=adt, layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(torch.randn(1, 1, K, N) / K**0.5, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
        truth = (ttnn.to_torch(a).double().reshape(M, K) @ ttnn.to_torch(w).double().reshape(K, N)).reshape(-1)
        flops = 2 * M * K * N
        peak = cores * 4096 * 1.35e9 / PHASES[fid]  # FLOP/s
        nbytes = (M * K * TILE_BYTES[adt] + K * N * TILE_BYTES[wdt] + M * N * TILE_BYTES[odt]) / 1024
        head = f"{M}x{K}x{N} {adt}x{wdt}->{odt} {fid}"

        def score(out):
            o = ttnn.to_torch(out).double().reshape(-1)
            return float((o - truth).norm() / truth.norm()), float((o * truth).sum() / (truth * truth).sum())

        if TRUTH:
            print(f"[mmt] == {name}: {head}", flush=True)
            for l1 in (False, True):
                k = _ckc(fid, l1)
                for tag, fn in (
                    ("linear auto", lambda: ttnn.linear(a, w, dtype=odt, compute_kernel_config=k, memory_config=mc)),
                    (
                        "minimal auto",
                        lambda: ttnn.experimental.minimal_matmul(
                            a, w, dtype=odt, compute_kernel_config=k, memory_config=mc
                        ),
                    ),
                ):
                    try:
                        ms, out = _timed(device, fn)
                    except Exception as ex:
                        print(f"[mmt]   {tag} packer_l1_acc={l1} FAILED {str(ex).splitlines()[0][:80]}", flush=True)
                        continue
                    rel, sc = score(out)
                    print(f"[mmt]   {ms:7.3f} ms  rel {rel:.3e}  scale {sc:.6f}  {tag} packer_l1_acc={l1}", flush=True)
                    ttnn.deallocate(out)
            ttnn.deallocate(a)
            ttnn.deallocate(w)
            continue

        ckc = _ckc(fid)
        cands = [("linear auto", "linear", None)]
        cands += [(t, "linear", c) for t, c in mm2d_configs(grid, M, K, N)]
        if odt == BF16 and adt == BF16:
            cands += [(t, "minimal", c) for t, c in minimal_configs(grid, M, N)]
        res = []
        for tag, kind, cfg in cands:
            if kind == "linear":
                fn = lambda: ttnn.linear(  # noqa: E731
                    a, w, dtype=odt, compute_kernel_config=ckc, memory_config=mc, program_config=cfg
                )
            else:
                fn = lambda: ttnn.experimental.minimal_matmul(  # noqa: E731
                    a, w, config=cfg, dtype=odt, compute_kernel_config=ckc, memory_config=mc
                )
            try:
                ms, out = _timed(device, fn)
                rel, sc = score(out)
                ttnn.deallocate(out)
                res.append((tag, ms, rel, sc))
            except Exception as ex:  # a blocking may not fit L1 / the op's constraints
                res.append((f"{tag} FAILED {str(ex).splitlines()[0][:50]}", float("inf"), float("nan"), float("nan")))
        res.sort(key=lambda r: r[1])
        base = next(r for r in res if r[0] == "linear auto")
        best = res[0]
        print(f"[mmt] == {name}: {head}, {calls} calls / chunk", flush=True)
        for tag, ms, rel, sc in res[:6] + ([base] if base not in res[:6] else []):
            util = flops / (ms * 1e-3) / peak * 100
            dram = nbytes / (ms * 1e-3) / 512e9 * 100
            print(
                f"[mmt]   {ms:7.3f} ms  math {util:5.1f}%  dram {dram:5.1f}%  x{base[1] / ms:4.2f}  rel {rel:.2e} "
                f"scale {sc:.5f}  {tag}",
                flush=True,
            )
        summary.append((name, calls, base[1], best[1], best[0], flops / (best[1] * 1e-3) / peak * 100))
        ttnn.deallocate(a)
        ttnn.deallocate(w)
    if summary:
        print("[mmt] == summary (ms per chunk = ms per call x calls)", flush=True)
        for name, calls, b, t, tag, util in summary:
            print(
                f"[mmt]   {name:10s} auto {b:6.3f} -> {t:6.3f} ms  ({util:5.1f}% math)  "
                f"chunk {b * calls:6.1f} -> {t * calls:6.1f} ms  {tag}",
                flush=True,
            )
