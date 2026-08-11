#!/usr/bin/env python3
"""Isolated single-op PCC + timing sweep for gemma-4-26B's residual rms_norm.

Reproduces the op exactly as models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py
_rms_norm builds it -- same shape (1,1,1,2816) bf16, same weight shape, same epsilon, same
HiFi4/fp32_dest_acc compute config, same create_sharded_memory_config + LayerNormShardedMultiCore
program config -- and sweeps:

  A  the divisor ladder under the MODEL'S OWN grid rule: (cores,1) if cores<=11 else (11, cores//11)
     including 88, the rung the v3 run never measured and the one v2 shipped.
  B  the SAME core count in a DIFFERENT rectangle (4x2, 2x4, 2x2, 1x2 ...) -- the model's rule
     can never produce these.
  C  the SAME grid with a DIFFERENT subblock_w, to separate accumulation blocking from placement.
  D  the interleaved baseline (what the incumbent runs).

Reference is float64 torch. Reported PCC is candidate-vs-reference (absolute), which is the same
quantity the stage's absolute oracle reports.
"""
import json
import statistics
import sys
import time

import torch
import ttnn

HIDDEN = 2816
TILE = 32
WIDTH_TILES = HIDDEN // TILE  # 88
EPS = 9.99999997e-7  # from shard_advise/sliding_attention/final_ir.mlir
ITERS = 30


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().to(torch.float64)
    y = b.flatten().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = x.norm() * y.norm()
    return 1.0 if d == 0 else float((x @ y) / d)


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x64 = x.to(torch.float64)
    rms = torch.sqrt((x64 * x64).mean(-1, keepdim=True) + EPS)
    return (x64 / rms) * w.to(torch.float64)


def model_subblock_w(block_w: int) -> int:
    return next(v for v in (4, 2, 1) if block_w % v == 0)


def model_grid(cores: int):
    return (cores, 1) if cores <= 11 else (11, cores // 11)


def build_cases():
    cases = []
    # --- A: the divisor ladder, model's own grid rule
    for c in (1, 2, 4, 8, 11, 22, 44, 88):
        bw = WIDTH_TILES // c
        gx, gy = model_grid(c)
        cases.append(dict(sweep="A ladder", cores=c, grid=(gx, gy), block_w=bw,
                          subblock_w=model_subblock_w(bw),
                          note="model rule" + ("  <-- NEVER MEASURED in the v3 run; v2 shipped it" if c == 88 else "")))
    # --- B: same core count, different rectangle. The model's rule cannot emit any of these.
    for c, gx, gy in ((2, 1, 2), (4, 2, 2), (8, 4, 2), (8, 2, 4), (22, 11, 2), (44, 11, 4)):
        if (gx, gy) == model_grid(c):
            continue
        bw = WIDTH_TILES // c
        cases.append(dict(sweep="B shape", cores=c, grid=(gx, gy), block_w=bw,
                          subblock_w=model_subblock_w(bw),
                          note=f"same {c} cores as A, rectangle {gx}x{gy} instead of "
                               f"{model_grid(c)[0]}x{model_grid(c)[1]}"))
    # --- C: same grid, forced subblock_w, to separate blocking from placement
    for c in (1, 11, 22, 44):
        bw = WIDTH_TILES // c
        gx, gy = model_grid(c)
        for sw in (4, 2, 1):
            if bw % sw or sw == model_subblock_w(bw):
                continue
            cases.append(dict(sweep="C blocking", cores=c, grid=(gx, gy), block_w=bw, subblock_w=sw,
                              note=f"model would pick subblock_w={model_subblock_w(bw)}"))
    return cases


def run_sharded(device, xt, wt, cc, cores, grid, block_w, subblock_w):
    gx, gy = grid
    mc = ttnn.create_sharded_memory_config(
        (TILE, HIDDEN // cores),
        ttnn.CoreGrid(x=gx, y=gy),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    xs = ttnn.to_memory_config(xt, mc)
    ws = ttnn.to_memory_config(wt, mc)

    def once():
        out = ttnn.rms_norm(xs, epsilon=EPS, weight=ws, compute_kernel_config=cc,
                            memory_config=mc, program_config=pc)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    out = once()
    got = ttnn.to_torch(out)
    times = []
    for _ in range(ITERS):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        once()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1e6)
    return got, statistics.median(times)


def run_interleaved(device, xt, wt, cc):
    def once():
        return ttnn.rms_norm(xt, epsilon=EPS, weight=wt, compute_kernel_config=cc,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

    got = ttnn.to_torch(once())
    times = []
    for _ in range(ITERS):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        once()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1e6)
    return got, statistics.median(times)


def main():
    torch.manual_seed(20260811)
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False,
    )
    results = []
    try:
        # Two input distributions. "unit" is a clean control; "residual" mimics a deep
        # residual stream -- wide dynamic range with a few large channels, which is what
        # decides whether a reduction re-association shows up in the 3rd decimal.
        inputs = {
            "unit": torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32),
            "residual": (torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32) * 8.0
                         + torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32).pow(3) * 40.0),
        }
        weight = 1.0 + 0.1 * torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32)

        cases = build_cases()
        print(f"device grid {grid.x}x{grid.y} = {grid.x*grid.y} cores; "
              f"tensor 1x1x1x{HIDDEN} bf16 = {WIDTH_TILES} tiles wide; eps={EPS}")
        print(f"{len(cases)} sharded cases x {len(inputs)} inputs, {ITERS} timed iters each\n")

        for iname, x in inputs.items():
            ref = reference(x.to(torch.bfloat16).to(torch.float32),
                            weight.to(torch.bfloat16).to(torch.float32))
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            wt = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            got, us = run_interleaved(device, xt, wt, cc)
            p = pcc(got.to(torch.float32), ref)
            results.append(dict(input=iname, sweep="D interleaved", cores=None, grid=None,
                                block_w=None, subblock_w=None, pcc=p, us=us, note="incumbent path"))
            print(f"[{iname}] D interleaved (incumbent)                          "
                  f"PCC {p:.8f}   {us:8.1f} us")

            for c in cases:
                try:
                    got, us = run_sharded(device, xt, wt, cc, c["cores"], c["grid"],
                                          c["block_w"], c["subblock_w"])
                    p = pcc(got.to(torch.float32), ref)
                    err = None
                except Exception as exc:  # a rejected shard spec is itself a result
                    p, us, err = None, None, f"{type(exc).__name__}: {str(exc)[:180]}"
                gx, gy = c["grid"]
                results.append(dict(input=iname, **c, pcc=p, us=us, error=err))
                lbl = f"{c['sweep']:<11} {c['cores']:>3}c {gx:>2}x{gy:<2} bw={c['block_w']:<2} sw={c['subblock_w']}"
                if err:
                    print(f"[{iname}] {lbl}   REJECTED  {err}")
                else:
                    print(f"[{iname}] {lbl}   PCC {p:.8f}   {us:8.1f} us   {c['note']}")
            print()
    finally:
        ttnn.close_device(device)

    out = "/tmp/normsweep-results.json"
    with open(out, "w") as fh:
        json.dump(dict(device_grid=[grid.x, grid.y], hidden=HIDDEN, eps=EPS, iters=ITERS,
                       results=results), fh, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
