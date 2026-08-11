#!/usr/bin/env python3
"""Follow-up to sweep.py, which found the norm clean at every grid to 8 decimals.

Three things sweep.py did not vary, each a candidate for the whole-layer 0.99457:

  E  weight=None.  final_ir.mlir shows TWO of the sliding layer's rms_norms with
     operandSegmentSizes <1,0,0> -- no weight operand.  One of them is the MoE ROUTER norm
     (_router_weights calls self._rms_norm(residual, None)), whose output feeds ttnn.topk.
     A router perturbation flips an expert selection, which is a discontinuity, not a rounding.
  F  dynamic range.  Real residual streams carry activation spikes orders of magnitude above
     the bulk.  bf16 sum-of-squares re-association is range-sensitive.
  G  batched dispatch timing.  sweep.py synced every iteration, so its microseconds are
     dispatch-dominated.  Here N enqueues then one sync, to get a per-op cost that can be
     compared across grids.
"""
import json
import statistics
import sys
import time

import torch
import ttnn

HIDDEN = 2816
TILE = 32
WIDTH_TILES = HIDDEN // TILE
EPS = 9.99999997e-7
BATCH = 200


def pcc(a, b):
    x = a.flatten().to(torch.float64)
    y = b.flatten().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = x.norm() * y.norm()
    return 1.0 if d == 0 else float((x @ y) / d)


def reference(x, w):
    x64 = x.to(torch.float64)
    rms = torch.sqrt((x64 * x64).mean(-1, keepdim=True) + EPS)
    out = x64 / rms
    return out if w is None else out * w.to(torch.float64)


def model_subblock_w(bw):
    return next(v for v in (4, 2, 1) if bw % v == 0)


def model_grid(c):
    return (c, 1) if c <= 11 else (11, c // 11)


GRIDS = [(2, (2, 1)), (4, (4, 1)), (8, (8, 1)), (11, (11, 1)),
         (22, (11, 2)), (44, (11, 4)), (88, (11, 8)),
         (8, (4, 2)), (8, (2, 4)), (4, (2, 2))]


def sharded_cfg(cores, grid, subblock_w=None):
    gx, gy = grid
    bw = WIDTH_TILES // cores
    mc = ttnn.create_sharded_memory_config(
        (TILE, HIDDEN // cores), ttnn.CoreGrid(x=gx, y=gy),
        ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w or model_subblock_w(bw),
        block_h=1, block_w=bw, inplace=False)
    return mc, pc, bw


def timed(device, fn):
    fn()
    ttnn.synchronize_device(device)
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            fn()
        ttnn.synchronize_device(device)
        reps.append((time.perf_counter() - t0) * 1e6 / BATCH)
    return statistics.median(reps)


def main():
    torch.manual_seed(20260811)
    device = ttnn.open_device(device_id=0)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    results = []
    try:
        base = torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32)
        spike = base.clone()
        spike[..., ::311] *= 3000.0          # sparse activation spikes, ~9 channels
        wild = base.clone() * torch.exp(torch.randn(1, 1, 1, HIDDEN) * 4.0)  # ~1e7 spread
        inputs = {"unit": base, "spiked_3000x": spike, "logspread_1e7": wild}
        weights = {"weighted": 1.0 + 0.1 * torch.randn(1, 1, 1, HIDDEN, dtype=torch.float32),
                   "no_weight": None}

        print(f"tensor 1x1x1x{HIDDEN} bf16, {WIDTH_TILES} tiles; {BATCH} enqueues per timing rep\n")
        for wname, w in weights.items():
            wt = None if w is None else ttnn.from_torch(
                w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for iname, x in inputs.items():
                xb = x.to(torch.bfloat16).to(torch.float32)
                ref = reference(xb, None if w is None else w.to(torch.bfloat16).to(torch.float32))
                xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                rng = float(xb.abs().max() / xb.abs()[xb.abs() > 0].min())

                def il():
                    return ttnn.rms_norm(xt, epsilon=EPS, weight=wt,
                                         compute_kernel_config=cc,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

                p0 = pcc(ttnn.to_torch(il()).to(torch.float32), ref)
                t0 = timed(device, il)
                results.append(dict(weight=wname, input=iname, cores=None, grid=None,
                                    pcc=p0, us=t0))
                print(f"[{wname:<9} {iname:<14} range 1e{rng:.0e}] interleaved            "
                      f"PCC {p0:.8f}  {t0:7.2f} us")

                for cores, grid in GRIDS:
                    try:
                        mc, pc, bw = sharded_cfg(cores, grid)
                        xs = ttnn.to_memory_config(xt, mc)
                        ws = None if wt is None else ttnn.to_memory_config(wt, mc)

                        def sh(xs=xs, ws=ws, mc=mc, pc=pc):
                            o = ttnn.rms_norm(xs, epsilon=EPS, weight=ws,
                                              compute_kernel_config=cc,
                                              memory_config=mc, program_config=pc)
                            return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)

                        p = pcc(ttnn.to_torch(sh()).to(torch.float32), ref)
                        t = timed(device, sh)
                        d = p - p0
                        results.append(dict(weight=wname, input=iname, cores=cores,
                                            grid=list(grid), block_w=bw, pcc=p, us=t))
                        print(f"[{wname:<9} {iname:<14}          ] {cores:>3}c {grid[0]:>2}x{grid[1]:<2} bw={bw:<2}  "
                              f"PCC {p:.8f}  {t:7.2f} us   vs interleaved {d:+.2e}")
                    except Exception as exc:
                        results.append(dict(weight=wname, input=iname, cores=cores,
                                            grid=list(grid), error=str(exc)[:160]))
                        print(f"[{wname:<9} {iname:<14}          ] {cores:>3}c {grid[0]:>2}x{grid[1]:<2}       "
                              f"REJECTED {type(exc).__name__}")
                print()
    finally:
        ttnn.close_device(device)
    with open("/tmp/normsweep2-results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote /tmp/normsweep2-results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
