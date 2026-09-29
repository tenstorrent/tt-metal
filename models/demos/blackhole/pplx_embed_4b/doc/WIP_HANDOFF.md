# WIP handoff (temporary — delete once picked up)

Saved 2026-09-29 while moving to a machine whose P150s open at 12×10. Branch `amorrison/high-batch-optim` at c61c7e3.

## Why we stopped

Every P150b on the previous box (8 chips, firmware bundle 19.8.1.0) opens with an **11×10 = 110-core** grid, with both
this checkout's build and `~/tt-blaze/tt-metal`'s. The profiles, rooflines and model configs assume 12×10 = 120, so no
measurement was taken there. `scripts/setup_harvesting.py` ("Current Loudbox", 2 disabled columns) would reflash every
board to 120 cores, but it is a shared box and it was not run.

**First thing on the new machine:** confirm the grid before benchmarking.

```
python -c "import ttnn; d=ttnn.open_device(device_id=0); g=d.compute_with_storage_grid_size(); print(g.x, g.y); ttnn.close_device(d)"
```

It must print `12 10`.

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. **(now) bs1 fused SwiGLU with K_block 40.** bs1 still runs FF1 + FF3 + a separate BinaryNg `mul` (1.88 ms, 12% of
   bs1). The fused path lost in NEGATIVE_RESULTS §58 (15.7 → 16.0 ms at 2,20,8 1×2, 214.4 µs standalone), but that was
   before §60 showed the pack-thread SwiGLU was half exposed at K_block 20 and K_block 40 hides it (bs8 −14%). bs1 is
   not power-capped, so a win counts in full.
2. **(next) Cheaper SwiGLU in FF1+FF3** (40% of the batched replay, biggest gap at every batch: 9.3 / 16.6 / 28.4 ms
   to roofline at bs8 / 16 / 32). The SFPU pass is ~1400 cycles/output tile (§60) and costs power even when hidden;
   under the power cap sustained only gets ~half of a cold gain (c61c7e3: cold −4.8/−5.3/−6.1%, sustained
   −2.5/−2.8/−2.8%). Try a cheaper SiLU; check STS-B.
3. **(later) SDPA DRAM traffic** (41–60% of its DRAM roof; Q/K/V round-trip through DRAM from the heads op). Keep the
   heads output in L1 for SDPA, or the head-major QKV write (#57722).

Also worth doing: `sustained_run.sh` reports AICLK as the median over the whole run (cold iterations included); take it
over the sustained window (iterations 15–29) so the clock-adjusted roofline is accurate. Add J/inference
(power × sustained time) as a ranking metric.

## Step 1 details

Bar to beat, per layer, from the Optimized bs1 profile (7dbf479): FF1 69.7 µs + FF3 69.7 µs + mul 52.3 µs ≈ **192 µs**.
The fused kernel must come in under that standalone, and then win e2e (its FF2 input and layout differ).

Standalone sweep (99 configs; K_block 20 / 40 / 80 over M 1/2/4, N 2/4/6/8/16, 1×W subblocks — §60 found only 1×W
subblocks win for the fused kernel):

```
S=""; for mb in 1 2 4; do for kb in 20 40 80; do for nb in 2 4 6 8 16; do for sw in 2 4 6 8; do
  [ $((nb % sw)) -eq 0 ] || continue; S="$S$mb,$kb,$nb,1,$sw;"; done; done; done; done; S=${S%;}
TT_VISIBLE_DEVICES=0 SWEEP="$S" ./python_env/bin/python \
  models/demos/blackhole/pplx_embed_4b/perf_tools/bench_ff13_sweep.py 1 20
```

Check whether bs1's FF1/FF3 compute config uses fp32 dest accumulation (`FF13_FP32=1` in the sweep; subblock width is
then capped at 4 — `MLP._bs1_fused_config` clamps it).

If a config beats ~192 µs, run e2e: `QWEN_FUSE_SWIGLU_BS1=1 QWEN_MM_BLOCK_FF13=M,K,N QWEN_MM_SUBBLOCK_FF13=1,W` against
the default, alternating rounds of `perf_tools/sustained_run.sh 1 <chip> 30 <tag> "<ENV>"` on the same chip (current
bs1: cold 15.6 / sustained 15.75 ms), then STS-B. If it wins, make those blocks the `_bs1_fused_config` defaults and
decide whether bs1 fuses by default.

## Build notes from the old box (may not apply on the new one)

- The system clang-20 there had no `llvm-ar`/`ld.lld`; a local LLVM 20 (`~/.local/llvm-20/usr/lib/llvm-20/bin`) had to
  be on PATH **before the first configure** (CMake caches the compiler's archiver then).
- The login shell exported `TT_METAL_HOME` / `TT_METAL_RUNTIME_ROOT` / `PYTHONPATH` for another checkout; the build's
  `precompile_fw` step writes firmware to `$TT_METAL_HOME/tt_metal/pre-compiled`. Point all three at this checkout.
