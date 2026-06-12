# minimal_matmul sweep — reproduce "main optimized baseline vs branch"

Self-contained kit to reproduce the perf comparison on any machine (built for moving the study from
Wormhole to **Blackhole**). Everything needed is committed on branch
`cglagovich/minimal-matmul-mcast-prefetch`; no separate `main` checkout/build is required.

## What it compares
- **baseline** = the *main optimized baseline*: minimal_matmul as plain unicast, swept over block sizes,
  best block per shape. Reproduced **on this branch** by pinning an explicit `MinimalMatmulConfig`
  (which gates off every branch feature — K-cap, block sizer, subblock maximizer, slicing, prefetch
  gate) **and** setting `TT_MM_NO_LARGE_LEVERS=1` (the only always-on branch addition is the N>=4096
  DRAM levers; this disables them). So baseline == main bit-for-bit on the dataflow path. Verified on
  Wormhole: branch+explicit+no-levers == main best-swept (e.g. 1024x6144x4608 845us vs main 847us).
- **branch** = the production auto path (no config): K-cap + block sizer + subblock DST-maximizer +
  auto core-grid slicing + mcast/prefetch gate.

speedup = baseline / branch (>1 = branch faster).

## The three branch optimizations under test (vs main 8/8/8 default blocking)
1. `K_block = min(K_block, K_tiles)` — no K padding on small-K shapes.
2. Auto block sizer — subblock-multiple M/N blocks, fewest-blocks/even tiebreak, L1-budget capped.
3. Subblock DST-maximizer — fill the half-sync DST (4 fp32 / 8 bf16) instead of capping a dim at 2.
(Plus the pre-existing auto core-grid slicing + mcast/prefetch gate for skewed/skinny shapes.)
All are gated behind `!config`. See `ttnn/cpp/.../minimal_matmul/device/minimal_matmul_program_factory.cpp`.

## Run it

```bash
# 0. get the branch (already on origin)
git fetch origin && git checkout cglagovich/minimal-matmul-mcast-prefetch
git submodule update --init --recursive          # if you switched from another commit

# 1. env + build  (Blackhole)
source /home/cglagovich/bh_env.sh                # sets ARCH_NAME=blackhole; use wh_env.sh on Wormhole
source python_env/bin/activate
bash build_metal.sh                              # CPP changed -> rebuild

# 2. sweep (each writes baseline_* and branch_* tracy dirs). ~30-50 min for the 65 big shapes.
bash tools/mm_sweep/run_sweep.sh tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big
bash tools/mm_sweep/run_sweep.sh tools/mm_sweep/shapes_ltx.txt /tmp/mm_bh_ltx

# 3. parse -> comparison table + geomean/wins/losses
python tools/mm_sweep/parse_sweep.py /tmp/mm_bh_big tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big/results.md
python tools/mm_sweep/parse_sweep.py /tmp/mm_bh_ltx tools/mm_sweep/shapes_ltx.txt /tmp/mm_bh_ltx/results.md
```

Run `run_sweep.sh` in the background (`nohup ... &`) and poll the progress log; each `<shape> <mode> done
pcc=...` line confirms a PCC pass. **Every line must show pcc>=0.99** — an auto-derived block that isn't
a subblock multiple silently corrupts (the auto path skips the host validator), so treat a missing/low
PCC as a real failure, not a profiler hiccup.

## Blackhole-specific checks (do these before trusting the numbers)
- **Grid**: the harness auto-uses `device.compute_with_storage_grid_size()`, so it adapts to BH's larger
  grid (vs WH 8x8). Both baseline and branch use the same device grid -> fair.
- **L1 budget**: the block sizer hardcodes `L1_CB_BUDGET = 1310720` (~1.25 MiB), sized for WH's 1.5 MB
  L1. BH L1 is >= WH, so this is *conservative but safe* (won't overflow; may leave a little perf on the
  table on BH). If you want BH-optimal blocking, bump that constant in the factory and rebuild — but for
  an apples-to-apples "does the speedup repro" check, leave it.
- **Baseline block seeds** assume M-on-rows; on BH's non-square grid the seeds for very skewed M>N shapes
  may be slightly off. If a baseline number looks suspiciously bad, widen the sweep:
  `FC_BPCM=1,2,4,8,16 FC_BPCN=1,2,4,8,16 FC_KBS=2,4,8,16,32 bash tools/mm_sweep/run_sweep.sh ...`.
- **fp32 acc** is on by default (`MM_FP32_ACC=1`), matching the WH study. Set `=0` for bf16-acc.

## Wormhole reference results (to compare BH against)
Branch vs main best-swept, the committed state of this branch (single n150, 8x8, bf16 in/out, fp32 acc):
- **big shapes (65):** geomean **1.36x**, 1 loss (`512x128x1536` 0.78x, an 18us overhead-bound shape).
- **LTX (17):** slicing wins up to ~2.35x on skinny shapes; non-skinny stay ~1.0x. See
  `/minimal_matmul_slicing_findings.md` and `/minimal_matmul_bigshape_study.md` (repo root) for the full
  WH tables and the investigation writeup.

If BH reproduces a comparable geomean with no PCC failures, the optimizations carry over.
