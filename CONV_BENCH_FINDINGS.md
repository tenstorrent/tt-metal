# GH #45995 — Diagnosing the Failure of Matmul-Helper Perf Benefits in Convolution

**Date:** 2026-06-04  **Device:** Wormhole n150 L (single chip)  **Branch:** `wransom/conv_bench`
**Build:** profiler-enabled (`ENABLE_TRACY=ON`, already compiled in the current `build_Release`; no rebuild needed)
**Measurement:** Tracy `DEVICE KERNEL DURATION [ns]`, warm (2nd of `run_twice`) `Conv2dDeviceOperation` row only; ≥3 reps/median.
**Submodules:** clean throughout. Working-tree delta: `test_conv_bench.py` (CONFIG made env-overridable, defaults unchanged) + this file. Fix committed as `fixup! 2ed07eab4b9` (`cffa73e34e4`).

---

## ⚠️ CORRECTION (2026-06-05) — the relaxation DOES win once unblocked

The original TL;DR below ("cannot execute on conv at all") described the branch **as-found**, where the validate gate blocked the relaxed shape. At the user's direction I (1) **relaxed that gate** for the `HelperRowMajor` path (`conv2d_device_operation.cpp`, gated to bench-trm only) and (2) discovered + fixed a **device hang the gate was masking** (a CB self-deadlock in the TileRowMajor+bias path; see below). With both fixes, **helper_trm runs correctly and is consistently faster** — overturning the "no win" conclusion:

| Config (HS, fp32_accum, abh=64) | per_core_N | helper_sbm (1×N) | helper_trm (2×2) | **trm vs sbm** | PCC (all modes identical) |
|---|---|---|---|---|---|
| B  320←16  64×64 b2 (real SD) | 10 | 57,658 ns | 53,665 ns | **−6.9%** | 0.99999623 |
| A  320←256 32×32 b2 | 10 | 339,287 ns | 309,855 ns | **−8.7%** | 0.99999475 |
| C  448←128 16×16 b8 | 14 | 245,262 ns | 223,660 ns | **−8.8%** | 0.99999489 |
| D  192←96  32×32 b4 (1×3→2×2) | 6 | 84,012 ns | 79,032 ns | **−5.9%** | 0.99999493 |

**Mechanism of the win:** `out_subblock_h = 2` halves the number of times each fp32 weight tile is re-read (weight reads ∝ `in0_num_subblocks = per_core_M / out_subblock_h`). The pack method is perf-free (B measured 53,916 ns TileRowMajor pre-fix vs 53,665 ns SubblockMajor+reblock post-fix, within 0.5%), so the gain is the **shape**, not the pack — confirming the 2026-06-02 read-model.

**CRITICAL REGIME CAVEAT:** every number here is the **fp32-weights** regime — `run_conv` converts weights to fp32 whenever `weights_dtype != bfloat16`, and the harness hardcodes `weights_dtype=None`. The win exists *because* weights are the expensive operand. Real heavy convs (e.g. SDXL UNet) use **bfloat8_b weights**, where the 2026-06-02 micro-bench showed `out_subblock_h>1` is **+10% slower**. So this win is **regime-specific (fp32/bf16 weights), not universal**, and the **bf8-weights regime is the open question** — the harness can't test it without exposing `weights_dtype` (currently hardcoded). Do not generalize "conv benefits from the relaxation" to bf8-weight convs without measuring.

**The hang that the gate was masking:** with the gate relaxed, helper_trm 2×2 ran correctly on shallow configs but **hung** deeper-K ones (`FDMeshCommandQueue completion reader queue is not empty` → `system_memory_manager.cpp:738`). Root cause (found via Watcher): a CB self-deadlock in the TileRowMajor+`fuse_bias` path — the bias-add does `wait_front(row_group_tiles)` + `reserve_back(row_group_tiles)` on the **same** `matmul_partials_cb`, which deadlocks when that group fills the CB. Fix (`conv_bmm_tilize.cpp`, gated to `tile_pack_row_major && fuse_bias`): pack the bias path SubblockMajor and untilize via `reblock_and_untilize` (4-tile granularity → no deadlock; numerically verified PCC ~0.99999). The TileRowMajor+no-bias path is unchanged. **Normal (non-bench) convs are byte-identical** — 112 ResNet50 conv cases pass. Both edits are committed in `fixup! 2ed07eab4b9`.

**Net for GH #45995:** the matmul-helper subblock relaxation *is* a real conv win (≈6–9% device-kernel) in the fp32/expensive-weight regime; it was hidden by an over-conservative op `validate()` **and** a latent CB-deadlock in the bias path, both now fixed. Whether it helps the bf8-weight heavy convs that dominate real models is the next thing to measure.

---

## TL;DR (the headline) — INITIAL as-found state, SUPERSEDED by the correction above

> The text below was written before the validate gate was relaxed; it accurately describes the branch as-found (relaxation gated off) but its "no win" conclusion is overturned by the correction above.

The matmul-helper's subblock relaxation **cannot execute on conv at all** on this branch — not "it runs but doesn't help."

- The relaxation only ever changes the subblock **shape** by choosing `out_subblock_h > 1` with `out_subblock_w < per_core_N`.
- That exact shape is **rejected, unconditionally, by the conv op's `validate()`** at
  `conv2d_device_operation.cpp:119-122` (HEIGHT_SHARDED) / `131-134` (BLOCK_SHARDED):
  `out_subblock_w == per_core_N || out_subblock_h == 1`. It is **not** gated on bench mode / `tile_pack_row_major`.
- The harness relaxed the **picker**, the **factory define**, and the **kernel** (which has a TileRowMajor plain-untilize path), **but not this op-level validate gate**. So `helper_trm`:
  - **equals `helper_sbm`** on every config where the tuner's TileRowMajor pick keeps `out_subblock_h == 1` (no real difference), and
  - **`TT_FATAL`s** on every config where it would actually differ (`out_subblock_h > 1`).
- Therefore there is **no relaxation perf number to measure** — the win-enabling configuration is **correctness-gated off**, on both HS and BS.

Two secondary, measurable results:

1. **The helper migration (`main` → `helper_sbm`) is perf-neutral** across every expensive real-model conv shape measured (≤±0.5% on 7 of 8; one cheap 1×1 conv shows +3.3% from a fixed untilize-path overhead, amortized away on the larger convs). Migrating conv to the matmul helper did **not** cost performance.
2. **The TileRowMajor pack itself is free** where it runs (`out_subblock_h == 1`): `helper_trm` is within ~0.1-0.5% of `helper_sbm` (and on one config slightly *faster*).

This corroborates and explains the prior conclusion ("conv shows no helper perf win"): for conv there is nothing to win, because (a) the configs where the relaxation could change the shape are rare and OOM-prone on n150, and (b) where they do exist, the conv op rejects the relaxed shape as a hard correctness gate — whereas matmul's output path natively supports the relaxed (`h>1`, row-major) shape and its validate permits it, which is why matmul gets the win and conv does not.

---

## Why the relaxation can or cannot change the shape (refined model)

The tuner picks `out_subblock (h, w)` with `h·w ≤ DST`, `h | per_core_M`, `w | per_core_N`.
- `DST = 4` when `fp32_accum=True`, `DST = 8` when `fp32_accum=False`.
- For HEIGHT_SHARDED, `per_core_N = out_channels / 32` (the full output width sits on every core).
  For BLOCK_SHARDED, `per_core_N = out_channels / 32 / grid_cols` (N is split across the grid → small).
- `per_core_M` in the `CONV_BENCH` log `= act_block_h / 32`.
- **SubblockMajor** additionally requires `w == per_core_N || h == 1`. **TileRowMajor** drops that.

The relaxation changes the shape (`SBM ≠ TRM`) **iff all of**:
1. `per_core_N > DST` (so `w` can't reach `per_core_N`; this is the harness's `weight_num_subblocks > 1` eligibility guard), **and**
2. `per_core_N` is **not divisible by DST** — otherwise an `h=1` subblock already reaches `DST` volume (`per_core_N=8, DST=4 → 1×4`), and SBM == TRM, **and**
3. `per_core_M ≥ 2` (`act_block_h ≥ 64`) so an `h>1` subblock is legal.

Confirmed on-device (the `CONV_BENCH … tuner would pick:` line):

| `out_ch` (HS) | `per_core_N` | SubblockMajor | TileRowMajor | differs? | reason |
|---|---|---|---|---|---|
| 256 | 8 | **1×4** | **1×4** | no | 8 divisible by DST=4 → `h=1` reaches DST |
| 288 | 9 | **1×3** | **1×3** | no | only {1,3} divide 9 (≤4); `h=2` needs `w≤2`∤9 |
| **192** | **6** | **1×3** | **2×2** | **YES** | `h=1` caps at vol 3; `2×2`=vol 4 |
| **320** | **10** | **1×2** | **2×2** | **YES** | `h=1` caps at vol 2; `2×2`=vol 4 |
| **448** | **14** | **1×2** | **2×2** | **YES** | `h=1` caps at vol 2; `2×2`=vol 4 |

(This refines the earlier "`per_core_N > DST` & `per_core_M > 1`" rule, which is necessary but not sufficient — `per_core_N=8` and `=9` satisfy it yet still produce no shape change.)

---

## How real models actually shard these convs (why the condition is rarely met)

- **High-channel convs (the ones with `out_ch` big enough for `per_core_N > DST`) are run BLOCK_SHARDED** in models (SDXL UNet/refiner/VAE, ResNet50 layer3/4, SD UNet). BS splits N across the grid, so `per_core_N` collapses:
  - **ResNet50 layer3 `256←256 14×14` as the model runs it (BS):** `per_core_N = 1`, `SBM=TRM=1×1` — the relaxation is a pure no-op (measured).
  - **SDXL UNet `768←768 64×64` (BS):** `per_core_N = 3`.
- **HEIGHT_SHARDED convs in models are usually low-channel** (`out_ch ≤ 128 → per_core_N ≤ 4`), so `per_core_N ≤ DST` and there is no relaxation to do.
- The bf16-accum regime models actually use (`fp32_accum=False → DST=8`) makes it *harder*: `per_core_N=8 → 1×8` for both (no diff), and the harness's own no-op guard rejects `helper_trm` there.

So in practice the relaxation almost never engages on a real model conv; to exercise it at all I had to **force HEIGHT_SHARDED + `fp32_accum=True` + `act_block_h≥64` + channel counts whose `per_core_N` is not a multiple of 4** (192/320/448). Even then it `TT_FATAL`s at the validate gate (below).

---

## Recommended test set & per-mode baselines

All runs: HS, `fp32_accum=True` (DST=4) unless noted, `act_block_h=64`, weights resolve to **float32** (the harness passes `weights_dtype=None`; `run_conv` converts any non-bf16 weights to fp32), `output=ROW_MAJOR`, `packer_l1_acc=OFF` (forced). Durations are median ns over 3 reps; spread was 0.1-0.5% on every cell (device timing is highly stable here).

### Group 1 — natural shape difference (`helper_trm` would change the shape → it FATALs)

| # | Model | Conv (B, out←in, H×W, k, s) | per_core_N | SBM→TRM | `main` ns | `helper_sbm` ns | sbm vs main | `helper_trm` | PCC (main=sbm) |
|---|---|---|---|---|---|---|---|---|---|
| A | SDXL/SD 320-ch class (derived) | 2, 320←256, 32×32, 3×3, s1 | 10 | 1×2→2×2 | 339,771 | 339,398 | **−0.11%** | **FATAL** `…:122` | 0.99999475 |
| B | **Stable Diffusion UNet (real shape)** | 2, 320←16, 64×64, 3×3, s1 | 10 | 1×2→2×2 | 57,455 | 57,736 | **+0.49%** | **FATAL** `…:122` | 0.99999623 |
| C | high-channel class (derived) | 8, 448←128, 16×16, 3×3, s1 | 14 | 1×2→2×2 | 245,403 | 244,827 | **−0.23%** | **FATAL** `…:122` | 0.99999489 |
| D | mid-channel class (derived) | 4, 192←96, 32×32, 3×3, s1 | 6 | 1×3→2×2 | 83,799 | 84,052 | **+0.30%** | **FATAL** `…:122` | 0.99999493 |

`helper_trm` fatal text (all four): `TT_FATAL @ conv2d_device_operation.cpp:122: out_subblock_w_ntiles == out_width_ntiles || out_subblock_h_ntiles == 1`. The tuner picks the relaxed `2×2`, the op rejects it before the kernel runs.

### Group 2 — no shape difference (`helper_trm` keeps `h=1` → it RUNS; measures pack-layout only)

| # | Model | Conv | per_core_N | shape | `main` ns | `helper_sbm` ns | `helper_trm` ns | verdict | PCC (all 3 equal) |
|---|---|---|---|---|---|---|---|---|---|
| E | **ResNet50 layer3 (real shape, HS)** | 8, 256←256, 14×14, 3×3, s1 | 8 | 1×4 | 262,873 | 263,024 (+0.06%) | 263,093 (+0.08%) | wash | 0.99999477 |
| F | **ResNet50 bottleneck downsample (real, HS)** | 20, 256←64, 56×56, 1×1, s2 | 8 | 1×4 | 36,770 | 37,987 (**+3.3%**) | 36,585 (−0.5%) | see note | 0.99999782 |
| Z0 | harness default (per_core_M=1) | 1, 256←256, 14×14, 3×3, s1 | 8 | 1×4 | 207,454 | 207,757 (+0.15%) | 208,440 (+0.48%) | wash | 0.99999483 |
| E′ | ResNet50 layer3, **bf16-accum** (real regime) | 8, 256←256, 14×14, 3×3, s1 | 8 (DST=8) | 1×8 | 317,877 | 318,026 (+0.05%) | **FATAL** (no-op guard) | wash | 0.99956331 |

**F note (the one non-neutral migration result):** on the cheap 1×1 downsample (37µs), `helper_sbm` is **+3.3% slower than `main`** while `helper_trm` matches `main` (−0.5%). Reproduced with reordered modes (helper_sbm first, still slowest): main 36,792 / helper_sbm 37,973 / helper_trm 36,637 ns. Cause: the helper's SubblockMajor untilize (`reblock_and_untilize`) carries a small **fixed** overhead vs `main`'s hand-written untilize; `helper_trm` skips it (plain untilize). On a single-K-block 1×1 conv this fixed cost is a visible fraction; on the larger convs (A/C/E, hundreds of µs) it is amortized to noise. **E′ `helper_trm` fatal** is the harness's *own* no-op guard (`conv2d_op_sharded_program_factory.cpp:961`, `weight_num_subblocks>1`), not the validate gate: at DST=8, `per_core_N=8 → 1×8 = 1×8`, so TRM is provably identical to SBM and the harness refuses to run it.

---

## Per-candidate verdict on the central question

- **A, B, C, D (natural diff):** The relaxation **cannot be evaluated** — `helper_trm` `TT_FATAL`s at the op validate gate. `main` ≈ `helper_sbm` (both SubblockMajor, ≤±0.5%): the **helper migration is a wash**. No relaxation win is obtainable without relaxing the validate gate *and* proving the kernel's TileRowMajor untilize path is numerically correct for `h>1`.
- **E, Z0, E′ (no diff):** All three modes are within ≤0.5% and numerically identical. The relaxation is a no-op here by construction (`h=1`); the helper migration is a wash.
- **F (no diff, high matmul-block count):** This is the WH regime where a per-call win would translate (huge M, many blocks). The relaxation is still a no-op (`1×4`). The only real delta is the helper's SubblockMajor untilize overhead (+3.3% on this cheap conv), which `helper_trm`'s pack path avoids — i.e. the only measurable "helper effect" here is a small **cost** on the SBM path, not a win, and it disappears on larger convs.

**Structural reason, tying it together:** a conv subblock win requires `out_subblock_h > 1`. That shape (with `w < per_core_N`) is exactly what the conv op `validate()` forbids, on both HS (`:122`) and BS (`:134`). Matmul wins from the helper because matmul's row-major output path supports `h>1` and its validate permits it; conv's untilize/reblock path is SubblockMajor-oriented (`reblock_and_untilize` is `static_assert` SubblockMajor-only) and its validate hard-gates `h>1`. Independently, a 2026-06-02 micro-bench showed forced `h>1` (2×4) was **+10-12% slower** in conv's bf8+bias regime anyway — so even if the gate were lifted, the win is regime-dependent and was negative for that regime.

---

## Auto-tuner audit (forced larger subblock) — tuner is NOT deficient

For the no-diff candidates the task calls for forcing a larger subblock the tuner didn't pick and checking whether it beats the auto pick (a tuner-left-a-win signal).

- **E, force `2×2` (helper_trm):** `TT_FATAL @ conv2d_device_operation.cpp:122`.
- **F, force `2×2` (helper_trm):** `TT_FATAL @ conv2d_device_operation.cpp:122`.

At `per_core_N=8, DST=4` the tuner's `1×4` is already at the DST volume ceiling; the only alternative TileRowMajor allows is a *reshape* at equal volume (`2×2`), which the op rejects. **No larger-volume subblock is legal, and the equal-volume reshape can't run.** Conclusion: **the tuner is not leaving a runnable win on the table** — it correctly avoids `h>1` precisely because the op would reject it. (BS `768←768` with `act_block_h=128` is a further example: the relaxation picks `4×1` tall-skinny and FATALs at `:134`.) **No tuner-deficiency flags.**

---

## Dropped / ineligible / OOM

**Ineligible by guard (surveyed in `tests/ttnn/nightly/.../conv/test_conv2d.py`):**
- **yolov10x** — every conv is depthwise (`groups == in_ch`, e.g. `320←320 groups=320`) → harness fatals (depthwise not wired). Whole family out.
- **segformer dwconv blocks** (`groups=128/256/640/576/960`) and its **WIDTH_SHARDED** convs → ineligible.
- **swin_s** (`96←3`, `per_core_N=3`), **model_k_256x256** (dilation>1, `out_ch≤64`), **vanilla-unet** low-channel rows → `per_core_N ≤ DST`, relaxation is a no-op.
- **1kX1k** (`out_ch=64 → per_core_N=2`; many use DRAM slicing the CONFIG can't express) → no-op / unsupported.
- **SDXL/SD/ResNet50 high-channel convs as the models run them (BLOCK_SHARDED)** → `per_core_N` collapses to 1-3, relaxation is a no-op (demonstrated on ResNet50 L3 BS → `per_core_N=1`).

**OOM on n150 (abh≥64 + fp32 weights + ROW_MAJOR untilize blow the 1.46 MB L1 / core):** these would have been the *expensive* relaxation-eligible convs but do not fit:
`SD 320←960 64×64 1×1` (deep K, the real expensive SD conv), `320←320 32×32`, `320←128 64×64 b2`, `192←192 28×28`, `448←256 16×16`, `RN50 256←256 28×28`, `vanilla-unet 288←288 60×80`.
This is itself part of the answer: the convs where the relaxation could matter *and* be expensive don't fit L1 on n150 once `act_block_h≥64` is forced — so the fitting natural-diff set is necessarily modest-FLOP (~0.75-3 GFLOP), though high wall-clock (57-340µs) due to fp32 weights + HiFi4 + ROW_MAJOR untilize + no l1_acc.

---

## What could not be measured, and recommended follow-ups (for the human to schedule)

1. **The relaxation's actual perf is unmeasurable on this branch** — it FATALs at the op validate gate before the kernel runs. To measure it one must (a) condition `conv2d_device_operation.cpp` validate (`:119-122` and `:131-134`) on the `tile_pack_row_major` path, **and** (b) verify the kernel's TileRowMajor plain-untilize path (`conv_bmm_tilize.cpp:632-645`) is numerically correct for `h>1` (the 2026-06-02 forced-2×4 test gave PCC ~0.50 on the older kernel; the new fallback is untested for `h>1` because the gate never lets it run). This is the harness gap to close before any conv relaxation perf claim is possible. **This is op/validate work, not a measurement — deliberately not done here (measurement-only task).**
2. If/when the gate is opened, the **Group-1 candidates (A/B/C/D)** are the ready-made benchmarks; expectation from prior data is *neutral-to-negative* in conv's bf8+bias regime and the gate is unlikely to surface a win.
3. The **`helper_sbm` +3.3% on the 1×1 downsample (F)** is worth a glance if cheap 1×1 convs matter — it's a fixed `reblock_and_untilize` overhead the TileRowMajor path avoids; benign on larger convs.

---

## Appendix — reproduction

CONFIG is env-overridable (defaults unchanged); drive with `CB_*` vars + `TT_CONV_BENCH_MODE`. Example (candidate A):

```
CB_BATCH=2 CB_OUT_CH=320 CB_IN_CH=256 CB_H=32 CB_W=32 CB_FILTER=3 CB_STRIDE=1 CB_PAD=1,1,1,1 \
CB_SHARD=HS CB_ABH=64 CB_FP32_ACCUM=true CB_BIAS=true \
TT_CONV_BENCH_MODE=helper_trm python -m tracy -r -p -v -m pytest \
  tests/ttnn/unit_tests/operations/conv/test_conv_bench.py
```

Read the warm conv duration from the newest `generated/profiler/reports/*/ops_perf_results_*.csv`, column
`DEVICE KERNEL DURATION [ns]`, 2nd `Conv2dDeviceOperation` row. `CONV_BENCH[...]` log line reports
`per_core_M`/`per_core_N`, the SubblockMajor vs TileRowMajor tuner picks, and the subblock actually used.
