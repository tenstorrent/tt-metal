# GH #45995 — Conv2d matmul-helper baselines (real-config data acquisition)

**Date:** 2026-06-05  **Device:** Wormhole n150 L (single chip)  **Branch:** `wransom/conv_bench`
**Measurement:** Tracy `DEVICE KERNEL DURATION [ns]`, warm (2nd of `run_twice`) `Conv2dDeviceOperation` row, ≥2 reps / median (device timing spread 0.0–0.2%).
**Scope:** conv2d only (the bench wiring is conv2d-specific; conv1d / conv_transpose2d / conv3d are separate ops, not wired).
**Data:** `conv_bench_data.csv` (full per-conv/per-mode rows). Submodules clean.

---

## TL;DR

Measuring **real model conv2d at each conv's real config** (its real weights dtype, accum, math fidelity, output
layout, packer_l1_acc, shard layout, bias — not a forced test regime), the matmul-helper **migration**
(`main` hand-written kernel → `helper_sbm` matmul-helper kernel, same tuner-picked subblock) is:

> **neutral-to-favorable: mean −0.90% across 35 heavyweight convs, range −6.8% → +1.3%, with the real wins
> (−2.6% to −6.8%) concentrated on large BLOCK_SHARDED convs, and the only regression a marginal +1.3%.**

i.e. swapping conv onto the matmul helper is **safe** (no meaningful regression anywhere measured) and a **real win**
on several heavyweight BLOCK_SHARDED convs. The third mode — `helper_trm` (the SubblockMajor→TileRowMajor subblock
*relaxation*) — is **structurally inapplicable to essentially all real convs** (see §4), so the real-config dataset
is a 2-mode (`main` vs `helper_sbm`) comparison; the relaxation only ever engages in a forced ROW_MAJOR-output regime
(prior-session study, summarized in §5).

---

## 1. Methodology — what is held vs varied

The conv_bench harness runs the conv compute kernel in 3 env-selected modes (`TT_CONV_BENCH_MODE`):
- `main` — main's verbatim hand-written (no matmul-helper) conv kernel.
- `helper_sbm` — the matmul-helper conv kernel, SubblockMajor (what every normal conv on this branch already uses).
- `helper_trm` — the matmul-helper kernel with the TileRowMajor subblock relaxation (ROW_MAJOR-output only; §4).

Earlier the harness *forced* ROW_MAJOR output + packer_l1_acc OFF + fp32 weights to make `helper_trm` runnable — but
those are concessions, not how models run convs, and that fp32-weights regime is exactly where the relaxation wins.
The harness was **rewired** (commit `ff9b84c95e0`) to pass each conv's **real** config through: `output_layout`,
`packer_l1_acc`, and `weights_dtype` are now real per-conv (the three bench forcings were lifted, all bench-gated →
non-bench convs byte-identical). So each row below is `main` vs `helper_sbm` **at the model's real conv config**, on
the same tuner-picked subblock — isolating the helper kernel's efficiency vs the hand-written one.

`main` and `helper_sbm` produce identical PCC per conv (verified every run); only device-kernel duration differs.

---

## 2. Cross-model summary (`helper_sbm` vs `main`, real config)

| model | convs measured | did-not-fit n150 | Δ min | Δ max | Δ mean | wins ≤ −2% |
|---|---|---|---|---|---|---|
| ResNet50 | 11 | 0 | **−6.8%** | +1.3% | −1.87% | 3 (−6.8, −5.8, −5.3) |
| SDXL UNet | 12 | 7 | −0.9% | +0.3% | −0.23% | 0 |
| vanilla UNet | 12 | 3 | **−2.6%** | +0.2% | −0.69% | 1 (−2.6) |
| **all** | **35** | **10** | **−6.8%** | **+1.3%** | **−0.90%** | 4 |

(Δ = (helper_sbm − main)/main; negative = helper faster.)

---

## 3. Per-family detail

### ResNet50 — bf8 weights, LoFi, fp32_accum=False, **packer_l1_acc ON**, TILE out, bias

| conv (out←in, H×W, k/stride) | shard | per_core_N | subblock | main ns | helper_sbm ns | Δ |
|---|---|---|---|---|---|---|
| 64←64 56² 3×3 | HS | 2 | 1×2 | 96,456 | 96,052 | −0.4% |
| 128←128 56² 3×3 s2 | HS | 4 | 2×4 | 58,342 | 57,902 | −0.8% |
| 128←128 28² 3×3 | HS | 4 | 2×4 | 58,374 | 57,773 | −1.0% |
| 256←256 28² 3×3 s2 | BS | 1 | 8×1 | 191,232 | 189,586 | −0.9% |
| 256←256 14² 3×3 | BS | 1 | 8×1 | 191,172 | 189,856 | −0.7% |
| **512←512 14² 3×3 s2** | BS | 2 | 4×2 | 106,262 | 99,034 | **−6.8%** |
| **512←512 7² 3×3** | BS | 2 | 4×2 | 106,120 | 100,016 | **−5.8%** |
| 256←64 56² 1×1 s2 | HS | 8 | 1×8 | 13,993 | 13,973 | −0.1% |
| 1024←512 28² 1×1 s2 | BS | 4 | 2×4 | 71,450 | 71,378 | −0.1% |
| **2048←1024 14² 1×1 s2** | BS | 8 | 1×8 | 58,219 | 55,158 | **−5.3%** |
| stem 64←16 115² 4×4 | HS | 2 | 1×2 | 384,888 | 389,843 | +1.3% |

### SDXL UNet — bf8 weights, HiFi2, fp32_accum=False, **packer_l1_acc OFF**, BS, bias, TILE out

| conv (out←in, H×W, stride) | per_core_N | subblock | main ns | helper_sbm ns | Δ |
|---|---|---|---|---|---|
| 768←1152 64² | 3 | 2×3 | 1,173,546 | 1,170,567 | −0.3% |
| 1536←1536 16² | 6 | 1×6 | 331,202 | 329,978 | −0.4% |
| 1536←1536 32² | 6 | 1×6 | 623,048 | 623,036 | −0.0% |
| 1536←2304 32² | 6 | 1×6 | 909,058 | 911,632 | +0.3% |
| 1536←3072 16² | 6 | 1×6 | 662,928 | 661,777 | −0.2% |
| 1536←3072 32² | 6 | 1×6 | 1,202,184 | 1,196,355 | −0.5% |
| 768←384 64² | 3 | 2×3 | 482,632 | 482,084 | −0.1% |
| 1536←768 32² | 6 | 1×6 | 335,952 | 335,306 | −0.2% |
| 768←768 64² | 3 | 2×3 | 790,420 | 783,630 | −0.9% |
| 1536←1536 32² s2 | 6 | 1×6 | 331,220 | 330,028 | −0.4% |
| 384←384 128² s2 | 2 | 4×2 | 379,606 | 379,852 | +0.1% |
| 768←768 64² s2 | 3 | 2×3 | 209,798 | 209,282 | −0.2% |

Did not fit n150 (7): 384←1152 128², 1536←1536 64², 768←1536 64², 768←2304 64², 384←384 128², 384←768 128²,
768←768 128² — these exceed n150 L1 (the model runs them on a 5×8 BS grid + activation-width DRAM slicing, which
this single-chip harness does not express; they show as OOM / `program.cpp:1476` L1-allocation). **Caveat:** the
fitting SDXL rows use n150's auto BS grid, not SDXL's exact 5×8 grid, so absolute numbers are n150-native baselines
for the real shapes.

### vanilla UNet — bf16 weights, LoFi, fp32_accum=False, **packer_l1_acc ON**, **no bias**, HS/BS, TILE out

| conv (out←in, H×W) | shard | per_core_N | subblock | main ns | helper_sbm ns | Δ |
|---|---|---|---|---|---|---|
| 32←3 480×640 | HS | 1 | 6×1 | 190,371 | 189,742 | −0.3% |
| 64←32 240×320 | HS | 2 | 2×2 | 72,340 | 71,553 | −1.1% |
| 64←64 240×320 | HS | 2 | 2×2 | 121,282 | 120,638 | −0.5% |
| 128←64 120×160 | HS | 4 | 2×4 | 43,514 | 42,924 | −1.4% |
| 128←128 120×160 | HS | 4 | 2×4 | 77,472 | 76,890 | −0.8% |
| 256←128 60×80 | HS | 8 | 1×8 | 82,972 | 82,494 | −0.6% |
| 256←256 60×80 | HS | 8 | 1×8 | 161,446 | 160,946 | −0.3% |
| 288←288 60×80 | HS | 9 | 1×3 | 214,670 | 214,192 | −0.2% |
| 512←256 30×40 | HS | 16 | 1×8 | 212,566 | 212,145 | −0.2% |
| **512←512 30×40** | BS | 2 | 1×2 | 164,758 | 160,402 | **−2.6%** |
| 256←512 60×80 | BS | 1 | 1×1 | 449,760 | 450,638 | +0.2% |
| 32←64 256×256 | HS | 1 | 8×1 | 82,770 | 82,404 | −0.4% |

Did not fit n150 (3): 32←32 480×640, 128←256 120×160, 64←128 240×320.

---

## 4. `helper_trm` (the subblock relaxation) — structurally N/A for real convs

`helper_trm` relaxes the SubblockMajor constraint (`out_subblock_w == per_core_N || out_subblock_h == 1`) to let the
tuner pick a larger subblock. It only changes anything when, on a HEIGHT_SHARDED conv, `per_core_N > DST` **and**
`per_core_N` is **not a multiple of DST** **and** `per_core_M ≥ 2`, **and** it only runs on the ROW_MAJOR-output
(untilize) path. Real heavyweight convs fail these:
- heavy convs are BLOCK_SHARDED (per_core_N collapses to 1–3) or use fp32_accum=False (DST=8, raising the bar);
- real channel counts cluster on multiples of 32 → per_core_N ∈ {2,4,8,16} (multiples of 4 → no shape change);
- they use TILE output (the relaxation is row-major-output-specific).

So on the real-config corpus `helper_trm` is a no-op / not-applicable and is excluded from §2–3. Where it *does*
engage (forced ROW_MAJOR + fp32 weights + the right HS channel count), it gives a 6–9% win — but that is a regime
real convs do not use (see §5). **Conclusion: there is no real-conv `helper_trm` data to collect on conv2d; the real
signal is the migration in §2.**

---

## 5. Background — the wiring journey (forced regime, prior sessions)

To make `helper_trm` runnable at all, two real bugs were found and fixed (committed): the conv op `validate()`
(`conv2d_device_operation.cpp:119-122/131-134`) hard-gated `out_subblock_h>1` — relaxed for the HelperRowMajor path;
and a CB self-deadlock in the TileRowMajor+bias kernel path (`conv_bmm_tilize.cpp`) — fixed by routing bias through
SubblockMajor+`reblock_and_untilize`. In the **forced** ROW_MAJOR + l1_acc-off + **fp32-weights** regime the
relaxation then gives a real ≈6–9% device-kernel win (out_subblock_h=2 halves fp32-weight re-reads). That regime is
not how models run convs (fp32 weights, ROW_MAJOR out), which is why the real-config dataset above is the
authoritative answer and the forced wins are noted only as mechanism.

---

## 6. Coverage / out of scope

- **Collected (heavyweight, eligible, fits n150):** ResNet50, SDXL UNet, vanilla UNet — 35 convs.
- **Structurally ineligible (not collected):** yolov4/v8/v9/v10, mobilenetv2, efficientnet heavy convs = **depthwise**
  (groups=in_ch → different kernel, harness fatals); segformer/swin heavy convs = **depthwise/width-sharded** (only
  low-FLOP non-depthwise stems are eligible).
- **Eligible but does not fit n150:** SDXL VAE (512²/1024² spatial, groups=1) needs DRAM activation slicing to fit —
  the harness can't express slicing, so they exceed L1; the SDXL UNet "did-not-fit" rows above are the same story.
- **Out of scope:** conv1d / conv_transpose2d / conv3d (separate ops, no bench wiring).

---

## Appendix — reproduction

Per-conv, env-driven (defaults match real usage: TILE out, l1_acc on). Example (ResNet50 L4):
```
CB_BATCH=20 CB_OUT_CH=512 CB_IN_CH=512 CB_H=14 CB_W=14 CB_FILTER=3 CB_STRIDE=2 CB_PAD=1,1,1,1 CB_SHARD=BS \
CB_WEIGHTS_DTYPE=bfloat8_b CB_OUT_DTYPE=bfloat16 CB_FP32_ACCUM=false CB_FIDELITY=LoFi CB_L1_ACC=true CB_OUT_LAYOUT=tile \
CB_BIAS=true  TT_CONV_BENCH_MODE=<main|helper_sbm> python -m tracy -r -p -v -m pytest \
  tests/ttnn/unit_tests/operations/conv/test_conv_bench.py
```
Read warm conv duration from newest `generated/profiler/reports/*/ops_perf_results_*.csv`, col
`DEVICE KERNEL DURATION [ns]`, 2nd `Conv2dDeviceOperation` row. Forced relaxation regime:
add `CB_OUT_LAYOUT=row_major CB_L1_ACC=false` and `TT_CONV_BENCH_MODE=helper_trm`.
