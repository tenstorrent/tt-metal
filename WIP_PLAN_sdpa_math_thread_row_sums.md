# WIP plan: `math_thread_row_sums`: benchmarks, docstring, reply to review

Temporary file: delete before the PR merges. Answers pavlejosipovic's third review comment on PR #59695.

## The review comment

The reviewer asked:
1. Which exact workload and configuration make the MATH-thread row sums faster: Q/K/V shape, chunk sizes, `reuse_kv`,
   math fidelity, `math_approx_mode`, `fp32_dest_acc_en`? The PR gives Blackhole latencies but not the fidelity.
2. Matched PACK-vs-MATH profiles at the same settings, ideally MATH/PACK busy or stall cycles.
3. Why Blackhole only? Blackhole has roughly 2× Wormhole's packer bandwidth, so why does moving work *off* PACK help
   there? The reviewer's premise: SDPA originally moved the row reduction onto PACK because MATH was the bottleneck.

## Decision (already made)

`math_thread_row_sums` **stays opt-in**. Whether it helps depends on fidelity and shape. It also costs some accuracy
wherever it's on: one 16-bit DEST accumulator per row, compared with the pack thread's L1 sums (SDXL PCC
0.9972 → 0.9968). The work below supplies the numbers for the docstring and the reply. It doesn't change the
default.

## What's already known

Sources: `amorrison/high-batch-optim`, `models/demos/blackhole/pplx_embed_4b/doc/NEGATIVE_RESULTS.md` §65 and the
`doc/POSITIVE_RESULTS.md` row "batched SDPA row sums on the math thread (09-30)".

- **Workload:** pplx-embed-4b batched prefill.
  - Q `[B, 32, 512, 128]`, K/V `[B, 8, 512, 128]`, bfp8, non-causal, `output_heads_concat=True`.
  - `reuse_kv=True`, q128/k512 (one K chunk). Grid 12×8 at bs8, 12×10 at bs16/32.
  - **LoFi**, `math_approx_mode=False`, `exp_approx_mode=True`, `fp32_dest_acc_en=False` (streaming kernel),
    `packer_l1_acc=True`.
- **Per-thread zones** (bs8 shape, compute only, one q128 unit). The per-row-group figures cover 8 score tiles:

  | | PACK row sums (before) | MATH row sums (after) |
  |---|---|---|
  | MATH Q·Kᵀ 2×4 subblock | ~690 | ~690 |
  | MATH `SUB` (mostly the `tile_regs_acquire` wait on PACK) | 725–750 | 265–420 |
  | PACK `EXP` (SFPU issued from the pack thread) | 500–540 | 500–540 |
  | PACK `SUB_EXP` packs (incl. 8 L1-acc row-sum packs) | 455–465 | 121–168 |
  | normalize (per unit) | 2,887 | 3,615 |
  | **unit total** | **20,302** | **18,766** |

  Probes (wrong output): no row-sum packs −11%, no exp −19%.
- **History:**
  - `666d33c8bc6` (2025) put the partial row sums on PACK via L1 accumulation. That kernel ran `exp_tile` on the
    **math** thread (`compute_common.hpp:398`).
  - The streaming kernel (`08a5992cf1d`, 2026-03) moved exp onto the **pack** thread (`exp_packthread_tile`,
    `compute_streaming.hpp:614`). Since then PACK carries exp *and* the row-sum packs.
- **Hypothesis to confirm:** which thread paces depends on MATH's matmul cycles per score tile vs. PACK's exp and
  row-sum cycles per score tile.
  - LoFi, `d=128` (Q·Kᵀ inner dim 4 tiles), approximate exp: PACK is the bottleneck. Measured.
  - HiFi2/HiFi4 (about 2×/4× matmul cycles), larger `d`: MATH is likely already the bottleneck, so the extra
    normalize work lands on it and the flag loses. Not measured.
  - `exp_approx_mode=False`: costlier exp on PACK, which pushes back toward PACK being the bottleneck. Not measured.
- **Blackhole only, as a matter of scope:** the batched normalize matmul (`matmul_block` with `rt_dim = sbh`,
  `kt_dim` = row stride, then a 1×1 re-init to avoid the V-matmul hang) was written and validated against Blackhole's
  LLK only. Wormhole has its own no-MOP matmul (`bb3abd495f3`), so a port is plausible but unmeasured. Packer bandwidth
  isn't the limit: the removed cost is pack-*thread* time (SFPU exp issue plus ~57-cycle single-tile L1-acc packs), not
  bytes.
- **Speedups recorded so far disagree:**

  | Source | bs8 | bs16 | bs32 |
  |---|---|---|---|
  | §65 | −5.9% | −7.0% | −5.7% |
  | PR body (K/V in L1: 202.9 / 312.8 / 570.6 → 188.1 / 286.8 / 528.5 µs) | −7% | −8% | −7% |
  | nanobind docstring | 6% | 7% | 7% |

  Re-measure and use one set everywhere. **Settled:** the re-measurement (device cycles, K/V in L1)
  gives −5.3 / −6.6 / −6.4%, which matches §65 (−5.9 / −7.0 / −5.7%) within noise. The PR body's
  −7 / −8 / −7% and the docstring's 6 / 7 / 7% were wall-clock µs, which drifts with AICLK.

## Benchmarks

### Tooling

On `amorrison/high-batch-optim` under `models/demos/blackhole/pplx_embed_4b/perf_tools/`:
- `bench_common_traced.py`: traced device time.
- `bench_sdpa_floors.py`: the model's reuse_kv q128 config, all-DRAM / K-V-L1 / all-L1 placements.
- `bench_sdpa_zones.py`: per-unit device-profiler zones, plus `--parse`.
- `sdpa_kernel_variants.py`: patched kernel trees, `zones` / `zconly`.

Check them out into the working tree uncommitted (`git checkout origin/amorrison/high-batch-optim -- <paths>`).

Added for this round (same directory, also uncommitted — they belong on `amorrison/high-batch-optim`):
- `bench_cycles.py` + `run_cycles.sh`: device cycles per call, flag off vs on, from traced replays. **Use this
  for the headline numbers**, not wall-clock µs.
- `bench_row_sums.py`: traced µs and PCC vs fp32 torch for a whole arm file, flag alternating per round.
- `bench_zones.py` + `run_zones.sh`: per-work-unit zones for the `zconly` / `zfconly` variants.
- `bench_denom.py`: per-row denominator error vs `S` (V=ones probe).
- `arms_all.json` / `arms_A2.json` / `arms_Z.json`: the arm matrix below.
- `showcyc.py`, `collect.py`, `zsum.py`, `zfsum.py`: summarise the above.
Run them from a directory with no `ttnn/` tree (kernel lookup tries the cwd first), with one `TT_METAL_CACHE` per
variant.

**Porting needed before the runs.** The variants script patches the kernels as they were *before* this PR's refactors:
`REUSE_KV` and `SDPA_SUM_ON_PACK` were defines then. Now they're compile-time args, and the define is gone.
- [x] Drop the `sumpack` variant. `math_thread_row_sums=False` is now the same-binary control.
- [x] Check that the `zones` / `zconly` / `nosum` / `noexp` regex patches still apply to the current
      `compute_streaming.hpp`. Fix the anchors where they don't.
- [x] Update the `DM_STUB` reuse detection: it reads `#ifdef REUSE_KV`, which should become the compile-time arg.
- [x] Add CLI arguments to `bench_sdpa_floors.py` / `bench_sdpa_zones.py` for `math_thread_row_sums`,
      `math_fidelity`, `exp_approx_mode` and the shape (`B`, `NQH`, `NKH`, `S`, `d`, q/k chunk, grid).

### Matrix

Every arm runs flag off vs. flag on, in the same binary. For each, record **device cycles per call** (traced
replays under the device profiler — wall-clock µs turned out to drift too much with AICLK; see Method below) and
the per-thread zones: phase zones and the unit total from `zconly`, and MATH `SUB` / acquire wait, PACK `EXP`,
PACK `SUB_EXP` from `zfconly` (the kernel's own `profiling_enabled` zones, which `zconly` does not turn on).

**A. pplx-embed-4b (the motivating workload).** q128/k512, `reuse_kv=True`, `exp_approx_mode=True`, bs8 on 12×8,
bs16/32 on 12×10, K/V in L1 (bs8/16) and all DRAM (bs32).
- [x] LoFi: re-measure the speedup and settle the inconsistent numbers above.
- [x] HiFi2: tests whether MATH takes over as the bottleneck.
- [x] HiFi4.

**B. Head dimension**, at the pplx batch-16 shape with LoFi. Change `d` only, keeping one K chunk.
- [x] `d` = 64, 128, 256. Larger `d` means more Q·Kᵀ cycles per score tile on MATH.

**C. Exp mode.**
- [x] pplx bs16, LoFi, `exp_approx_mode=False`.

**D. SDXL 1024×1024 on Blackhole** (`models/demos/stable_diffusion_xl_base/tt/model_configs/model_configs_1024x1024BH.py`,
`tt_attention.py`). These are real single-K-chunk shapes at the accuracy-sensitive end. Grid 11×10, q128,
`exp_approx_mode=False`, `math_approx_mode=False`, `fp32_dest_acc_en=False`.
- [x] Self-attention in down/up blocks 2: 1024 tokens, k1024 (one K chunk), 20 heads, `d=64`, **LoFi** (the model's
      setting). Confirm the batch (CFG) in the model.
- [x] The same shape at **HiFi2**. ~~The model uses HiFi2 for the down-block-1 / up-block-0 self-attention, but
      those blocks run 4096 tokens with k512, so the flag doesn't apply there.~~ **Wrong:** `get_sdpa_config`
      gives `512_K` only to `down_blocks.1` / `up_blocks.1`. `up_blocks.0` is HiFi2 *and* gets `1024_K` at 1024
      tokens, so the flag does apply there — it is a real HiFi2 single-K-chunk shape. Measured at ~+2%.
- [x] Cross-attention: 77-token K, k128 (one K chunk), LoFi.

**E. Batch 1 (pplx).**
- [x] q256/k512 on 8×8 and packed q192/k512 on 11×8, LoFi. Does the gain hold with about one unit per core?

**F. Wormhole.** Out of scope unless the reply should argue for or against a port. If so, it would mean porting the
batched normalize to the WH no-MOP matmul first.

### Accuracy (each arm in A–D)

- [x] PCC and max error vs. fp32 torch, flag off vs. on.
- [x] Per-row denominator error statistics vs. `S`: the docstring's "S=1024 std 0.007 → 0.012" claim.
- [x] SDXL module PCC tests at the model config with the flag forced on, to repeat the 0.9972 → 0.9968 data point:
      `test_module_tt_transformermodel`, `test_module_tt_crossattnmidblock2d`, `test_module_tt_crossattnupblock`.

### Expected outcome

Two tables:
1. the speedup by (fidelity, `d`, exp mode);
2. per-thread zones at LoFi vs. HiFi2 showing where the bottleneck moves.

If HiFi2 doesn't move it, revisit the hypothesis before writing the reply.

## Docstring update (`sdpa_nanobind.cpp`, `math_thread_row_sums`) — DONE

Rewritten from the measurements below: it now states the regime (which thread paces the softmax),
gives the speed numbers with the full config *and the placement*, covers all three fidelities plus the
`d` and DRAM-vs-L1 axes, and replaces the unverifiable denominator-std claim with what the probe
actually shows. See the diff on `sdpa_nanobind.cpp`.

**Still to do before merge:** update the PR description to use the same numbers, and delete this file.

## Reply to the review comment — DONE

Final text in `REVIEW_REPLY.md` at the repo root (also temporary; paste into the PR thread and delete).

## Results (measured 2026-10-14, Blackhole P150b)

Build: `amorrison/sdpa-row-sums-wip` @ `ca00c8b1f59`, `./build_metal.sh --release -c` (Tracy on).

### Method

**Latency is reported in device cycles, not wall-clock µs.** Blackhole's AICLK moves with load, and the
pplx batch-16 shape sits between two clock states (the reason `perf_tools/ab_multi.sh` exists), so the same
config re-measured minutes later moves by up to 9% in µs. Device cycles do not depend on the clock: the
same arm measured twice under different names (`A-pplx-bs16-LoFi` and `B-pplx-bs16-LoFi-d128`) agrees to
0.03 pp in cycles while the wall-clock readings of the same two runs disagree by 6 pp.

Per arm: one traced SDPA call, 11 blocking replays, first dropped; cycles = span of the `-FW` zones per
replay, median over replays. Run-to-run spread within an arm ("stab" below) is 0.2-0.6% for the pplx
shapes. Flag off and on are the same binary — `math_thread_row_sums=False` is the control.

Zones: `sdpa_kernel_variants.py` `zconly` (phase zones, compute-only) and `zfconly` (the kernel's own
`profiling_enabled` zones, compute-only, sized to one work unit per core).

### A. pplx-embed-4b batched prefill — fidelity

Q `[B, 32, 512, 128]`, K/V `[B, 8, 512, 128]`, bfp8, non-causal, `output_concat_heads`, `reuse_kv`,
q128/k512 (one K chunk), `exp_approx_mode=True`, `math_approx_mode=False`, `fp32_dest_acc_en=False`,
`packer_l1_acc=True`. 12x8 at bs8, 12x10 at bs16/32. K/V in L1.

| fidelity | bs8 | bs16 | bs32 |
|---|---|---|---|
| **LoFi** | **-5.3%** | **-6.6%** | **-6.4%** |
| HiFi2 | +7.1% | +7.3% | +7.4% |
| HiFi4 | +8.7% | +8.8% | +8.6% |

Negative = faster with the flag on. This settles the three disagreeing numbers in the plan: the docstring's
"6 / 7 / 7% faster" is close but the shape of the curve is flat, not rising, and the sign flips at HiFi2.

#### Placement matters as much as batch

The earlier bs32 figure was measured all-DRAM, where the op is DRAM-bound and the compute saving is hidden:

| placement | bs8 | bs16 | bs32 |
|---|---|---|---|
| K/V in L1 | -5.3% | -6.6% | -6.4% |
| all DRAM | -5.6% | -1.6% | -0.8% |

So the gain is a *compute* gain: it survives only while the kernel is not waiting on DRAM. At bs16/32 with
K/V in DRAM it nearly vanishes. Quote the flag's speedup with the placement stated.

### B. Head dimension (bs16, LoFi, one K chunk)

| d | delta |
|---|---|
| 64 | **-8.8%** |
| 128 | -6.6% |
| 256 | **+4.2%** |

Larger `d` means more Q·Kᵀ cycles per score tile on the math thread, which is exactly the axis that moves
the bottleneck. At d=256 the flag already loses at LoFi.

### C. Exp mode (bs16, LoFi)

| exp_approx_mode | delta |
|---|---|
| True | -6.6% |
| False | -6.1% |

Exact exp costs the pack thread more, so the plan expected a *larger* gain. It is slightly smaller. The exp
is not what the flag removes from PACK — the row-sum packs are (see the zones below), and their cost is
unchanged by the exp mode, so the relative gain is near-identical.

### D. SDXL 1024x1024 on Blackhole

20 heads, d=64, CFG batch 2, grid 11x10, `exp_approx_mode=False`, `math_approx_mode=False`,
`fp32_dest_acc_en=False`, bf16 in L1. These arms are noisier (8-18% replay spread) than the pplx ones;
each figure below is the median of three independent runs.

| shape | fidelity | delta |
|---|---|---|
| self-attn, 1024 tokens, k1024 (one chunk) | LoFi | ~-1% |
| self-attn, 1024 tokens, k1024 (one chunk) | HiFi2 | ~+2% |
| cross-attn, 77-token K, k128 | LoFi | ~-2% |

**Correction to the plan:** the plan assumed the HiFi2 self-attention blocks never hit the single-K-chunk
path. They do. `get_sdpa_config` gives `512_K` only to `down_blocks.1` / `up_blocks.1`; `up_blocks.0` is
HiFi2 *and* gets `1024_K` at 1024 tokens, so the flag applies there. It is roughly +2% there.

The flag is close to neutral on every SDXL shape: d=64 and exact exp push toward PACK-bound (which helps),
but only 2 batch x 20 heads of work and `nlp_concat_heads` done separately leave less compute to save.

#### SDXL module PCC (1024x1024, Blackhole)

| test / case | what the flag touches | flag off | flag on | CI bar | on |
|---|---|---|---|---|---|
| `transformermodel` `down_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9990458 | 0.9990448 | 0.998 | passes |
| `transformermodel` `down_blocks.2` (1024 tok, k1024) | LoFi self-attn + cross-attn | **0.997167** | **0.996846** | 0.997 | **fails** |
| `crossattnmidblock2d` (1024 tok, k1024) | LoFi self-attn + cross-attn | **0.9972232** | **0.9966934** | 0.9968 | **fails** |
| `crossattnupblock` case 0 = `up_blocks.0` (1024 tok, k1024) | **HiFi2** self-attn + cross-attn | 0.9897058 | 0.9847438 | 0.968 | passes |
| `crossattnupblock` case 1 = `up_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9930153 | 0.9928541 | 0.993 | **fails** |

Run with the flag forced on through a temporary env hook in `tt_attention.py` (reverted afterwards).

Three things to take from this:

- Row 1 is the clean control for the *self-attention* path: 8 K chunks there, so the only eligible call in the
  block is its 77-token cross-attention, and the block moves by 1e-6.
- Rows 2 and 3 are the headline accuracy cost and reproduce the docstring's "0.9972 -> 0.9968" exactly. Both
  fall below their CI bar.
- Row 5 contradicts the docstring's old "with a few K tiles it makes no measurable difference". The only thing
  the flag touches in `up_blocks.1` is the same 77-token cross-attention as row 1, yet the block moves 1.6e-4
  and crosses its bar. That bar is passed by only 1.5e-5 with the flag off, so this is a knife-edge test rather
  than a large regression — but "no measurable difference" was too strong, and the docstring now says so.
- Row 4 has the largest absolute drop (5e-3) and still passes, because its bar is loose (0.968).

### E. Batch 1 (pplx, LoFi, no reuse_kv)

| config | delta |
|---|---|
| q256/k512 on 8x8 | ~-5% (noisy: one work unit per core, 36-70% replay spread) |
| packed `pack_gqa_heads` q192/k512 on 11x8 | **-6.6%** (spread 3%) |

The gain holds at batch 1. It does not need `reuse_kv` — it is a per-unit compute saving.

### Zones: where the bottleneck is, and where it moves

#### Phase zones, per work unit (`zconly`, compute-only, pplx bs8, 12x8, mean cycles over 3072 units)

| phase | LoFi math off/on | LoFi pack off/on | HiFi2 math off/on | HiFi2 pack off/on |
|---|---|---|---|---|
| P1 QK+exp rowgroup #1 | 3015 / 2998 | 2889 / 2857 | 5091 / 4978 | 5038 / 4939 |
| P1 QK+exp rowgroup #2 | 5443 / **4555** | 6005 / **4833** | 6183 / 6174 | 6176 / 6160 |
| P1 reduce max #1 | 231 / 230 | 679 / 673 | 232 / 235 | 1227 / 1224 |
| P2 exp rg1 + PV rg0 | 5691 / **4555** | 5670 / **4556** | 5981 / 5970 | 5933 / 5683 |
| P2 PV rg1 | 2965 / 2703 | 213 / 207 | 4607 / 4586 | 212 / 207 |
| P2 normalize #1 | 1440 / **1828** | 3637 / **4024** | 1604 / **2618** | 5871 / **6870** |
| P2 normalize #2 | 1428 / **1812** | 1438 / **1830** | 1502 / **2493** | 1504 / **2495** |
| **unit total** | **20295 / 18753** | 20298 / 18752 | **25252 / 27113** | 25248 / 27106 |
| | **-1542 (-7.6%)** | | **+1861 (+7.4%)** | |

At **LoFi** the two softmax phases get ~1100-1170 cycles cheaper **on both threads** — the signature of a
pack-thread bottleneck being relieved: MATH was waiting, so PACK's saving shortens the phase for everyone.
Normalize grows +388 per row group (+775 per unit). Net -1542.

At **HiFi2** the same phases barely move (-9, -16, -11 on MATH) — PACK is no longer the limiter, so taking
work off it buys nothing. Normalize still grows, and by more (+1014 and +991, i.e. +2005 per unit, because
the normalize matmul itself runs at HiFi2). Net +1861.

#### The kernel's own zones (`zfconly`, one work unit per core, mean cycles per occurrence)

Per `sub_exp_block_bcast_cols` call, which covers 8 score tiles (2x4 subblock):

| thread / zone | LoFi off | LoFi on | HiFi2 off | HiFi2 on |
|---|---|---|---|---|
| MATH `SUB` (mostly the `tile_regs_acquire` wait on PACK) | **701** | **390** | **293** | **293** |
| MATH `Q@KT MM+Pack` | 1353 | 1367 | 1903 | 1903 |
| PACK `EXP` (SFPU issued from the pack thread) | 558 | 556 | 608 | 676 |
| PACK `PACK SUB_EXP` (incl. the 8 L1-acc row-sum packs) | **486** | **147** | **479** | **146** |
| MATH `ROW_NORM` (per row group) | 1578 | 1999 | 1727 | 2716 |
| MATH `NORM_MATMUL_RECIP` | 459 | 1357 | 414 | 1850 |
| **MATH `TRISC-KERNEL`** | **25456** | **23943** | **30178** | **31915** |

This is the whole mechanism in one table:

- The flag removes the **same ~339 cycles** of row-sum packs from PACK at both fidelities (486 -> 147 and
  479 -> 146). The saving on the pack thread does not depend on fidelity.
- What changes is whether MATH was **waiting for it**. At LoFi MATH idles 701 cycles per call in
  `tile_regs_acquire`; removing PACK's packs cuts that to 390 (-311, i.e. almost all of PACK's saving turns
  into real time). At HiFi2 MATH's wait is only 293 and **does not move at all** — its own Q·Kᵀ has grown
  from 1353 to 1903 cycles and is now the long pole, so PACK's saving is absorbed by slack.
- Meanwhile the normalize matmul that the flag *adds* to MATH costs +898 (LoFi) / +1436 (HiFi2) cycles of
  `NORM_MATMUL_RECIP` per row group. At LoFi that is less than what MATH recovers; at HiFi2 it is more.

So the predictor is **MATH's Q·Kᵀ cycles per score tile vs. PACK's exp + row-sum cycles per score tile**,
exactly as the plan hypothesised, and the fidelity sweep, the head-dim sweep and the zone data all agree.

### Accuracy

#### Op-level PCC vs. fp32 torch

Across every arm in A-E the PCC moves in the **5th decimal**: pplx LoFi 0.99925 -> 0.99923, HiFi2
0.99961 -> 0.99959, SDXL self-attn LoFi 0.99911 -> 0.99909, SDXL cross-attn 0.99919 -> 0.99918. On its own
a single SDPA call looks unaffected.

#### Per-row denominator error vs. S

Probe: V = all ones, so every output element is exactly 1.0, and both arms share the **same** numerator
matmul — therefore `(out_on - out_off)` is purely the difference between the two denominators. LoFi, bfp8,
8 heads, d=128, one K chunk, q128.

| S | std(out_off - 1) | std(out_on - 1) | std of the denominator difference |
|---|---|---|---|
| 256 | 0.01500 | 0.01449 | **0.00614** |
| 512 | 0.01500 | 0.01465 | **0.00761** |
| 1024 | 0.01654 | 0.01593 | **0.00982** |

Two things to be careful about here:

1. **The denominators do diverge more as S grows**: 0.0061 -> 0.0098 from S=256 to S=1024, a factor 1.6
   over that range. That supports the docstring's mechanism (one 16-bit DEST accumulator rounds the whole
   running sum on every K tile, while the pack thread's per-position L1 sums average their rounding out).
2. **But this probe cannot say the math-thread sum is the worse of the two.** Each arm's total deviation is
   dominated by the shared LoFi numerator matmul error (std ~0.015-0.017), and the flag-on arm is not
   measurably worse on it — if anything marginally better. The docstring's "~1.6x larger (std 0.007 ->
   0.012)" is not reproduced as an off-vs-on ratio at fixed S; 1.6x is what the off-vs-on *difference* grows
   by between S=256 and S=1024.

S >= 2048 could not be measured: a single K chunk that large does not fit (program creation fails), so in
practice the error cannot keep growing past roughly S=1024 at d=128 — the flag's own eligibility rule
(`k_chunk_size >= s`) bounds its exposure.

#### What actually costs accuracy

The single-call PCC barely moves, but the error is systematic, so it accumulates. Of the five SDXL
1024x1024 module PCC cases above, three fall below their CI bar with the flag on. That, not the single-op
PCC, is the reason to keep this opt-in.

### Summary for the docstring

- **Helps** when the pack thread paces the softmax: **LoFi**, small head dim, and the data already on-chip.
  Best measured case -8.8% (bs16, d=64, LoFi, K/V in L1); the pplx shape is -5 to -7% across bs8/16/32.
- **Neutral to harmful** otherwise: **+7% at HiFi2 and +9% at HiFi4** on the same shape, **+4% at LoFi with
  d=256**, and only -1 to -2% once the op is DRAM-bound (bs16/32 all-DRAM) or small (SDXL).
- Costs 3e-4 to 5e-3 of PCC at SDXL module level; three of five 1024x1024 module cases fall below their bar.
- Also update the PR description with these numbers, then delete this file and `REVIEW_REPLY.md`.
