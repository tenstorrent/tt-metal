# L1 residency on the denoise hot path (dg-08, #47465)

Status: current for the full-canvas RMSNorm (the only shipped norm path); provenance-only for the
activation-L1 levers and the HF-fidelity replay, both measured on the deleted token-gather MoE.
Owns: full-canvas RMSNorm — mechanism, win, ULP delta, fp32-accumulation fix, and the 2026-07-30
overturn of its flip gate (absorbed from the deleted `norm_fullcanvas_flip_gate.md`).
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).
Over the 100-line cap: two open contradictions, four traps and two repro pins are not cut for length.

## Shipped: the full-canvas RMSNorm

Nothing selects it — it is the only path. `DG_NORM_FULLCANVAS` was **deleted 2026-07-30**
(`613d8dfd21b`); `tests/test_denoise_forward.py` asserts the gate cannot come back.

**Mechanism.** DiffusionGemma chunked the 256-row canvas into 8x 32-row slices
(`_chunked_norm_forward` / `_rms_norm_dram`) **only** to hit gemma4 RMSNorm's width-sharded fast path
(`rms_norm.py::_forward_sharded`, `block_h=1`, 32-row-only); `norm.forward` on 256 rows falls to the
slow plain-interleaved path. That cost 7 extra slices + 1 DRAM concat + 7 extra sharded-norm launches +
8 I2S/S2I round-trips **per norm call**, at ~6–8 norm calls/layer x 30 layers. The shipped path runs ONE
256-row width-sharded `rms_norm` (`block_h=8`) reusing `norm.tt_weight`, handing the L1 output straight
back; RMSNorm is per-row independent of `block_h`. The reclaimable glue it removes is the chunked-norm
`Slice` (2.5 ms/6L) + `Concat` (1.26 ms/6L) + the redundant `LayerNorm` launches.

**The win — three separate measurements, none superseding the others arithmetically:**

| harness / configuration | reading |
|---|---|
| traced 30L, seed 0 (this pass) | **@48 17.855 → 20.676 t/s (+15.8%)**; @12 49.841 → 61.476 t/s (+23.3%) |
| `sweep_denoise_arms.sh`, device Gumbel, concat MoE ([winter borrow](winter_borrow_20260727.md)) | 238.3 → 195.7 ms/step (−17.9%) |
| the shipped commit | −20.4%/block |

The @48 block delta (−1.956 s/block, 14.3376 → 12.3815 s) is ~13x the ~1.5% run-to-run block noise. The
isolated per-norm micro (8.6–9.8x weighted, 4.8x on the weightless `moe.router.norm` branch) came from
the deleted `bench_norm_fullcanvas.py`, whose PCC column is discredited below; the timings stand.

## The numerical delta, and the fp32-accumulation fix

Measured on QB2 at the shipped shape `[1,1,256,2816]` bf16 by `tests/test_device_norm_fullcanvas.py`:

| arm | rows differing | elements | rel p99 | rel max | bf16 ULP max |
|---|---|---|---|---|---|
| weighted, `block_h=8` vs 8x `block_h=1` (same 88-core grid) | 61/256 | 19.43% | 1.14e-2 | **2.24e-2** | **5.73** |
| scaleless, 8-core/`block_w=11` vs 88-core/`block_w=1` | 79/256 | 24.80% | 1.06e-2 | 1.56e-2 | 4.00 |

> **OPEN CONTRADICTION (unexplained):** the per-norm delta is stated as **~2e-6 / PCC 0.999998**
> (`bench_norm_fullcanvas.py`, deleted 2026-07-30 — it computed PCC as an fp32 dot product over ~720K
> elements and reported values ABOVE 1.0 in its own table, 1.000015 and 1.000050, so its resolution
> floor is ~5e-5) and as **5.73 bf16 ULP / rel max 2.24e-2** (`tests/test_device_norm_fullcanvas.py`).
> Four orders of magnitude apart; not explained.

**ROOT CAUSE + FIX.** `ttnn.rms_norm` accepts a `compute_kernel_config` and nothing in DiffusionGemma or
gemma4 ever passed one, so every norm ran ttnn's default `fp32_acc = false`. Switching only the
accumulator (same grid, block_w, fidelity, approx mode) moves the two row counts from **13.0% of
elements disagreeing to 0 of 69,206,016 over 96 device slices** (rate < 1.4e-8), is **2.8x more accurate**
against an fp64 reference over the same bf16 inputs (rmse 5.43e-3 → 1.94e-3), and costs 0.088 → **0.086
ms** per 256-row norm (−2.3%, free).

> **TRAP:** fp32 accumulation is free *here* only because these configs land on `block_w=1` /
> `subblock_w=1`, where halving DST capacity has nothing to take away. "fp32 is slow" is right for wide
> output blocks and wrong for this shape.

> **OPEN CONTRADICTION (unexplained):** with fp32 accumulation wired into BOTH row counts a 10-question
> device pair still diverged completely (0/10 byte-identical, 91 vs 109 blocks), even though the norm is
> bit-identical to <1.4e-8, the shape census shows the model only ever calls it at 256x2816 with no
> fallback, and the flag had exactly one reader. Those three facts cannot all hold; not explained. The
> most likely scope gap is the input distribution — every rate measurement used benign gaussian inputs
> and real activations are heavy-tailed, where a sum of squares is outlier-dominated. The production
> patch is parked at `/home/zni/dg_runs/fp32_norm_production.patch`; measurements in
> `tests/test_device_norm_fullcanvas.py`.

**BUG FIXED 2026-07-30.** `_chunked_norm_forward` tested `with_scale is False` AFTER attempting the
full-canvas path, so the MoE router's weightless norm was silently re-sharded from 8 cores/`block_w=11`
to 88 cores/`block_w=1`. Correcting it does NOT make the two row counts bit-identical — the weighted
delta is the larger of the two.

## How the flip gate was overturned (2026-07-27 gate → shipped 2026-07-30)

**VOID-PREMISE RULE.** The gate ran with `DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1`, i.e. on the
token-gather denoise MoE deleted in `7417bd7d69d` for not converging, so `committed_match = 0.145` was
measured between two arms on a broken baseline and cannot be cited either way. The tell is that it
describes BOTH trajectories as "coherent-then-degenerate". **Generalizes: every decision-fidelity number
measured on the deleted token-gather path is void — the same mistake voided the pad-fix revert.**

Gate numbers as historical record only: committed clean-argmax match 0.145 against a >= 0.95 bar
(rejected-bfp8 reference 0.227), mean per-step Gumbel argmax agreement 0.544 (min 0.144), mean
accept/renoise IoU 0.504 (min 0.0) vs bfp8 0.501, mean per-step entropy PCC 0.659 (min 0.259) vs bfp8
0.631, mean sampled-canvas agreement 0.889 (min 0.770).

**Overturning evidence (latency only):** the 198-question run with the norm on scored **71.21%** at 0
empty replies and 0 responses over the 2% non-Latin threshold. It was previously written up here as
beating **66.67%** "for the previous full run on the same questions" — corrected 2026-07-30: **NOT BUDGET-MATCHED — this comparison does not support a quality claim.** The 71.21% run used `max_gen_toks=13824`; the 66.67% run used **5632**, a 2.45x smaller budget. On a thinking-mode reasoning task the budget decides whether the chain of thought finishes, so the +4.5 pp cannot be attributed to the norm. The full-canvas norm's effect on score is **unmeasured**. The -20.4%/block latency win is unaffected (per-block latency is a per-block measure, and the longer run's longer contexts bias it conservatively). The only budget-matched TT-vs-reference reading is on 11 drift-selected prompts at 5632 (TT pads-hidden 8/11 vs A100 7/11) -- an enriched subset, not a population estimate. There is no 198-question budget-matched comparison. The "27% shorter answers" objection was a 10-question artifact — −10% at 71, gone at 198. The
71-question GPQA prefix (2026-07-29) moved score 76.06% → 78.87%, guard kills 2 → 1, degenerate 2 → 1,
drift-any 3 → 4 (1 fixed / 2 new), and on the >2% non-English gating metric 0/71 → 1/71.

**REPRODUCTION** (the only recorded invocation; env: see [plan](../../plan.md)):
`doc/datatype_sweep/decision_agreement.py run --num-layers 30 --max-denoising-steps 16 --seed 0 --output
<path> --label <chunked|fullcanvas>`, then `... compare --ref <chunked> --cand <fullcanvas>`. The
original also pinned `DG_SPARSE_MOE`, `DG_SPARSE_MOE_TUNED` and `DG_NORM_FULLCANVAS`, all deleted, so as
written the two arms are now the same run. `norm_fullcanvas_flip_agreement.json` is committed in this
directory; the `traj_{chunked,fullcanvas}.pt` trajectories stay in the run scratchpad. The ~85%
committed-token divergence is the #48291 bf16 chaos-amplification class, not a full-canvas bug —
[decision fidelity](../decision_fidelity/README.md).

## Rejected activation-L1 levers (provenance: token-gather MoE; full entries in the [refuted list](../REFUTED.md))

| lever | verdict |
|---|---|
| HIGH-1/2 gather + down output L1 (`DG_MOE_L1`, no reader today) | **WASH** — isolated MoE fwd −3.2% (gather_matmul 0.101 → 0.043 ms) but traced e2e −0.6% @48 / +0.4% @12 at bit-identical `committed_sha`; those DRAM writes already overlap adjacent compute under trace |
| MED-5 gate/up L1 | **no-op by construction** — `batched_experts` (weight-bound at M=1 tile, 62% of the MoE) does not move, so the MoE is weight-traffic-bound |
| HIGH-3 residual-stream L1 | **coupled** — every consumer of the 256x2816 residual takes a DRAM-interleaved input, so pinning it alone only inserts a reshard per boundary; pays only as a whole-layer L1 stack (OPT-003 residual-contract rule) |
| MED-6 attention L1 | the `to_memory_config(..., DRAM)` force at `diffusion_attention.py:400-411` is a **guarded passthrough no-op**; a real L1-sharded SDPA is blocked by the flash-SDPA CB clash |
| MED-7 mask L1 | disp/comb/disp_t masks are ~2 MB and the ops sub-ms — no material headroom |
| layout conversion generally | ALL `InterleavedToSharded` + `ShardedToInterleaved` device-FW is **~1.34 ms over 6 layers** (~4 ms/step at 30L, <3% of the step) |

## Method, baselines, residual risk

**MEASUREMENT SUBSTITUTE.** With `ENABLE_TRACY=OFF` there is no `tt-perf-report` op CSV, so the approved
ranking metric is the traced Metal capture/replay path plus synchronized per-op device-time tables
(`time.perf_counter` + `ttnn.synchronize_device`); evidence class `hardware-profiler-limited`. The four
benches this pass named (`bench_moe_l1_residency.py`, `bench_norm_fullcanvas.py`, `bench_moe_l1_e2e.py`,
`bench_lever_e2e.py`) are all absent from `doc/optimize_perf/`; `bench_norm_fullcanvas.py` was deleted
2026-07-30 and replaced by `tests/test_device_norm_fullcanvas.py`. Committed mirror:
`l1_residency_summary.json`.

**Historical traced baselines** for reproducing the harness: @48 17.86–18.13 t/s / 14.12–14.34 s block /
`a9f0d18709b07d1e`; @12 49.8–53.2 t/s / 4.81–5.14 s block / `24393ba7aad6077c`; block latency varies
~1.5% run to run.

**WATCHER SCOPE.** Verified only on a short smoke (4 steps / 1 block, `TT_METAL_WATCHER_DISABLE_ETH=1`,
zero violation strings in `generated/watcher/watcher.log`); a full @48 multi-block soak was never run.

**GATE TRAP.** `git diff-tree fbabe620f21 -- models/demos/gemma4/` is EMPTY; the literal `git diff main
-- models/demos/gemma4/` reads non-empty only because local `main` is ~842 commits stale — the automated
no-shared-edits gate false-positives on a stale `main`.

## Absolute HF-fidelity replay (provenance — ran on the deleted token-gather MoE)

| comparison | committed clean-argmax | per-step argmax | accept IoU | entropy PCC | canvas agreement |
|---|---:|---:|---:|---:|---:|
| chunked vs HF | 0.168 (43/256) | 0.541 | 0.100 | 0.027 | 0.146 |
| full-canvas vs HF | 0.160 (41/256) | 0.555 | 0.109 | 0.042 | 0.148 |
| chunked vs full-canvas | 0.145 (37/256) | 0.544 | 0.504 | 0.659 | 0.889 |

**REPRODUCTION PIN** (env: see [plan](../../plan.md)): prompt `"Explain what a diffusion language model
is in one sentence."`, seed 0, one 256-token canvas, 30 layers, 16 denoise steps, early halt disabled,
checkpoint revision `0f28bc42f588fbd8f71e08102b1c3960298a1358`, CPU-only HF via `demo/replay_hf_tt.py`
with `reference/generate.py::{make_replay_canvas_init_fn,make_replay_noise_fn}`. In the clean-argmax
regime the torch replay must inject an all-zero Gumbel tensor (`sampled == argmax`) plus the gate's
random-renoise stream at `seed + 1000`, or the trajectories are not comparable. The full 30L/16-step HF
reference run cost 49.9 s trajectory / 54.5 s process wall at 41.7 GiB peak RSS — no reduced fallback
needed.

## Roofline context (owned elsewhere)

- Weight bytes, not incremental KV, set the denoise floor; all 128 experts are active at S=256 and the
  step re-reads 13.27 GiB/chip (~88.6% MoE experts) — [work log](work_log.md) (measured here: the
  all-128 bf16 weight floor is ~12.3 ms/step at 1024 GB/s peak against a measured ~0.23–0.27 s/step, so
  the step is op-efficiency-bound well above the bandwidth floor).
- ~235 GB/s practical per-chip denominator and the terminal argmax/entropy 18-bit-index bf16 wall:
  [non-MoE roofline](nonmoe_roofline/README.md). Why the token-gather MoE was deleted:
  [winter borrow](winter_borrow_20260727.md). The sum-of-device-FW overlap trap:
  [op profile](whole_gen_opprofile/README.md). Current denoise per-step cost:
  [optimize_perf hub](README.md).
