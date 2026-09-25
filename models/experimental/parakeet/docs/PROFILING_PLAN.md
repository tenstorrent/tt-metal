<!-- SPDX-FileCopyrightText: © 2026 Abror Shopulatov -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Profiling plan (bringup PASSED 2026-09-25; execute one hypothesis per work unit)

Goal: rank synchronized time in (a) subsampling convolutions, (b) attention, (c) TDT decode
round-trips, then test optimization hypotheses against matched same-precision measurements.
Protocol for every number: `mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1`,
warm sync-bounded medians, bringup-suite cases mounted via public_input_stage=bringup (/input).

## Measured anchors (this porting, current source source hash...)

CPU FP32 baselines (`baseline/cpu_fp32_*.json`) vs TT bf16-effective bringup PASS
(a measurement, median case p50 0.0852 s, tokens exact 8/8):

| case | CPU FP32 transcribe med (s) | TT p50 (s) | ratio | TT nrmse |
| --- | --- | --- | --- | --- |
| short | 0.234 | 0.0580 | 4.0x | 0.0181 |
| silence | 0.218 | 0.0557 | 3.9x | 0.0233 |
| tone | 0.248 | 0.0606 | 4.1x | 0.0331 |
| single0 | 0.226 | 0.0825 | 2.7x | 0.0074 |
| single1 | 0.330 | 0.0879 | 3.8x | 0.0061 |
| batch | 0.424 | 0.1084 | 3.9x | 0.0067 |
| reverse | 0.424 | 0.1060 | 4.0x | 0.0067 |
| long | 0.910 | 0.2540 | 3.6x | 0.0115 |

Recorded context (not a target): CUDA A100 FP32 bringup median case p50 0.0820 s
(`gpu-precision-parakeet-5929b627da7f`); its bf16 run failed long/tone.

## Matched-comparison method (applies to every hypothesis)

- Absolute reference: CPU FP32 per-case medians in `baseline/cpu_fp32_smoke.json` /
  `baseline/cpu_fp32_bringup.json` (same case, same suite, same protocol, same host).
- Optimization deltas: same TT build precision declaration (bf16 weights / fp32 acts+acc),
  before/after on the same cases; never compare across different precisions or suites.
- Tool: `benchmarks/profile_transcribe.py --input /input --cases <cases> --precision bf16`
  (wraps encode_device/_decoder_step/_joint/_masked with device syncs; adds upload/readback/
  idle-sync round-trip microbench). Component checks: `tests/diag_layers.py`,
  `tests/diag_sub_split.py` (`--conv-share` mode for H2), `tests/diag_decode_ab.py`
  (smallest relevant components).
- Identity: measured on unchanged source (backend.py sha256 8b0d33e9..., re-verified in-job;
  profile script sha256 fad1846c50b94329287395ea76149e19c2a3461c619a6e9875217819aec7c45f;
  conv-share harness tests/diag_sub_split.py cce61835f64c18b4ac44c6d318e4469f1b3966151b4d01f44efffd501c87bf12).

## Hypotheses

- **H1 — TDT decode round-trips dominate end-to-end latency.**
  Baseline artifact: `baseline/cpu_fp32_smoke.json` `short` transcribe_median_s 0.2382 vs
  encode median 0.1786 (CPU decode overhead ~0.060 s); `baseline/cpu_fp32_bringup.json`
  per-case `transcribe_median_s` minus `encode_s` median for longer cases (e.g. `long` 0.910);
  TT anchor: bringup per-case p50s above.
  Method: TT per-step cost = (transcribe p50 − encode p50)/(steps+1) from the instrumented run;
  compare with the same quotient from `baseline/cpu_fp32_*.json`; compare against the
  upload/readback/idle-sync round-trip floor. Accept H1 if decode share > 50% of transcribe p50.
- **H2 — Subsampling conv stack dominates encoder time.**
  Baseline artifact: `baseline/cpu_fp32_bringup.json` per-case `encode_s` arrays
  (e.g. silence median ~0.169 s, tone ~0.193 s).
  Method: sync-bounded timing of conv-only prefix via `tests/diag_sub_split.py`; share of
  `encode_device` p50. Accept if conv prefix > 40% of encoder time.
- **H3 — Attention/rel-pos cost scales superlinearly with input length on device.**
  Baseline artifact: CPU encode scaling long-vs-short from `baseline/cpu_fp32_*.json`
  (bringup `long` vs smoke `short` encode medians).
  Method: TT `encode_device` p50 for `long` vs `short` (same run); compare the ratio to the
  CPU ratio; investigate layout only if TT ratio exceeds CPU ratio by >1.5x.
- **H4 — Per-step host round-trip floor bounds decode throughput.**
  Baseline artifact: round-trip microbench in `benchmarks/profile_transcribe.py` output
  (upload/readback/idle-sync), interpreted against H1 step counts.
  Accept if measured floor x syncs-per-step explains >= 80% of TT per-step cost.

## Rules

- One hypothesis per work unit; record measured artifact + verdict (accepted/rejected) below.
- No full corpus during development; short focused device commands only.
- Never rerun a failed experiment without a new hypothesis or changed evidence.

## Results (fill after runs)

| Hypothesis | Artifact (job id) | Verdict | Notes |
| --- | --- | --- | --- |
| H1 decode round-trips dominate | f10e82f4f7264a2a9f1a8645f349dcef (92.1s, rc 0, /input bringup, bf16-eff, unchanged source) | REJECTED as stated (accepted only for long outputs) | decode share: long 63.7%, batch 42.9%, short 40.1% — >50% only on long; encoder still dominates short/batch |
| H2 subsampling conv stack dominates encoder | 67056e998f324de1a1c045193c02bc20 (60.4s, rc 0, /input bringup, bf16-eff, unchanged source 8b0d33e9 verified in-job; harness cce61835) | REJECTED | conv-prefix share of encode_device p50: short 28.0% (11.42/40.85 ms), batch 22.8% (13.91/61.08 ms); conformer blocks hold ~72-77% |
| H3 attention/rel-pos superlinear length scaling | c202eca71a224033bffcd6d9fba34beb (72.8s, rc 0, /input bringup, bf16, repeats=10, unchanged source 8b0d33e9 verified in-log) | REJECTED | TT encode long/short p50 ratio 2.55x (91.6/35.9 ms) vs CPU 3.19x (0.5701/0.1786 s); pre-registered layout trigger 1.5x CPU = 4.79x not approached (TT/CPU = 0.80x) |
| H4 host round-trip floor bounds decode throughput | f10e82f4f7264a2a9f1a8645f349dcef (H1 capture) + c202eca71a224033bffcd6d9fba34beb (H3 side capture); docs-only closure, no dedicated device run | REJECTED | floor 87.7 us/step (42.8+29.0+15.9) = 5.0-7.3% of per-step (H1); 140.3 us/step (82.4+38.9+19.0) = 8.1-13.5% (H3); both far below the >= 80% bar; no sync-batching candidate |

### H1 detail (2026-09-25, `benchmarks/profile_transcribe.py --input /input --cases long,short,batch --precision bf16`, warm sync-bounded medians, repeats=5)

TT (p50): long T'=205 steps=89 transcribe 0.2472 s, encode 0.0899 s, decode 0.1574 s,
1.77 ms/step; short T'=29 steps=19 transcribe 0.0566 s, encode 0.0339 s, decode 0.0227 s,
1.20 ms/step; batch T'=55 steps=26 transcribe 0.1049 s, encode 0.0599 s, decode 0.0450 s,
1.73 ms/step. Instrumented: long encode 92.0 ms, _decoder_step n=82 avg 1.71 ms,
_joint n=89 avg 0.63 ms, host_other 12 ms; short encode 36.8 ms, _decoder_step n=10 avg
2.32 ms, _joint n=19 avg 0.86 ms; batch encode 61.7 ms, _decoder_step n=25 avg 2.48 ms,
_joint n=26 avg 0.96 ms, _masked n=10 avg 0.30 ms. Round-trip floor: upload_ids 42.8 us +
readback_argmax 29.0 us + sync_idle 15.9 us ~= 87.7 us/step = 5.0–7.3% of per-step cost.

CPU FP32 (`cpu_fp32_bringup.json`; short also vs `cpu_fp32_smoke.json`): decode long
0.3396 s (3.82 ms/step, share 37.3%), short 0.0600 s (3.16 ms/step, share 25.7%; smoke
0.0596 s, 3.14 ms/step, share 25.0%), batch 0.0786 s (3.02 ms/step, share 18.5%).

TT-vs-CPU FP32 quotients (CPU/TT): transcribe long 3.68x, short 4.13x (smoke 4.21x),
batch 4.04x; encode long 6.34x, short 5.12x, batch 5.77x; decode-loop long 2.16x,
short 2.64x, batch 1.75x.

Reading: TT encode is disproportionately faster than TT decode (5.1–6.3x vs 1.7–2.6x
CPU FP32), so the decode share roughly doubles vs CPU (25.7→40.1% short, 37.3→63.7% long)
and crosses 50% only on the longest outputs. The ~88 us host round-trip floor explains
<8% of the 1.2–1.8 ms per-step cost, so per-step cost is per-step device work/dispatch
inside `_decoder_step`+`_joint`, not host upload/readback.

Resulting H2 (next unit, unchanged method): encoder dominates short/batch end-to-end
(encode share 59.9% short, 57.1% batch), so keep H2 as defined — measure the subsampling
conv-only prefix share of `encode_device` p50 via `tests/diag_sub_split.py` on short and
batch (accept if > 40%); secondary read-out: whether the remaining non-conv encoder time
concentrates in attention/rel-pos (feeds H3's long-vs-short ratio check).

### H2 detail (2026-09-25, `tests/diag_sub_split.py --conv-share --cases short,batch --repeats 10`, warm sync-bounded medians, default traced encode path)

| case | frames | T' | tp | encode_device p50 (ms) | conv-prefix p50 (ms) | conv share |
| --- | --- | --- | --- | --- | --- | --- |
| short | 225 | 29 | 32 | 40.85 | 11.42 | 28.0% |
| batch | 440 | 55 | 64 | 61.08 | 13.91 | 22.8% |

Conv prefix = `Backend._subsample` + residual cast, exactly `encode_device`'s prologue on
unchanged source (untraced in production; the encoder trace covers only the conformer
blocks), timed with the same sync-before/sync-after median protocol as
`benchmarks/profile_transcribe.py`. Identity in-job: backend.py 8b0d33e9...,
tt/ttnn_parakeet.py fdfba2a2..., harness tests/diag_sub_split.py cce61835... (harness-only
addition; no measured-source edits during the unit).

Sensitivity: H1 (repeats=5, different run) measured encode p50 short 33.9 ms / batch
59.9 ms; with that denominator the short share would be 33.7% (batch 23.2%) — still below
the 40% bar on both, so the REJECTED verdict does not hinge on run-to-run encode variance
(batch denominators agree within 2%).

Reading: the subsampling conv stack is a minority of encoder time; ~72–77% of
`encode_device` is the 22 conformer blocks (attention + FFN + conv modules). Conv-prefix
cost grows only ~22% from short to batch (11.42 → 13.91 ms) while frames roughly double,
so the blocks carry the length scaling — H3's long-vs-short encode ratio vs the same CPU
quotient is the right next probe. Known caveat: allocating `_subsample` buffers while an
encoder trace is live emits a ttnn allocator warning (harness-induced; no trace execution
was interleaved with those allocations; job rc 0, numbers sane).

### H3 detail (2026-09-25, `benchmarks/profile_transcribe.py --input /input --cases long,short --precision bf16 --repeats 10`, warm sync-bounded medians, unchanged source)

| path | T' | steps | transcribe p50 (s) | encode p50 (s) | encode instrumented (ms) |
| --- | --- | --- | --- | --- | --- |
| TT long | 205 | 89 | 0.2462 | 0.0916 | 91.75 |
| TT short | 29 | 19 | 0.0558 | 0.0359 | 36.69 |

- TT encode long/short ratio: 2.55x by p50 (0.0916/0.0359), 2.50x instrumented
  (91.75/36.69).
- CPU FP32 pre-registered quotient (per Hypotheses): bringup `long` encode median
  0.5701 s / smoke `short` encode median 0.1786 s = 3.19x.
- Pre-registered rule: investigate layout only if TT ratio > 1.5x CPU ratio = 4.79x.
  Measured 2.55x is 0.80x of the CPU ratio — the trigger is not approached, so no layout
  investigation is warranted.
- Run-to-run sensitivity: H1 (repeats=5, separate run) measured encode long 89.9 ms /
  short 33.9 ms = 2.65x; agrees with H3's 2.55x within ~4%, so the verdict is robust to
  encode p50 variance.
- Reading: both CPU FP32 and TT encode scale sublinearly against the T' ratio
  (205/29 = 7.07x) — per-invocation fixed overhead amortizes with length and there is no
  evidence of superlinear attention/rel-pos growth on device; TT length scaling is more
  favorable than CPU FP32 (encode CPU/TT quotients: long 6.22x, short 4.98x). H3
  REJECTED; conditional optimization-candidate naming not triggered; no source edits.
- Side capture for H4: this run's round-trip floor upload_ids 82.4 us + readback_argmax
  38.9 us + sync_idle 19.0 us = 140.3 us/step = 8.1% of long per-step (1.74 ms) and
  13.5% of short per-step (1.04 ms).

### H4 detail (2026-09-25, docs-only closure; data already captured in H1 and H3 jobs; no dedicated device run)

- Pre-registered rule (Hypotheses above): accept only if measured round-trip floor x
  syncs-per-step explains >= 80% of TT per-step cost.
- H1 capture (`--cases long,short,batch`, repeats=5): upload_ids 42.8 us + readback_argmax
  29.0 us + sync_idle 15.9 us = 87.7 us/step against per-step costs long 1.77 ms,
  batch 1.73 ms, short 1.20 ms → the 3-component sync floor explains 5.0% (long),
  5.1% (batch) and 7.3% (short).
- H3 capture (`--cases long,short`, repeats=10): upload_ids 82.4 us + readback_argmax
  38.9 us + sync_idle 19.0 us = 140.3 us/step = 8.1% of long per-step (1.74 ms) and
  13.5% of short per-step (1.04 ms).
- The two floors differ by ~60 us (~1.6x microbench run-to-run variance), yet even the
  higher floor is ~6x below the 80% acceptance bar on its worst case (13.5%). Per-step
  cost remains dominated by device work/dispatch inside `_decoder_step` + `_joint`
  (H1 instrumented 1.71–2.48 ms/step). A hypothetical sync-batching optimization could
  recover at most the floor itself (~0.09–0.14 ms/step, < 13.5% even at the high-floor
  capture) — not an actionable candidate under the matched-measurement rules.
- Verdict: REJECTED; no sync-batching candidate; no source edits.

**Closure (2026-09-25): H1–H4 are all closed with zero ACCEPTED optimization hypotheses —
a valid measured outcome, not a gap.** The measured picture: encoder conformer blocks
dominate short/batch end-to-end (~57–60% of transcribe), TDT decode device work exceeds
50% only on the longest outputs, the subsampling conv prefix is a 23–28% minority of
encode time, TT length scaling is sub-CPU-linear (2.55x vs 3.19x, layout trigger not
approached), and the 3-component host sync floor explains < 14% of per-step decode cost.
 no fixed speedup or gratuitous edits are required; no optimization edits
follow from profiling. Recorded limits and remaining caveats live in this file (H1–H4
details), `docs/README.md` and `docs/PROFILING_PLAN.md`. Next: package hardening, then the
submit decision.
