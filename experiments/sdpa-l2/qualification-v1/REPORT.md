# SDPA numerical qualification v1

**Neither candidate qualifies against the agreed numerical/structural contract.** The complete 660-input synthetic matrix was evaluated on one Blackhole P100A (yyzo-bh-26, reservation 215262): 1320 requested mode cases, with unsupported FP32 requests explicitly excluded rather than run on fallback. Performance and end-to-end model evaluation were not tested.

| Mode | Pass | Fail | Unsupported |
|---|---:|---:|---:|
| fast | 344 | 316 | 0 |
| accurate | 169 | 261 | 230 |

Counts are per input case/mode; every tested head must pass. Unsupported does not count as a pass. Long-context references cover 512 stratified query rows/head; Q<=2048 cases check every query row. Full device operations, not sampled-device operations, were executed. These are absolute acceptance failures, not a main-versus-candidate regression comparison.

## Results by test group

Cells are pass / fail / unsupported.

| Group | Fast | Accurate streaming |
|---|---:|---:|
| normal | 80 / 0 / 0 | 15 / 25 / 40 |
| stress | 24 / 276 / 0 | 72 / 228 / 0 |
| structural | 90 / 40 / 0 | 80 / 0 / 50 |
| boundary | 150 / 0 / 0 | 2 / 8 / 140 |

## Standard normal inputs

Maximum per-head L2 percent and minimum defined per-head PCC, across both head counts and all five seeds. These are worst cases, not means.

| Length | Fast L2 % / PCC | Accurate L2 % / PCC | Accurate pass / fail / unsupported |
|---|---:|---:|---:|
| 2,048 | 2.577202 / 0.99967067 | — / — | 0 / 0 / 10 |
| 8,192 | 2.633451 / 0.99966095 | — / — | 0 / 0 / 10 |
| 25,920 | 2.598265 / 0.99968473 | — / — | 0 / 0 / 10 |
| 32,768 | 2.666457 / 0.99967681 | 0.511406 / 0.99998689 | 2 / 8 / 0 |
| 65,536 | 2.790921 / 0.99967149 | 0.514010 / 0.99998680 | 5 / 5 / 0 |
| 75,600 | 2.796088 / 0.99967861 | — / — | 0 / 0 / 10 |
| 131,072 | 2.994674 / 0.99967029 | 0.519221 / 0.99998651 | 2 / 8 / 0 |
| 262,144 | 3.297717 / 0.99968798 | 0.503128 / 0.99998734 | 6 / 4 / 0 |

The previous approximately 0.49% aggregate FP32 result does not imply every head is below 0.5%. This qualification exposes that margin issue directly. The 25,920 and 75,600 FP32 results from earlier experiments were fallback measurements and are deliberately NOT reused here.

## Stress inputs

Each distribution is tested at 32K and 256K, H=5 and H=10, five seeds. Common-mode rows combine offsets -32,-8,+8,+32 applied to only that tensor.

| Distribution | Fast max head L2 % | Accurate max head L2 % | Fast pass / fail | Accurate pass / fail |
|---|---:|---:|---:|---:|
| scaled_low | 3.543969 | 0.416742 | 20 / 0 | 20 / 0 |
| scaled_qk | 6.032065 | 0.964279 | 0 / 20 | 0 / 20 |
| outliers | 6.265224 | 0.933626 | 4 / 16 | 0 / 20 |
| common_q | 48.361599 | 7.139098 | 0 / 80 | 0 / 80 |
| common_k | 18.526317 | 1.223285 | 0 / 80 | 0 / 80 |
| common_v | 1.031790 | 0.113696 | 0 / 80 | 52 / 28 |

| Distribution | Fast min PCC / worst row L2 % | Accurate min PCC / worst row L2 % |
|---|---:|---:|
| scaled_low | 0.99970565 / 4.516924 | 0.99999108 / 0.519725 |
| scaled_qk | 0.99821044 / 22.001601 | 0.99995383 / 3.829646 |
| outliers | 0.99806220 / 42.426764 | 0.99995643 / 6.887660 |
| common_q | 0.87897484 / 138.429415 | 0.99749528 / 35.229881 |
| common_k | 0.98311923 / 92.036341 | 0.99992593 / 4.144508 |
| common_v | 0.00583899 / 2.346918 | 0.01946564 / 0.155782 |

Row denominators use the agreed 1%-of-head-RMS floor. Raw unfloored row summaries, maximum absolute error, and raw maximum elementwise relative error are also in the JSONL. Maxima apply to tested rows, not every long-context output.

## Structural checks

| Input | Fast pass / fail / unsupported | Accurate pass / fail / unsupported |
|---|---:|---:|
| zero_v | 30 / 0 / 0 | 20 / 0 / 10 |
| constant_v | 0 / 30 / 0 | 20 / 0 / 10 |
| uniform | 30 / 0 / 0 | 20 / 0 / 10 |
| cancellation | 30 / 0 / 0 | 20 / 0 / 10 |
| single_key | 0 / 10 / 0 | 0 / 0 / 10 |

fast: maximum structural error where measured = 3 BF16 ULP.

accurate: maximum structural error where measured = 0 BF16 ULP.

## Common-V rounding-floor audit

The original proposal exempts undefined PCC for constant outputs. The online scorer initially missed the constant-actual exception; adjudicate.py applies it to the preserved raw metrics. It does not change thresholds, waive defined PCC failures, or alter any other gate.

- fast: ideal FP64-reference-rounded-to-BF16 oracle: 49 / 31 / 0; device output exactly matches that oracle in 0/80 cases; 80/80 cases fail the common-V residual budget.
- accurate: ideal FP64-reference-rounded-to-BF16 oracle: 49 / 31 / 0; device output exactly matches that oracle in 56/80 cases; 0/80 cases fail the common-V residual budget.

Defined PCC still rejects some ideally rounded results: the residual signal can be poorly resolved in BF16 even when ordinary L2 is at its rounding floor. These remain formal failures under the current contract, flagged separately from additional operator error. A floor-aware PCC rule needs agreement before treating this as a release contract.

Seven FP32 common-V cases exactly match the rounded oracle yet formally fail defined PCC. Conversely, three device cases pass through the constant-output/undefined-PCC exception while the ideally rounded reference has defined PCC and fails. This is a weakness in the current criterion, not evidence that those device outputs are more accurate than the ideal reference. No defined PCC failure was waived in the counts above.

## Coverage and implementation provenance

- Fast uses compensated BF16 streaming, untouched Q, Q/K chunks 128/512, and unchanged input buffering. A qualification-only host patch enables compensation below the default 64-K-chunk cutoff.
- Accurate uses the retained improved FP32 streaming algorithm and its six-bit Q preprocessing/1.0027 scale compensation. K chunks are 1024, or 512 where that permits streaming (33,280). No FP32 fallback was executed.
- The three negative FP32 probes verify host rejection for short/padded inputs. Explicit-mask structural requests are unsupported by both specializations; their guard probes are recorded separately.
- Numerical kernels were not tuned or changed. The qualification-only host patch, source hashes, build logs, input hashes, and output hashes are retained.
- One harness-only interruption came from assuming centered K was always exactly BF16-representable. The runner was corrected to account for requantization, and the affected case was retried. Historical records remain in raw results; this was not a hardware/kernel failure.
- **Not qualified:** Galaxy hardware, two model-family Q/K/V capture sets, and an additional release holdout suite. Captures were requested but not available; no Blackhole Galaxy cards were available at discovery. The available Wormhole Galaxies cannot run these Blackhole-only specializations unchanged.
- Causal/GQA/D!=128/multibatch behavior is outside the proposed v1 scope. Performance and model-score evaluation were explicitly excluded.

## Artifacts

- [Frozen contract and matrix](SPEC.md), [reproduction procedure](REPRODUCE.md).
- [Accepted scores](accepted-results.jsonl), [raw measurements and initial scores](results.jsonl).
- [Accepted rounding oracle](accepted-oracle.jsonl), [raw oracle](rounding-oracle.jsonl).
- [Qualification host patch](qualification-host.patch), [source hashes](SOURCE-SHA256.txt).
- [Summary](summary.json), [completeness/provenance audit](audit.json).

## Verification and final build state

- All **1320 requested mode cases** are present: **1090 executed, 230 unsupported**, with no missing cases or unresolved execution errors. The one historical harness error is preserved in [audit-raw.json](audit-raw.json), and its case was successfully retried.
- **218 full-output trace equality checks** and **100 full-output physical-padding invariance checks** passed. All 430 supported mode pairs use matching original input hashes and reference positions.
- Three short/padded FP32 guard probes and six explicit-mask guard probes rejected requests without executing fallback. The explicit-mask probes are additional feature-support checks, not part of the 1320-case totals: see [guard-probes.log](guard-probes.log) and [mask-guard-probes.log](mask-guard-probes.log).
- The FP64-reference/scoring self-tests, Python syntax checks, Black checks on all nine active Python scripts, patch applicability check, and `git diff --check` passed. The qualification host build and all device JITs completed successfully.
- After qualification, the retained default factory was restored and forcibly rebuilt with `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install`. The build log confirms recompilation of the transformer host object. All four operator-source hashes match their pre-qualification values; see [RESTORED-SHA256.txt](RESTORED-SHA256.txt).
- Both modes were rerun at N=32768,H=5,D=128,seed=1234. Full-output hashes, sampled-output hashes, input hashes, and all per-head metrics exactly match their qualification results. See [restored-smoke.jsonl](restored-smoke.jsonl) and [restored-checks.json](restored-checks.json). This verifies restoration, not a numerical qualification pass: the known FP32 per-head L2 miss remains unchanged.

The allocated machine is left on the retained improved build with its original dispatch guards, not on the qualification-only host override. No numerical kernel changes or commits were made for this qualification.
