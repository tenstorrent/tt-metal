# Stage Review

Verdict: clean-pass

Independent review of Stage 2, fused decoder for `Qwen/Qwen3.8-27B`, on the
live `mvasiljevic/qwen38-full-bringup` worktree, base `41bdb276313`,
2026-09-11. Reviewed final implementation SHA256:
`3f324562925d42fb0c2997672f86fe68926a21869d3f2d351d1f5c3ba9792035`.
The final full-context run completed during review, with clean device teardown
at 21:19:57 UTC. No reviewer hardware execution was performed.

## Required Work

None. All four items from `stage_review_1.md` are resolved: final-source
latencies are reproduced, uneven batch grouping is tested and its discovered
accuracy failure repaired, untilize fusion has an adapted rejection with a
passing control, and the handoff documents identify current evidence.

The stage owner may perform the required stage-owned local checkpoint commit
and record its SHA. That post-review action is not represented as already done.

## Other Concerns

- The corrected decoder is not universally equivalent at PCC >=0.995 to the
  erroneous frozen functional output. The added B3/S257 changed-token case
  has corrected/frozen PCC **0.99181628**. I independently loaded the saved
  tensors and verified identical original inputs, changed inputs, and stock-HF
  oracle tensors across all three controls. Corrected/stock-HF PCC is
  **0.99988180**, versus **0.99362123** for the pre-fix fused path and
  **0.99300992** for the frozen functional path; RMSE falls from 0.080175
  to 0.009825. This is an evidenced accuracy repair under the supplied
  contract's requirement to explain material deltas, not an oracle waiver or
  evidence of regression. The source, README and AutoFix report retain this
  distinction. Subsequent stages must retain the corrected normalization.
- The packed and split-projection alternatives are effectively tied in the
  recorded corrected-policy comparisons. Three paired trials give packed
  medians 2.493290, 2.493872, 2.491243 ms and split medians 2.489839,
  2.492960, 2.493221 ms. Pooled medians over 90 samples each are
  **2.492811/2.492334 ms**. The winner reverses; a 0.019% pooled difference
  does not demonstrate a consistent advantage that requires switching the
  default. Retaining packed is supported by these measurements, without
  claiming it decisively outperforms split. Op count alone was not used.
- This is decoder-layer evidence. Replays restore prefix state between
  samples; continuation tests separately check accumulated prefix behavior.
  The results do not establish continuous full-model generation, serving,
  multi-device operation, or accuracy across an entire layer stack.

Independently checked final evidence:

| Requirement | Result |
|---|---|
| Independent runtime | `FusedDecoder` directly implements both attention kinds. No functional runtime dispatch or host compute fallback was found. The delivered regression tests patch functional construction to fail. |
| Public boundaries and grouping | Three final synthetic tests pass, with zero failures/errors/skips: linear B2/B3 and full B2, each at 1,2,3,31,32,33,127,128,129,257,31. Continuation, changed inputs, deterministic replay and full-attention page-table refresh pass. Native scan grouping of 2+1 is exercised by B3. |
| Real uneven batch | `corrected_batch3_linear.json` contains all seven requested lengths, ending at 257, with stock-HF changed-token minimum 0.99988180. No seed or threshold was changed to pass. |
| Representative equivalence | Current B1/S128 fused/functional PCC: linear prefill/decode 0.99846750/0.99994248; full 0.99996179/0.99997485. All exceed 0.995. The exceptional changed-token delta is documented above. |
| Context retained | Both current-source context JSONs contain 4097,262143,262144,31, matching their console rows and source/baseline hashes. Every valid next-token replay passes. Maximum public prefill remains 262144; the runner does not attempt an extra out-of-contract token after that prefill. |
| Long-context accuracy | Worst conservative prefill HF bounds are linear 0.99615411 and full 0.99601026, both at 262143. Exact-limit prefill bounds are 0.99623798/0.99606463. Final-valid-context decode bounds are 0.99874594/0.99743386. All exceed 0.995. |
| Watcher | Four current-source real-weight B1/B32, S257 continuation runs pass. The actual watcher logs match their hashes and have no fatal/exception/invalid/overflow/out-of-bounds/error/corruption/sanitizer matches. |
| Runtime policy | Final raw profiler rows show BF16 projection inputs/weights/output, HiFi4, exact math and FP32 destination accumulation. SiLU appears in the matmul program's fused-activation attribute. No selected final matmul is classified SLOW. |
| Trace and movement | All eight measured profiler windows contain only device operations on device 3; all decode rows carry trace ID 0. Forward/capture guards reject Torch operations and host conversions. Remaining conversion/layout operations have explicit native consumers and assessed alternatives. |
| Portable evidence | All 37 archived candidate hashes verify. Both log archives pass CRC and byte comparison against local originals; all 125 log index entries verify. All 261 entries in the final pre-review artifact index verify. Raw op CSV gzip contents match their originals. |

Recomputed warmed wall medians from the actual five prefill and 30 decode
samples in each benchmark JSON:

| Kind | Prefill before / after, ms | Traced decode before / after, ms |
|---|---:|---:|
| Linear | 4.848313 / 3.074463 | 3.350835 / 2.494085 |
| Full | 3.554313 / 2.754356 | 2.443808 / 2.260432 |

The selected implementation is faster than the functional baseline in both
modes for both kinds. Timed intervals exclude state restoration and readback;
each decode sample is checked bitwise. Final benchmark sidecars match the
reviewed source. Historical faster linear measurements with the failing norm
policy are clearly distinguished from valid final candidates.

Raw signpost windows, report CSVs and `performance_final.json` agree:

| Kind / mode | Device operations before / after | Kernel sums before / after, microseconds |
|---|---:|---:|
| Linear prefill | 88 / 32 | 4600.646 / 2865.165 |
| Linear decode | 100 / 38 | 3011.400 / 2466.776 |
| Full prefill | 52 / 29 | 3400.790 / 2590.584 |
| Full decode | 57 / 32 | 2375.970 / 2223.064 |

These kernel sums exclude gaps and are not substituted for wall latency.

## Hard-Check Gaps

- The long-context HF values are conservative correlation-angle bounds using
  frozen functional-stage HF results, not freshly executed HF runs. I checked
  the bound calculation, unchanged functional source, baseline artifact hashes,
  seeded input/page construction, complete public requests and every final
  result row. This is valid evidence for the supplied fusion-equivalence
  contract. A smaller bound after the accuracy correction does not itself
  establish worse actual HF accuracy: the old functional result is the common
  reference in that conservative construction.
- PCC is aggregate over logical output elements. The B3 diagnostic additionally
  records per-user PCC and absolute errors. Existing evidence covers relevant
  changed branches and tails; an additional format, profiler run or long soak
  is not required merely to strengthen presentation.
- The final page sentinel assertion explicitly checks unowned pages after
  prefill. It is not a separate post-decode sentinel assertion. Direct output
  checks, per-user changed positions, page-table rerouting, disjoint cache-update
  core sets and clean watcher runs reveal no concrete corruption concern.
- Reviewer validation was source/artifact inspection and small CPU analysis
  only. Python files parse successfully with the AST parser. No reviewer
  TTNN import, device open/reset, target test, profiler, server, dependency
  installation or native build occurred. The scope is Python/docs only; no
  C++/CMake build was necessary. Stage-owner formatting/check claims are
  recorded separately in the work log.

## Anomaly Ledger

- Observed anomaly: B3 changed-token PCC failed despite passing original-token
  replay and most boundary cases.
  Evidence: `uneven_batch3_linear.log`, functional control log, `AUTODEBUG.md`,
  `AUTOFIX.md`, and saved `probe_batch3_*.json/.pt` controls.
  Affected path: linear common RMSNorm and the prefix/changed-token arithmetic.
  Control or comparison: frozen functional also fails; identical stock HF
  tensors were verified across runs. Eager and traced output/state are bitwise
  equal, and user 0 rather than the leftover user dominates the error. HF cache
  rounding changes the reference output by only PCC 0.99999750.
  Likely subsystem: inherited RMSNorm compute precision.
  Investigation performed: audited the unmodified primary oracle and diagnostic
  state swaps; recomputed saved-tensor PCC/RMSE; checked isolated norm controls,
  final-source probes and the repeated original real B3 sweep. FP32 folded
  weights alone fail to fix it; explicit existing compute config fixes it.
  Resolution: fixed in fused linear attention; the frozen-functional mismatch
  remains explicitly controlled and is not called universal equivalence.

- Observed anomaly: applying the norm correction to full attention increased
  decode latency without a demonstrated accuracy failure requiring it.
  Evidence: `fixed_full_benchmark.json`, `corrected_full_benchmark.json` and
  the final conditional `_norm` implementation.
  Affected path: full-attention common normalization.
  Control or comparison: blanket-config decode median 2.296987 ms versus
  retained native-default final 2.260432 ms; full accuracy/context gates pass.
  Likely subsystem: native norm configuration cost.
  Investigation performed: inspected the scoped change and measured policy
  comparison, then checked current full-context and profiler evidence.
  Resolution: fixed by restricting the accuracy repair to linear attention.

- Observed anomaly: matmul-untilize candidates produced the wrong layout or
  failed decode PCC after executable adaptations.
  Evidence: untilize projection/matmul/layout logs, explicit-output N3/N4 logs,
  archived sources and `split_projection_control_linear.json`.
  Affected path: row-major QKV production for native causal convolution.
  Control or comparison: adapted untilize forms fail at PCC 0.856359,
  0.848129 or 0.815826; the same split with tiled output followed by explicit
  conversion passes. The archived explicit-output/control diff isolates
  untilize/output allocation rather than a different recurrence algorithm.
  Likely subsystem: native untilize-output contract/implementation for this
  geometry and precision policy; no broader native root cause is asserted.
  Investigation performed: checked the adaptations, native entry points,
  failed logs, passing control and corrected-policy paired measurements.
  Resolution: controlled rejection; retaining the explicit conversion is earned.

- Observed anomaly: earlier norm/head-concat/convolution candidates had API,
  sharding or padded-batch failures.
  Evidence: historical candidate logs/snapshots and successful adapted JSONs.
  Affected path: dedicated convolution, full Q/K norm and decode head flatten.
  Control or comparison: convolution config namespace corrected; B1 norm uses
  supported block sharding and larger batches use interleaved norm; adapted
  decode concat passes but is slower than direct flatten.
  Likely subsystem: native API and layout constraints.
  Investigation performed: checked the selected branches, adapted candidates,
  final B1/B2/B32 coverage and native scan/head-output contracts.
  Resolution: fixed or controlled; first API failures were not used as final
  rejection evidence. No unassessed applicable graph-fusing pattern was found.

- Observed anomaly: historical files named final/delivered and earlier context
  bounds initially disagreed with the current source/report.
  Evidence: source sidecars, final benchmark/context JSON and the documentation
  updates made during review.
  Affected path: evidence provenance and handoff accuracy.
  Control or comparison: corrected/verified/watcher_final/tracy_final records
  now match current source; old sections/candidate policies are labeled;
  context manifest points at current verified artifacts.
  Likely subsystem: documentation finalization.
  Investigation performed: recomputed medians, profiler totals, bounds and
  hashes, reported stale details, and reread corrected documentation.
  Resolution: fixed.

- Observed anomaly: host/device-discovery and profiler warnings occur in
  successful logs, and closely timed candidates change ranking.
  Evidence: unknown motherboard bus-ID fallback, subset-MMIO advisory,
  historical clock-settling warnings, unused-variable JIT warnings, optional
  viewer-copy failure and mixed-column CSV parser warnings.
  Affected path: platform discovery/tooling and measurement variability.
  Control or comparison: device 3 is consistently profiled; all expected
  signposts/rows and timings are present; complete raw artifacts and clean
  watcher runs remain available; paired samples are retained without selecting
  favorable individual minima.
  Likely subsystem: environment/tooling, not demonstrated model corruption.
  Investigation performed: classified warning signatures, checked raw/filtered
  totals, actual trace IDs and dtype attributes, log hashes and clean teardown.
  Resolution: controlled within the recorded single-device environment.

## Scope Inspected

- Goal/skill paths: supplied Stage 2 contract and AGENTS instructions;
  `.agents/skills/{stage-review,graph-fusing,tt-device-usage}/SKILL.md`;
  installed `tt-model-bringup/0.1.4/skills/stage-review/SKILL.md`, including
  affected-branch and awkward-tail coverage requirements.
- Artifact paths: fused README, work log, patterns, first review, AutoDebug and
  AutoFix reports, corrected benchmarks/paired candidates, original failure and
  control logs, saved probe tensors, synthetic JUnit/JSON, watcher logs/summaries,
  both verified context JSON/logs, all final raw/profile CSV windows and tables,
  environment and context manifests, candidate/log archives and hash indexes.
- Code paths: complete `tt/fused_decoder.py`; fused runner, regression tests,
  context runner, B3 probe, untilize candidate and three stage shell runners;
  functional baseline interface and input generation; archived candidate diffs;
  native delta-rule adapter/head-major outputs, gated norm contract and matmul
  untilize/activation entry points.
- Commands run: read-only `cat`, `sed`, `rg`, `tail`, `git status/diff`;
  Python standard-library JSON/CSV/XML/AST/hash/ZIP/statistics analysis; CPU-only
  Torch loading and correlation/error comparison of saved diagnostic tensors.
  The sole reviewer mutation was this report. No further agent was spawned.

## Residual Risk

The evidence covers the pinned environment and representative single-device
linear/full decoder layers. Maximum context is tested at B1; B32 is tested at
257 tokens, so not every batch/context combination is demonstrated. Global
optimality across new kernels, all matmul geometry/precision choices or future
runtime versions is not claimed. Within this stage's existing-native graph
fusion scope, the selected path is supported by correctness, measured latency,
adapted alternatives, current profiler evidence and an explicit anomaly record.
