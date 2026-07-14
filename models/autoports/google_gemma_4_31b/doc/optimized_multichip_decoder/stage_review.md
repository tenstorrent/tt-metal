# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Faster real-weight candidates that pass the declared PCC gate were rejected under an undocumented stricter rule.
  Evidence: `tests/test_multichip_decoder.py:39` sets the acceptance bar to `PCC_THRESHOLD = 0.995`. The real-weight, full-trace QKV block-7 candidate passed at PCC 0.999841 and 0.472533 ms, versus selected block-3 at PCC 0.999962 and 0.477638 ms (`candidates/qkv_geometry.log:76-88`). Attention BFP4 likewise measured PCC 0.997274 and 0.472763 ms (`candidates/precision_fidelity_retry1.log:60`), versus the same sweep's BFP8 baseline at 0.479273 ms (`candidates/precision_fidelity.log:60`). Persistent BFP8 CCL passed at PCC 0.999906/0.999849 and 0.476668/0.528138 ms (`candidates/persistent_async_ccl_bfp8.log:60-66`). README/work-log prose rejects these for falling below an "accepted baseline", although they exceed the code's accepted PCC threshold and no separate minimum is defined.
  Why this matters: The optimize and stage-review contracts prohibit selecting a slower higher-precision/configuration path when a faster real-target-weight candidate passes the minimum accuracy bar. Some measurements predate the final lifecycle repair, so they cannot simply replace the default, but they also are not validly rejected.
  Required next step: Define one evidence-backed acceptance bar consistently in tests and reports, then rerun the faster QKV block-7, attention-BFP4, and BFP8-CCL candidates on the current final topology/lifecycle for both layer kinds, including non-aligned, batch-32, trace-replay, watcher, and final-default latency evidence. Keep the fastest accepted cumulative policy or document a model-visible correctness failure/op-contract blocker.

- P1: The dominant final packed-MLP geometry was not swept under its selected topology/output policy.
  Evidence: The 7/8/12/14/21/24/28/42/56/84-core sweep assigns `_mlp_geometry_policy(num_cores)` (`tests/test_multichip_decoder.py:766`), whose default base is `DEFAULT_OPTIMIZATION_POLICY` (`:411`), reverting the selected `DEFAULT_MULTICHIP_OPTIMIZATION_POLICY` packed gate/up BFP8-output contract. Thus `candidates/mlp_geometry_extended_retry2.log` and the paired 14-vs-24 run measure the separate gate/up family. The final profiler instead shows a dominant packed `32 x 5376 x 10752` BFP4/LoFi-to-BFP8 matmul at about 103 us, marked `SLOW` (`tracy/final/*_decode/perf_report.csv`). Only the selected 14-core packed configuration is measured; material packed geometries are absent.
  Why this matters: Geometry evidence from a different projection topology/output dtype does not establish the best geometry for the final dominant matmul. This is the exact precision/topology/geometry mixing forbidden by the optimize contract.
  Required next step: Preserve the packed BFP8-output policy while sweeping legal core/shard/block geometries for the packed gate/up row, including a precision-locked 14-vs-24 comparison and other material legal grids/divisors; compare whole-MLP and whole-layer traced latency and PCC, then reproduce the winner as the final default.

- P2: The claimed BF16 CCL payload is not reconciled with the measured MLP reduction row.
  Evidence: README/context claim two BF16 reductions and `collective_payload_dtype: bfloat16`. In `_tp_allreduce`, no typecast occurs when `communication_dtype == bfloat16`; therefore the packed MLP's BFP8 partial is passed directly to `all_reduce_async`. Both final decode CSVs show the second reduction as `BFP8, BF16 => BF16`, with `Input 0 Datatype=BFLOAT8_B`, while the attention reduction is `BF16, BF16 => BF16`.
  Why this matters: The contract requires measured proof of activation/CCL dtype. The current profiler proves a mixed input/output contract, not the unqualified two-BF16-payload claim, and this ambiguity affects the comparison to the rejected BFP8-CCL family.
  Required next step: Establish from the op contract which dtype is actually communicated/accumulated, explicitly cast if BF16 input payload is intended, and update code/report/candidate comparisons so the selected CCL policy matches the measured runtime rows.

## Other Concerns

- `_PERSISTENT_CCL_POOLS` strongly retains each closed mesh and its TT tensors by `id(mesh_device)` with no teardown. The reviewed single-process fixtures use one module-scoped mesh, so no failure is demonstrated, but repeated mesh open/close in a long-lived process can accumulate stale pool entries. Add lifecycle ownership/cleanup coverage during remediation.

## Hard-Check Gaps

- The advertised-position test allocates the full cache and exercises position 262,143, determinism, replica equality, and finiteness, but it does not populate long history or compare PCC. Prior-stage context evidence remains referenced and the optimized stage did not alter cache layout, so this is not independently blocking here.
- The absent stage commit and pending status are expected administrative items and were not used to fail the review.

## Anomaly Ledger

- Observed anomaly: Linear fused MM+RS hung.
  Evidence: `candidates/fused_mmrs_hang_triage.txt.gz`, `fused_mmrs_model_shape*.log`.
  Affected path: rejected candidate only, not final Linear async all-reduce default.
  Control or comparison: Ring exact-shape retry passed PCC 0.999963; complete Ring family was later rerun on current source.
  Likely subsystem: fused program's Ring-kernel/topology contract.
  Investigation performed: triage, reset/health recovery, Ring adaptation, full-family lifecycle/autofix runs.
  Resolution: controlled.

- Observed anomaly: Initial packed batch-32 and Ring/full-prefill runs collided with retained L1 scratch.
  Evidence: failing candidate XML/logs plus `evidence/batch32_autofix.xml` and final suite.
  Affected path: early candidates.
  Control or comparison: tail-24 shared scratch and M>1 separate branch pass ordered batch-32 sliding/full tests and final watcher run.
  Likely subsystem: persistent L1/CB lifecycle.
  Investigation performed: autodebug/autofix and current-source reruns.
  Resolution: fixed for the final default.

## Scope Inspected

- Goal/skills: Stage 05 contract supplied by the orchestrator; `.agents/skills/stage-review/SKILL.md`, `.agents/skills/optimize/SKILL.md`, and `.agents/skills/tt-device-usage/SKILL.md`.
- Artifacts: README, work log, context contract, perf accounting, candidate matrix and raw logs/XML, final correctness/latency/context/watcher evidence, Tracy provenance, enriched source CSV, advice-enabled reports, filtered CSVs, and hashes.
- Code: `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, `models/demos/gemma4/tt/precision.py`, and `tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py`, including live diffs.
- Commands: read-only `git status/diff`, `find`, `rg`, `sed`, XML/CSV parsing, SHA-256 verification, and Python syntax compilation. No TT hardware or server command was run.

## Residual Risk

- Final-default functional evidence itself is strong: 11 standard-suite passes, four final latency passes, two advertised-position passes, four separate watcher passes, and four profiler-window passes; profiler hashes verify. Historical candidate failures were not misclassified as final-default failures.
- The verdict remains `more-work-needed` because the selected default is not yet proven to be the fastest correct cumulative policy under the stage's own PCC gate and final topology.
