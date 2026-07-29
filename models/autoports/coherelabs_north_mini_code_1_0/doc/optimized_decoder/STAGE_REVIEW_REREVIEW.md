# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Complete and retain the per-role sparse output-subblock search, then
  profile the promoted default.
  Evidence:
  The promoted 1x2 geometry is real and beneficial: the retained cumulative
  candidate improves traced layer-1 b1 decode from 0.791673 to 0.725390 ms
  and seq128 prefill from 13.990679 to 13.544480 ms. Authentic layer-1/layer-4
  decode passes, and the final 40-pass normal and watcher suites cover the
  promoted default.

  The search itself is incomplete. `SPARSE_SUBBLOCK_HYPOTHESIS.md:40-59`
  identifies legal isolated gate/up/down controls and legal 1x3/1x4
  candidates, and requires independent role selection before a cumulative
  run. The only retained performance files are a 1x1 all-role control and a
  cumulative all-role 1x2 result; there are no isolated gate, up, or down
  timings, no 1x3/1x4 results, and no exact blockers for those legal rows.
  Moreover the current named “isolated” policies at
  `tests/test_optimized_decoder.py:75-96` start from
  `OptimizationConfig()`, whose defaults are already cumulative 1x2
  (`tt/optimized_decoder.py:76-87`). Therefore
  `gate_g12_b2_s2`, `up_g12_b2_s2`, and `down_g32_b2_s2` no longer isolate
  one role; each leaves the other two roles on the promoted 1x2 geometry.

  No post-promotion Tracy/`tt-perf-report` tree exists. All retained sparse
  profiler rows under `tracy/review3_selected/` describe the superseded
  24/24/64-core 1x1 programs, so no current runtime row proves the promoted
  12/12/32-core 1x2 gate/up/down contracts or reconciles current advice.
  The optimize skill requires separate legal geometry sweeps for each
  dominant role and final measured runtime rows.
  Why this matters:
  Sparse matmuls dominate primary b1 MoE time. A winning cumulative 1x2
  candidate proves an improvement, but it does not establish that 1x2 is the
  best legal per-role combination or satisfy final-profile signoff.
  Required next step:
  Make isolated policies start from an explicit 1x1 control. Measure gate,
  up, and down independently, including the documented legal 1x3/1x4 rows,
  or retain exact failures/blockers. Rebuild the best cumulative default and
  collect separate advice-enabled b1 decode and seq33/128 prefill profiles
  proving BFP8/LoFi, `in0_block_w=16/16/12`, the selected grids, and selected
  output subblocks. Rerun correctness/watcher only if the default changes.

- P1: Batch-32 still has no routed active-expert runtime.
  Evidence:
  The current threshold remains 32, and `_sparse_moe()` still sends b32
  decode/prefill to `_dense_expert_moe_chunk()`
  (`tt/optimized_decoder.py:94,1419-1424`). `ROUTED_MOE_HYPOTHESIS.md`
  credibly exhausts model-local alternatives: full-surface sparse matmul is
  17.831-21.896 ms, the fast `moe_compute` API exposes only a rolling
  two-expert buffer, and its complete consumer requires fabric. The required
  compact persistent output or local-only combine must be implemented in
  shared TTNN, outside the authorized model-only stage files.
  Why this matters:
  The current optimize checklist explicitly requires routed active-expert
  execution with no dense all-expert runtime path. AutoFix establishes an
  exact shared-API blocker, but the required implementation path remains
  absent.
  Required next step:
  Add a shared-TTNN compact routed output or fabric-free local combine, wire
  it into the decoder, and prove authentic PCC, stable trace replay, watcher
  cleanliness, no dense branch, and b32 no-regression. Alternatively, the
  stage owner must explicitly change the goal/skill contract; the present
  model-only scope cannot produce a clean pass against the current contract.

## Other Concerns

- The sparse performance comparison still uses deterministic synthetic
  weights/activations. Current authentic correctness covers the promoted
  default, but the required role-specific rerun should use recorded target
  routing activations where practical.

## Hard-Check Gaps

- No current profiler artifact describes the promoted sparse geometry.
- No retained isolated-role or legal 1x3/1x4 sparse candidate result exists.
- The shared TTNN batch-32 routed-output capability remains unavailable.
- This rereview did not run hardware; it inspected the supplied current
  normal/watcher and candidate artifacts.

## Anomaly Ledger

- Observed anomaly:
  BFP4 attention was previously rejected on random activation and old
  geometry.
  Evidence:
  `candidates/review_attention_precision/results.xml` now contains two passing
  selected rows and 16 authentic BFP4 failures at b1/b32 on final topology;
  QKV-only PCC is 0.99458730, O-only 0.99474771, and cumulative 0.99258065.
  Affected path:
  Layer-0 attention precision.
  Control or comparison:
  Selected BFP8/LoFi passes current focused, full, and watcher suites.
  Likely subsystem:
  Attention-weight quantization.
  Investigation performed:
  Inspected policy separation, checkpoint activation construction, final
  topology assertions, cache-consuming trace path, and JUnit failures.
  Resolution:
  fixed. The prior attention P1 is closed.

- Observed anomaly:
  Sparse 1x1 rows were dominant and had untried output-subblock advice.
  Evidence:
  Promoted 1x2 source/candidates improve b1 and pass current suites, but the
  documented independent/larger legal matrix and current profiling are
  absent.
  Affected path:
  Active-expert b1 decode and prefill.
  Control or comparison:
  Retained 1x1 control versus cumulative 1x2 candidate.
  Likely subsystem:
  Sparse matmul geometry search/evidence.
  Investigation performed:
  Compared hypothesis matrix, config defaults, named test policies, candidate
  inventory, JSON policy fields, and profiler directories.
  Resolution:
  more-work-needed.

- Observed anomaly:
  Batch 32 selects dense all-expert execution.
  Evidence:
  Current dispatch source plus `ROUTED_MOE_HYPOTHESIS.md` and
  `AUTOFIX_CURRENT.md`.
  Affected path:
  Layer-1/layer-4 b32 decode and prefill.
  Control or comparison:
  Exhausted dynamic/static/packed sparse and fused compute families.
  Likely subsystem:
  Shared TTNN routed-output/combine contract.
  Investigation performed:
  Rechecked dispatch and the exact model-local/API blockers.
  Resolution:
  more-work-needed.

- Observed anomaly:
  Promoted source needed a fresh full watcher run.
  Evidence:
  `artifacts/current_after_autofix.xml` and
  `current_after_autofix_watcher.xml` each contain 56 cases: 40 passes, 16
  expected opt-in DRAM-candidate skips, zero failures/errors. The 3,247-line
  `watcher/current_after_autofix/generated/watcher/watcher.log` has no fatal,
  invalid-NoC/CB, overflow, sanitizer, timeout, hang, trip, kernel/watcher
  error, or assert signature.
  Affected path:
  Complete promoted optimized decoder.
  Control or comparison:
  Normal versus watcher executions.
  Likely subsystem:
  Device/runtime safety.
  Investigation performed:
  Parsed both XMLs and independently scanned the watcher log.
  Resolution:
  fixed.

## Scope Inspected

- Goal/skill paths:
  current `.agents/skills/{optimize,stage-review}/SKILL.md`, prior
  `STAGE_REVIEW_CURRENT.md`, HEAD `72ad2c0d193`, and functional checkpoint
  `78dbd88bec7`.
- Artifact paths:
  `AUTODEBUG_CURRENT.md`, `AUTOFIX_CURRENT.md`,
  `SPARSE_SUBBLOCK_HYPOTHESIS.md`, `ROUTED_MOE_HYPOTHESIS.md`, attention and
  sparse candidates, final normal/watcher XML, watcher log, README, work log,
  and retained profiler trees.
- Code paths:
  optimized decoder, performance harness, correctness tests, and
  prefill-geometry tests.
- Commands run:
  read-only Git inspection, `rg`, `find`, `sed`, `nl`, JSON/XML parsing, AST
  parsing, policy comparison, and watcher signature scan. No hardware,
  server, reset, profiler, or vLLM command was run.

## Residual Risk

- Attention precision is now earned, and final promoted correctness/watcher
  evidence is strong.
- The verdict remains `more-work-needed` only for the incomplete dominant
  sparse search/profile closure and the explicit shared-TTNN batch-32 routed
  output blocker.
