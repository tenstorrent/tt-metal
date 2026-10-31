# Stage Review

Verdict: more-work-needed

Independent near-complete review of stage 04, `multichip-decoder`, on 2026-10-31.
Reviewed the live worktree on `mvasiljevic/gemma4-31b-fresh-pipeline-v0811`, based
on `87140921cfc65695b74f0981b3c19b4182e46e61`. Paths below are relative to
`models/autoports/google_gemma_4_31b_it` unless otherwise stated. This is an
inspection result before the owner's final validation/profile queue completes;
it does not claim that the pending jobs failed.

## Required Work

- P2: Complete acceptance on the frozen selected implementation.
  Evidence: `tests/run_multichip_validation.py:17` defines boundary, default-batch,
  context and watcher groups. At inspection, completed final JSONs existed for
  sliding 262143/262144, full 262143, both B32 short stress cases, and both
  cyclic4097 cases. There were no completed final boundary/batch/watcher results,
  `doc/multichip_decoder/stack.json`, or full-attention 262144 result. Existing
  final manifests matched `tt/optimized_decoder.py` but did not match current
  `tt/multichip_decoder.py`, `tests/run_multichip_decoder.py`, or the candidate
  source. The owner identifies production changes as formatting and harness
  changes as additional audit/cache-snapshot checks, and plans a final rerun.
  Why this matters: `multichip/SKILL.md`, Evidence To Leave, requires these
  correctness, stacking, batch, watcher and trace gates; earlier artifacts do
  not establish the final changed source without a semantic provenance bridge.
  Required next step: Finish the selected boundary cases, B1/32/16/8 for both
  kinds, remaining maximum-context case, watcher and direct two-layer interface
  test. Preserve exact source manifests and successful exits. Reuse unaffected
  evidence only with a concrete explanation of the intervening source changes;
  otherwise rerun affected cases. Keep trace allocation tracking enabled with
  program-cache allocations included, and rerun the relevant stress after any
  persistent-buffer change.

- P2: Exercise cyclic decode across the physical-cache wrap.
  Evidence: `tt/multichip_decoder.py:341` introduces the modulo-aware update and
  SDPA path with a 2048-token physical pool. `tests/run_multichip_validation.py:22`
  starts the cyclic stress at 4097 and advances only 128 steps; the maximum
  context cases repeatedly use position 262143. The boundary group at line 18
  does not enable cyclic storage. Thus the current acceptance artifacts exercise
  prefill wrapping and decode after a wrap, but not advancing decode through it.
  Why this matters: the new cache-addressing branch must preserve the previous
  sliding window while reusing physical pages. The model-bringup changed-path
  rule requires the affected boundary, rather than only a long passing input.
  Required next step: Add a tracked cyclic stress case beginning at 2047 and
  advancing through 2048, preferably B32 with private shuffled pools and distinct
  rows. Compare against the unbounded optimized baseline, retain per-replay PCC,
  and verify the output before and after wrap. No additional long soak is needed.

- P2: Measure persistent buffers for the selected repeated collective path.
  Evidence: `tt/multichip_decoder.py:246` explicitly passes
  `persistent_output_buffer=None` to all-gather; line 283 uses ordinary minimal
  reduce-scatter. The candidate's persistent buffers are attached to the
  different fused-matmul/RS family, not the selected RS/AG sequence.
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.cpp:221`
  supports a supplied output buffer. `optimize/SKILL.md`, OPT-009, explicitly
  requires a persistent/preallocated candidate or an exact blocker for material
  repeated decode CCL rows.
  Why this matters: the selected layer pays these collective boundaries after
  both row projections; the fused-family result does not isolate buffer planning
  on the selected topology.
  Required next step: Measure legal persistent outputs/intermediates on the
  selected Ring/BF16 decode family, with stable setup-time allocation and explicit
  alias lifetimes. Record any API that cannot accept such buffers precisely.
  Compare complete traced layer time and correctness, include the CCL tuning and
  buffer fields in the final report, and retain the faster correct path.

- P2: Compare packed gate/up on the new TP4 dimensions.
  Evidence: `tt/multichip_decoder.py:330` always launches separate gate and up
  projections; `tests/run_multichip_candidates.py:15` has no packed-MLP candidate.
  The prior single-chip comparison in `doc/optimized_decoder/README.md:122` is
  for different projection dimensions. The local padded TP4 output is 5632,
  rather than the baseline global intermediate width 21504.
  Why this matters: `optimize/SKILL.md`, OPT-010, requires packed versus split
  gate/up including splitting, activation, movement and downstream costs. The
  changed dimensions can change the winning geometry and launch-cost tradeoff.
  Required next step: Test a TP4 packed gate/up candidate at the selected
  BFP4/LoFi policy against the existing split-plus-fused-GELU-multiply path.
  Include split/layout cost and the down/collective boundary in whole-layer
  timing. Reject it with measured loss or a precise adapted op-contract blocker.

- P2: Supply same-shape one-reader controls for selected TP4 projections.
  Evidence: `tt/multichip_decoder.py:186` selects two readers for the attention
  and down projections, and lines 199-200 select two for padded gate/up.
  `tests/run_multichip_candidates.py:15` tests QK three readers and alternate
  geometry, but has no one-reader controls for the final padded shapes. The
  earlier gate/up one-reader path uses unpadded 5376 output channels and therefore
  changes both shape and reader count. The prior single-chip shapes are also
  different.
  Why this matters: `optimize/SKILL.md`, OPT-015, requires same-shape one/two/legal
  three-reader comparisons. TP4 produces narrower per-bank reads, where more
  readers are not automatically faster.
  Required next step: Add precision-locked one-reader controls for the selected
  local projection geometries and preserve legal third-reader evidence for QK.
  Record per-bank tiles/physical bytes, the selected reader count, correctness
  and traced whole-layer effect. Do not require an illegal third reader for
  16- or 22-tile bank shards. Use the final profiler rows to verify the actual
  selected dtype/fidelity and reader count.

- P2: Finish final performance evidence and reconcile stage records.
  Evidence: `doc/multichip_decoder/README.md:39` describes final profile tables,
  but no `tracy_final/**/performance.json` existed at inspection. The telemetry
  packet at `bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z/telemetry/packets/9630fa21-6741-4688-bae0-7db3f63850d7.json`
  still contains only the initial sliding profile and explicit pending reasons.
  `doc/context_contract.json:134` records decoder weights 4,188,190,720 and
  windowed total 10,819,909,632 bytes/device, whereas the current
  `doc/multichip_decoder/memory_plan.json` records 4,411,965,440 and 11,043,684,352.
  `doc/agentic_serving.json` remains stage 03. README line 39 names
  `optimized_decoder/tracy_final` for the baseline, but the selected final
  full-attention baseline is `optimized_decoder/tracy_final_k32`, as recorded in
  the existing agentic-serving measurements.
  Why this matters: the stage contract requires final default-run whole-layer
  performance, speedup/efficiency, all default batches, and current context and
  workload records. Initial candidate timing cannot stand in for the final path.
  Required next step: Produce separate warmed prefill/decode device profiles for
  both kinds at B1/32/16/8, with raw CSV provenance and readable plus CSV
  tt-perf-report output. Include all layer work and internal gaps; compare against
  the selected same-workload baseline, compute speedup and TP4 efficiency, and
  classify remaining material movement. Verify final defaults reproduce the
  chosen candidate. Reconcile memory/context/workload/work-log records and the
  exact telemetry packet, retaining 4096/128/B1 separate-kind headline rows and
  the target concurrency beside them. A local checkpoint follows a later clean
  review; its absence at this pre-pass point is expected.

## Other Concerns

- `doc/multichip_decoder/AUTODEBUG_CCL8.md` and `AUTOTRIAGE.md` intentionally
  preserve earlier unresolved diagnoses. The final work log should link those
  historical findings to their later controlling runs, so readers do not mistake
  them for unresolved final-path failures. The existing barrier and matching
  Ring controls support those dispositions.
- I did not find a concrete mathematical sharding defect in the inspected
  production source. Projection ownership, local Q/K/V head counts, per-shard
  inert MLP padding, row reduction before norms, and BF16 residual restoration
  agree with the documented TP4 contract. This does not waive the required runs.

## Hard-Check Gaps

- The acceptance runner writes its JSON before its final PCC assertion. Result
  files alone therefore are not pass tokens. The serial wrapper checks child
  exit status; final closeout should preserve that status or the successful log
  with the JSON. Existing saved PCC values reviewed here exceed 0.995.
- Manifests hash source but not the input fixture. Pinned weight loading and the
  recorded-activation paths are visible in source; no evidence showed a fixture
  mismatch. Fixture provenance is an audit improvement, not a new required
  runtime test or a reason by itself to reject this stage.

## Anomaly Ledger

- Observed anomaly: Explicit mapped device upload stalled before decoder work.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, and
  `sliding_rep4096_hosttilize.json`.
  Affected path: setup transfer of 4096-row activations.
  Control or comparison: single-chip omission of mapper succeeded; host tilize
  then device transfer allowed both single-chip and TP4 comparison.
  Likely subsystem: mapped upload/device tilization.
  Investigation performed: staged upload isolation and owner-checked recovery.
  Resolution: controlled; setup avoids the failing path. No kernel fix is claimed.

- Observed anomaly: BFP8-CCL prefill PCC was NaN; diagnostic mocking exhausted L1.
  Evidence: `sliding_fastnorm8.json`, `AUTODEBUG_CCL8.md`,
  `sliding_nan_diag2.json`, `sliding_ccl8_barrier.json`, `sliding_rep8_fixed.json`.
  Affected path: experimental prefill CCL and diagnostic tensor lifetimes.
  Control or comparison: BF16 CCL passed; a plain diagnostic callback removed
  mock retention; ordinary BFP8 with an RS barrier passed real-weight PCC.
  Likely subsystem: synchronization plus a separate diagnostic ownership bug.
  Investigation performed: source lifetime audit, finite component controls,
  barrier adaptation and subsequent ordinary runs.
  Resolution: controlled by passing barrier-enabled runs; the final chosen
  buffer policy still needs the final tracked stress and watcher gate above.

- Observed anomaly: Fused matmul/RS stalled on Linear fabric.
  Evidence: `triage_fused.txt`, `AUTOTRIAGE.md`, `sliding_fused_ring.json`.
  Affected path: rejected fused row-projection candidate.
  Control or comparison: matching Ring fabric/topology executes and passes PCC;
  its approximately 870 us host traced decode is slower than the selected family.
  Likely subsystem: fused collective/fabric configuration.
  Investigation performed: preserved live stacks and adapted sharded residual
  and padded output contracts, then matching-fabric retry.
  Resolution: controlled; the family is not rejected on the initial hang alone.

- Observed anomaly: Fused AG matmul BFP8 projection output fails prefill PCC.
  Evidence: `sliding_fused_ag_adapted2.json` prefill PCC 0.9849548692;
  `sliding_fused_ag_bf16.json` prefill PCC 0.9999389716.
  Affected path: rejected fused-AG/sharded-residual candidate.
  Control or comparison: BF16 projection output repairs PCC; both variants take
  about 1095 us host traced decode, versus about 437 us for replicated Ring.
  Likely subsystem: projection-output precision in that fused family.
  Investigation performed: real-weight precision control and compatible
  distributed norm/residual consumers.
  Resolution: controlled; the BF16 legal family loses the measured target.

## Scope Inspected

- Goal/skill paths: repository `AGENTS.md`, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`; installed stage-review, multichip,
  agentic-serving, optimize, tt-enable-tracing, tt-device-usage, model-bringup,
  and tt-review-core skills. Model-bringup skill root:
  `/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.22/skills`.
- Artifact paths: stage README, mesh plan, work log, context contract, memory
  plan, agentic-serving record, completed final correctness/stress JSONs and
  manifests, topology/precision/geometry candidate JSONs and manifests, initial
  performance JSON/table, diagnosis/recovery reports, and the specified packet.
- Code paths: complete `tt/multichip_decoder.py`, inherited optimized decoder
  forward/allocation/normalization/projection/cache methods, multichip candidate,
  validation, stack, profiling, replay selection, profile summarization,
  memory-plan and recovery scripts; baseline pinned weight loaders and trace
  helper; local collective persistent-output implementation. Read local
  `tech_reports/LLMs/llms.md` multi-device/optimization guidance and
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Commands run: read-only `cat`, `sed`, `rg`, `nl`, `wc`, Git status/branch/HEAD,
  and small standard-library Python analyses of artifact JSONs/source hashes.
  No hardware, reset, server, TTNN import, long test, publication, or messaging
  outside the agent team was performed. Only this review report was written.

## Residual Risk

- Review covers a live worktree; later edits require affected evidence and
  rereview. The owner has acknowledged these required items and is continuing
  implementation/validation; that acknowledgement is not completion evidence.
- Existing long-context evidence is strong but bounded: sliding 262144 output
  PCC 0.9999246485 and decode PCC 0.9999517928; full 262143 output PCC
  0.9999838289 and decode PCC 0.9999905049. Long comparisons cover final rows
  against the completed single-chip optimized control, not every output row.
- This decoder stage has no generator or serving adapter. Token feedback,
  cross-request serving behavior, full-model memory peaks and serving registration
  are not established here and were not invented as stage-04 pass requirements.
