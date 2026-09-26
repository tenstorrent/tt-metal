# Stage Review

Verdict: **more-work-needed**

Stage 04, `multichip-decoder`, `google/gemma-4-26B-A4B-it`. Independent,
source-and-artifact review of the live worktree on branch
`gemma-4-26b-a4b-it`, starting commit
`2e3a1779d3271bf33f0083cb03c9bb56feddc03d`, optimized baseline `a9259624f2`.
Reviewed runtime SHA256:
`a7c8370500079afd74548628186a565c758a2fd4b37b07993988938ea4479efb`.
No hardware commands, implementation edits, or additional reviewers were used.

The current documentation correctly calls this an incomplete attempt. The
paired EP candidate has useful correctness evidence; it is not an accepted
optimized decoder or a stage pass. No additional model arithmetic defect was
demonstrated by this inspection. The failed Watcher gate and the incomplete
capability/performance gates below remain required work.

## Required Work

- **P1: Resolve the fabric exit failure before claiming Watcher acceptance.**
  Evidence: `ep_watch_noinline.log` prints all numerical checks and
  `EP_PROBE_PASS`, then captures the device 0, virtual Ethernet core `(28,25)`
  subordinate packet-tag assertion at `20:41:10.171`. The probe prints its pass
  marker after Python mesh close (`tests/probe_multichip_expert_parallel.py:166-169`),
  but the process subsequently aborts. Both `ccl_watcher.log` and
  `ccl_watcher_single_erisc.log` pass BF16/FP32 reductions and then abort at the
  same core's firmware heartbeat wait. The latter explicitly enables the
  supported single-ERISC override. Firmware discovery reports `19.9.0`.
  Why this matters: correct numerical output and a successful no-payload mesh
  smoke do not establish the required clean kernel/process handoff. Source
  confirms the zero-tag assertion after `kernel_main` in
  `tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:41-51`; the router teardown
  drains transactions without an explicit tag clear, whereas the neighboring
  mux clears tags after its barrier. The proposed infrastructure cause is
  supported, but its repair has not been applied or verified.
  Required next step: obtain a repaired infrastructure revision or separately
  authorized infrastructure work, then rerun the minimal CCL control, original
  EP probe, and final decoder checks with all Watcher features and normal
  process exit. Do not suppress checks or classify this as a false positive.
  `AUTOFIX_watcher.md` records an unsuccessful scoped workaround, so this is an
  actual external scope blocker, not an uninvestigated first failure. This
  review does not authorize C++ changes or certify a proposed patch.

- **P1: Validate the preserved multichip capability contract on the final path.**
  Evidence: the current EP paired JSONs cover one request, 4096 prefill tokens,
  128 advancing decode positions, and eight local K/V comparisons per layer
  kind. The context contract now accurately records the largest exercised
  multichip context as 4224 and the target 262144 as unvalidated. Short length
  65 controls exist for earlier TP candidates. The batch/prefix harness
  (`tests/test_multichip_contracts.py`) and two-layer handoff harness
  (`tests/test_multichip_stack.py`) have no execution results. No multichip
  maximum/non-aligned maximum-context or request-reuse stress result exists.
  Why this matters: inherited single-chip orchestration does not validate the
  changed local-head ownership, per-rank cache mapping, collectives, trace
  signatures, and cumulative stack behavior at those boundaries.
  Required next step: after the infrastructure gate is repaired, run the
  contract's maximum and non-aligned contexts for both layer kinds, paged
  prefix/slot preservation and request reuse, larger logical batches, direct
  stack handoff, and repeated/stress execution with fallback guards. Record
  source-matched results for the selected path. Preserve 262144 unless a hard
  physical limit and the largest feasible value are actually demonstrated.

- **P2: Finish topology/geometry selection and reproduce the selected default.**
  Evidence: recomputed EP host-wall medians give prefill speedups
  `2.2040718` sliding and `2.1704462` full, but decode speedups only
  `0.8816024` and `0.9373927`; four-device efficiencies are respectively
  `0.551018/0.220401` and `0.542612/0.234348`. Sliding TP v2's 864.36 us decode
  is also slower than its paired 824.99 us TP1 baseline. The only measured
  hidden-sharded residual control uses full attention at length 65, and the
  fused-CCL harness is explicitly unrun. `expert_parallel=False` remains the
  runtime default; no final winner is declared.
  Why this matters: the stage's practical multichip optimization goal has not
  been established, and the short residual control does not reject the
  headline-workload sharded/fused families. A prefill win alone does not prove
  the primary traced-decode optimization. The documentation appropriately
  refrains from claiming that it does.
  Required next step: complete the already-planned compatible residual/CCL
  families and material role-specific geometry/precision comparisons under
  the target workload, address applicable native-profile advice, then select
  and rerun the default against the strongest correct candidates and paired
  optimized TP1 baseline. A measured, explicit target tradeoff is acceptable;
  neither an unmeasured hybrid nor an isolated component win is final evidence.

- **P2: Obtain final target native profiles and matched rooflines.**
  Evidence: `profile_v0.json` and `profile_v0/provenance.json` describe an older
  sliding TP v0 candidate at 4096/1. The raw CSV independently reproduces
  2229 prefill operations per rank and a maximum full-rank span of
  `825575.946667 us`; one decode replay has 139 operations per rank and a
  maximum span of `999.845185 us`. These match `whole_layer.json` exactly.
  Human tables and report CSVs exist, contain advice, and expose the old
  sparse projection geometry. They do not profile final EP/TP v2 or the full
  attention target. The telemetry packet correctly leaves target device time
  and utilization null.
  Why this matters: valid diagnostic profiling cannot establish the final
  selected dtype/fidelity, whole-layer device latency, bandwidth/compute
  utilization, or target performance for both representative kinds.
  Required next step: profile warmed prefill and traced decode separately for
  the final selected path and required workload, retain human and CSV reports
  with exact source/command provenance, and reconcile whole-layer device time,
  host time, and the declared roofline estimates. Continue to leave unknown
  target telemetry null until matching evidence exists.

## Other Concerns

- The two CCL controls show heartbeat timeouts, not second independently
  captured packet-tag assertions. Their same immediate register cause remains
  an inference. `AUTOTRIAGE_watcher.md` and `AUTOFIX_watcher.md` preserve this
  distinction and do not overstate it. The generic minimum-firmware suffix in
  `tt_metal/llrt/llrt.cpp:588-594` is unconditional on that timeout; it does not
  contradict the recorded firmware version.
- The memory plan was corrected during review to include the three full-length
  BF16 prefill buffers. CPU arithmetic reproduces 8,556,380,160 cache bytes,
  6,067,486,720 weight/state-bound bytes, 19,858,358,272 resident/reserve bytes,
  and a 24,287,543,296-byte conservative peak per device. The hypothetical dual
  layout's 29,085,106,176-byte peak is explicitly unselected and unimplemented.
  `memory_audit.md` correctly requires shared per-kind RoPE, tied embeddings,
  and prompt release of old layer/setup buffers. These calculations are not
  allocator measurements or maximum-context execution proof.
- The formatted runtime differs from the measured EP snapshot only by
  formatting and top-level import ordering; removing import nodes gives an
  identical AST, and the measured snapshot hash matches both EP results.
  `source_provenance.json` and the revised README/work log now state that
  narrower equivalence rather than claiming exact AST identity.

## Hard-Check Gaps

- Existing device-only guards cover the measured forwards and trace capture;
  they are useful evidence for those shapes. They do not replace the missing
  final context, batch, stack, and stress runs.
- The paired runner uses identical real checkpoint layers and recorded input
  fixtures, compares every replicated device output, refreshes input/position
  buffers between replay steps, and checks bitwise repeated replay output.
  The EP paired results contain 129 output PCC values and eight local-cache
  PCC values per layer kind. Recomputed minima and medians match
  `candidate_summary.json` and the telemetry accuracy rows.
- Baseline/older TP artifacts lack the newer embedded runtime-hash field.
  Their retained snapshots and chronology provide candidate context; the
  source-hashed EP results provide stronger linkage for the current body.
  This does not invalidate the declared diagnostic use of older artifacts.
- Existing pre-commit logs show applicable hooks passing. Python/docs-only
  changes do not require a C++ build. No test or build was rerun by this
  reviewer. This verdict cannot satisfy the clean-pass gate or authorize a
  stage-completion commit; preserving an incomplete attempt is a different
  action.

## Anomaly Ledger

- Observed anomaly: numerical EP success followed by packet-tag assertion.
  Evidence: `ep_watch_noinline.log`, firmware exit assertion and router source.
  Affected path: fabric kernel/process teardown with Watcher enabled.
  Control or comparison: model-free RS also fails teardown; single-ERISC
  fallback does not fix it; a normal recovered mesh open/close succeeds.
  Likely subsystem: fabric exit-state cleanup, with shared CCL-control cause
  inferred rather than captured directly.
  Investigation performed: AutoTriage/source inspection and scoped AutoFix
  controls; no C++ repair tested.
  Resolution: **more-work-needed**, external scope blocker.

- Observed anomaly: EP improves prefill but regresses traced decode.
  Evidence: both paired EP JSONs and independently recomputed medians above.
  Affected path: active expert execution plus collectives.
  Control or comparison: paired OptimizedDecoder and prior TP candidates.
  Likely subsystem: changed expert ownership/geometry, sparse scanning and
  communication; no single cost attribution is proven for final EP.
  Investigation performed: TP geometry revisions, EP probe and paired target
  runs; final topology and native attribution remain open.
  Resolution: **more-work-needed**, no unsupported performance win accepted.

- Observed anomaly: apparent slow profiler close and an earlier overlapping
  recovery/reduced-check retry.
  Evidence: `AUTOTRIAGE.md`, `AUTOFIX_ep.md`, later preserved recovery logs.
  Affected path: experiment lifecycle.
  Control or comparison: profiler eventually completed; the overlapped retry
  did not execute EP and is excluded from acceptance. Subsequent CCL controls
  and recovery were recorded separately; final normal mesh smoke completes.
  Likely subsystem: host shutdown/instrumentation and experiment orchestration.
  Investigation performed: saved triage and bounded reset/list/mesh recovery.
  Resolution: **controlled for the stated diagnostic use**; does not resolve
  the independent Watcher failure.

## Scope Inspected

- Goal/skill paths: supplied Stage 04 contract;
  `.agents/skills/{stage-review,multichip,optimize,tt-device-usage}/SKILL.md`.
- Artifacts: stage README/work log, mesh and fused-CCL plans, context contract,
  memory plan/audit, candidate and paired JSONs, runtime snapshots/provenance,
  AutoFix/AutoTriage reports, raw EP/CCL Watcher and recovery logs, preserved
  log hashes, native v0 raw CSV, human reports, CSVs and whole-layer windows;
  telemetry packet
  `bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/cd88dda8-3baa-459f-9ff7-6beb4847565d.json`.
- Code: `tt/multichip_decoder.py`, inherited optimized/fused decoder and runtime
  audit paths, paired runner, batch/prefix and stack harnesses, EP/CCL probes,
  multichip performance summarizer, Gemma4 mesh CCL helpers, and cited firmware,
  router, mux and heartbeat-wait source.
- Commands: read-only `git status/rev-parse/branch`, `find`, `cat`, `sed`,
  `grep`, `nl`, and small Python standard-library scripts to recompute PCC
  summaries/timing ratios, AST/hash provenance, memory arithmetic, compressed
  log hashes and native firmware timestamp windows. `rg` was unavailable.
  Only this report was written by the reviewer.

## Residual Risk

All accepted claims remain limited to their named candidate runs. No text
generation/full-model/vLLM result is claimed or required from this decoder-only
stage. The final hardware path, maximum context, cumulative resource ownership,
and performance remain unvalidated. After the external repair and outstanding
contract work, a new independent review must return clean-pass before stage
completion is claimed.
