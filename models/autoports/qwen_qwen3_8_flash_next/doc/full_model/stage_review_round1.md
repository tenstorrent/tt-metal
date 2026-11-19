# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 6, full-model, for
`Qwen/Qwen3.8-Flash-Next`. Reviewed the live main worktree at
`2589dcde879a81213c03e5f5eaef5c85f68dc6eb`, including the uncommitted stage
implementation. This verdict accepts the scoped stage evidence; it does not
claim a vLLM implementation, broader model quality, or completed checkpoint
commits. The stage owner must perform the already-required local checkpoint
and record its SHA after this review. The run policy is **never push**.

## Required Work

- None. No remaining correctness, original-contract, or required-evidence
  blocker was found in the inspected implementation and completed artifacts.
  No new hardware run or tuning is required by this review.

## Other Concerns

- **Controlled active-width sensitivity remains a limitation, not an equality
  pass.** The current forced-43 cross-width comparison still fails its frozen
  PCC diagnostic. The original HF backend also fails that comparison under
  matched logical histories. I independently loaded both saved tensor files:
  FP64-centered PCC is approximately 0.990030/0.988163 whole/minimum-row for TT
  and 0.992530/0.987765 for HF. These differ slightly from the original scorer's
  floating-point values but preserve the same verdicts. Both sides select
  deterministic argmax IDs `[71111, 77693]` at that comparison. I also checked
  actual TT sampled tokens against the saved TT and HF logits: all 62 repeated
  draws are TT maxima, and all 31 distinct history/position rows are in the
  matched HF top five. This controls the current numerical anomaly without
  waiving an HF accuracy bar. The larger historical TT43 top-five loss is not
  claimed reproduced by HF; that artifact stays failed. Exact same-history
  migration and final-state checks remain separate requirements and pass.
  `doc/eval_readiness/limitations.json` records `original_model: "same"` with
  appropriately limited backend and evidence scope. I accept this classification
  under the stage-review known-limitation rule, not as a human waiver.

- **Text quality is established within the observed window.** Both AIME
  completions and all six greedy HF/TT pairs were read directly, together with
  sampled seed-0 and seed-7 outputs for every prompt. The visible text is
  coherent; no mechanical collapse, unrelated language switch, or demonstrated
  feedback corruption appeared. For example, both AIME outputs restate the
  walking-time problem and start distance/speed plus coffee-time reasoning;
  both translation outputs reach the requested French answer. The frozen
  120-example blind comparison covers ten matched samples per backend for each
  of six prompts and records zero failures per side. Thinking and truncation
  commonly prevent a complete answer at 128 tokens on both backends. This is
  correctly recorded as a shared limitation, not proof of mathematical or code
  answer correctness. The saved text precedes B>1 factoring; direct AST
  comparison with the archived text-run generator shows the B1 forward,
  prefill, and sampling path is retained, with B1 decode dispatch moved through
  an equivalent helper. The model wrapper change is formatting-only according
  to the matching before/after source manifest.

- **Capacity and performance scopes remain important.** Full48 B32 correctness
  is at C512; full48 B1 maximum-context execution is at C262144. No Cartesian
  B32-by-262144 claim is supported. The extra worker-pool pages are calculated
  charges against measured allocator snapshots, not executions of 545/8194
  blocks. The final reduced profiler is eager, short-history attribution and
  must not become a full48 traced component decomposition. These limitations
  are already explicit in the current README, context contract, capacity
  reconciliation, and performance audit. No new sweep is needed to restate
  these boundaries.

## Hard-Check Gaps

- The final stage checker verifies output degeneracy, context metadata, and
  serving-ledger structure; it cannot by itself prove cache ownership, seed
  identity, trace lifetime, or correct maximum-context completion. Those claims
  have additional source and raw-result support inspected here.
- The maximum-context monitor did not retain a numeric process exit. This is
  not an unfinished required execution: the final JSON and attempt record
  agree, the driver completion marker follows explicit generator/mesh close,
  and the log records completed UMD teardown. No numeric exit is inferred.
- The text runner exited 1 in its final status reporter after all requested
  outputs and device teardown. The raw error is retained. The reporter-only
  repair has CPU evidence and later acceptance runs execute the corrected
  final marker successfully. Completed saved outputs remain valid qualitative
  evidence; the historical process is not relabeled exit 0.
- Existing artifacts are sufficient for this stage. Wider prompt diversity,
  near-end context performance, future serving allocation, and exact full48
  critical-path attribution are limitations of coverage, not missing original
  stage gates.

## Anomaly Ledger

- Observed anomaly: Cross-active-width PCC failure, including historical TT43
  top-five disagreement.
  Evidence: `shared_pool_generator_v5.json`,
  `current_geometry_sampler_v1.json`, `shared_pool_hf_control_v2.json`,
  `shared_pool_hf_control_v3_rescore.json`, `batch_geometry_disposition.md`,
  `batch_reduction_AUTODEBUG.md`, `AUTOFIX.md`.
  Affected path: Reduced real layers 0/1/3, different active-width histories.
  Control or comparison: Original BF16 HF with matched logical histories;
  same-history repeat/migration; current direct HF top-k and sampled-token
  checks; full48 B32 accuracy and state checks.
  Likely subsystem: Geometry-dependent arithmetic and subsequent recurrent
  history, with the TT reduction producer investigated in saved fixtures.
  Investigation performed: Source inspection and independent saved-logit
  comparisons; current sampling maxima and HF membership verified above.
  Resolution: Controlled current PCC sensitivity. Historical larger failure
  remains historical failed evidence; no threshold or precision relaxation.

- Observed anomaly: B32v1 combined final-state disagreement for all requests.
  Evidence: `batch_hf_seed_contract.md`, `batch32_hf_factored_v1.json`,
  `batch32_hf_factored_v2.json`, current common formatter and generator setter.
  Affected path: Baseline-versus-migration probe state comparison.
  Control or comparison: Scalar seed initializes lane 0 only; corrected probe
  supplies an explicit seed for every request and retains all state fields.
  Likely subsystem: Test caller seed ownership.
  Investigation performed: Inspected setter/probe contract and independently
  compared v2 final request and per-field digest maps for all 32 requests.
  Resolution: Fixed in the harness. V2 maps are exactly equal; no state field
  was excluded and production scalar-seed semantics were preserved.

- Observed anomaly: Retained-prefill divergence at nonaligned splits.
  Evidence: `rms_repair_review.md`, `rms_native_regression.log`,
  `prefill_factored_full48_v1.json`, `readiness/prefill_resume_check.json`.
  Affected path: Native floating row reduction, expert rounding, and odd-tail
  singleton versus multi-row continuation.
  Control or comparison: Native row-placement regression; full48 standard
  resume with negative controls; exact all48 B2 starts 31 and 68.
  Likely subsystem: Row-dependent reduction order and retained tail state.
  Investigation performed: Inspected the Blackhole FloatAdd-only repair, the
  explicit hash-keyed stochastic-rounding option, and compact checkpoint
  save/restore/remap lifetime. The default MoE rounding remains unchanged for
  other callers; the target selects deterministic rounding without changing
  formats/fidelity. Native regression reports ten passing cases, including
  BF16, integer, and max siblings. Final all48 gates use the repaired source.
  Resolution: Fixed. Standard resume starts 32/68/256 pass; both negative
  controls fail at 32/68. The existing short-window rule omits controls at 256.

- Observed anomaly: Tracker rejection of a live allocation.
  Evidence: `serving_probe_v7.json`, `serving_watcher_v2.json`,
  `trace_alloc_tracking/positive_control.log` and current generator source.
  Affected path: Deliberate unsafe-allocation control before replay.
  Control or comparison: Removing the deliberate live temporary permits the
  final clean replay; all 16 serving checks pass with program-cache tracking
  included.
  Likely subsystem: Expected tracker behavior, not an unclassified production
  corruption warning.
  Investigation performed: Inspected the generator-owned sampling capture and
  setup ordering. It bypasses the common wrapper's broad allocation exemption;
  no model-wide exemption or disabled program-cache tracking is used.
  Resolution: Controlled negative test. Historical temporary prefill-row
  lifetime failures are repaired in the current path.

- Observed anomaly: Maximum-context outer-monitor interruption and historical
  text reporter exception.
  Evidence: `max_context_closure.json`, final maximum-context log,
  `acceptance_text_v1_process_status.json`, final acceptance driver source.
  Affected path: Process/reporting provenance.
  Control or comparison: Completed outputs, explicit close, final markers,
  UMD teardown, reporter repair and subsequent successful final marker.
  Likely subsystem: Monitoring/reporting, not model execution.
  Investigation performed: Checked final source digests, driver ordering, and
  completion records without starting any job.
  Resolution: Controlled maximum-context exit-code gap; fixed reporter.

- Observed anomaly: Thinking/cap truncation, profiler metadata inconsistencies,
  and historical malformed CSV exports.
  Evidence: `qualitative/verdict.json`, actual HF/TT outputs,
  `terminal_final_analysis.md`, `terminal_final_perf_v1/coverage.json`,
  `terminal_csv_export_diagnosis.md`, Tracy commit
  `390ef12730f570dfec54b6407f3735d03213d1c0`.
  Affected path: Text-window interpretation and profiler reporting.
  Control or comparison: Blind matched HF controls; repaired CSV export and
  exact operation coverage; source/adjacent shapes for attribution.
  Likely subsystem: Shared model behavior and instrumentation metadata.
  Investigation performed: Read actual text and inspected repaired exporter
  scope; independently summed the compact final profiler CSVs.
  Resolution: Controlled text/shape limitations; CSV repair completed. No
  FLOP/bandwidth claim relies on inconsistent BinaryNg shape annotations.

## Scope Inspected

- Goal/skill paths: Original
  `/home/ttuser/qwen38-flash/multigoal_logs/06-06-full-model.prompt.txt`;
  installed 0.1.24 `stage-review`, `full-model`, `tt-enable-tracing`,
  `tt-device-usage`, `qualitative-check`, and startup instructions;
  repository `TraceCorrectness.md`. The user's cap/no-new-tuning instruction
  was treated as a scope constraint, not a correctness waiver.
- Artifact paths: Model README; `doc/full_model/README.md` and `work_log.md`;
  serving/context/limitations ledgers; current all48 accuracy, performance,
  maximum-context, B2/B32 and reduced serving/watcher results and logs;
  resume artifact; HF reference construction/cache metadata; qualitative
  outputs/blind verdicts; current geometry and HF saved tensor artifacts;
  seed, fallback, native-repair, capacity, and performance audits;
  final profiler CSVs/coverage; formatter/source manifests; final checker log.
- Code paths: `tt/model.py`, `tt/generator.py`, `tt/factored_decode.py`,
  `tt/cache_pool.py`, `tt/slot_state.py`, `tt/prefill_checkpoint.py`,
  `tt/device_seed.py`, `tt/token_feedback.py`, `tt/indexed_rows.py`;
  current `multichip_decoder.py` diff; changed native reduction and MoE files;
  acceptance, HF reference/control and geometry-probe drivers; both common
  sampler interfaces; pinned plugin interface/capability documentation.
- Commands run: Read-only `rg`, `sed`, `cat`, `tail`, `git status/diff/show`,
  and short Python JSON/CSV/source-AST/saved-tensor analyses. Current source
  digests independently match all recorded entries in final accuracy, perf,
  maximum-context, B32v2, and watcher artifacts. No device, server, reset,
  experiment, implementation edit, raw-capture archive, or raw-capture hash
  was performed. Only this report was written.
- Accuracy verification: Final B1 reports 97/100 prefill and 96/100 traced
  teacher top-one, with 100/100 top-five/top-100 in both. Independently summed
  B32 records give 2645/2704 top-one and 2704/2704 top-five/top-100 for each
  history; final request/per-field maps are exact across all 32 requests.
- Trace/capacity verification: Current generator prepares supported physical
  buckets and state-copy helpers before capture, retains fixed backing storage,
  uses common sampling with persistent `tt_out_tok`, and advances positions
  on device. Request reset skips caller-owned shared K/V; explicit setup owns
  pool rebinding and poisoned-allocation failure handling. Reduced serving,
  all48 B2, and B32 records establish retained IDs/program counts and dormant
  state. The maximum-context run reaches 262144 in all 48 QSA layer/rank
  observations after 8196 prefill replays and one decode, with finite terminal
  logits on all four ranks.
- Performance verification: Clean public 128/128 generation reports
  7038.672 ms TTFT and 138.754269 ms/token, or 7.206985 tokens/s/user.
  Its 127 steady steps have zero token/position/page uploads, captures,
  releases, or resets. Independent CSV sums reproduce 11.2568075 ms reduced
  model, 0.42926325 ms split greedy, and 2.88656275 ms force-argmax. Sampler
  dominance is not supported. The separate model-only measurement is correctly
  labeled, and historical C128 layer estimates are not represented as current
  C262144 wrapper timing.
- Repository state: Plugin checkout is clean at
  `042a94863ad27d88207bcf71abdf67122c6f4c27`; tt-transformers checkout is clean.
  Tracy's stage-owned local commit is recorded. Unrelated untracked
  fused-decoder artifacts were untouched. Parent stage checkpoint is pending
  as required after review, and must isolate stage-owned files.

## Residual Risk

- The stage establishes the stated text autoregressive generator contract on
  the tested four-chip Blackhole mesh, not vLLM/server correctness. The pinned
  serving ledger has 63 rows, 42 supported generator equivalents and 21
  documented unsupported declarations, with no missing rows. KV-only prefix
  caching is excluded by recurrent state; retained-state continuation passes.
- Full48 B32 predictions derive from one correlated AIME reference. Existing
  controls cover request/state behavior and the required top-k bars, rather
  than broad benchmark accuracy. Long-context acceptance proves execution and
  capacity, not semantic quality over 262144 tokens.
- Native floating reduction changes have scoped regression and full-model
  evidence; this is not an exhaustive proof for every possible NaN payload,
  signed-zero pattern, or downstream kernel consumer.
- Clean-pass permits the stage's prescribed local commit/metadata closure.
  It does not authorize pushes, a precision-policy change, or promoting any
  historical failure into a pass.
