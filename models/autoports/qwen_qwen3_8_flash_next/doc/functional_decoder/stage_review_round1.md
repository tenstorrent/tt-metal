# Stage Review

Verdict: more-work-needed

Independent review, round 1 of at most 3, of the functional-decoder stage for
`Qwen/Qwen3.8-Flash-Next`. Reviewed the live `main` worktree based on
`d7e788cd88199501a8546ddb7bc381a601af47a9`. This verdict concerns the decoder stage
only. One required performance-evidence gate remains open. I found no additional
required implementation repair in the inspected runtime, numerical, trace-safety,
Watcher, or capacity evidence.

## Required Work

- **P2: Produce the required clean timing measurements separately from allocation-tracking and runtime-audit runs.**

  **Evidence:** The active [tt-enable-tracing skill](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/tt-enable-tracing/SKILL.md:207)
  explicitly requires: “Collect performance numbers separately with tracking and
  diagnostics disabled, recording the settings for each run.”
  [profile_trace_decoder.py](../../tests/profile_trace_decoder.py) lines 69–70
  rejects allocation tracking being disabled; line 91 records it as enabled.
  Lines 259–261 execute the measured replay inside `runtime_audit()`, and line
  278 records that audit. [report_trace_performance.py](../../tests/report_trace_performance.py)
  lines 29–30 requires both audit and allocation tracking to be enabled.
  All six final `tracy/layer{0,1,3}_{prefill,decode}_device_trace_v3/profile.json`
  records contain `allocation_tracking: true` and `runtime_audit: true`, consistent
  with their capture provenance. The final README numbers and
  [performance_evidence_offline_v3.json](performance_evidence_offline_v3.json)
  are derived from these runs.

  **Why this matters:** These artifacts substantiate instrumented trace execution
  and output/state equality, but do not satisfy the explicitly required clean
  performance regime. This is a missing gate, not a claim that the reported native
  device intervals have been proved inflated or incorrectly converted. Native
  timing, CSV units, and the existing numerical comparisons otherwise agree.

  **Required next step:** Add a clean timing mode that verifies allocation tracking
  and diagnostic/audit modes are disabled and records their effective settings.
  Run this in a fresh process, retaining the device profiling needed to measure
  the trace. Keep eager/replay output and state comparisons outside measurement.
  Update the offline reporter so final performance acceptance requires the
  explicit clean regime rather than the current tracking-enabled condition.
  Recapture the six representative layer 0/1/3 prefill/decode measurements, generate
  the corresponding CSVs and human reports, and update final performance bindings,
  README, and work-log references. Preserve the current tracking-enabled runs as
  safety/correctness evidence. This finding does not require repeating unrelated
  full-context, numerical-control, Watcher, or capacity runs if their runtime
  sources remain unchanged. A fresh diff-only round-2 review should inspect this
  fix, the six replacement measurements, and affected gates.

## Other Concerns

- **Fresh contribution sensitivities are controlled, not literal strict passes.**
  I accept their classification for this decoder stage after inspecting the
  independent controls and the failing rows. Every accepted literal whole-output
  row retains PCC >= 0.995; the required literal resumed-prefill and decode
  contribution gates also pass. Some fresh residual-subtracted contribution rows
  do not. The changed-accumulation and actual-packed-lookup HF controls reproduce
  those specific deviations, while TT/control comparisons pass. They neither
  force router choices nor inject HF state into TT. This is model-specific
  numerical evidence allowed by the contract; it is not the skill's automatic
  unchanged-HF/same-settings exemption. The explicit
  `literal_hf_strict_passed=false` records and
  [limitations.json](../eval_readiness/limitations.json) correctly preserve that
  distinction. It remains a downstream full-model accuracy risk.

- **The 262144 claim is a state-context claim with reduced prefix computation.**
  The long runs exercise the actual stateful attention/PLE paths on all prefix
  rows and then the complete decoder at the final position. They do not prove
  full MoE computation at every one of 262144 prefix rows. The functional-decoder
  skill permits a reduced layer harness; source inspection supports the
  composition here because the omitted decoder/MoE operations are token-local
  and use the same bounded physical chunks already exercised by public decoder
  tests. I accept this scoped gate, as separately represented in
  [context_contract.json](../context_contract.json), without promoting it to a
  full-model or complete long-prefix result.

- **Capacity results apply to their recorded allocation profiles.** The PLE
  single-request full-prefill maximum of 4768 and normal batch maximum of 5 have
  concrete boundary and physical evidence. They are separate profiles, not a
  guarantee that maximum length, maximum batch, arbitrary live traces, and every
  allocation history fit simultaneously. Watcher's smaller batch limit is
  correctly treated as instrumentation overhead. This does not justify reducing
  the supported 262144 state-context contract.

## Hard-Check Gaps

- The performance validator enforces the opposite of the required timing regime.
  Thus its passing result cannot close the clean-performance gate. Required Work
  above covers the correction; merely changing the README label would not close it.
- The numerical and capacity summaries deliberately leave some decisions to
  review. I checked the row-specific controls, source allocation rules, physical
  bounds, and exact source-transition constraints rather than treating summary
  `passed` fields as verdicts. No new evidence format is required for those gates.
- Synthetic coverage contains four complete decoder cases for layers 0/3 and two
  PLE arithmetic cases, rather than a complete synthetic PLE decoder case. Real
  full PLE decoder evidence supplies the functional coverage reviewed here. More
  synthetic composition coverage would be useful for future CI, but is not an
  additional blocker for this stage.

## Anomaly Ledger

- **Observed anomaly:** GDN batch-32 fresh request 16 contribution PCC is
  0.9906083941 despite literal whole-output PCC >= 0.995.
  **Evidence:** [linear_final_batch32_v4.json](linear_final_batch32_v4.json),
  [gr_same_operand_rounding_cpu.json](gr_same_operand_rounding_cpu.json), and
  [gr_backend_sensitivity_cpu.json](gr_backend_sensitivity_cpu.json).
  **Affected path:** Attention gated-residual down projection followed by routing.
  **Control or comparison:** Changing only independent HF residual-down
  accumulation to FP64 reproduces approximately 0.99061060 against literal HF;
  TT/control at the implicated row is approximately 0.99998516. The exact dot
  product 2.1640585417408147 lies just below the BF16 midpoint 2.1640625. The
  independent FP32/FP64 result rounds to 2.15625 while the pinned CPU BF16 backend
  rounds to 2.171875, changing one selected expert.
  **Likely subsystem:** Backend rounding amplified by a discrete routing boundary.
  **Investigation performed:** Inspected the actual failing row, operand/rounding
  evidence, independent HF control implementation, strict short/resume cases, and
  literal decode contribution minimum 0.9999741912. No forced router selection.
  **Resolution:** controlled, within the scoped numerical qualification above.

- **Observed anomaly:** PLE fresh contribution failures occur in batch request 2,
  live-trace fresh lengths 65/129, and the retained 3707-token run.
  **Evidence:** [ple_final_batch4_controlled_v4.json](ple_final_batch4_controlled_v4.json),
  [ple_private_batch5_normal_v1.json](ple_private_batch5_normal_v1.json),
  [public_continuation_layer1_v7.json](public_continuation_layer1_v7.json),
  [ple_public_backend_matched.json](ple_public_backend_matched.json),
  [ple_public_hf_3707_private_v1.json](ple_public_hf_3707_private_v1.json), and
  [ple_prefill_backend_3707_matched_v1.json](ple_prefill_backend_3707_matched_v1.json).
  **Affected path:** Actual packed PLE lookup plus sensitive projections/routing.
  **Control or comparison:** Batch request 2 row 0 is 0.9949769974 versus literal
  HF; the actual-packed-lookup HF control reproduces 0.9949601293 and TT/control
  is 0.9999611974. Lengths 65/129 have literal failing contributions
  0.9949381948/0.9947543740; a uniform independent packed-lookup plus six FP64
  projection control reproduces failures at the same rows, with TT/control
  approximately 0.99995285/0.99995047. At length 3707 row 55, literal TT/HF is
  0.9947104757, packed HF/literal HF is 0.9947631043, and TT/control is
  0.9999113071. The control's separate literal failure at row 33 remains recorded.
  **Likely subsystem:** Deliberate storage approximation and accumulation
  sensitivity amplified by routing.
  **Investigation performed:** Inspected control construction, actual-cache
  parity checks, per-row values, retained strict failures, exact repeat checks,
  literal resume/decode gates, and the successful strict 4767/4768 boundaries.
  Controls run independent HF execution and fixed policies across cases; no
  per-row oracle substitution or HF-state injection into TT was found.
  **Resolution:** controlled, not an unchanged-settings original-HF exemption.

- **Observed anomaly:** The earlier QSA full-context run had global PCC
  0.9999747948 but row 105584 failed at approximately 0.99442369. A uniform FP64
  alternate fixed that row while introducing a different failing row.
  **Evidence:** Historical `qsa_final_device_full_boundary.json`,
  [qsa_blocked_projection_full_cpu_v1.json](qsa_blocked_projection_full_cpu_v1.json),
  [ordered_projection_device_v3.json](ordered_projection_device_v3.json),
  [qsa_ordered_device_full_boundary_v1.json](qsa_ordered_device_full_boundary_v1.json),
  [qsa_ordered_repaired_row_cpu_v1.json](qsa_ordered_repaired_row_cpu_v1.json), and
  [qsa_ordered_full_decoder_v1.json](qsa_ordered_full_decoder_v1.json).
  **Affected path:** Indexer projection rounding near the sparse selection cutoff.
  **Control or comparison:** The final projection implements the pinned reference
  accumulation order uniformly. CPU comparison covers all 262144 x 640 outputs;
  device coverage exercises Q=1..32 at ten starts with repeated replay. The new
  from-empty attention run checks all 262144 rows, minimum 0.9982089996, with no
  failures; repaired row 105584 is approximately 0.99997596. Final full decoder
  changed-input A/B/A/B replay has minimum whole/contribution PCC
  0.9999422431/0.9999366403 and exact replay/state checks.
  **Likely subsystem:** Accumulation order at a discrete top-k boundary.
  **Investigation performed:** Read ordered projection kernels, source integration,
  failed controls, full per-row summaries, and final boundary checkpoint lineage.
  The accepted export comes from the successful normal full run; recovery-origin
  prefix and HF-prefix-window records remain diagnostic history.
  **Resolution:** fixed.

- **Observed anomaly:** The original chunk-input implementation accumulated
  programs/allocations across varied lengths, undermining capacity extrapolation.
  **Evidence:** `chunk_input_native_growth.json`,
  [chunk_input_final_v6.json](chunk_input_final_v6.json),
  [chunk_output_final_v5.json](chunk_output_final_v5.json), and generic input/output
  kernels under `tt/`.
  **Affected path:** Public arbitrary-length prefill and trace-live helper paths.
  **Control or comparison:** Historical native slicing added 338 programs and
  5537792 bytes over 324 cases. The replacement reader/writer evidence covers
  644 cases each, all tail counts 1..32, with three fixed programs and 49152 bytes
  without growth; negative and output-canary checks also pass.
  **Likely subsystem:** Shape-dependent program/cache allocation.
  **Investigation performed:** Inspected runtime offset/count use, padding masks,
  writer placement, fixed signatures, and public changed-input/resume/interleaved
  trace tests for all three layer kinds.
  **Resolution:** fixed.

- **Observed anomaly:** The full advertised PLE hidden input cannot coexist with
  the resident representation, and length 4769 / normal batch 6 fail allocation.
  **Evidence:** [final_capacity_qsa_closure_v4.json](final_capacity_qsa_closure_v4.json),
  [final_batch_capacity_qsa_closure_v4.json](final_batch_capacity_qsa_closure_v4.json),
  [capacity_contract_final_assessment.md](../../tests/capacity_contract_final_assessment.md),
  [ple_public_hf_4768_private_v1.json](ple_public_hf_4768_private_v1.json), and
  [ple_public_hf_4767_private_v1.json](ple_public_hf_4767_private_v1.json).
  **Affected path:** Full public PLE prefill and tuple batching on this device.
  **Control or comparison:** Advertised-length BF16 hidden input alone requires
  5368709120 bytes versus 514902528 bytes observed free after resident model/state.
  Actual 4768 full prefills repeat successfully, and 4767/4768 literal per-row
  whole/contribution checks pass. The fresh 4769 failure identifies the required
  contiguous allocation and smaller available per-bank extent. Residue checks
  cover all 32 tails with stable post-probe baselines. For batch 6, the measured
  resident, trace, state, and simultaneously live expert buffers yield a lower
  bound 12072320 bytes above total DRAM before counting other allocations.
  **Likely subsystem:** Resident storage, contiguous allocation, and overlapping
  activation/trace lifetimes.
  **Investigation performed:** Read source buffer lifetimes, physical arithmetic,
  successful/failing boundary records, fixed-program proof, and the distinction
  between proxy residue checks and complete boundary executions. The reduced
  long-context runs still establish the advertised state context.
  **Resolution:** controlled for the explicit allocation profiles; no universal
  optimality or simultaneous maximum-length/maximum-batch claim accepted.

- **Observed anomaly:** The ordinary host Tracy capture overflowed source-location
  capacity, and the streaming-profiler option was unsupported on the recorded
  firmware; final capture instead uses native whole-trace intervals.
  **Evidence:** README performance history, six final v3 capture/report directories,
  native trace protocol/parser code, CSVs, and provenance.
  **Affected path:** Performance instrumentation, not decoder numerical execution.
  **Control or comparison:** Native warm/measured trace markers, recorded actual
  110 participating cores, successful drains, raw CSV conversion, and eager/trace
  output plus all-state equality. Independent CSV inspection reproduces the
  reported milliseconds. These are aggregate trace measurements, not invented
  per-op timings.
  **Likely subsystem:** Host instrumentation capacity and firmware support.
  **Investigation performed:** Inspected capture/report harnesses, protocol,
  profiler scope, units, report provenance, and archived CSV payloads.
  **Resolution:** controlled for the capture method. The separate enabled-tracking
  problem below remains open.

- **Observed anomaly:** All final timings were collected with allocation tracking
  and runtime audit enabled despite the required separate clean regime.
  **Evidence:** Required Work above and all six final profile records.
  **Affected path:** Final warmed prefill/decode performance gate.
  **Control or comparison:** Safety runs and profiler conversion checks pass, but
  no final clean timing counterparts exist.
  **Likely subsystem:** Harness/validator contract selection.
  **Investigation performed:** Compared the active skill with harness conditions,
  measured contexts, actual profile fields, provenance, and reporter acceptance.
  **Resolution:** more-work-needed.

- **Observed anomaly:** Some accepted numerical/Watcher evidence predates final
  formatting and the QSA-only constructor integration change.
  **Evidence:** `plans/format19_transition_v1.json`, exact source-transition
  validators/snapshots, `final_gate_bindings.json`, and
  [final_watcher_summary_format_v2.json](final_watcher_summary_format_v2.json).
  **Affected path:** Evidence-to-final-source identity.
  **Control or comparison:** The recorded clang-format transition is confined to
  formatting and braces around existing single statements. The QSA constructor
  integration has an exact two-replacement proof and is unreachable for linear
  layers 0/1. Shared runtime hashes are constrained; historical hashes are retained.
  **Likely subsystem:** Evidence provenance after source cleanup/integration.
  **Investigation performed:** Read final sources and transition artifacts; hashed
  all 26 final runtime files against final bindings. Watcher summaries explicitly
  distinguish semantic closure from byte identity.
  **Resolution:** controlled; not a claim that the old measured binary is the
  byte-identical final binary.

## Scope Inspected

- **Goal/skills:** [review_request_v1.md](review_request_v1.md); active stage-review,
  functional-decoder, tt-enable-tracing, and tt-device-usage skills; relevant
  model-bringup startup requirements and trace-correctness guidance.
- **Implementation:** All 26 runtime files under `tt/`, including public
  `FunctionalDecoder`, GDN recurrence/projection kernels, PLE hash/full packed
  embedding, QSA paged/selected cache and ordered projection, routed experts,
  generic chunk reader/writer, and setup/runtime separation. Compared the pinned
  configuration and relevant installed `qwen4_exp` HF reference implementation,
  including layer-kind normalization, gated residuals, and PLE ordering.
- **Correctness and trace evidence:** Current strict short, public continuation,
  and batch probes for layers 0/1/3; actual failed rows and independent control
  implementations; changed/repeated replay, state restoration, page-table remap,
  history-negative controls, trace-live independent requests, exact corruptible
  output acknowledgments, and the allocation-tracker positive control. The
  representative traced runs use tracking with program-cache allocations included.
  Runtime audits reject Torch and host conversion paths. No broad capture/forward
  allocation exemption was found.
- **Long context and physical limits:** GDN/PLE full-context records and reduced
  harnesses; QSA successful full per-row run and final full-decoder bridge; strict
  4767/4768 public PLE records; capacity source reviews, residue evidence, actual
  failure records, and normal batch-5/batch-6 evidence. GDN prefix minimum is
  0.9999791980; PLE prefix attention-contribution minimum is 0.9957735538. Their
  final complete decoder contribution minima are 0.9999969599 and 0.9978260896.
- **Watcher and synthetic evidence:** Rehashed and rescanned all three final
  Watcher logs, checking complete dumps/detach and no failure-pattern hits;
  inspected separate-profiler provenance and synthetic six-case summary. No
  failures/errors/skips in that synthetic summary.
- **Performance and provenance:** Six final v3 raw/profile/CSV/human reports,
  capture/report code, offline gate, source bindings, formatting/dependency
  transition constraints, final inventory v3 and owner semantic inspection v2.
  Owner reports were treated as claims. Independently checked archive manifest
  identities/counts and ten selected compressed payloads, including all six final
  performance CSVs and representative numerical records, against original bytes.
- **Repository/check scope:** All staged changes were confined to the model root.
  Complete staged precommit log `/tmp/qwen_all_staged_pre_review_hooks_v3.log`
  reports exit 0. Local checkpoint commits are intentionally pending passing
  review under the supplied run plan. No push is authorized.
- **Commands:** Read-only `rg`, `cat`, `sed`, `nl`, `wc`, Git status/identity/diff
  inspection, and small Python standard-library JSON/CSV/hash/archive analyses.
  No hardware commands, TT imports/device opening, servers, resets, or new tests
  were run. The only review-created file is this report.

## Residual Risk

- This is artifact/source review, not an independent hardware rerun. Recorded
  correctness and safety evidence is extensive but finite. Public caller
  preconditions still include valid positions, page mappings, and sufficient
  state capacity; arbitrary invalid metadata is not a supported execution claim.
- Packed lookup and rounding-sensitive routing can accumulate differently across
  a complete model. The accepted decoder controls do not establish downstream
  generation quality or evaluation accuracy. No generated text exists in this
  stage, so full-model qualitative or serving conclusions are not implied.
- Whole-trace timings are batch-1 representative functional measurements, with
  the clean-regime replacement still required. They establish neither optimized
  performance nor individual operator attribution.
- Full-model residency, all-layer execution, serving request lifecycle, and
  multi-device behavior are later stages. No conclusion here extends the tested
  single-device decoder profiles to those scopes.
