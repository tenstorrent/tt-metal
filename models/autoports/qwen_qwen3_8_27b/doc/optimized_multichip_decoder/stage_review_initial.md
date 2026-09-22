# Stage Review

Verdict: more-work-needed

Independent read-only review of Stage 5, optimized-multichip-decoder,
Qwen/Qwen3.8-27B, on the live `mvasiljevic/qwen38-full-bringup` worktree
starting at `1623f9cb595dd1e87cb17abc6f3449838c1762ee`.
Reviewed model SHA-256:
`bb18fc368864fe622d202398c209958e85d47da46570a7333699b8de22d294e5`.
This verdict describes the evidence available during review; it does not
certify the subsequently repaired source or unfinished follow-up runs.

## Required Work

- P2: Remove the redundant identity casts introduced by short prefill.
  Evidence: `tt/multichip_decoder.py:553` unconditionally invokes
  `ttnn.typecast(x, configured_activation_dtype)` in the new `prefill_1d`
  branch. The selected policy and these inputs are BF16. Unlike the guarded
  inherited `OptimizedDecoder._linear`, this emits device work:
  `tracy/final_profile_l3/device0_prefill_perf_report.txt` rows 110, 133,
  139, and 144 show BF16-to-BF16 `TypecastDeviceOperation` with approximately
  7, 3, 7, and 6 microseconds of kernel time, plus dispatch gaps.
  Why this matters: these conversions do not change model semantics, add four
  operations per layer, and violate the optimize skill's final audit against
  unnecessary conversions. The affected short-prefill path has variable
  regressions in the reported host timings.
  Required next step: compare `x.dtype` with the requested dtype before
  casting. Validate both layer kinds and an affected stacked/batched case,
  including strict PCC and trace replay, and inspect a new short-prefill
  profile to confirm the identity conversions disappeared. Refresh final
  source/evidence associations after the edit.

- P2: Reconcile the final report and evidence index after the repair and
  supplemental validation.
  Evidence: README still headlines `after_default_*` and says validation is
  in progress. The installed-runtime `after_final_*` reports now exist with
  exit 0 and empty policy overrides. Their prefill medians are
  1.500125/1.184672/2.676882 ms and decode medians are
  .427397/.307933/.705262 ms for linear/full/stack, respectively; these differ
  from the README values. `final_validation_index.json` validates 51 older
  final artifacts and does not yet include these installed-default runs or
  the new prefill-trace evidence. The original full-attention short-prefill
  trace report has a 2.009047 ms median with substantial host outliers;
  `profile_trace_prefill_l3.json` instead reports .772044 ms and a device
  span of approximately .743738 ms. The owner is collecting a normal-run
  control for this discrepancy. The anomaly ledger still says final native
  validation is pending even though the 18-case final regression exists.
  Why this matters: the user requires final numbers from the final default
  path, actionable profiler advice resolved, and classified anomalies.
  Passing older artifacts must not silently certify changed source.
  Required next step: finish the targeted repair/control runs, identify their
  exact source and installed binaries, update the README headline and index
  to the final selected default, record the trace timing control and the GDN
  comparison's measured disposition, and close stale present-tense pending
  statements. Obtain a clean rereview before the local checkpoint commit.

## Other Concerns

- The new native factory signature changes an internal C++ symbol. Python's
  legacy fourth `core_range_set` argument remains accepted, while a fifth
  coordinate argument is added. Build/install evidence correctly includes
  both native components; consumers of old binary artifacts must rebuild.
- Multi-reader remote coordinates are explicitly rejected. Heterogeneous
  harvesting and other architectures are source-reviewed, not hardware
  validated. These limits are disclosed and do not contradict this local
  four-chip Blackhole target.
- The precision experiments do not use synthetic PCC to veto real-weight
  winners. BFP4 KV has a real raw-cache parity delta; output and replay gates
  remain active in diagnostic reruns, and those reports are excluded from
  final acceptance. The final BFP8 cache policy preserves the accepted state
  contract.

## Hard-Check Gaps

- The index checker validates model source hashes but relies on separate
  source and installed-library provenance files for dependencies. Those
  files were inspected; requiring a new runtime counter format is not
  necessary. They must remain consistent after the requested edit.
- The checker only explicitly asserts the absence of the ETH watcher-disable
  variable. The watcher launcher removes every `TT_METAL_WATCHER_DISABLE*`
  variable, and saved full-watcher cases show successful process teardown.
  This is sufficient existing evidence, rather than a new blocking test.
- Baseline runs predate the wrapper's exit-marker addition. Their successful
  recorded logs and coherent repeated one-reader controls support comparison;
  their JSON alone is not presented as final watcher-clean evidence.

## Anomaly Ledger

- Observed anomaly: identity BF16 prefill conversions.
  Evidence: the new branch and four profiler rows cited above.
  Affected path: selected bounded short prefill.
  Control or comparison: inherited projection path guards activation casts.
  Likely subsystem: model dispatch.
  Investigation performed: direct code and actual per-device table review.
  Resolution: more-work-needed.
- Observed anomaly: full-attention short-prefill trace host outliers.
  Evidence: `trace_prefill_l3_s128.json`, `profile_trace_prefill_l3.json`.
  Affected path: optional prefill-trace measurement.
  Control or comparison: .772044 ms profile replay versus 2.009047 ms
  original normal median; ordinary decode in the original run also has
  millisecond outliers.
  Likely subsystem: host timing/scheduling; the precise cause is not proven.
  Investigation performed: owner-supplied profile and retained raw samples;
  longer normal control was still follow-up work at review time.
  Resolution: more-work-needed until classified in the final evidence.
- Observed anomaly: native multi-reader mesh-coordinate assertion and
  padding-only reader storage assertion.
  Evidence: `AUTODEBUG_dram_mesh.md`, `AUTODEBUG_dram_tail.md`, native diffs,
  final 18-case mesh/cache/placement regression and real-weight model runs.
  Affected path: DRAM-sharded matmul.
  Control or comparison: per-rank data, readers 1/2/3, TP4 and offset TP2,
  three output geometries and three retained allocation generations.
  Likely subsystem: descriptor construction and output write assignment.
  Investigation performed: coordinate-aware adapter/binding inspection,
  physical-device lookup, writer drain and full-shard allocation inspection.
  Resolution: fixed for the supported local mesh.
- Observed anomaly: model gates pass but full-watcher process aborts on close.
  Evidence: `AUTOTRIAGE_watcher_os_teardown.md`, failed exit 134 records,
  saved ERISC assertion/PC evidence, router diff, passing repeat and four
  final full-watcher exit 0 cases.
  Affected path: fabric router teardown.
  Control or comparison: empty watcher mesh closes; traffic triggers sticky
  packet-tag exit assertion; repaired traffic runs close normally.
  Likely subsystem: owned-NoC packet-tag cleanup.
  Investigation performed: barriers, tag API, peer synchronization and
  kernel-exit assertion source inspected alongside recorded diagnosis.
  Resolution: fixed for the tested fabric configuration; assertions remain.
- Observed anomaly: profiler reports approximately 101% matmul utilization.
  Evidence: final tables, runtime reader attributes, `final_matmul_rows.csv`.
  Affected path: performance interpretation.
  Control or comparison: profiler assumes eight workers; native configured
  execution uses sixteen/twenty-four workers.
  Likely subsystem: profiler denominator.
  Investigation performed: native descriptor and accounting source review.
  Resolution: controlled; original tables preserved and corrected supplemental
  values are clearly separated.
- Observed anomaly: narrow conversion rejection, stale installed native
  library, wrapper exit 2, shared-memory warnings, and nanobind teardown
  diagnostics.
  Evidence: `anomaly_ledger.md` and associated failure/retry/control logs.
  Affected path: rejected split-projection candidate, native materialization,
  experiment wrapper, and test environment.
  Control or comparison: padded/sliced candidate passes but loses; both native
  components installed; frozen wrapper repeat exits 0; baseline timings
  reproduce after shared-memory cleanup; existing unit-mesh tests show the
  same shutdown diagnostics and exit 0.
  Likely subsystem: the documented isolated mechanisms.
  Investigation performed: source adaptation and recorded controls inspected.
  Resolution: controlled; failed artifacts are not final acceptance evidence.

## Scope Inspected

- Goal/skill paths: parent-provided original Stage 5 contract;
  `.agents/skills/stage-review/SKILL.md`, `optimize/SKILL.md`,
  `tt-enable-tracing/SKILL.md`, `tt-device-usage/SKILL.md`, and
  `tech_reports/LLMs/llms.md` section 4.
- Artifact paths: stage README/work log/anomaly ledger, collective and
  inter-layer contracts, 51 indexed final artifacts and their hashes/exit
  markers, context/capacity plan, native build/install/test logs, full watcher
  records, two 100-step stress reports, final and supplemental timing reports,
  six final profile families with per-device tables, performance accounting,
  212-row BFP4/LoFi geometry search, 60 topology-family results, precision and
  prefill matrices, raw-archive manifest, and AutoDebug/AutoFix/AutoTriage
  reports. Selected raw rows and source provenance were inspected directly.
- Code paths: model and runner diffs, inherited optimized projection and
  logical-tail paths, capacity/index/accounting/sweep/profile/watcher helpers,
  GDN comparison shim, native factory/header/nanobind/fabric teardown diffs,
  native regression tests, and mesh descriptor adapter/binding behavior.
- Commands run: read-only `git status`, `git diff`, `git diff --check`, `rg`,
  `sed`, `cat`, and Python 3 standard-library analysis over JSON/CSV/hash
  artifacts. The default `python` command was unavailable; `python3` was
  used. No tests, builds, TTNN imports, linked probes, or device operations
  were run by this reviewer. Only this review report was written.

## Residual Risk

- The 31-case model regression, strict states/cache checks, batches through
  32, two evolving 100-step stress cases, five max-context probes, four full
  watcher traffic cases, and native cache/placement tests are substantial
  recorded evidence. They do not establish arbitrary topology, concurrency,
  harvesting, or full-model/serving behavior outside this decoder stage.
- The reviewed inter-layer profile directly connects final residual add to
  the next input norm on all four devices. No collective or reshard is
  inserted at that boundary. Coherent sharded families were measured through
  the next consumer, with comparison-only gathers outside timing.
- No additional implementation defect was found in the native repairs or
  selected decode path. The redundant prefill casts and final evidence
  reconciliation above prevent clean-pass at this review point.
- After clean rereview, the stage owner must create the required local
  stage-owned checkpoint commit(s), record branch/SHA in the work log, and
  avoid pushing. Commit creation follows clean review under the skill; the
  currently uncommitted worktree is not independently used to fail review.
