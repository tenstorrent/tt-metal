# Stage Review

Verdict: clean-pass

Fresh independent follow-up, round 2 of at most 3, for stage 1 functional-decoder,
`Qwen/Qwen3.8-Flash-Next`. This is a diff-only review of the round-1 finding,
the three changed profiling/test files, and their replacement evidence. It does
not re-review or extend the original stage scope.

## Required Work

None. No new correctness, contract, or missing-evidence defect was found in the
reviewed changes.

**Closed — round-1 P2: Produce the required clean timing measurements separately
from allocation-tracking and runtime-audit runs.**

- [profile_trace_decoder.py](../../tests/profile_trace_decoder.py) checks both the
  effective tracker flags and explicit zero environment settings before device
  setup. `--clean-timing` selects `nullcontext` for every former runtime-audit
  wrapper. The measured region contains trace execution and synchronization;
  output/state readback and equality checks occur after profiler drain. Native
  profiler instrumentation remains enabled to measure the trace.
- All six `tracy/layer{0,1,3}_{prefill,decode}_device_trace_v4/profile.json` records
  have `measurement_regime=clean_timing`, `allocation_tracking=false`,
  `allocation_diagnostics=false`, `runtime_audit=false`, explicit allocation
  tracking/tracebacks values of `0`, and Watcher unset. Each actual capture argv
  contains `--clean-timing`. The twelve capture/report provenance records are
  finished with exit code 0; their argv match the command plan, timestamps show
  serial separate invocations, and their log hashes match the files.
- Both warm and measured replay outputs are finite and exactly equal to eager
  output. Every recorded state field also agrees: GDN convolution/recurrent
  state; PLE convolution/token history in addition to GDN state; and QSA K/V,
  raw history, compressed keys, position, and page table.
- [report_trace_performance.py](../../tests/report_trace_performance.py)
  rejects enabled or missing clean-regime flags, nonzero or missing allocation
  environment settings, enabled Watcher, unknown regimes, and failed replay
  comparisons. Historical tracked records retain their original validation
  branch. All six final performance gate definitions explicitly require the
  clean regime and disabled flags/settings.
- [clean_timing_cpu_v1 provenance](provenance/clean_timing_cpu_v1.json) binds the
  three current changed source files. Its log records 10 passing tests. Small
  independent in-memory checks also confirmed rejection of 10 enabled/missing
  clean-regime metadata cases, without importing TTNN or using devices.
- [performance_evidence_offline_v4.json](performance_evidence_offline_v4.json)
  records successful native parsing and exact report regeneration for all six
  runs. I independently checked source/artifact hashes, both native replay
  endpoint inventories against the recorded 110-core markers, and recomputed
  the measured first/last intervals from the raw CSV clock frequency. The
  resulting milliseconds agree with the summaries, filtered report CSVs,
  human-readable tables, README, and work log:

  | Layer | Prefill, 32 tokens (ms) | Decode after 32 tokens (ms) |
  |---|---:|---:|
  | 0 | 424.137608889 | 47.199168889 |
  | 1 | 455.555837037 | 48.118907407 |
  | 3 | 969.702974074 | 249.930357037 |

- All 26 current runtime source hashes equal those recorded in the previous v3
  captures and the v4 captures. The three preserved `*.before.txt` files match
  the v3 source hashes. Historical v3 records and bindings remain available as
  instrumented correctness/safety evidence; they are not the final timing rows.

## Other Concerns

The three previous concerns retain their round-1 dispositions. They are closed
as stage blockers, with these limitations still applicable:

- **Fresh contribution sensitivities remain controlled, not literal strict
  passes.** No runtime change or new evidence here changes the prior numerical
  assessment. `literal_hf_strict_passed=false` and the distinction between
  deliberately changed HF controls and unchanged literal HF remain documented
  in [limitations.json](../eval_readiness/limitations.json) and the README.
  Downstream full-model accuracy remains unproven.
- **262144 remains the scoped state-context result with reduced prefix
  computation.** The current README and [context contract](../context_contract.json)
  continue to distinguish reduced stateful prefix work from complete decoder/MoE
  computation at every prefix row. This review does not promote that claim.
- **Capacity remains specific to the recorded allocation profiles.** PLE's
  prepared full-prefill range through 4768 and normal batch maximum of 5 remain
  separate profiles, without a claim that their maxima and arbitrary live
  traces fit simultaneously.

## Hard-Check Gaps

None introduced by this change. The offline performance record predates later
whole-manifest bookkeeping and therefore has a different manifest digest. Its
six performance gate paths, predicates, and required artifact sets still match
the live definitions exactly. The latest
[pre-review inventory](pre_review_gate_inventory_v4.json) matches the current
manifest digest and records 48 satisfied declared checks, with only independent
review and local checkpoint commit pending. The owner inspection is not used as
an independent verdict.

## Anomaly Ledger

- **Observed anomaly:** Round-1 final performance used allocation tracking and
  runtime audits.
  **Evidence:** Preserved v3 profiles and [round-1 review](stage_review_round1.md);
  the six v4 profiles, actual invocation records, and native CSV/report results.
  **Affected path:** Representative complete-layer prefill and traced decode
  timing.
  **Control or comparison:** Unchanged runtime and shapes, separate clean
  processes, exact eager/replay output and state comparisons outside timing.
  **Likely subsystem:** Measurement regime.
  **Investigation performed:** Source diff, effective settings, invocation and
  source identity checks, replay protocol and native interval reconstruction.
  **Resolution:** Fixed. V4 supplies clean timing; v3 remains historical safety
  and correctness evidence.

- **Observed anomaly:** Capture logs report a one-chip P300/custom-cluster
  configuration and an unknown motherboard using the bus ID as tray ID.
  **Evidence:** All six v4 capture logs and the recorded setup explanation in
  [work_log.md](work_log.md).
  **Affected path:** Device discovery metadata.
  **Control or comparison:** The same authorized single-visible-chip setup and
  custom mesh descriptor were used previously; all new captures complete with
  the expected device and 110-worker inventory.
  **Likely subsystem:** UMD/platform discovery.
  **Investigation performed:** Compared recorded settings, warning text, final
  process status, and native worker inventories.
  **Resolution:** Controlled for this single-device scope; no new device-health
  failure appears in these runs.

- **Observed anomaly:** Human reports label the aggregate operation unclassified
  and show architectural maximum 120 cores in the header versus 110 in the row.
  **Evidence:** The six v4 TXT/CSV reports; the same warning exists in v3.
  **Affected path:** Generic performance-report presentation.
  **Control or comparison:** The row is explicitly a whole-trace aggregate;
  110 cores are independently inventoried, and raw cycles reproduce its time.
  **Likely subsystem:** Report categorization and architecture metadata.
  **Investigation performed:** Inspected actual tables, CSVs, native endpoints,
  and the documented absence of per-op, utilization, or roofline claims.
  **Resolution:** Controlled. These presentation fields do not change the
  reported trace interval.

## Scope Inspected

- **Goal/skills:** [review_request_v1.md](review_request_v1.md),
  [review_request_v2.md](review_request_v2.md), the round-1 findings, and the
  installed `stage-review`, `tt-enable-tracing`, and relevant
  `functional-decoder` requirements under
  `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`.
- **Code:** The three changed files and their exact pre-change snapshots under
  `tests/diagnostic_sources/clean_timing_v1/`; supporting read-only inspection
  of the existing protocol, native report parsing, and tracker flag definitions.
  Runtime identity was checked by hash rather than re-reviewing all runtime code.
- **Artifacts:** Six complete v4 profile/report sets; capture/report command
  plan, execution inventory, provenance and logs; v3 provenance and preserved
  bindings; CPU and offline validation records; live gate bindings;
  `final_semantic_inspection_v3.json`; `pre_review_gate_inventory_v4.json`;
  affected README/work-log sections and retained limitation declarations;
  `/tmp/qwen_round2_staged_hooks.log`.
- **Commands:** Read-only `cat`, `sed`, `rg`, Git status/HEAD inspection, Python
  diffs, and small `python3 -B` JSON/hash/CSV/in-memory-validator analyses.
  No target tests, TTNN imports, hardware operations, servers, or resets were
  performed. The only reviewer write is this report.
- **Workspace/push state:** Live branch `main`, base
  `d7e788cd88199501a8546ddb7bc381a601af47a9`. Before this report, all 2226 staged
  paths were model-local and no unstaged changes existed. The supplied staged
  hook log passes. The run plan leaves the local checkpoint commit until after
  a passing review and authorizes no stage-agent push; this review made neither.

## Residual Risk

- Timings describe one warmed replay per layer/mode, batch 1, at the stated
  precision and shape. They include internal device gaps and native profiler
  instrumentation. They establish neither a latency distribution nor full-model
  or serving performance.
- The original controlled numerical sensitivities, reduced-prefix context scope,
  and allocation-profile capacity limits remain unchanged. The clean timing
  runs supplement the separate safety/correctness evidence; they do not replace it.
- Checkpoint creation and recording its SHA remain the stage owner's final
  administrative step after this passing review.
