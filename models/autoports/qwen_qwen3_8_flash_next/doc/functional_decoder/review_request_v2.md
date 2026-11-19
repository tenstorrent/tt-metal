# Independent functional-decoder follow-up review, round2

Use `$stage-review` as a fresh independent **diff-only** reviewer, round2 of at most3.
Same target and original contract as `review_request_v1.md`; branch main, live worktree
based on d7e788cd88199501a8546ddb7bc381a601af47a9, no commits/push yet.
Do not re-review the whole stage. Do not spawn another reviewer. Read-only local
artifact/source analysis; no hardware, TT device imports, servers, resets or long tests.
You may write only `doc/functional_decoder/stage_review_round2.md`.

Read the stage-review and tt-enable-tracing skills at the absolute paths in the first
brief. Check each previous finding closed/open with evidence and check changed files
for new defects. Reports are claims; inspect actual code/settings/native measurements.

## Previous Required Work and Other Concerns, verbatim

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


## Changes since round1

No runtime file changed. Three test/harness files changed:
- `tests/profile_trace_decoder.py`: explicit clean timing, runtime tracking/diagnostic
  guards and effective metadata, no runtime-audit wrappers in clean mode; equality
  comparisons stay outside measurement.
- `tests/report_trace_performance.py`: validates clean mode flags/settings while
  retaining the historical tracked-record branch. Final gates explicitly require clean.
- `tests/test_trace_profile_cpu.py`: enabled/missing diagnostic regime and bad-state
  negative coverage;10 tests pass in `clean_timing_cpu_v1`.

Exact pre-change files are `tests/diagnostic_sources/clean_timing_v1/*.before.txt`;
compare to current files for the diff. Final runtime source hashes remain the26
identities in `final_gate_bindings.json`.

Six fresh serial measurements and six report conversions: all layer0/1/3 prefill/decode
`tracy/layer{layer}_{mode}_device_trace_v4/`. Each capture uses `--clean-timing`,
`TT_METAL_TRACE_ALLOC_TRACKING=0`, `TT_METAL_TRACE_ALLOC_TRACEBACKS=0`, noWatcher and
retains native trace profiler measurement. Inspect actual effective flags, eager/replay
output/state equality, profiler protocol and report metadata. Exact executed argv/env
and closed status are in matching `provenance/layer{layer}_{mode}_device_trace_v4{,_report}.json`.
`plans/qwen_perf_v4_commands.json` and `qwen_perf_v4_execution.json` provide serial plan
and execution inventory. All captures/reports are closed before this brief is dispatched.

`performance_evidence_offline_v4.json` reparses/re-renders all six records successfully.
`final_gate_bindings.json` points to v4 with explicit clean-regime/disabled diagnostics
predicates; old v3 gate definitions are retained in
`plans/performance_v3_historical_bindings.json`. README/work_log now use v4 final timings
and preserve old results as history. New archives and Markdown table views preserve
raw exact TXT/CSV/log/provenance bytes; no old evidence is overwritten. Latest owner
inspection/inventory records have the highest vN filename and are not verdicts.

## Commands producing new evidence

CPU: `python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.test_trace_profile_cpu`
through `tests.run_recorded`, bounded180s; exact argv in `provenance/clean_timing_cpu_v1.json`.
Hardware/report: twelve exact commands and environment settings in
`plans/qwen_perf_v4_commands.json`, individual `run_recorded` records listed above.
Offline: `tests.validate_trace_evidence --gate-manifest .../final_gate_bindings.json
--output .../performance_evidence_offline_v4.json`, exact argv in matching provenance.
Scoped format/hooks and final full staged precommit pass before dispatch.

Return the skill verdict/findings/anomaly/scope/residual-risk format. Original known
numerical/capacity concerns remain scoped as reviewed; this follow-up does not promote
those controls to literal passes or claim full-model accuracy. Local commit remains
pending a passing verdict. Never push.
