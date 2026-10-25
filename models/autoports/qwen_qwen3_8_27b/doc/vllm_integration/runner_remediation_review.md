# Stage Review

Verdict: clean-pass

Independent review on 2026-10-25 of Stage 09 runner remediation for
Qwen/Qwen3.8-27B, on the live `exp26-qwen38-bringup` worktree relative to
implementation checkpoint `425000e4bd982a48db43a6963e03359d07b509b0`.
This verdict accepts the user-authorized concrete refutation of the reported
critical context cap. It does not assert an aggregate checker exit of zero.
The original implementation acceptance remains recorded in `stage_review.md`.

## Required Work

None for this remediation. The retained host fixture and runtime evidence
establish that the reported 256-token cap was a fixture-scope error. Documentation
now reports the actual aggregate exit 1 and its remaining advisory causes. No
runtime code changed, and no runtime remeasurement is required by this correction.

## Other Concerns

- The unchanged stage gate independently returns **1**, with zero critical
  context findings and seven agentic-serving advisories: six duplicate weight
  families and the configured KV-pool shortfall. Both the `agentic-serving`
  skill and `check_agentic_serving.py` explicitly classify explained shortfalls
  as advisory. README, work log and current telemetry retain that distinction.
- The pool is still 786432 tokens against the estimated 32-by-32768 working set
  of 1048576 tokens. Existing evidence describes a configured layout limitation,
  not a proven hard physical maximum or a reduction of advertised model context.
  The remediation does not claim the larger workload was measured.
- Existing limitations in the prior accepted review remain: the B32 long-prefill
  pause, compatibility sampling scope, preparation-only memory checkpoints and
  limited process visibility. None is changed or waived by this report.

## Hard-Check Gaps

- `check_context_contract.py:148` recursively checks every matching JSON key in
  selected artifact trees without considering the artifact's `scope`. Its
  `checked_files` includes `readiness_vllm` and excludes `tests`. Relocating this
  explicitly synthetic host fixture to `tests/evidence` therefore changes scan
  membership, but its unchanged contents and corroborating runtime evidence make
  this a legitimate evidence classification correction. This is not a general
  exemption for runtime configurations or failed hardware evidence.
- Text advisories remain for plugin policy tests and their copies in retained
  checker logs. The non-strict context checker itself returns zero for those;
  the aggregate exit 1 comes from the memory ledger advisories.

## Anomaly Ledger

- Observed anomaly: three critical context findings report 256 below 262144.
  Evidence: the supplied `09-09-vllm.check-1.log` is byte-identical to
  `readiness_vllm/runner_gate_reproduced.log`; all three findings name the host
  preparation fixture's event signatures.
  Affected path: artifact scanning, not serving configuration.
  Control or comparison: the fixture already states host-only AST extraction,
  fake tensor/trace operations and no hardware validation. Its original
  `preparation_events.md` description states the same. Both final B1 and B32
  server manifests specify 262144; API and EngineCore logs record native context;
  adapter initialization rejects context different from the checkpoint contract.
  Investigation performed: inspected scanner source and adapter context
  propagation; compared relocated bytes against the implementation checkpoint.
  SHA256 is `c5aec0171f39c95b35bbd59b98a3391ea557a7fe4b190949f3efbe361c3b7a3e`.
  Resolution: controlled by concrete counterevidence; fixture preserved unchanged
  under `tests/evidence`, provenance recorded, references updated.

- Observed anomaly: retained log warnings mention a 16384-token model length.
  Evidence: `readiness_vllm/plugin_state_after.log:89` and adjacent warnings.
  Affected path: plugin host policy tests using `_FakeModel` in
  `tests.test_chunked_prefill_policy`.
  Control or comparison: each warning appears inside a named passing pytest
  case; final production server logs record 262144.
  Investigation performed: read the surrounding test names, fake-model logging
  and actual final server configuration.
  Resolution: controlled; original warning evidence remains intact.

- Observed anomaly: aggregate checker remains nonzero after the critical finding
  is removed.
  Evidence: independent unchanged `09-vllm.check.sh` invocation returned 1;
  `runner_gate_remediated.log` records the same seven ledger advisories.
  Affected path: advisory acceptance/reporting.
  Control or comparison: installed skill and checker define explained weight
  copies and pool shortfalls as advisory, and the user allows concrete refutation.
  Investigation performed: inspected the original goal, current work log,
  checker exit aggregation and advisory implementation.
  Resolution: controlled and disclosed; no exit-zero claim is made.

## Scope Inspected

- Goal: persistent run `20261023-astra-medium-exp26-clean/09-09-vllm.prompt.txt`
  and original `09-09-vllm.check-1.log`, plus the supplied remediation contract.
- Skills: installed version 0.1.21 `stage-review`, `vllm-integration`,
  `agentic-serving`, `tt-device-usage`, and model-bringup startup/evidence rules.
  Enabled model-bringup and autodebug entries were checked in `.codex/config.toml`.
- Changed documentation, prior `stage_review.md`, fixture/provenance README,
  retained failure and remediation logs, context contract, adapter initialization
  and generator construction, checker source and plugin policy-test log.
- Both `final_b1_command.json` and `full_b32_performance_command.json`: all 44
  recorded source hashes in each match the current tree, including plugin and
  model implementation. The live diff contains no runtime implementation changes;
  the plugin worktree is clean.
- Final B1 raw and normalized benchmark artifacts: one completed request, zero
  failures, 4096 input/128 output tokens, concurrency one. Raw TTFT is
  445.5456119030714 ms; `1000 / mean_tpot_ms` is 36.24361633151652 tokens/s/user.
  Performance manifests disable tracking, watcher, profiler and host sampling.
- Current telemetry packet
  `telemetry/packets/b0421e58-420d-49e8-acf6-5bb002034c03.json` in the persistent
  run: schema/run/stage/attempt/kind/workload match its adjacent template exactly;
  TTFT and decode rate equal the raw-derived B1 values exactly. Device-time and
  roofline fields remain null with explicit missing reasons. No new measurement
  is implied by this reused evidence.
- Read all twelve serving qualitative completions and the six corresponding
  standalone controls/rendered prompts. Outputs remain coherent and topical;
  visible thinking text and length truncation are also present in controls.
  No new output anomaly invalidates the prior accepted qualitative evidence.
- Commands: read-only `git status/diff/show`, `cat`, `sed`, `rg`, short Python
  standard-library hash/JSON comparisons, and the unchanged host-only stage
  checker. No TTNN import, hardware operation, server, request or benchmark was
  launched. Only this review report was written by the reviewer.

## Residual Risk

This is a review of the remediation against the accepted implementation and
retained evidence, not a fresh execution of the full serving contract. It does
not establish every batch-by-context combination or remove the documented pool
shortfall. The concrete refutation is sufficient for the reported critical
finding; future work must continue to report advisory status accurately.
