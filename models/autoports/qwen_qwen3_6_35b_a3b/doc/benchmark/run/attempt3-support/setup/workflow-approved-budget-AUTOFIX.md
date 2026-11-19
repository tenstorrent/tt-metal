# Benchmark accuracy HTTP timeout repair

## Diagnosis and failing contract

A timed benchmark's accuracy requests must remain eligible to complete until its
remaining client deadline. The parent already bounds the evaluator subprocess,
including queued requests, verification, scoring and reporting. The evaluator's
hardcoded 3600-second HTTP timeout could cancel an answer earlier: the preserved
bringup failure recorded `TimeoutError`, 31 of 32 responses, 3616.3302 client
seconds and a 7200-second client budget. This is a transport failure, not an
accuracy measurement. The independent timeout audit and source inspection agree
on the causal chain: `run.py` applies the parent deadline, but `evaluate.py`
passes a separate fixed timeout to `LocalChatCompletion`.

The preferred repair passes the remaining client budget to the HTTP backend.
Increasing the fixed constant would still fail for a larger configured budget;
disabling timeouts globally would remove standalone evaluation's finite bound.
Neither alternative is needed.

## Repair

- `runtime/benchmark_stage/run.py` calculates one accuracy subprocess deadline,
  rounds its remaining duration up to whole seconds and passes both together.
  HTTP begins after this calculation, so its timeout cannot precede the parent
  deadline. Existing parent cancellation and separate setup accounting remain
  authoritative.
- `runtime/benchmark_stage/evaluate.py` and `__main__.py` accept
  `request_timeout_seconds` / `--request-timeout-seconds`. Standalone evaluation
  defaults to 3600 seconds per request. Nonfinite, zero, negative and nonnumeric
  values are rejected before task loading. Positive fractions round up because
  upstream lm-eval casts the timeout to an integer; rounding down below one
  second would disable aiohttp's total timeout. Results retain the effective
  timeout in `benchmark_stage.request_timeout_seconds`.
- CPU tests cover shared and sequential execution with budgets above and below
  3600 seconds, standalone CLI forwarding, invalid inputs, and the real upstream
  HTTP backend's effective integer timeout and recorded metadata using a local
  fake endpoint.
- Both plugin manifests bump 0.1.24 to 0.1.25. Both catalogues, root README and
  the benchmark run contract document the behavior. These edits and tests are
  confined to this editable skills repository.

## Validation

No packages were installed. The existing evaluation environment supplies
lm-eval; the existing Python environment supplies pytest through an appended
site-packages path. Automatic pytest plugins are disabled to keep these CPU
checks separate from device tooling.

```sh
env -u HF_MODEL PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ../benchmark-eval-env/bin/python -c 'import sys; sys.path.append("../tt-metal/python_env/lib/python3.12/site-packages"); import pytest; raise SystemExit(pytest.main(["tests/", "-q"]))'
python3 scripts/sync_review_plugin.py --check
git diff --check
claude plugin validate . --strict
claude plugin validate plugins/tt-model-bringup --strict
python3 scripts/check_plugin_versions.py 834cda7
```

- Full suite: **744 passed, 1 skipped**, 49.78 seconds.
- Package sync: passed; 15 review skills current. Imported-file packaged hashes
  remain current; changed runtime/reference files are plugin-authored.
- Whitespace check and both Claude validations: passed. The plugin validator
  reports the pre-existing optional missing-icon advisory.
- The installed Codex CLI has no plugin validation command.
- Initial full-suite invocation with ambient `HF_MODEL` retained: 738 passed,
  1 skipped, 6 failed. All six context-contract fixtures inherited the active
  model's cached 262144 context while expecting their synthetic 8192 context.
  Unsetting that lane-specific variable makes the unchanged tests pass; no
  context gate was modified. An earlier direct pytest invocation also found
  pytest absent from the evaluation environment, resolved by read-only reuse of
  the existing pytest installation as above.

## Remaining uncertainty

This repair has not been installed into the live 0.1.24 plugin or exercised by a
new device run. CPU tests establish deadline propagation and HTTP configuration,
not whether the pending real-model answer would complete within the full client
budget. The failed run and its incomplete measurements remain failure evidence.
No hardware access, new inference, device resets, process termination or remote
pushes were performed for this repair.

# Explicit owner-approved client-budget override

The default 1800-second client bound per configured batch remains unchanged.
The runner's owner subsequently approved a separately bounded attempt requiring
28800 client seconds. Previously `batch_plan` rejected any larger value before
launch, even with explicit owner authorization. The shared plan now accepts
`owner_approved_client_budget` with finite positive numeric `seconds` and
nonblank `approved_by`, `reason`, and `source` strings. A separately provided
`budget_seconds` must agree exactly. This is recorded provenance, not an
external approval authentication mechanism.

The runner and checker use that same plan and copy the approval into the summary;
the report prints its bound, approver, reason and source. The checker rejects
missing or changed approval provenance, an exhausted approved clock, and missing
measurements. Setup keeps its own bound, all measurement completeness checks
remain, and no partial-pass mechanism was added. The timeout repair above stays
intact. Only `plan.py`, the checker error wording, report rendering, focused
regressions and required documentation/catalogue/manifests changed. Both plugin
manifests are now 0.1.26.

Validation reuses the CPU-only commands above, with the version check based on
`f61cb046`. Focused tests initially passed 196 cases, followed by the full suite
with the additional provenance regression. Package sync, all 64 imported-file
hashes, whitespace checks and both Claude host validators passed; the existing
missing-icon advisory remains. No installed-cache changes, hardware work,
package installation or pushes were performed. Actual completion of the
owner-approved run is outside these CPU checks.

Full regression result: **766 passed, 1 skipped in 50.33 seconds**.
