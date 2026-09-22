# AutoFix: repeated startup and worker cleanup

## Starting Evidence

See `AUTOTRIAGE_repeated_startup.md` for exact artifacts and source paths.
Failing command: reduced `tests/run_vllm_stage.sh --stages serve
--sampling-profile smoke`, followed later by the same control with
`--no-async-scheduling`; QWEN_VLLM_TEST_LAYERS=0,3, TP4 Ring1x4, context262144.
Both failures occur in `open_mesh_device` before model load.

## Hypothesis Experiments

1. **Runner forcibly killed the successful server and caused next startup failure.**
   Inspection: packaged `_shutdown`, `reduced_server_runner.log`, and successful
   raw server shutdown. Result: SIGTERM wait completed, UMD closure completed;
   no forced-kill marker. Verdict: refuted for this normal successful run.
   The API-only fallback still does not prove all descendants exited after
   exceptional startup. No runner changes made.

2. **Framework worker shutdown fails to explicitly close model/mesh.**
   Inspection: `EngineCore.shutdown -> UniProcExecutor.shutdown ->
   WorkerWrapperBase.shutdown -> WorkerBase.shutdown` reaches a no-op because
   TTWorker overrides only `__del__`. Verdict: verified lifecycle defect;
   causal relation to heartbeat remains uncertain. Focused experiment: host
   fakes invoking the real extracted lifecycle methods, holding worker/runner
   references so Python destruction cannot stand in for explicit close. Assert
   optional model trace release precedes mesh close; repeat shutdown/destructor
   is harmless; partial initialization still closes an existing mesh; missing
   model close is supported. Runtime serve/stop/reopen remains necessary.

   Implemented in adjacent vLLM `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py`:
   explicit shutdown drains the existing async controller, calls optional model
   close, and attempts mesh teardown in `finally`; ownership is detached so
   repeated shutdown/destructor calls cannot double-close. `__del__` delegates
   best-effort. Partial init no longer skips mesh close. Successful mesh close
   has distinct begin/end log messages. The parent's fabric payload change is
   preserved. The coordinating agent owns Qwen adapter close forwarding.

   Exact focused command from tt-metal root:

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python_env/bin/python -m pytest -q \
     -p no:cacheprovider -c /dev/null \
     --confcutdir=../vllm/plugins/vllm-tt-plugin/tests \
     ../vllm/plugins/vllm-tt-plugin/tests/test_worker_shutdown.py
   ```

   The before command omitted `-p no:cacheprovider`; it ran the same test file.
   `worker_shutdown_host_before.log`: 5 failed, 1 passed. Explicit framework
   shutdown performed no cleanup; destructor omitted async/model cleanup;
   partial init skipped mesh close. `worker_shutdown_host_after.log`: 6 passed.
   Tests execute unchanged lifecycle method bodies extracted with AST from real
   framework/base/worker sources, with fake collaborators and a retained runner
   cycle. They import neither TTNN nor target model modules. This verifies call
   order and ownership behavior, not hardware completion.

3. **Foreign telemetry intervened between runs.**
   Inspection: supervisor event at 00:37:24 after UMD close at 00:37:05 and before
   mesh open at 00:37:41. Source logs TERM attempts without joining processes.
   Verdict: plausible confounder, unverified cause. Proposed experiment: record
   timestamped host/container PID mappings and KMD owners across a bounded
   restart; request an operator-controlled telemetry-free interval if needed.
   No foreign process signals or service mutations performed.

4. **Unresolved base/Metal Ethernet firmware transition issue.**
   Inspection: two identical failing stop/heartbeat transitions, older and newer
   heartbeat values, successful bounded reset plus exact mesh smoke. Verdict:
   unresolved. Proposed experiment after recurrence: preserve the live process
   and obtain explicit-device Ethernet status/callstacks with the device-usage
   procedure; existing triage skipped all those checks. Capture run flags,
   firmware state, core mappings, current PCs, and both heartbeat samples before
   reset. Do not infer cause from reset recovery or lower model parameters.

## Final Status

The worker lifecycle defect is repaired with a failing-before/passing-after host
regression. The cause of the original heartbeat symptom remains unresolved;
later finite shutdown/reopen checks are recorded below.
No device command, server launch, process signal, or telemetry mutation was run
by this investigator. Parent-owned recovery produced another exact Ring1x4
`MESH_SMOKE_OK` at 00:41:00-01 (`control_recovery_mesh.log`), before this worker
change; it is not verification of the repair.

`git diff --check` passed in both checkouts. No C++/CMake files changed, so no
build is required. Ruff check/format could not run because Ruff is unavailable
in the provided Python environment. Black check passed for the new test file;
whole-worker Black check requested unrelated changes to existing Python 3.13
assert formatting and the parent-owned fabric block, while leaving the lifecycle
hunk unchanged. Those unrelated formatting changes were not applied.

Native atexit already attempts RISC cleanup, but does not make explicit worker/
mesh teardown redundant. Runtime evidence now provided by the coordinating agent:

- `packed_sampling_trace_failure.log:333,348`: both explicit mesh-close markers
  after an allocation-tracker exception at 00:52:43 UTC; coordinator reported no
  remaining stage owner/process. `explicit_shutdown_reopen.log` proves exact
  Ring1x4/8192 mesh open/close **without reset** at 00:54. An initial probe had
  invalid constructor kwargs and failed before device open; the corrected
  attribute-assignment probe is the successful evidence.
- `reduced_final_server.log:276-277`: both explicit close markers during normal
  idle SIGINT shutdown at 01:04:39, followed by UMD close at 01:04:40.031. The
  subsequent full32 launch opened the same mesh without reset.

Thus explicit exceptional and normal worker shutdown have runtime proof, and
reopen without reset passed. These finite runs do not conclusively rule out
intermittent firmware or telemetry interference as the cause of the earlier
heartbeat failures. A separate API-parent-only cancellation defect orphaned an
EngineCore during model startup; see `AUTODEBUG_startup_cancel.md` for that new
lifecycle diagnosis. This investigator performed no hardware operations.
