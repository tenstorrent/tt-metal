# AUTOTRIAGE: repeated Ethernet startup failure

## Diagnosis

The observed failure is failure to return active Ethernet core Device 0 / virtual
29-25 to base firmware during mesh initialization, before any model layer loads.
The evidence does **not** establish why its heartbeat stopped. A separate,
source-proven lifecycle defect exists: `TTWorker` implements device cleanup only
in `__del__`, while vLLM calls its inherited no-op `shutdown()`. Fixing this gap
is warranted, but is not yet a demonstrated fix for the heartbeat failure.

Investigation used the local AutoTriage and AutoFix skills. No hardware commands,
server launches, signals, or implementation changes preceded this diagnosis.

## Triage Evidence

- `startup_heartbeat_failure.log:64-65`: at 00:26:20.661 UTC, heartbeat remains
  `0xdcba04c0` after the 20-second startup wait. Port/Rx/PCS are up, train status
  is `0x2`, postcode `0xc0dea000`, reset PCs `0x357c`/`0x9330`, reset `0x47000`.
- `control_startup_heartbeat_failure.log:64-65`: at 00:38:01.965, the same core
  and register signature recur, heartbeat now remains `0xdcba58c0`. This is
  `TTWorker.init_device -> open_mesh_device -> RiscFirmwareInitializer.reset_cores`,
  not a decode, sampling, allocation, or model-layer operation.
- `triage/summary.txt`: inspector data was unavailable; device checks were skipped
  because neither inspector nor explicit device selection was provided. This
  capture proves no RISC-V stop site, NoC/CB state, ARC health, or root cause.
- `reduced_async_server.log`: all 11 completion requests returned HTTP 200. API
  shutdown begins at 00:37:04, UMD close completes at 00:37:05.151, and application
  shutdown completes. `reduced_server_runner.log` says the API exited within the
  runner's SIGTERM wait. There is no SIGKILL evidence for this successful run.
- `recovery_mesh_tp4.log`: exact FABRIC_1D_RING on four devices initialized and
  printed `MESH_SMOKE_OK` at 00:28:31-33 after the recorded reset. Subsequent
  recovery logs likewise contain successful reset/list/mesh evidence; these are
  recoverability evidence, not causal experiments.
- `../pipeline-supervisor.log:172-175`: telemetry cleanup attempts at 00:37:24
  occur after successful serving exit and before the next mesh open at
  00:37:41.383. A further attempt occurs at 00:39:24 after failure.

## Source Evidence

Paths prefixed `../vllm` are the adjacent serving checkout. The packaged runner
is under `../codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4/runtime/readiness_check/`.

1. `tt_metal/llrt/llrt.cpp:557-617` samples the base-firmware heartbeat, repeatedly writes
   zero to the Metal active-ERISC run flag, barriers, and reads heartbeat until
   it changes or the 20-second timeout expires. This is an exit/heartbeat state
   transition, not a CCL producer/consumer packet count. Host produces the stop
   flag; Metal Ethernet firmware must yield/exit; base firmware produces a
   changing heartbeat. Which transition failed is not captured.
2. `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:215-242,377-388,464-486`
   calls the check during initial reset, before firmware launch completes. The
   detected bundle is 19.8.0; the generic minimum-version message (18.10.0) does
   not prove an outdated bundle.
3. Packaged `run_vllm_server.py:710-729` SIGTERMs only the API child, waits 15
   seconds, then kills only that PID on timeout. It does not directly account
   for EngineCore descendants and labels any completed wait "Terminated
   cleanly" without checking its return code. That is an observability/process
   ownership weakness, but the normal successful shutdown did not take its
   SIGKILL branch.
4. `../vllm/vllm/v1/engine/core.py:949-965,1027-1040,579-584` handles SIGTERM with
   SystemExit and invokes executor shutdown in `finally` after completed init.
   `uniproc_executor.py:135-137` delegates to `worker_base.py:206-208`, which calls
   worker shutdown. `WorkerBase.shutdown` at 170-172 is a no-op. The pre-fix
   `TTWorker` at `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py:498-509`
   overrides only `__del__`. Its single `suppress(AttributeError)` also skips
   mesh closure if `model_runner` was never assigned during partial init.
5. The destructor calls `close_mesh_device`, which explicitly closes submeshes
   and mesh and resets fabric. Native mesh close reaches
   `DeviceManager::close_devices` (`device_manager.cpp:680-750`), performing
   dispatch, fabric, profiler, and CQ teardown in order. Fabric teardown sends
   termination to master routers (`fabric_firmware_initializer.cpp:349-420`).
6. Native atexit is present: `metal_context.cpp:500-507` destroys contexts;
   `MetalContext::~MetalContext` calls `teardown`, whose RISC teardown waits for
   heartbeat and asserts cores. This compensates in part for a missing Python
   destructor, so "no Python close means firmware never stopped" is unsupported.
   However, `DeviceManager::~DeviceManager` calls `Device::close` directly; this
   fallback is not the ordered dispatch/fabric teardown in `close_devices`.
   UMD closure alone therefore does not prove explicit mesh shutdown occurred.
7. `../pipeline_supervisor.sh:57-73` stops telemetry servers and SIGTERMs found
   collectors. It logs "terminated" even if the signal fails and does not wait
   for collector exit. Its logs prove attempted intervention, not collector
   lifetime, device ownership, or a corrupting register write. The process
   namespace PID mismatch also prevents joining host and container PIDs by
   number alone.

## Downstream Effects

EngineCore startup exceptions, API startup failure, runner SIGKILL on the failed
launch, and a later C++ abort during MetalContext destruction are downstream of
the initial heartbeat timeout. The first launch's stale EngineCore is a cleanup
consequence; it cannot explain the initial timeout without preceding evidence.
Nanobind leak diagnostics support retained Python/native objects at interpreter
exit, but are not proof of an active fabric kernel or of heartbeat corruption.

## Proposed Fix

Implement explicit, idempotent `TTWorker.shutdown()` on the framework's existing
shutdown path: invoke an optional model `close()` while the mesh is alive, drop
the runner, then call the existing mesh close helper. Make `__del__` a best-effort
fallback to that method and handle partial initialization. The Qwen adapter must
forward `close()` to its canonical generator so trace release is explicit.
Prove ordering and repeat-call behavior with host fakes before using hardware.

Do not modify native heartbeat timeout, firmware flags, or installed runner
based only on this evidence. Compare fresh-reset serve/stop/reopen runs with
recorded explicit-close markers, then isolate foreign telemetry through the
operator if recurrence remains. These tests discriminate cleanup causality from
intervening activity; they must not be run concurrently with serving.

## Uncertainty

There are no valid live Ethernet call stacks or per-core run-flag samples. The
prior successful shutdown may already have completed native RISC teardown; the
00:37:24 telemetry interval is a real confounder. No telemetry service changes
or firmware fixes are justified by correlation alone. Heartbeat causality and
reliable restart behavior require runtime verification by the coordinating agent.
