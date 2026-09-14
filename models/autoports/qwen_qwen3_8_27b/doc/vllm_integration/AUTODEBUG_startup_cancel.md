# AutoDebug: orphaned EngineCore after startup cancellation

## Diagnosis

The packaged readiness runner owns only the API launcher PID. Cancellation
during readiness polling invokes its `finally` cleanup, but that cleanup sends
SIGTERM and then SIGKILL only to the API launcher. When the API has not finished
building its engine context, it may not propagate shutdown to EngineCore before
the 15-second limit. EngineCore then survives reparented to PID 1. The stage
wrapper has no independent descendant ownership or exit cleanup.

This is distinct from TTWorker's previously repaired explicit-close defect:
worker cleanup only runs when EngineCore itself reaches shutdown.

## Evidence

- `full32_server_runner.log`: KeyboardInterrupt was raised inside
  `_wait_for_server` at its polling sleep. The runner attempted SIGTERM of
  API PID 652301, timed out after 15 seconds, and sent SIGKILL. Its archived
  log tail ended at EngineCore 652449 `LOAD_LAYER 12`.
- `full32_cancelled_startup.log:69-132`: EngineCore 652449 continued through
  `LOAD_LAYER 63` without API readiness. The coordinating agent observed it
  reparented to PID 1 with device ownership after runner/API exit, and sent
  SIGTERM to this confirmed stage-owned process. That ownership observation is
  supplied by the coordinator; this investigator did not signal or probe it.
- The earlier work-log inference that SIGINT had no effect is superseded by
  the runner's KeyboardInterrupt traceback. The signal did interrupt the runner;
  the surviving EngineCore made cancellation appear ineffective.
- Packaged `readiness_check/run_vllm_server.py:199-250` launches an API child
  with a copied environment. Lines 985-1004 and 1095-1097 wrap readiness polling
  in `finally: _shutdown(server_proc, server_log)`. Lines 710-729 inspect and
  signal only `server_proc`. No descendant traversal or process-group ownership
  is present. Its serve-only SIGTERM handler is installed after readiness at
  731-755; this is another reason outer lifecycle handling must start earlier.
- Stage `tests/run_vllm_stage.sh` directly launches the packaged module and has
  no EXIT/INT/TERM cleanup. The installed package remains read-only.

## Proposed Repair

Add a stage-owned Python process guard invoked with `exec` by the shell wrapper.
Install signal handlers before launch, mint a fresh random launch marker, and
export that marker only into the runner's subprocess environment. The packaged
runner already copies environment into API children, and subprocess descendants
inherit it; the marker therefore survives reparenting.

On normal/error exit or cancellation, locate only same-user live `/proc`
processes with that exact marker. Revalidate identity and marker using pidfds
before SIGTERM, wait for a bounded interval, and report exact remaining PIDs
with a failing exit code. Do not SIGKILL leftover device processes, clear locks,
or reset devices. A surviving process needs preserved logs/triage and coordinated
recovery. Preserve unrelated processes and previous launches with different
markers. Forward cancellation to the runner first so its own graceful cleanup
can finish, then handle any owned survivors independently of their PPID.

## Focused Verification Plan

Use host-only dummy subprocesses and fake `/proc`/signal seams. Reproduce an
API-style parent that exits while its child remains alive; verify the guard
reclaims the marked orphan on ordinary exit and on SIGINT/SIGTERM. Keep a
separate sentinel with another marker alive throughout. Verify exact environment
matching, PID-identity revalidation, bounded reporting of a TERM-ignoring child,
and preservation of the original command exit status. These tests open no TT
devices and do not import TTNN. Runtime startup cancellation remains a separate
coordinator-owned check.

## Related Worker Repair Runtime Evidence

`packed_sampling_trace_failure.log:333,348` contains both explicit TTWorker
mesh-close markers after an allocation-tracker exception at 00:52:43 UTC.
`explicit_shutdown_reopen.log` records a successful four-device Ring1x4 / 8192
mesh open/close without reset at 00:54. `reduced_final_server.log:276-277` records
both worker-close markers on normal idle SIGINT shutdown at 01:04:39, followed
by successful UMD close. The subsequent full32 mesh opened without reset. These
confirm worker-close behavior for those runs; they do not establish the cause
of earlier intermittent Ethernet heartbeat failures.

## Repair and Verification

Implemented `tests/vllm_process_guard.py`, with the stage shell script using
`exec` to put the guard at its own PID. The guard prints its PID, runner PID,
and a new `QWEN_VLLM_LAUNCH_ID` on every invocation. Signals should target that
guard PID. It forwards cancellation as SIGINT so the packaged runner can reach
its `finally` even during readiness polling, allows 20 seconds for the runner,
then sends SIGTERM only to same-user processes carrying the exact launch marker.
It waits up to 30 seconds and reports any remaining PIDs. A survivor changes an
otherwise successful exit to failure. Original command failure and signal exit
statuses are retained. There is no guard SIGKILL or device/reset operation.

The guard is a Linux child subreaper, so orphaned grandchildren are reparented
to it and reaped rather than leaving zombies at PID 1. Ownership decisions still
require the launch marker; PPID or process names alone never authorize a signal.
Environment entry matching is exact. PID birth time and marker are revalidated
after opening a pidfd, and signals target that pidfd to exclude PID reuse races.
The standalone Python lacks `os.pidfd_open` and `signal.pidfd_send_signal`; the
initial host test caught this. The final implementation uses the same existing
glibc APIs through ctypes, without installing dependencies.

Exact host test command from tt-metal root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python_env/bin/python -m pytest -q \
  -p no:cacheprovider -c /dev/null \
  --confcutdir=models/autoports/qwen_qwen3_8_27b/tests \
  models/autoports/qwen_qwen3_8_27b/tests/test_vllm_process_guard.py
```

`startup_cancel_host.log`: **7 passed in 1.29 seconds**. The tests execute the
packaged runner's actual extracted `_shutdown` against a dummy parent and prove
its child survives, then verify repaired ordinary-exit/SIGINT/SIGTERM cleanup,
fresh markers despite an inherited old marker, preservation of an unrelated
sentinel, orphan reaping, exact marker matching, PID birth revalidation, and
bounded failure reporting while leaving a TERM-ignoring dummy alive. The test
fixture separately terminates its own intentionally surviving dummy processes;
the production guard never escalates to SIGKILL.

`bash -n tests/run_vllm_stage.sh`, Black with `--target-version py312` on both
new Python files, and `git diff --check` passed. No C++ or CMake change requires
a build. This investigation performed no hardware operation or real server
launch. Parent-owned runtime startup cancellation remains to be verified with
the guard. SIGKILL of the guard itself, or descendants deliberately removing
their launch marker, cannot be covered by this bounded cooperative cleanup.
