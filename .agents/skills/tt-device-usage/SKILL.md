---
name: tt-device-usage
description: Use TT hardware safely during tt-metal model bringup, tests, profiling, serving, reset/recovery, tt-smi health checks, tt-triage hang capture, watcher/profiler separation, and T3K ARC/ERISC/remote-Ethernet recovery. Use whenever a stage touches Tenstorrent devices or must decide whether a hardware failure is recoverable.
---

# TT Device Usage

Use this skill for the hardware-facing parts of model bringup. It covers only non-obvious TT device practices: command serialization, reset/list quirks, recoverable ARC/ERISC failures, hang triage, and when a stage may call itself blocked.

## Basic Rules

- Run TT hardware-facing commands one at a time: `tt-smi`, resets, TTNN import/open-device probes, watcher runs, Tracy/device-profiler runs, tests, benchmarks, and serving jobs.
- Close devices before starting the next device-facing command. Kill stale processes from the same run before reset or retry.
- Keep watcher and profiler evidence in separate runs. Do not combine `TT_METAL_WATCHER` with device-profiler or Tracy collection.
- Do not profile live vLLM serving stages. Use serving benchmark JSON and logs instead.
- Preserve evidence before cleanup: work logs, README files, benchmark JSON, server logs, compact perf summaries, and exact failing commands. Do not delete `CODEX_HOME`, auth/config, completed stage artifacts, or the repo state.

## Reset And Health Checks

Use bounded commands so a bad device state cannot consume the whole stage:

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
sleep 20  # let fabric/Ethernet re-establish before re-listing (else a spurious recheck forces an extra reset)
timeout 60 tt-smi -ls --local
```

After reset, check that the expected devices are visible. On a T3K this usually means all eight chips. If reset returns but not all expected devices or Ethernet links come back, run the bounded reset sequence one more time before escalating. A first reset can leave part of the mesh or Ethernet fabric missing.

When device listing looks healthy, prove the mesh can actually open and close before resuming the stage:

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Adjust `MeshShape` for non-T3K targets. Record the final device list and mesh-smoke result.

## ARC / ERISC / Remote Ethernet Recovery

Treat these as recoverable infrastructure faults until recovery fails:

- `Timeout waiting for Ethernet core service remote IO request`
- `ETH core heartbeat check failed`
- `Unexpected ERISC Response Flags`
- `Read 0xffffffff from ARC scratch`
- ARC lock/readback waits
- `tt-smi -ls --local` hanging or failing
- `tt-smi -r` hanging or failing
- TTNN device open failing before model code runs

Recovery sequence:

1. Stop only risky or stale processes from this run: `EngineCore`, `vllm.entrypoints`, `run_vllm_server`, model benchmark processes, stale `pytest`, stale Tracy, and stale profiler jobs.
2. Do not start more profiler, watcher, serving-adapter profiler, or `ttnn.ReadDeviceProfiler(mesh)` work while hardware is unhealthy.
3. Run the bounded list/reset/list sequence above.
4. If listing is incomplete after the first successful reset, run the bounded reset sequence once more.
5. If devices are visible, clear stale TT UMD locks only after confirming no live process from this run owns the devices. Then run the mesh smoke.
6. If reset, listing, or mesh smoke still fails, ask the monitor/operator for a physical Docker-host reboot and reservation re-acquire. If the current agent explicitly owns experiment monitoring or machine recovery, reboot the host directly and repeat list/reset/list plus mesh smoke after reconnecting.
7. Resume the same stage from preserved state. In the multigoal flow, use `--resume-stage <stage_number>` instead of restarting completed earlier stages.
8. Record this as infrastructure recovery, not a model correctness or performance result.

Do not mark a model stage blocked because a board briefly became undiscoverable. A model stage may block on hardware only after this recovery path fails or requires operator intervention the agent cannot perform.

## Hangs And tt-triage

If a device job appears hung, collect triage evidence before killing it unless the machine is already unreachable or triage itself hangs.

Prefer the LLM-readable report:

```bash
mkdir -p models/autoports/<model>/doc/<stage>/triage
timeout 180 tools/tt-triage.py \
  --llm-output \
  --llm-output-path models/autoports/<model>/doc/<stage>/triage/tt-triage.txt \
  --triage-summary-path models/autoports/<model>/doc/<stage>/triage/triage-summary.txt
```

Useful focused variants:

```bash
timeout 120 tools/tt-triage.py --llm-output --run=dump_callstacks --run=dump_running_operations --run=check_eth_status
timeout 120 tools/tt-triage.py --llm-output --run=dump_watcher_ringbuffer --run=check_arc
```

If dependencies are missing, install the repo-local triage requirements in the active environment:

```bash
python -m pip install -r tools/triage/requirements.txt
```

Read `tools/triage/tt-triage.md` for command details and available scripts. Keep the triage output with the stage evidence.

After capturing triage, use `$autofix` for the failure if ordinary log reading does not explain it. `$autofix` will run `$autodebug` when needed. Give it the failing command, console log, `tt-triage.txt`, current stage work log, and relevant source paths. Do not declare a stage blocked for a hang until `$autofix` has tried and failed, unless the remaining blocker is unavailable hardware or operator-only recovery.

## Evidence To Record

For any hardware recovery or hang, record:

- failure signature and exact command that exposed it;
- active processes killed, if any;
- `tt-smi` list/reset/list commands and exit status;
- whether a second reset was needed for missing devices or Ethernet links;
- whether locks were cleared and why that was safe;
- mesh-smoke command and result;
- tt-triage output path for hangs;
- whether `$autofix` was used, and its conclusion;
- resumed stage/thread/log path.
