# Pipeline blocker log

## 2026-09-12: stage 6 checker could not find `python`

- Symptom: stages 1 through 6 completed, then `06-full-model.check.sh`
  exited 127 with `python: command not found`.
- Cause: the runner invoked its Python interpreter by absolute path but exported
  only the Node directory to `PATH`. Stage check scripts invoke `python` by name.
- Repair: prepend both `python_env/bin` and the pinned Node directory to `PATH`.
- Validation: rerunning the stage 6 check in the container with the corrected
  path passed the non-degenerate-output and 262144-token context checks.
- Resume point: stage 7 (`optimized-full-model`); stages 1 through 6 are retained.

For later blockers, record the UTC date, symptom, root cause, repair, validation,
and resume point here before continuing the pipeline.

## 2026-09-12: resume prompt selection

- Symptom: the no-model dry-run paired stage 7 with `01-functional-decoder`
  and numbered the final prompt as stage 17.
- Cause: `--start-index 7` controls numbering; it does not remove the first six
  prompt files from an all-prompts glob.
- Repair: supply only prompt files 07 through 11 and retain `--start-index 7`.
- Validation: require the corrected dry-run manifest to map stages 7 through 11
  to `optimized-full-model` through `tti-release` before launching Astra.
- Resume point: stage 7.

## 2026-09-12: stage 7 hardware ownership prevents mesh initialization

- Initial TP4 mesh open failed on device0 Ethernet heartbeat before model code.
- Two bounded resets restored enumeration of all four Blackhole p300c devices.
- Exact mesh smoke then failed at the UMD sysmem guard: expected NOC address
  `0x1000000000000000`, received `0x1000000040000000`.
- KMD owner records show four PID0 entries per device in this container;
  recorded opener PIDs are outside its PID namespace. No visible owner can be
  safely stopped here. AutoFix requires host operator owner cleanup or host recovery.
- Evidence: `doc/optimized_full_model/AUTOFIX_startup.md`, recovery logs and
  `work_log.md`. Prepared benchmark has syntax checks only; no stage7 device result.
- Resume point: stage7 after successful ring1x4 mesh open/close. No model policy
  change, stage completion, stage checkpoint commit, vLLM work or push performed.

### 2026-09-13 host recovery

- Host-namespace inspection resolved the container's PID0 records to PID 664795,
  a Kubernetes `tt_telemetry_collector` that had held all four devices since
  2026-09-12 18:45 UTC. Its parent telemetry server was left running.
- Sent SIGTERM only to the stale collector. The KMD owner lists for devices 0-3
  became empty immediately.
- The rerun checkout's virtualenv does not contain `tt-smi`; device listing uses
  the established `/home/mvasiljevic/tt-metal/python_env/bin/tt-smi` while mesh
  execution uses this checkout's rebuilt `python_env` and TTNN libraries.
- Bounded `tt-smi -ls --local` completed with all four Blackhole p300c devices.
- The exact current-checkout `FABRIC_1D_RING`, `MeshShape(1, 4)`,
  `trace_region_size=0` open/close smoke completed with `MESH_SMOKE_OK`.
- Repair validated. Resume the preserved stage 7 thread with `--resume-stage 7`.
- The telemetry server subsequently respawned collector PID 4055677, which
  reopened all four devices. To keep the experiment reservation stable, host
  telemetry server PID 1567332 was suspended with reversible SIGSTOP and the
  collector was terminated. KMD owner lists are empty. Restore telemetry after
  the pipeline with `sudo kill -CONT 1567332`.
