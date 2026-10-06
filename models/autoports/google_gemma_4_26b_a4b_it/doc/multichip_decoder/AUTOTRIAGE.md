# AUTOTRIAGE

## Diagnosis

- **The apparent close hang was a finite shutdown/profiler drain followed by substantial host-side report processing; this run does not demonstrate a decoder or fabric deadlock.** The runner returned from `close_mesh_device`, wrote `profile_v0.json`, completed UMD shutdown, and generated its final ops CSV without intervention. Attribution of the entire quiet interval specifically to profiler draining remains an inference, because no close-entry/exit timestamps or host stack were captured.
- Scope: stage 04, `google/gemma-4-26B-A4B-it`, TP4 sliding decoder, length 4096, one traced decode step, profiling enabled. Primary artifacts are `profile_v0.log`, `profile_v0.json`, and `triage/{profile_close.txt,profile_close_summary.txt,capture.log}` in this directory. Investigation used the repo-local AutoTriage and AutoFix instructions. No device commands, process intervention, or implementation changes were performed by this investigator.

## Triage Evidence

- `profile_close.txt` contains only system information. `profile_close_summary.txt` and `capture.log` report missing `/tmp/tt-metal/inspector`; default `--dev=in_use` could not determine devices. Running-op, callstack, CB, NoC, Ethernet, ARC, binary-integrity and watcher checks were all skipped. This capture proves **neither a device stall nor device health**. The tool's generic “Metal runtime is not running” diagnostic is not an independently observed process state.
- `profile_v0.log` gives the following forward-progress sequence, all on 2026-09-26:

  | Time | Observed event |
  | --- | --- |
  | 20:26:39.524 | `PERF_DECODE_END`, followed by `TP_DONE 4` |
  | Before 20:27:04.863 | Result dictionary printed, which occurs after mesh close in this runner |
  | 20:27:05.680 | UMD reports device close completed and cluster destructor completed |
  | Before 20:27:06.835 | Tracy saves its trace: 44,690,573 zones, 111.88 MB compressed |
  | 20:27:18.975 | Host report exports complete; host ops import starts |
  | 20:28:08.366 | Device ops analysis begins |
  | 20:28:27.142 | Final ops CSV generation completes |

- The generated host timing CSV is 4,666,752,410 bytes; host op data is 146,119,516 bytes. The final report is `profile_v0/reports/tp4/2026_09_26_20_28_12/ops_perf_results_tp4_2026_09_26_20_28_12.csv`. These observations support substantial finite host processing rather than a silent device wait after 20:27:18.
- `profile_v0.json` records the expected TP4/4096/one-step traced workload. Its empty PCC list and `passed: null` reflect the TP4-only profile run; its existence proves runner completion through close, not paired numerical acceptance.

## Source Evidence

- `tests/run_multichip_decoder.py:111` synchronizes uploaded decode inputs; line 114 executes the trace with `blocking=True`; line 119 reads the result back. Lines 126–129 release the trace and print `TP_DONE`. Line 131 closes the mesh in `finally`. Only then do lines 137–138 write the JSON and print the result. The observed result output therefore directly refutes a persistent stop in line 131. The earlier `TP_DONE` alone would not have been sufficient.
- `tt_metal/distributed/mesh_device.cpp:957` implements `MeshDeviceImpl::close_impl`. It requests the last profiler read at line 967, clears CQs at line 1018, shuts down any realtime profiler at line 1024, and releases scoped devices at line 1050. These are separate lifecycle phases, not decoder work.
- `tt_metal/impl/profiler/tt_metal_profiler.cpp:1189` implements `ReadMeshDeviceProfilerResults`. With profiling enabled it finishes each CQ at lines 1220–1222, reads the device profiler results at lines 1246–1248, dispatches per-device processing at lines 1251–1255, and waits for the processing pool at line 1258. Thus a Python stack at close can represent host profiler processing after kernels have completed.
- `tt_metal/impl/device/firmware/profiler_initializer.cpp:80` requires dispatch and fabric teardown before final dispatch-core profiler reads and clock synchronization. `post_teardown` invokes `cleanup_device_profilers` at line 108. `tt_metal/impl/profiler/profiler_state_manager.cpp:108` launches one dump/context-destruction thread per device and joins all of them at lines 128–129. Required final profiler work is already present; adding another generic finish or drain is not a supported fix.
- `tools/tracy/__main__.py:441` waits for the test subprocess to exit. Only afterwards does it wait for capture completion at line 447 and invoke report generation at lines 486–493. Therefore report-generation messages prove that the test subprocess, not just its model loop, finished. The wrapper may remain active for host-only work.
- `tools/tracy/__init__.py:162` and line 173 announce two completed CSV exports before `process_ops` runs. `tools/tracy/process_ops_logs.py:434` parses op metadata and lines 524–527 load the large timing CSV with pandas. The observed `DtypeWarning` at line 527 is a parsing warning in this host path, not a hardware error.

The relevant completion ledger is:

| Producer | Consumer/completion boundary | Evidence in this run |
| --- | --- | --- |
| Traced decoder/CQ | Blocking replay and host tensor reads | Completed before `TP_DONE 4` |
| Device profiling records | Last profiler read and per-device processing pool | Close returned; result JSON written |
| Dispatch/fabric final profiling data | Profiler teardown, dump threads, joins | Test process exited; UMD shutdown completed |
| Tracy event stream | Capture subprocess | Saved trace and exited before report generation |
| Host CSV exports | Python op-log import and report writer | Final ops CSV emitted at 20:28:27.142 |

There is no captured stuck fabric send, route, credit, TRID, semaphore or CB state from which to build a meaningful packet-route ledger. Inventing such a root cause would exceed the evidence.

## Downstream Effects

- The quiet interval between `TP_DONE` and the next log line can look like a close deadlock because the marker precedes teardown and that teardown includes synchronous profiler work.
- The later approximately 49-second op-log import is an independent host postprocessing phase. It cannot be attributed to running model kernels once the wrapper has reaped the test process.
- The GUI trace-copy warning uses the default `generated/profiler/.logs` location even though capture was saved under this stage's custom output directory. Source at `tools/tracy/__main__.py:451` selects that default path. This is a separate, nonfatal GUI-copy path issue: the custom-output capture exists and report generation succeeds. It does not explain the earlier close interval and needs no decoder change.

## Proposed Fix

- **No model, collective, or teardown implementation fix is justified.** The persistent-hang hypothesis is refuted by uninterrupted completion. Keep the current implementation and record this as a false hang alarm in the AutoFix work log.
- The minimal discriminating experiment was read-only observation of the same run: inspect the result JSON, subsequent UMD/capture messages, and final report artifact. It passed without resetting devices, killing processes, or changing code. No additional device rerun is needed to classify this event.
- If shutdown latency itself becomes a separate problem, add timestamped markers immediately before and after the existing close call in a future targeted run. Compare the same workload with and without the device profiler, changing only profiling. A host stack during the interval can distinguish `ReadMeshDeviceProfilerResults`/dump-thread processing from fabric termination. Perform this only if new evidence warrants a latency investigation; do not add arbitrary sleeps or an extra synchronization to the model.
- For a future **persistent** stall, capture valid Inspector artifacts and a supported explicit device selection before drawing device conclusions; the current triage capture is inadequate for that purpose. This is a future evidence requirement, not authorization or a reason to rerun hardware now.

## Uncertainty

- The approximately 26-second interval from `PERF_DECODE_END` to UMD close completion includes close and possible interpreter/runtime teardown. Without direct close timing or host stacks, the contribution of individual profiler and fabric teardown functions cannot be quantified.
- Large trace/export volume explains the later report-processing burden, but no baseline establishes whether this volume or shutdown latency is optimal. This report makes no performance-improvement claim.
- Final report generation proves progress and available artifacts; the main agent still owns exit-status confirmation and performance/accuracy analysis. The failed triage capture supplies no independent hardware-health certification.
