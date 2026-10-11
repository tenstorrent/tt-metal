# Current GDN hardware-counter capture

The counter experiment preserves the current resident FP32 recurrence and
packed-L1 epilogue. It is a synthetic component diagnostic, not new model
throughput or evaluation qualification. The measured candidate remains
20.8846 TSU at B16/32K/TP4 and 23.8762 TSU at B16/16K.

## Attempts and recovery

1. Attempt 1 launched at 2026-10-11 04:17:48 UTC. Its mask-zero hardware
   control passed all eight cases and 48 kernel invocations, with clean close.
   The new analysis then rejected two raw-log paths: native Tracy retains the
   canonical `.logs/profile_log_device.csv` and a report-directory copy.
   The fix explicitly reads the canonical path. No counter pass ran.
2. Attempt 2 recovered the original control's complete 192 rank/call records
   without rerunning hardware. Native planning chose L1 bank 0 plus FPU,
   mask 9, for the first counter pass. JIT BRISC firmware exceeded its text
   region by eight bytes: segment size `0x2208`, limit `0x2200`. No model
   kernel executed. The test receipt was failed/unclean, despite Tracy's
   wrapper returning PASS. The independent controller rejected it.
3. Attempt 3 uses the native planner with a stricter one-group limit and
   ten passes: FPU, pack, unpack, instruction and six Blackhole L1 banks.
   It retains the original control, requires identical state/output hashes
   on every pass, and ends with another mask-zero control. An exclusive-lock
   Galaxy reset completed successfully before this attempt. At the retained
   04:31:51 UTC observation, the FPU pass was active; no counter result was
   yet established. Original receipts and sources remain preserved.

The active handle at that observation was
`qwen38-gdn-counter-v3-20261011.service`, PID 903650, invocation
`de815ce36490419f9f184d4b36ac6fdb`. Recheck systemd for current state. It survives
SSH/session disconnect, has a two-hour hard deadline, a 128-GiB process memory
limit and a 16-GiB aggregate artifact guard. Per-capture raw files retain the
existing 1-GiB maximum. Native installations, precision and serving defaults
were not changed. JIT artifacts are host-local and disposable.

## Validation and interpretation

The latest frozen source passed 770 CPU tests, one skip and 104 subtests;
the allocated-device test collected successfully. Local focused parser/phase
tests passed 26 cases. Pre-commit passed. These checks do not establish that
all native counter groups fit or return useful hardware records.

The collector uses the pinned architecture's counter arrays and name decoder.
It requires every requested counter on every active core of all 192 target
rank/call records, with positive reference intervals, exact signpost coverage
and no duplicate records. State and output hashes must match the mask-zero
control. Missing/invalid results stop the queue; they are not silently dropped.

Counter intervals bracket TRISC1 on compute cores. L1 request/grant counts
describe the L1 interface, not physical DRAM bandwidth. Ratios from different
passes must not be summed as simultaneous activity. The before/after timings
will quantify counter overhead; they are not model TPOT. A matched streaming
ceiling and full-model critical-path reconciliation remain separate work.

`attempt-01`, `attempt-02`, and `attempt-03` contain exact launch arguments,
source manifests, unit checks, logs and captured status. `original-capture`
preserves the first collection unchanged. `capture.json` records the latest
observation represented here. Compressed source files retain the launch's
diagnostic additions; all frozen model source paths/hashes are in the manifests.

## Metadata export correction and fourth attempt

Attempt 3 subsequently passed all eight hardware cases, 48 invocations,
exact output comparisons and clean close with a single FPU group. Its counter
analysis failed because all 95,664 counter records had empty metadata. The
CSV retained the low payload word but not the counter type in its trailer;
do not assign types by row order or treat those records as utilization data.

The pinned native profiler's `DeviceProfiler::dumpResults` calls
`processDeviceMarkerData` only when `is_mid_run_dump` is false. That processing
populates counter type/value/reference metadata. Our ordinary `--profile-ops`
wrapper requests mid-run dumps, so it is unsuitable for these counters.
The new `--profile-counters` mode keeps Python tracing and device UI pushes
disabled but retains final native metadata processing. Ordinary phase capture
behavior is unchanged. No installed library change is required for this test.

Attempt 4 uses this mode and preserves all earlier attempts. At04:47:16UTC,
`qwen38-gdn-counter-v4-20261011.service` was active with PID924056 and invocation
`d2427244c3c449feb7c4fe9d2f245d47`, running its first FPU pass. The preceding
hardware close was clean, so this launch did not reset the Galaxy. Completion
and meaningful counter coverage still need verification. The two-hour/16-GiB
bounds remain. `recovery-capture.json` and `attempt-04` record this observation;
`source-final` retains the corrected harness and wrapper source.


## First verified counter pass

The corrected FPU pass completed and passed full inventory checks: all three
FPU/SFPU/MATH counters are present on 23,040 active-core operation records,
covering 192 target rank/call records. Exact state/output hashes match the
mask-zero control. The collector automatically advanced to the pack pass.
See `first-verified-pass` for its full compressed summaries, hardware result,
source hashes and captured live queue. Later passes are not credited yet.

For B16, skip-padding, unannotated kernels, the per-rank median math activity
ratios are 0.3185-0.3194 for recurrence and 0.4759-0.4771 for epilogue. Corresponding
SFPU activity is 0.2706-0.2714 and 0.4365-0.4375. These reference-window ratios are
not physical DRAM utilization, peak FLOP efficiency or full-model critical-path
shares. Pack/unpack, instruction and L1 passes plus before/after overhead controls
remain necessary before assigning the inactive portion to a bottleneck.
