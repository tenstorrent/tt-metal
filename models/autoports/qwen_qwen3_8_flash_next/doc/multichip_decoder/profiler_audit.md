# Missing profiler device IDs: CPU source and capture audit

2026-11-13, CPU only. Audited the original completed capture at
`/home/ttuser/qwen38-flash/multichip_profiles/mesh_l0/.logs` and repository
profiler source. No hardware calls or implementation/processor edits.

The missing device IDs are actual capture loss from full profiler DRAM
buffers. They are not a cumsum-only instrumentation defect, signed counter
overflow, or loss introduced by the operation-only Tracy export.

`profile_mesh_l0.log:15026` and subsequent lines report:

> Profiler DRAM buffers were full, markers were dropped! ... bufferEndIndex = 12000.

The warning explicitly recommends fewer operations per capture or reading
the device profiler more often. Source: `tt_metal/impl/profiler/profiler.cpp:1650-1660`.

## Independent raw-data comparison

I parsed original `tracy_ops_data.csv` with its actual semicolon/backtick
format, resolved cached host operation messages, parsed
`cpp_device_perf_report.csv`, and streamed the raw1.1GB
`profile_log_device.csv`. Only compact counts were printed; no large file
was loaded or modified.

| Evidence | Distinct (device, operation ID) pairs |
|---|---:|
| Original host device-operation messages | 19420 |
| Original raw device CSV | 8548 |
| C++ device report | 8548 |
| Host IDs missing from raw device CSV | 10872 |
| Host IDs present in raw but absent from C++ report | 0 |
| Raw device IDs not in original host metadata | 0 |

Raw CSV parsing encountered zero malformed lines. Raw and C++ IDs match
exactly. The missing host IDs span30 operation types, including2248
Typecast,1480 Slice,1240 BinaryNg,1060 ReshapeCodegen,1052 Reduce,400
AllGather,248 Matmul,132 GenericOp,48 Accumulation, and20 MoeFusedSwiGlu.
This is not a special case confined to `AccumulationDeviceOperation`.

The first missing ID1028096 is rank0's cumsum along dim3 of an FP32 tensor
with logical shape[1,1,1,512], padded[1,1,32,512]. Host metadata contains
ordinary accumulation reader/writer/compute kernels. That exact ID has zero
raw device markers. Neighbor IDs1027072 and1029120 each have2180 raw
markers;1031168 has zero,1032192 has300. Different worker cores exhausting
their buffers at different times explains holes rather than one clean
global cutoff.

Host ID range is1024..4971523; raw and C++ range is1024..3251203. None is
near signed32-bit overflow. Per-device ID encoding also explains adjacent
rank IDs1028096..1028099; it is not evidence of corruption.

The measured signpost windows are completely missing device timing:

| Original host signpost window | Host device-op records | Present in device report |
|---|---:|---:|
| PERF_PREFILL | 2964 | 0 |
| PERF_DECODE | 1120 | 0 |

Consequently this capture cannot yield a valid measured-window op report.
The8548 device rows belong to earlier activity. Forcing the legacy parser
cannot reconstruct markers absent from the raw CSV. Do not drop missing
operations, synthesize durations, or reuse earlier warmup rows as measured
window timing.

## Drain semantics and recovery

`ttnn.ReadDeviceProfiler(mesh)` is suitable for bounded capture batches:

1. `ttnn/cpp/ttnn-nanobind/device.cpp:766-770` calls
   `ReadMeshDeviceProfilerResults(..., ProfilerReadState::NORMAL, metadata)`.
2. `tt_metal/impl/profiler/tt_metal_profiler.cpp:1197-1269` finishes every
   hardware CQ, reads each mesh device, processes results, and waits for
   the host processing pool. The ordinary Blackhole path chooses DRAM
   input at lines927-940.
3. `DeviceProfiler::readResults` in
   `tt_metal/impl/profiler/profiler.cpp:2579-2584` reads control buffers,
   reads the profiler DRAM bank, then calls `resetControlBuffers`.
   That routine at1466-1503 creates fresh zero-initialized control vectors,
   restores required address/core metadata, clears per-RISC active-buffer
   and read cursors, writes the reset controls, and resets active indices.
   It resets collection capacity; it does not erase the application's
   operation-ID counter or accumulated CSV history.
4. Raw device CSV writer at `profiler.cpp:1213-1218` opens an existing file
   with `std::ios_base::app`. C++ analysis writer at
   `profiler_analysis.cpp:664-669` does the same for
   `cpp_device_perf_report.csv`. `profiler.cpp:2471-2475` also appends each
   batch's translated results to the in-memory history. `dumpDeviceResults`
   writes the current batch and then clears only its host marker buffer.

Owner's proposed cadence is supported by this source: drain after model
setup, after each initial forward, after each warmup, and after closing
each measured signpost window. Keep drains outside latency windows. A
single forward could still overflow if it individually exceeds a
per-core buffer; require zero full-buffer warnings and complete host/device
ID coverage in the next capture, rather than assuming the cadence suffices.
If a forward alone exceeds capacity, narrow the profiled token scope or
use separately labeled component windows with drains between them.

Use a fresh output folder so append behavior does not mix independent
process captures. Retain the operation-only binary export workaround for
the separate unescaped multiline generic-source CSV issue: it fixes host
CSV syntax, not device marker loss. The original all-zone export and failed
capture remain useful provenance, not accepted performance evidence.

No device reset is indicated by this profiler-buffer exhaustion. Hardware
health and the successful decoder run are separate from timing completeness.

## Single-chip follow-up: one prefill exceeds the buffer

The subsequent capture at
`/home/ttuser/qwen38-flash/multichip_profiles/baseline_l0/.logs` used drains
after setup, each initial forward, each warmup, and each closed measured
window. It still reported full12000-entry buffers immediately after the
cold step0 prefill (`profile_baseline_l0.log:9140`) and after measured
prefill. Thus a whole forward can itself exceed capacity.

An independent streamed raw-ID audit found:

| Evidence | Distinct device/op IDs |
|---|---:|
| Original host messages | 6942 |
| Raw device CSV | 6676 |
| C++ device report | 6676 |
| Host IDs absent from both device sources | 266 |
| Raw/C++ ID mismatches | 0 |

| Measured window | Host device-op records | Present | Missing |
|---|---:|---:|---:|
| PERF_PREFILL | 1212 | 1159 | 53 |
| PERF_DECODE | 215 | 215 | 0 |

The first missing ID1087488 is the GR injection matmul with logical
input[1,1,1,10240] and weight[1,1,10240,4], on the one-token tail. It has
zero raw markers. The1212-operation prefill executes the existing32+1
logical chunk sequence (`functional_decoder.py:157-169`). Missing records
span13 operation types, including110 Typecast,35 Slice,25 Reduce, and10
Matmul. This is intra-forward overflow, not a device-ID mapping mismatch
or a cumsum instrumentation issue. Forcing legacy CSV processing cannot
recover these missing markers.

The implemented test-only recovery in `probe_multichip_decoder.py` imports
`profile_fused_ops.py:bounded_profiler`, which wraps selected host operation
entry points, checks the host operation counter, and synchronizes/drains
after64 counter increments with a recursion guard. It runs only in profiler
mode. The report records `profiler_drain_operation_ids` and explicitly labels
the scope as segmented per-operation timings; separate trace runs supply
latency. This avoids changing decoder arithmetic or public sequence-length
contracts. An alternative drain after each32-token `_chunk` is narrower but
would still need proof that one chunk remains below per-core capacity.

Drains can lie inside broad phase signposts used to select operation rows.
They therefore introduce wall-clock and inter-operation gaps: those broad
intervals are phase membership, not uninterrupted latency measurements.
Individual device-op durations remain the intended profiling evidence.
Do not interpret summed host intervals or drain-inflated op-to-op latency
as decoder throughput.

Fresh `profile_baseline_l{0,1,3}_segmented` and
`profile_mesh_l{0,1,3}_segmented` captures/reports are pending acceptance at
this update. Require complete measured-window host/device joins, zero
overflow warnings, and human-readable/CSV report review before marking
profiling complete. Existing partial or finished files alone do not imply
that these gates passed. Raw failed baseline and mesh captures are preserved.

## Layer3 segmented capture: unmatched endpoint at final parsing

CPU source audit, 2026-11-13. Capture
`/home/ttuser/qwen38-flash/multichip_profiles/mesh_l3_segmented` failed with
wrapper exit4 / child SIGABRT. The exact console evidence is
`profile_mesh_l3_segmented.log:4490-4575`. Initial prefill/decode whole-output
PCCs were0.9996135235,0.9997360706,0.9997849464,0.9998460412, all finite.
Both measured signpost windows closed before the fatal at19:38:46.339.
This is a failed profiler capture, not an observed decoder numerical failure.

The fatal is `profiler.cpp:2210`: an NCRISC `NCRISC-KERNEL` ZONE_END,
marker43314, execution UID `(10237954, 9223372036854775808,
9223372036854775808)`, chip2, physical core(1,2), timestamp2327872274441,
operation `TypecastDeviceOperation`, has an empty nesting stack. The marker
source is `tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc:70`. The large trace
sentinels in this UID are not evidence of signed operation-ID overflow.

Unlike the earlier buffer-overflow captures, this console contains **zero**
`Profiler DRAM buffers were full`, `markers were dropped`, or
`Device profiling never finished` messages. The branch at
`tt_metal/impl/profiler/profiler.cpp:2207-2216` also confirms that the device
profiler's `had_dropped_markers` flag was false when the fatal fired.
Do not relabel this as established DRAM overflow.

Source gives a narrower diagnosis than a physical root cause: the final
host-side marker set is unbalanced. `dumpDeviceResults(false)` validates
all per-core/RISC marker sets at`profiler.cpp:2499-2508` **before** CSV writing
at2518 and device pushing at2520. Consequently this failed directory holds
the41MB host Tracy capture and source-location files, but no raw device CSV
or C++ device report. It cannot establish whether the start was lost on
device, omitted during buffer parsing, or discarded earlier in host marker
collection. No missing-marker repair or decoder-core change is justified
by the available evidence.

The Python64-operation wrapper already synchronizes before each drain
(`tests/profile_fused_ops.py:40-68`). The normal mesh read additionally
finishes every hardware CQ, reads devices, then waits for per-device parsing
tasks (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1229-1266`). It delegates
to the debug-dump thread instead of starting a second reader when that mode
is enabled. The current source therefore does not support a claim that two
ordinary Python drains overlap. DRAM reads/reset and marker parsing remain
separate operations (`profiler.cpp:2579-2596`); without raw marker evidence,
a device flush/reset boundary issue is only a hypothesis.

Minimal diagnostic retry, owned by the parent device runner:

- Keep the existing synchronized64-operation cadence initially. There is
  no overflow evidence requiring more frequent resets. If a second trial
  is needed, use32 as a controlled cadence change and record actual drain
  operation IDs; neither cadence is a correctness proof.
- Set`TT_METAL_PROFILER_MID_RUN_DUMP=1` and
  `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1`, while retaining device CSV
  writing and C++ postprocessing. The normal read then calls
  `dumpDeviceResults(true)` after each completed drain
  (`tt_metal_profiler.cpp:996-1051`). This emits bounded raw evidence and
  analyses before clearing each marker set (`profiler.cpp:2511-2524`),
  avoiding dependence on one final accumulated nesting-validation pass.
  This is a diagnostic capture mode, not a repair of a missing start.
- Disabling device pushes **alone cannot fix this assertion**: its guard
  is inside`pushTracyDeviceResults` at`profiler.cpp:2667`, after the failing
  validation. Combining it with mid-run dumping avoids pushing potentially
  unbalanced device zones to Tracy; host operation metadata/signposts and
  device CSV remain the reporting inputs.
- Accept only if measured-window host/device joins are complete, every
  reported duration has valid start/end support, overflow warnings remain
  absent, and the human-readable tables are credible. Preserve failed and
  retry directories separately. A successful exit alone is insufficient,
  because mid-run mode skips the final nesting validator for discharged
  marker sets. Continue using separate trace runs for latency.

No devices were accessed or core profiler/decoder source changed for this
audit. The retry is pending; no successful layer3 profiler evidence is
claimed here.

### Follow-up: duration validation and rollover evidence

The fatal timestamp has a useful exact decomposition:
`2327872274441 = (542 << 32) + 9`. Its low word is nine cycles after a
32-bit rollover. On Blackhole,
`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:253-257` reads the high
clock word, writes it to the marker, then separately reads the low word.
There is no high/low/high consistency retry in this function. A rollover
between those reads would combine the old high word with the new low
word, making that endpoint appear exactly`2^32` cycles early. Markers are
sorted by timestamp (`TracyTTDeviceData.hpp:253-269`), so this can put an
end before its matching start and produce the observed empty-stack fatal.
The low-word value is strong supporting evidence for this hypothesis;
without the raw matching start, it is not proof. This is distinct from a
normal consistent64-bit timestamp crossing a low-word boundary, which
would sort correctly, or a drain resetting the wall clock (the audited
reset changes profiler control buffers, not the device wall clock).

The selected next action is therefore a fresh **ordinary segmented capture**
with the original final nesting validation enabled, followed by the strict
measured-window host/device join. This supersedes the earlier suggestion
to enable mid-run dumping on the first retry. A successful fully validated
capture avoids bypassing that check and needs no custom endpoint checker.
The following mid-run analysis/check recipe is contingency only if the
ordinary retry repeats the failure; it has not been adopted or executed.

Mid-run dumping preserves every marker **present in the host marker set**
in append-only raw CSV (`profiler.cpp:1191-1259`), including operation ID,
trace ID/counter, core, RISC, timer ID, type, name, and timestamp. It cannot
guarantee that every emitted device marker reached that set. Its duration
construction is not equivalent to the skipped nesting validator:
`profiler_analysis.cpp:305-364` groups by program execution UID, selects
the first qualifying start and last qualifying end across participating
cores/RISCs, and asserts only aggregate`start <= end`. An orphan on one
core can be masked by valid endpoints on another core. Therefore even a
complete host/device operation join and nonzero durations are insufficient.

The retry's explicit offline acceptance check must stream raw CSV and:

1. Select the exact host operation IDs inside each measured signpost
   window, keeping device IDs separate and checking every expected mesh
   rank. Include the full trace ID/counter in identity even though this
   capture is eager. Keep duplicate device/op report rows visible rather
   than silently replacing them in a dictionary.
2. Group kernel endpoints by`(device, core_x, core_y, RISC, operation ID,
   trace ID, trace counter, timer ID, zone name)`. For the ordinary outer
   `*-KERNEL` zones require exactly one start and one end, matching identity,
   with`start <= end`; diagnose repeated instances explicitly rather than
   pairing arbitrary extrema. Report orphan starts, orphan ends, duplicate
   endpoints, reversed timestamps, and unexpectedly absent observed
   core/RISC groups. Check firmware pairs too if firmware durations are
   to be used. Pair counts alone cannot detect a wholly missing core/RISC;
   compare the observed participating cores with available program metadata
   and the same program's intact warmup invocations.
3. Recompute each device/program kernel envelope from those validated
   pairs: earliest kernel start and latest kernel end. Compare its exact
   cycle endpoints against the C++ report, then compare the duration using
   CSV chip MHz and the C++ rounding rule. Check the separate DM-start
   envelope against its configured BRISC/NCRISC/ERISC start set when using
   that column. Finally verify the merged worst-rank duration from all
   device rows. These definitions are in
   `tt_metal/tools/profiler/cpp_device_analyses.json`.
4. Reject any measured operation with malformed endpoints, absent ranks,
   unresolved participating-core omissions, inconsistent envelopes, or
   implausible rollover-sized durations. Do not silently add`2^32` to a
   suspect endpoint or drop the operation. Retain diagnostics for unmeasured
   setup/warmup failures separately; acceptance is explicitly limited to
   the measured per-operation attribution windows proven intact.

Passing these checks permits use of the measured per-operation durations;
it does not prove the profiler's underlying orphan bug repaired. The raw
retry, endpoint-check counts/failures, and checked window IDs must be kept
with provenance. Separate uninterrupted trace evidence remains the source
for latency and speedup.

## Final validated retry

`profile_final_mesh_l0`, `profile_final_mesh_l1`, and
`profile_final_mesh_l3` all completed with exit zero. Their ordinary
nesting validation and strict host/device operation join were retained;
`TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY` and mid-run dump were not used.
The final QSA retry therefore supersedes the contingency endpoint-check
recipe above. All three final mesh and all three segmented baseline logs
contain no overflow warning. Exact raw captures and operation-export
provenance live in `/home/ttuser/qwen38-flash/multichip_profiles/`.
Final advice-enabled human tables and CSVs are indexed by the provenance
files under `tracy/final_mesh_l{0,1,3}`. These are segmented operator
measurements; uninterrupted trace latency remains in `performance_summary.json`.
