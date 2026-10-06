# AutoDebug: selected sliding profiler abort

2026-09-27. Fresh delegated source/log audit under AutoFix. No TTNN import,
hardware access, reset, target execution, runtime edit, or assertion change.
The auxiliary AutoDebug CLI was reported blocked by missing `bwrap`; this
fresh-context audit uses the delegated AutoFix route instead of retrying that
infrastructure. Existing model/runtime changes were left untouched.

## Finding

**A concrete 32-bit overflow in profiler host-buffer sizing is the leading
cause hypothesis.** It is activated by the failing 250000 program buffer count
and avoided by the earlier successful 100000 count. The surviving host trace
ends during profiler DRAM readback, before evidence of marker validation.
This is substantially better supported for this capture than the earlier
Blackhole timestamp-rollover hypothesis. Source establishes an undersized
vector and out-of-bounds writes; a controlled runtime rerun remains necessary
to verify that this explains this particular abort.

Do not repeat the 250000-count capture unchanged before resolving this sizing
hazard. The smallest workload-preserving control is **100000 program support,
4096 prefill, 128 traced decode, identical selected BF16 policy, and unchanged
final marker validation**, with unbuffered Python and faulthandler enabled.
No performance result from the failed capture is valid.

## Direct observations

- `profile_selected_sliding.log:25227–25239`: PERF_DECODE and PERF_DECODE_END,
  then TP_DONE 4, then shell `Aborted (core dumped)`. No fatal-marker endpoints,
  C++ exception explanation, run.json, or C++ device CSV survive.
- Runner `tests/run_multichip_decoder.py:327–329` prints TP_DONE immediately
  before `ttnn.close_mesh_device(mesh)`; run.json is written at line380.
- The failed compiler commands define
  `PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=12000000`, `NUM_L1_BANKS=110`, and
  `NUM_DRAM_BANKS=8`. `profile_v0.log` instead defines 4800000; the stage work
  log records its 100000-count, 4096/1 command and successful completion.
- Surviving host CSV `profile_selected_sliding/.logs/tracy_ops_times.csv:4694–4695`
  records completed `readResults-3-LAST_FD_READ-DRAM` (start85707465171ns,
  duration693937394ns) and `readResults-2-LAST_FD_READ-DRAM`
  (start86401481926ns,duration735295471ns).
  Lines7733–7734 and27472–27473 contain their completed DRAM-read zones.
  Latest completed zones are seven successive `FDMeshCommandQueue::finish_nolock`
  calls, ending at87775518172ns, consistent with the following device read.
  No completed processResults/dumpDeviceResults zones were found.
- Host zones are exported on completion, and abrupt termination can lose the
  tail; absence alone is not a stack trace. Nonetheless the surviving trace
  specifically supports early readback and provides no affirmative evidence
  of the earlier marker-validation failure.
- Raw device CSV was explicitly disabled, so its absence is expected even
  for a passing capture. Missing cpp_device_perf_report.csv after child abort
  causes the secondary Python postprocessing error.

## Source chain and concrete arithmetic

1. `tt_metal/impl/profiler/profiler_state_manager.cpp:58–80` calculates bytes
   per RISC per supported program. The failed compile has48 bytes/program ×
   250000 =12000000 bytes/RISC. This macro is **bytes, not uint32 words**:
   `tt_metal/tools/profiler/kernel_profiler.hpp:62` divides it by sizeof(uint32_t)
   to obtain the word count.
2. `tt_metal/llrt/metal_soc_descriptor.cpp:369–388` counts TENSIX plus Ethernet
   cores and rounds their count up per DRAM view. Blackhole has5 processors
   per core (`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:34`,
   `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:466`).
3. `tt_metal/impl/profiler/tt_metal_profiler.cpp:788–798` forms uint32 bank bytes
   as bytes/RISC ×5 ×ceil(core_count/8), then passes bank bytes and8 banks to
   `DeviceProfiler::setProfileBufferBankSizeBytes`.
4. **Earliest inconsistent calculation:**
   `tt_metal/impl/profiler/profiler.cpp:2392–2394` takes two uint32_t arguments
   and executes `profile_buffer.resize(size * num_dram_banks / sizeof(uint32_t))`.
   Multiplication is evaluated before division; `size * num_dram_banks`
   wraps in32bits before sizeof promotes the division to size_t.
5. Readback at `profiler.cpp:1359–1382` iterates all DRAM banks, copying the
   full unwrapped bank size to `&profile_buffer[profile_buffer_idx]`, advancing
   the index by bank_size/4 each time. No vector bound check intervenes.
   This writes past the allocation before any marker parsing or validation.
6. `tt_metal/distributed/mesh_device.cpp:964–967` starts this LAST_FD_READ in
   mesh close. `tt_metal_profiler.cpp:1246–1255` reads **every device first**,
   then queues parsing. Thus read3/read2 completion plus a subsequent read
   tail fits buffer corruption before processResults starts.

The failed compile proves110 logical compute/L1 banks and8 DRAM banks.
`tt_metal/jit_build/jit_device_config.cpp:42–45` derives NUM_L1_BANKS from
logical compute cores, so110 is a **lower bound**, not an exact count of the
SoC's TENSIX-plus-Ethernet profiler core set. The exact live SoC descriptor
was not preserved. Blackhole HAL reserves for up to140 TENSIX plus14 ETH
(`bh_hal.cpp:44–49`), so cores_per_bank lies between14 and20. Every possible
value in that range causes the failing allocation overflow; every value is
safe at100000. The16-core example below is illustrative, not a recovered
runtime descriptor. Representative values:

| Cores per bank | Program count | Bytes per bank | Required vector bytes | Actual vector bytes after wrap |
| --- | ---: | ---: | ---: | ---: |
|14|100000|336000000|2688000000|2688000000|
|14|250000|840000000|6720000000|2425032704|
|15|100000|360000000|2880000000|2880000000|
|15|250000|900000000|7200000000|2905032704|
|16|100000|384000000|3072000000|3072000000|
|16|250000|960000000|7680000000|3385032704|
|17|100000|408000000|3264000000|3264000000|
|17|250000|1020000000|8160000000|3865032704|
|20|100000|480000000|3840000000|3840000000|
|20|250000|1200000000|9600000000|1010065408|

At16cores/bank, failed readback first exceeds its vector within bank3
(zero-based): bank3 writes[2880000000,3840000000), but allocated bytes end at
3385032704. The seven/eight-bank reads continue well past that bound. Two
completed device reads do not make those writes safe; heap mappings and
corruption detection can delay termination. The exact abort mechanism still
needs a stack trace if the safe-count control does not resolve it.

Host-only arithmetic reproduction, executed using Python stdlib:

```python
from ctypes import c_uint32
for count in (100000, 250000):
    bank = 48 * count * 5 * 16
    print(count, bank, bank * 8, c_uint32(bank * 8).value)
# 100000 384000000 3072000000 3072000000
# 250000 960000000 7680000000 3385032704
```

The intended four-chip raw host allocation is26.88–38.4GB, not105.6GB:
12000000 is already a byte count. `/proc/meminfo` at audit time reports
261425568kB total, about97911908kB available. The visible cgroup has
memory.max=max and all memory.events oom/oom_kill counters0; its peak is
53740199936bytes. These are present-time observations, not a historical RSS
trace, but offer no positive evidence of OOM. Linux OOM-kill is also normally
SIGKILL rather than the observed shell SIGABRT. Buffer overflow ranks first.

## Minimal controlled next experiment

Parent owns hardware scheduling. Preserve failed artifacts and use a new
output directory. Retain the complete original model workload and flags,
changing only `--op-support-count 250000` to100000; export
`PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1` for diagnostic inheritance into
Tracy's child. Adding `-X faulthandler` or `-u` only to the outer Python is
insufficient because `tools/tracy/__main__.py:418` rebuilds the child command
from sys.executable without those interpreter options.

The Tracy `--check-exit-code` flag (`__main__.py:155–159,441–444`) suppresses
misleading postprocessing after failed child exit; it preserves successful
report generation. Final marker checks remain enabled with ordinary `-r -p`,
no SUM/accumulate/trace-only modes and no mid-run dump option. Disabled raw
file/device-Tracy dumps may remain; C++ CSV generation is independent.

There is no source proof from host metadata alone that100000 supports every
per-core marker in this 128-replay workload. Earlier stage4096/128 captures
passed with100000, which is useful prior evidence, not a substitute for this
run's completeness checks. Require child/parent exit0, normal close/run.json,
nonempty cpp_device_perf_report.csv, complete prefill and128 decode windows,
consistent per-replay op counts, and no dropped/full-buffer marker warnings.
If those fail, do not accept the capture or increase above the safe allocation
bound; inspect observed program counts and drain via normal ReadDeviceProfiler
outside measured windows only if separately justified, preserving final checks.

If safe-count control still aborts, keep faulthandler output and obtain a
native backtrace under an available debugger. `gdb` is absent in this
environment; `/usr/bin/lldb-20` is present. Do not install tooling, mutate
core_pattern, or re-enter250000 readback merely to obtain an abort stack.
A native debugger would need to follow the model child or launch the
`--no-capture-tool` child directly alongside the ordinary capture tool;
debugging only the outer Tracy parent misses the aborting process.

## Hypothesis verdicts and intervention boundary

- **uint32 host allocation overflow: established source defect; leading
  explanation of this abort, pending safe-count runtime control.** Smallest
  source fix is widening before multiplication, e.g. size_t(size) * banks /4,
  plus a meaningful host-side overflow regression. No C++ change is made by
  this report; a source fix requires the repository's prescribed build.
- **Blackhole timestamp rollover:** possible general profiler defect, but no
  marker endpoints or validator stack in this capture. Old fused-stage report
  cannot establish this capture's cause. Demoted.
- **Profiler marker capacity exhaustion:** unproven; failing250000 actually
  introduces a host overflow and does not justify another increase. Lower
  count must still satisfy normal completeness checks.
- **General host OOM/allocator failure:** no positive evidence; units correction
  removes the105.6GB argument. Native stack needed if still suspected.
- **Missing C++ CSV / GUI copy warning:** downstream issues. GUI copy uses the
  default generated path even though capture used a custom output folder;
  the actual .tracy file exists in the custom folder. Neither explains abort.

## Host-metadata capacity cross-check

`profile_selected_abort_host_metadata.json` is a stdlib CSV/JSON extraction
of the failed capture, using the exporter delimiter/quote convention from
`tools/tracy/process_ops_logs.py:434–435`. Every device has 5632 device-op
metadata records:5506 outside capture and126 inside capture. There are 126
TRACE_ENQUEUE_PROGRAM records,128 TRACE_REPLAY records, and one each BEGIN,
END,RELEASE per device. All four signposts survive and zero metadata JSON
records failed to parse. Estimated program submissions per device are
5506 +126×128 =21634, or21760if conservatively counting capture execution
as well. This supports100000 as a practical buffer choice. Optional device
markers can consume more than the guaranteed allocation per program, so this
host count is not proof against marker drops. Acceptance must still verify
126 program rows per replay,128 replay windows per device, and final normal
validation/no dropped-marker warning.

The CPU sizing reproduction additionally covered cores_per_bank14, 15, 16, 17,
18, and 20. At the architectural maximum20,100000 allocates 3840000000 bytes
without overflow, while250000 needs 9600000000 bytes but allocates 1010065408.
Thus the safe-count finding does not depend on inferring an exact board grid.

## Status

The report was persisted before the controlled hardware retry. The parent
then ran the 100000-count control. Its child closed normally and wrote
run.json with unchanged runtime SHA256 beginning 8b59370c, the original 4096/128
workload and selected BF16 attention CCL policy. The 16,841,898-byte C++ device
CSV is present. No C++ implementation change or validation bypass was used.

Host-only integrity checks are saved in
`profile_selected_sliding_100k/capture_integrity.json`. For each of four
devices, the C++ CSV contains 21634 program rows, exactly 128 replay sessions
(numbered 1–128),126 rows per session, and the same operation-ID set in every
session. All 2585 host-expected prefill operations were recovered per device.
There are no duplicate device keys, missing/nonpositive firmware spans, or
malformed host-op metadata records. The log contains no fatal, abort,
dropped-marker/full-buffer warning, or traceback. The final exported CSV passed the same window/program completeness checks
in `profile_selected_sliding_100k/final_capture_integrity.json`, and the parent
confirmed outer exit 0. The whole-layer summary also completed with exit 0.
The closed experiment and exact command/artifact hashes are recorded in
`AUTOFIX_profile_selected_abort.md`.

This controlled contrast verifies a practical configuration workaround while
preserving workload and capture integrity, and strongly supports the concrete
host-allocation overflow as the original failure's cause. The general C++
large-buffer sizing defect remains unpatched. This report does not grant a
model-stage pass or infer performance improvement.
