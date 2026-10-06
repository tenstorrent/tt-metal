# AutoFix: full-attention profiler teardown

Source-only diagnosis, 2026-09-25. No device commands, kernel changes, assertion
changes, or runtime changes were made for this investigation. The parent owns
the serial hardware rerun. This report concerns capture integrity; it does not
grant a stage pass or claim usable performance results from the failed capture.

## Finding

The strongest source-supported hypothesis is a **Blackhole wall-clock rollover
sampling error**, not exhausted profiler DRAM. The failing firmware-start
timestamp is exactly 200 cycles before the low 32-bit counter rolls over.
The profiler reads the clock words in the reverse of the documented Blackhole
latch order. That can move the following kernel-start timestamp approximately
`2**32` cycles into the past, which produces the observed nesting mismatch.

The complete missing kernel-start record is unavailable because marker
validation aborts before raw-device or C++ performance CSV output. Consequently
this is a strongly localized source hypothesis, not a recovered raw-marker
proof. The cleanest control is another unchanged, fully validated capture at
a fresh clock phase. No supported configuration found here guarantees repair
of the underlying timestamp-read ordering.

## Exact evidence

`profile_full_final.log:737–815` records successful real-weight full-attention
prefill and decode before teardown:

- Prefill PCC: `0.9994185024977819`.
- Decode endpoints: position 4096 `0.999885166997054`; position 4223
  `0.9998933678407687`; repeated trace output equal.
- `PERF_DECODE` and `PERF_DECODE_END` both appear; runtime audits and program
  cache guard pass. The host capture completes, but the model child aborts.
- Fatal: `profiler.cpp:2246`, start/end marker IDs do not match, UID
  `(4117507, 0, 18)`, chip 3, physical core `(11, 2)`, `TRISC_1`.

| Endpoint on the error stack | Marker ID | Decimal clock | Hex clock |
| --- | ---: | ---: | --- |
| `TRISC-FW` start | 9116 | 14946486189880 | `0xd97ffffff38` |
| `TRISC-KERNEL` end | 12726 | 14946486249152 | `0xd980000e6c0` |

`0xffffff38 = 2**32 - 200`. The endpoint span is 59,272 cycles and crosses
the low-word carry from high word `0xd97` to `0xd98`.

The surviving host metadata identifies the actual program:
`tracy/full/raw_final/.logs/tracy_ops_data.csv:441673`. UID `4117507` is a
cached `MatmulDeviceOperation`, hash `9703732986422432002`, BF16 input/output,
HiFi2, FP32 destination disabled, packer L1 accumulation enabled. Inputs are
logical `[1,1,1,2112]` and `[1,1,2112,2816]`; the first input pads its token
dimension to 32. This matches the shared MLP down projection. Its compute
binary is `bmm_large_block_zm_fused_bias_activation/14342646819721868678/`.
It is not a newly introduced native softmax or cache-gather program.

## Source chain

1. `tt_metal/hw/inc/internal/tt-1xx/risc_common.h:245–251` documents that reading
   `WALL_CLOCK_L` freezes the upper word for the subsequent `WALL_CLOCK_H` read.
   Blackhole's `c_tensix_core.h:505–509` implements **LOW, then HIGH** and labels
   the LOW read “latches high.” The Blackhole LLK helper
   `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:589–594` uses this order.
2. `tt_metal/tools/profiler/kernel_profiler.hpp:233–248` uses HIGH first on
   non-Quasar targets, stores the marker's first word, then reads LOW. The HIGH
   value may therefore belong to the previous LOW sample. A first sample after
   a rollover can combine the previous high word with the new low word.
3. `profileScopeGuaranteed`, in the same file around 751–799, writes firmware
   and kernel endpoints into reserved slots. Firmware start is in
   `trisc.cc:150`; kernel start/end scope is in `trisck.cc:86`. The kernel start
   follows firmware setup, so a firmware start only 200 cycles before carry is
   a concrete opportunity for the subsequent kernel start to cross it.
4. `profiler.cpp:1800–1840` reconstructs the timestamp directly as
   `(uint64_t(time_H) << 32) | time_L`, with no rollover repair. Markers are
   sorted by timestamp (`TracyTTDeviceData.hpp:231`).
5. `processDeviceMarkerData`, `profiler.cpp:2180–2260`, maintains a nesting
   stack. When it reaches the kernel end, the firmware start is on top because
   the kernel start is absent from its correct chronological location. This is
   exactly the reported mismatch.
6. `dumpDeviceResults`, `profiler.cpp:2492–2550`, validates markers before
   `generateAnalysesForDeviceMarkers` and before writing raw device data.
   Thus absent `cpp_device_perf_report.csv` is a consequence of the abort.
   The later Python fallback assertion about absent device logs is secondary.

## Supported rerun controls, in order

1. **Repeat the same full workload and capture settings into a new artifact
   directory.** Keep the same selected fusion, 4096-token prefill, 128 decode
   positions, trace, correctness checks, op-support count 100000, and final
   validation. A new process starts the program at a different clock phase;
   a successful run discriminates a transient sampling collision from a
   deterministic graph/program failure. Preserve the old failed logs. A retry
   is not a permanent profiler fix; accept its report only after normal device
   close, exit status zero, nonempty C++ device CSV, and complete signpost rows.
2. **If needed, isolate compiled instrumentation without changing arithmetic:**
   give the retry a new `TT_METAL_CACHE` directory and set
   `TT_METAL_DISABLE_PRECOMPILED_FW=1`. These are supported in
   `rtoptions.cpp:443–447,1756–1760` and
   `jit_build/build_env_manager.cpp:250–267`. Do not delete the shared cache.
   This tests binary consistency and also changes capture phase; it cannot
   establish that stale binaries caused the failure. Current
   `jit_build/build.cpp:852–858` already compares build-state flags/defines, so
   stale instrumentation is a secondary hypothesis, not the leading diagnosis.
   `--no-op-info-cache` only disables compressed host op metadata; it does not
   refresh firmware or kernel binaries.
3. **If buffer pressure is independently observed, drain between phases with
   `ttnn.ReadDeviceProfiler(mesh)`, keeping `TT_METAL_PROFILER_MID_RUN_DUMP`
   unset.** The binding is `ttnn/cpp/ttnn-nanobind/device.cpp:629–633`.
   `ReadMeshDeviceProfilerResults` finishes command queues before reading;
   `DeviceProfiler::readResults` reads DRAM and resets device control indexes;
   parsed markers remain on the host for the final validation pass. Suitable
   harness-only boundaries are after warm prefill and before `PERF_PREFILL`,
   after `PERF_PREFILL_END`, and before `PERF_DECODE`. These calls belong outside
   trace capture and outside measured signposts. This reduces accumulated
   device-buffer pressure and changes scheduling; it does not repair a
   malformed timestamp that was already sampled. No such harness edit was
   made in this investigation.
4. Raw device output can be enabled by omitting
   `--disable-device-data-dump-to-files` in a diagnostic rerun. It preserves
   more evidence **if final validation succeeds**; the same failure still
   happens before that file is written. Disabling push-to-Tracy may remain:
   C++ CSV generation is independent of raw-file and device-Tracy output.

There is no source reason to increase `--op-support-count` as the first fix.
The failed compile already uses `PROFILE_KERNEL=1` and
`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=4800000`. The count sizes DRAM, not the
reserved endpoint slots in L1 (`profiler_state_manager.cpp:43–68`). No
“Profiler DRAM buffers were full, markers were dropped” warning is present.
The fatal also means `had_dropped_markers` was false; recognized drops instead
take the explicitly tolerant path. Buffer exhaustion is therefore not proven.

## Controls that do not satisfy the requested report

- Do not enable `--dump-device-data-mid-run` as a workaround for this fatal.
  `dumpDeviceResults(true)` skips `processDeviceMarkerData`, then clears the
  accumulated map. A resulting report can evade the failing validation rather
  than demonstrate marker integrity.
- Do not enable accumulate mode: the Tracy CLI rejects its combination with
  `-r`, and C++ per-program analysis returns early because op IDs are absent
  (`tools/tracy/__main__.py:193–198`, `profiler.cpp:2467–2471`).
- Trace-only, SUM, and performance-counter modes change the collected evidence
  or analysis path. In particular SUM/performance counters disable the usual
  C++ runtime analysis in `tools/tracy/__main__.py:335–351`; they are not
  equivalent replacements for the required ordinary per-op capture.
- Do not disable the assertion, claim timings from the successful host-only
  capture as device timings, fabricate missing CSV rows, or infer a model
  correctness regression from this teardown failure.

## Verification boundary

The timestamp hexadecimal conversion and op-to-UID mapping were checked with
host-side source/log reads. No silicon, reset, target process, executable
profiler reproduction, or kernel build was used. The parent must record the
rerun outcome before treating the performance report as complete. If another
capture fails, preserve the next endpoint timestamps: proximity to a low-word
rollover is the direct discriminator for this diagnosis.

## Observed rerun outcome

`batch64_profile_full.log` exits0 with the same ordinary per-op instrumentation,
100000-op buffer count, uncached op metadata, and final C++ marker validation
enabled. No mid-run mode, assertion bypass, cache deletion, reset, or timestamp
repair was used. The final report at `tracy/full_batch64/whole_layer.json` contains
one complete prefill window and128 complete, equally sized decode replay windows.
The failed capture is excluded from all performance and telemetry values.

`tt-smi -ls --local` and `post_profiler_smoke.log` passed after the failed process;
subsequent correctness/watcher/profiling runs pass. The clock-read ordering remains
a source-backed hypothesis rather than a patched profiler fix. The stage anomaly
is controlled by discarding the invalid capture and requiring a fresh fully
validated capture; no model-output or watcher error accompanied it.
