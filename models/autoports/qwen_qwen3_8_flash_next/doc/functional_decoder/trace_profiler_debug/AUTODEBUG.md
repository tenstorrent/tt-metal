# AutoDebug: whole-trace profiler produces no device records

Inspection date: 2026-11-12. Checkout: `d7e788cd88199501a8546ddb7bc381a601af47a9`.

## Findings

**The device timing evidence is missing, although the numerical replay checks passed. The supplied artifacts do not identify the first failing profiler boundary. No environment-only repair is established by this inspection.** In this checkout the recorded profiling options compile the correct instrumentation, trace dispatch arms it, and the host explicitly reads both DRAM and residual L1. Neither disabled Tracy push, disabled C++ analyses, a missing final DRAM push, nor the 128-program setting sufficiently explains a completely empty device CSV.

The smallest justified model-local repair is to make profiling acceptance depend on actual device endpoints and preserve profiler state **before** destructive reads. Retain the current numerical/runtime implementation and classic profiler settings. The evidence does not justify a firmware change, larger buffer, or repeated full-layer run before locating the failing boundary.

This was inspection only: no target execution, device access, firmware change, reset, source edit, or AutoDebug relaunch. This report is the only file written. Streaming remains excluded as requested; the management firmware version in the log is not evidence that classic trace instrumentation is unsupported.

## Direct observations and revision consistency

Full paths below are relative to the repository root. Short `tests/*.py` references mean `models/autoports/qwen_qwen3_8_flash_next/tests/*.py`; `kernel_profiler.hpp` means `tt_metal/hw/inc/api/debug/kernel_profiler.hpp`; `profiler_common.h` means `tt_metal/hw/inc/hostdev/profiler_common.h`. Short profiler C++ filenames refer to `tt_metal/impl/profiler/`, and `build.cpp` refers to `tt_metal/jit_build/build.cpp`. Artifact shorthand:

- **P**: `models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/ple_whole_trace_prefill_v1/`
- **L**: `models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/logs/ple_whole_trace_prefill_v1.log`

| Observation | Evidence and limit |
|---|---|
| Raw device CSV contains zero data rows, exactly 246 bytes. | `P/raw/.logs/profile_log_device.csv:1–2`; independently counted with Python's CSV reader, skipping the architecture line. Header says Blackhole, 1350 MHz, max compute cores 120. |
| Eager versus both replays is bitexact for output and all four persistent states. | `P/profile.json:41–60`. This establishes the recorded numerical checks, not profiler emission. |
| The decoder harness is the recorded revision. | Its current SHA-256 equals `P/profile.json:28`: `ce8a986ecc5e0b2f9455476a2de40a3ea7102b879537d6399195bb795a0398ee`. |
| Profiling environment is recorded. | `P/profile.json:19–26`: DEVICE_PROFILER=1, TRACE_PROFILER=1, PROGRAM_SUPPORT_COUNT=128, CPP_POST_PROCESS=0, DISABLE_PUSH_TO_TRACY=1, TRACE_TRACKING=1. Allocation tracking is separately recorded and checked by the harness. |
| Instrumentation was actually compiled. | All 691 compiler command lines in L have `PROFILE_KERNEL=5` and per-RISC host buffer size 6144; 9 are FW_BUILD and 682 KERNEL_BUILD. BRISC firmware command at L:231, TRISC at L:30. Compiler output is not a loaded-firmware checksum. |
| The workload closed; the outer reporting command failed afterward. | L:19782 records successful checks; L:19785–19787 records device-driver close. L:19824/L:19862 reports malformed host CSV; final exit is 1 at L:19865. Do not describe the entire profiling command as successful. |
| The capture wrapper has changed since that run. | L's traceback references its earlier 63-line version; the current wrapper has operation-only re-export at `tests/profile_trace_capture.py:20–37`. This is a separate host parser repair and cannot manufacture device records. |
| Board bundle is reported as 19.8.1. | L:17. No firmware capability experiment was performed. |

Current model files are untracked in this checkout; the decoder content hash is the stronger artifact-to-source correspondence. Neither per-drain control snapshots nor the run's loaded-library map is retained.

## Producer, completion, and consumer ledger

### 1. Configuration and build are consistent with trace-only profiling

`tt_metal/jit_build/build.cpp:354–371` starts with option 1 and ORs `PROFILER_OPT_DO_TRACE_ONLY` (bit 2, defined in `tt_metal/hw/inc/hostdev/profiler_common.h:9–14`), yielding **5**. The log agrees. `tools/tracy/__main__.py:274–281,304–335` propagates the relevant CLI settings and trace tracking; the wrapper explicitly sets CPP_POST_PROCESS=0 before importing Tracy (`tests/profile_trace_capture.py:93–98`).

A tempting cache diagnosis does not survive the full hash path. Although the initial runtime compile hash does not list every profiler setting, the final JIT environment key hashes effective `defines_` (`build.cpp:547–566`), and target build-state hashing does too (`build.cpp:698–720`). PCH's umbrella contains standard headers, not `kernel_profiler.hpp` (`tt_metal/hw/inc/internal/pch.h:5–27`); its define exclusion does not demonstrate disabled profiling. The recorded compile commands already provide stronger evidence than a generic clean-cache suggestion.

Read-only inspection of the cached BRISC ELF named by L and the current host library also found agreement: Tensix profiler control address `0xce0`, data address `0xde0`, trace-status word `0xdc8`, 64 control words, five 2048-byte RISC buffers. Cached BRISC contains `init_profiler` at `0x4a9c` and `trace_only_init` at `0x5004`. A sampled compiled BRISC kernel uses the same status/marker addresses. Firmware `traceCount`/`wIndex` are imported by that kernel outside its own data/BSS clearing region. These checks refute the inspected layout/BSS mismatch hypothesis; they do not prove which libraries/ELFs were loaded in the completed run.

Binary checks used `riscv-tt-elf-nm -C -n` and `riscv-tt-elf-objdump -Cd` from `runtime/sfpi/compiler/bin/`. The cache root was `/home/ttuser/qwen38-flash/tt-metal-cache/tt-metal-cache2937208258715348137/`; the sampled kernel was `kernels/writer_untilize_col_parallel/18394968165793265653/brisc/brisc.elf`, with kernel data/BSS beginning at `0xffb00c80`, above imported profiler globals `0xffb00048`/`0xffb00060`. Source contracts: `tt_metal/hw/inc/internal/firmware_common.h:63–74` and `tt_metal/hw/toolchain/main.ld:232–239`. Host `nm`/`objdump -s` inspected Blackhole Tensix offset arrays at `0x5c56c0`, `0x18118c0`, `0x1811bb0` in `build/lib/libtt_metal.so`, interpreted using `tt_metal/llrt/hal/generated/dev_msgs_impl.hpp:11–29,342–368`.

### 2. Replay GO, not an additional user launch flag, arms trace collection

`FDMeshCommandQueue::enqueue_trace` submits the replay buffer (`tt_metal/distributed/fd_mesh_command_queue.cpp:1418–1425,1583–1612`). `tt_metal/impl/trace/dispatch.cpp:104–142` sends `RUN_MSG_REPLAY_TRACE` and waits for worker acknowledgements before executing the trace buffer. BRISC handles that signal with `DeviceIncrementTraceCount()` and `DeviceTraceOnlyProfilerInit()` (`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:546–561`).

`kernel_profiler.hpp:983–995` pushes the preceding replay if `traceCount>0`, increments the replay ordinal, stores it as the host counter, arms status `0x80000000`, and invalidates the L1 run sentinel. The guaranteed FW scope advances status to `0x40000000`; the first kernel scope advances it to `0x20000000`. Exits update fixed end slots; BRISC makes `ID_HH=0` valid and publishes its device end index (`kernel_profiler.hpp:769–815`). All Tensix RISCs use aggregate buffer index zero in this mode (`:108–112`).

The launch `enables` field selects processors (`tt_metal/hw/inc/hostdev/dev_msgs.h:195–198`) and is used as a validity condition at `brisc.cc:574`. There is no extra harness trace-profiler launch flag to set. Ordinary eager/capture calls do not themselves arm this trace state machine, so their lack of trace rows is not diagnostic.

### 3. Replay completion and profiler transfer are separate boundaries

The harness warm replay is blocking (`tests/profile_trace_decoder.py:194–200`); the measured replay synchronizes explicitly (`:205–210`). `ReadMeshDeviceProfilerResults` also finishes command queues, reads device data, processes it, and waits for host processing (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1229–1266`). Thus missing host synchronization is not supported by the inspected sequence.

Trace-only scope destruction **does not call `finish_profiler()`**: current endpoints stay in L1 (`kernel_profiler.hpp:800–815`). The next replay's `quick_push()` can transfer the preceding replay to DRAM. Its early exits include linked NoC command buffers, zero DRAM profiler address, or an existing dropped-data bit; capacity can also reject the write (`:623–640,690–707`). Successful push flushes posted writes and advances the host end index.

These gates can explain missing **DRAM** data. They cannot alone explain this CSV: trace-only host reads and processing explicitly select `DRAM_AND_L1` (`tt_metal_profiler.cpp:931–935,1041–1042`). `profiler.cpp:2591–2599` snapshots control, DRAM, and L1 **before** resetting controls. `profiler.cpp:1553–1561` already exempts trace-only from the legacy zero-host-end early return. The current last replay therefore need not be pushed to DRAM to be readable.

### 4. The host has no relevant CSV-suppression switch enabled

At final dump, `processDeviceMarkerData()` runs irrespective of CPP_POST_PROCESS (`profiler.cpp:2499–2516`), translating FW/KERNEL hashes into `TRACE-FW`/`TRACE-KERNEL` for the aggregate lane (`:2180–2195`). CPP_POST_PROCESS only gates analyses; file writing is still enqueued (`:2514–2522`). DISABLE_PUSH_TO_TRACY only exits GUI pushing (`:2663–2670`). CSV serialization writes a header and then all retained parsed markers (`:1190–1257`); there is no zone-name export filter here. Unrecognized hashes retain unidentified markers (`:1916–1921`), rather than discarding every row.

Close invokes the profiler's final dump and waits for it (`tt_metal/impl/device/firmware/profiler_initializer.cpp:102–108`; `profiler_state_manager.cpp:154–175`). Under this path, a header-only file means the final export had no parsed markers. The artifacts do not tell whether control indices were empty or payloads failed the parser's run-start contract.

### 5. Capacity and counter expectations

Without SUM, sizing is `2 words × (2 ID markers + 4 guaranteed markers) × 4 bytes × 128 = 6144 bytes/RISC` (`profiler_state_manager.cpp:62–94`). L1 is 512 words/2048 bytes (`profiler_common.h:167–174`). FW/KERNEL trace endpoints overwrite four fixed marker slots, rather than adding a set for every operation. Two replays entail one replay-start push; a single bounded L1 chunk fits 6144 bytes. The layer's 28,073 raw operation-ID span is therefore not evidence of overflowing a 128-program trace buffer. Optional marker loss remains distinguishable through drop bits, not proven here.

Trace-only deliberately skips decoding the device trace counter (`profiler.cpp:599–600,1723–1726`); zero returns invalid trace metadata (`:1924–1927`). Consequently raw `trace id` and `trace id counter` columns can be **blank by design**. The replay ordinal instead comes from `ID_LL`, exported as encoded `run host ID` (`kernel_profiler.hpp:989`; `profiler.cpp:1727–1730`; encoding in `tt_metal_profiler.cpp:1174–1182`). The model analyzer already uses that ordinal (`tests/report_trace_performance.py:122–125`). Blank trace-ID columns do not explain missing endpoint rows.

## Minimal model-local repair and discriminating checks

These are proposed harness changes/follow-ups, not changes performed or hardware-validated fixes.

1. **Separate numerical success from timing evidence.** `profile_trace_decoder.py:213–217` checks only finiteness and exact output/state equality. Keep that evidence, but require successful device analysis before declaring profiling successful. After close/CSV serialization, require nonempty, complete aggregate endpoints for both replay ordinals, matching per-core coverage and interval ordering. Reuse `report_trace_performance.py:107–187`; it already rejects the supplied empty CSV at the endpoint-coverage check. Its coverage check only compares **observed** core sets, so additionally compare against an independently established inventory of participating cores: the same missing core in both replays otherwise escapes detection. For the canary, require its exact known core set. Do not check file size immediately after `ReadDeviceProfiler`: normal reads accumulate parsed records in memory; final close writes them.

2. **Record pre-drain state in the model harness.** Capture after synchronization and immediately before every `ReadDeviceProfiler`; its reset zeros trace status/end indices (`profiler.cpp:1466–1499`). Use read-only translated-core access, whose coordinate contract is documented at `ttnn/cpp/ttnn-nanobind/cluster.cpp:126–145`. Record exact device/core coordinates and addresses along with raw words. Derive addresses/offsets from the matching installed HAL/dev-msg layout, rather than silently assuming the values inspected here apply to another build. No writes to profiler memory are needed.

   | Value | Current control/aggregate word | Expected evidence after a participating completed replay |
   |---|---|---|
   | Host end, device end | control 0, 24 | Device end at least 12; host end may be zero on first replay. |
   | DRAM address, flat ID, core count | control 50, 54, 55 | Match initialized HAL routing/buffer configuration. |
   | Dropped, done, trace status | control 56, 57, 58 | Trace status `0x20000000`; done is not a reliable trace-completion requirement because normal finish is bypassed. |
   | Run sentinel and identity | aggregate words 0–3 | `ID_HH=0`, `ID_HL=0`, ordinal in `ID_LL` (1 or 2). See first-replay identity caveat below. |
   | FW start/end, kernel start/end | aggregate words 4–11 | Actual timestamped marker pairs, not untouched zero-timer placeholders. |

   Current indices follow `profiler_common.h:109–142`; marker slots follow `:26–40`. Sample known participating cores first, then compare all participating cores if necessary; idle-core samples alone cannot test emission.

3. **Use the snapshot to select the next intervention.**

   | Snapshot result | What it distinguishes / next inspection |
   |---|---|
   | No armed status or real markers on participating cores | Follow replay GO coverage, the actually loaded BRISC binary, and whether sampled addresses/cores match execution. More CSV options cannot fix absent production. |
   | Status stuck at FW_START or KERNEL_START | Identify the missing scope transition and its actual enabled processor/kernel path. Numerical completion alone does not identify which instrumentation was consumed. |
   | Status ALL_ENDS, device end >0, valid sentinel and timestamps | Compare direct bytes with the host fast-dispatch L1 read and parser inputs. If the completed raw CSV is still empty, the failure is downstream of marker production. |
   | Markers present but no zero/zero run sentinel | Parser silently skips ordinary zone endpoints before a sentinel (`profiler.cpp:1689–1717`). Investigate initialization/overwrite; preserve the actual bytes. |
   | Positive device end but zero host end on replay two | Test `quick_push` rejection conditions separately. The host should still be able to consume L1. |

4. **Do a tiny, independent trace canary before loading the full layer in a future authorized run.** Use a separate process/output directory and the same classic options, one known participating Tensix program, two replays and the same drains. Check endpoint serialization after close. A failed canary localizes the problem outside decoder math and avoids another full-layer load. A passed canary establishes a useful controlled contrast; then inspect the model's state snapshots. This investigation did not execute the canary.

5. **Preserve effective configuration and binary provenance.** Keep DEVICE_PROFILER=1, TRACE_PROFILER=1, PROGRAM_SUPPORT_COUNT=128, CPP_POST_PROCESS=0, DISABLE_PUSH_TO_TRACY=1, TRACE_ALLOC_TRACKING=1 and the dedicated output path set before imports. Record all relevant `TT_METAL_*` variables, imported module paths, loaded `libtt_metal.so` path/hash, and actual JIT paths/flags. The current environment subset omits some relevant controls (`profile_trace_decoder.py:84–86`). Keep streaming, watcher, mid-run dump, dispatch profiling, and accumulate excluded by the existing harness. Do not add profiler sync as a remedy: its source declares sync unsupported with active fast dispatch through a debug assertion (`tt_metal_profiler.cpp:632–634`), which does not enforce the restriction in Release.

   A fresh model-local `TT_METAL_CACHE`, with `TT_METAL_DISABLE_PRECOMPILED_FW=1`/`TT_METAL_FORCE_JIT_COMPILE=1` if a canary still needs cache exclusion, is a **diagnostic contrast**, not an evidenced repair (`rtoptions.cpp:507–514,905–910,1995–1999`). Do not delete shared caches or use a full-layer recompile as the first check.

## Other Potential Issues — not the established cause of zero rows

- **First-replay L1 flat identity is incomplete before the first successful push.** `init_profiler()` zeros ID fields and preserves only trace bits (`kernel_profiler.hpp:172–181`); `trace_only_init()` does not populate the flat ID. `quick_push()` fills it at `:654–656`. Thus the first warm L1 read can contain flat ID zero even on other cores, violating the host sanity assertion at `profiler.cpp:1781–1790`. This is a concrete source contract discrepancy, but it does **not** predict header-only output in the inspected Release build: `TT_ASSERT` is non-enforcing without DEBUG (`tt_stl/tt_stl/assert.hpp:155–167`; `build/CMakeCache.txt:234` says Release), and marker attribution uses the sampled core. In a DEBUG build it instead predicts an assertion on a participating nonzero-flat-ID core. Keep it secondary.

- **Reads change profiler state and can permit duplicate/stale re-export.** Control reset clears status/indices while preserving L1 bytes and firmware `traceCount`; the next replay can push the previously read L1 again. Retained marker sets deduplicate identical records (`profiler.cpp:1994–2015`), so this does not erase all data. Removing the warm drain alone is not an established fix: subsequent correctness/reset device work can extend the warm trace's ending slots while ALL_ENDS remains armed. The current harness deliberately drains before that work (`profile_trace_decoder.py:198–203`). A separate pre-created unmeasured flush trace could test a DRAM-only contrast, but must share required core/GO coverage and run outside measurement. It is not needed by the documented DRAM_AND_L1 path and should not replace pre-drain diagnosis.

- **Loaded-artifact mismatch or external overwrite remains unproved.** Inspected current host/firmware layouts agree, and library build timestamps precede the run. The original loader map and per-drain data are absent. Do not promote an unknown `LD_LIBRARY_PATH`, stale cache, mailbox overwrite, or firmware-version story without that correspondence.

## Claim review

The headline was rechecked against the complete sequence: eager/capture without trace rows is expected; completed replays should expose L1 endpoints; host reads include L1 and wait for completion; final dump renames endpoints and serializes them independently of both disabled options. Capacity and cache checks do not supply the missing causal link. The first-read identity defect was demoted because it predicts a DEBUG assertion rather than the observed Release empty CSV. The host CSV parser failure was kept separate because it occurs after the independent raw-device failure.

The unresolved fact is concrete: **what were the participating cores' end indices, sentinel words and trace status immediately before the warm and measured drains?** The saved artifacts do not contain those values. Until they are observed, identifying a specific producer/drain defect or promising an environment-only repair would exceed the evidence.
