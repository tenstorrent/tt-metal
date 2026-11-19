# AutoDebug: v4 profiler abort after completed traced decode

## Leading finding

**The legacy device profiler has an unsafe high/low wall-clock read. A torn timestamp at the low-word rollover is the strongest explanation for this abort.** The reported end timestamp, `7838315315204`, is exactly `(1825 << 32) | 4`. The reader can combine high `1825` from before rollover with low `4` from after rollover, backdating an otherwise valid FW end by `2^32` cycles. Host marker pairing sorts by timestamp, so the end can precede its own start and trigger the reported empty-stack fatal.

This is a directly established source defect and a highly specific match to the observed timestamp, **not a proved reconstruction of this individual device event**. The failed run has no final device CSV containing the corresponding start. A valid timestamp four cycles after rollover plus an independently lost start remains possible. No hardware reproduction, device access, reset, model execution, or source edits were performed.

The intervention boundary is the profiler's timestamp sampling, not the decoder's BinaryNG arithmetic. Stage scope forbids shared/native edits. There is no identified existing environment switch that makes this legacy timestamp read coherent. A fresh, validated retry and timing-only harness controls are permitted mitigations/discriminators; they are not a deterministic repair of the primitive.

## Revision and direct evidence

Investigation checkout: native Git HEAD `4c643884bf51a39fc1166e0b86b86bd373bedfee`, with an existing dirty model/test worktree. Current and failed-v4 model SHA256:

`b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`

Current, passing-v3, and failing-v4 `tests/run_decoder.py` SHA256:

`b8242ff98ce14e7465f9aae093f1f85767a73bc4d0905eab5066c472f200bbeb`

The failure directory is `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/profile_v4_failed_01/`. Paths below are repository-relative. References to `profiler.cpp` mean `tt_metal/impl/profiler/profiler.cpp`; references to `kernel_profiler.hpp` mean `tt_metal/hw/inc/api/debug/kernel_profiler.hpp`.

Direct observations:

- `profile_0.log` is an appended history. The final invocation begins at line 5325, `2026-11-13 06:02:44.215 UTC`; earlier successful runs in that file are separate sessions.
- Final measured decode signposts are lines 7681–7682, `06:03:06.643` through `06:03:08.379`. The fatal follows at line 7683, `06:03:08.506`.
- The offending marker is `ZONE_END`, `TRISC-FW`, static ID `55138`, UID `(15456256, 0, 9)`, chip 0, physical core `(2,3)`, `TRISC_0`, timestamp `7838315315204`, op `BinaryNgDeviceOperation`. The fatal identifies `profiler.cpp:2210` and firmware `tt_metal/hw/firmware/src/tt-1xx/trisc.cc:153`.
- Final `correctness.json` was written at `06:03:08.380797995`, between the end signpost and abort. It records 16 measured replays, deterministic correctness replays, decode PCC `0.9999779064754709`, and a passing runtime fallback audit. The harness writes this JSON before `close_mesh_device`, so its presence does not establish successful profiler close.
- Final provenance records support count `20000`, device profiler `1`, no-op-info-cache `1`, and watcher/allocation-tracking unset. The environment whitelist in `tests/provenance.py:33–44` omits other profiler/debug switches. Git HEAD and package version are not a native-binary hash; the recorded TTNN package version is `0.1.dev1+ga464944`.
- Final `raw/.logs/tracy_ops_data.csv` contains 171 captured operations and 171 trace-program enqueue messages, with 20 replay messages: four before the measured window and 16 within it. This is host evidence, not certification of complete device marker data.
- No `Profiler DRAM buffers were full` / `markers were dropped` warning appears in the supplied log. The failure is not a reported allocator limit or watcher assertion.
- Passing `doc/fused_decoder/profile_v3/profiler_completeness.json` records 181 operations per layer-0 replay, 78 per layer-3 replay, 20 total / 16 measured replays for each, identical sequences, and no drops. These replay counts were also checked against preserved operation CSVs. Passing-v3 model SHA256 is `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`; its harness, recorded native HEAD, package versions and listed profiler settings match v4.

The supplied correctness/watcher pass is consistent with a profiler-only fault. This investigation did not rerun or independently certify the separate watcher suite.

## Source causal chain: clock sampling to delayed fatal

### 1. Source path indicated by the recorded configuration uses the unsafe reader

`trisc.cc:153` opens `DeviceZoneScopedMainN("TRISC-FW")`. With the observed compile setting `PROFILE_KERNEL=1`, this is the ordinary fixed-slot `profileScopeGuaranteed<hash,0>` path (`kernel_profiler.hpp:1052–1064`). Its constructor and destructor write the FW start and end through `mark_time_at_index_inlined` (`:792–815`). Both endpoints are unconditional in this ordinary mode.

For Blackhole, that function executes (`kernel_profiler.hpp:99–100,253–257`):

```cpp
// HIGH_INDEX == 1; LOW_INDEX == 0
p_reg = reinterpret_cast<...>(RISCV_DEBUG_REG_WALL_CLOCK_L);
marker_header = ... | (p_reg[1] & PROFILER_MARKER_TS_HIGH_MASK);
marker_low = p_reg[0];
```

There are two separate volatile MMIO reads, with a profiler-header store between them, and no high-word verification. `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h:195–197,228–229` distinguishes:

| Register | Offset | Meaning for this reader |
| --- | --- | --- |
| `WALL_CLOCK_L` / `WALL_CLOCK_0` | `0x1F0` | low word, index 0 |
| `WALL_CLOCK_1` | `0x1F4` | live high word, index 1 used by legacy profiler |
| `WALL_CLOCK_H` / `WALL_CLOCK_1_AT` | `0x1F8` | sampled high word, index 2 |

Sibling code supplies concrete contract evidence:

- `tt_metal/hw/inc/internal/tt-1xx/risc_common.h:235–240` explicitly says reading L freezes the upper word for readback, then reads L followed by H.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:505–509` reads L first, with the comment `latches high`, followed by H.
- `tt_metal/hw/inc/internal/profiler/kernel_profiler_streaming.hpp:87,155–162` explicitly identifies index 1 as live, index 2 as snapshot, and reads L then snapshot H.
- `tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp:47–67` uses the same live high register as the legacy path, but retries high/low/high until the two high reads agree.

Thus the issue is not that high-before-low is inherently invalid for all registers. **This reader selects the live high register and omits the verification required to combine it coherently with low.**

### 2. The observed number matches the torn-read failure shape

Static arithmetic on the actual logged value gives:

```text
reported end = 7838315315204 = (1825 << 32) + 4
candidate coherent end       = (1826 << 32) + 4 = 7842610282500
difference                   = 4294967296 cycles
```

For a concrete illustrative sequence, not recovered device evidence:

```text
FW start:          (1825 << 32) + 0xFFFFFF38 = 7842610282296
read end live H:   1825
clock rolls over
read end L:        4
true end:          7842610282500  (204 cycles after illustrative start)
recorded end:      7838315315204  (4294967092 cycles before illustrative start)
```

A cheap CPU-only arithmetic/sorting check reproduced this ordering reversal and an empty-stack end-first outcome. It demonstrates the causal mechanism, not a hardware reproduction. At the nearby passing device CSV's reported 1350 MHz, one low-word epoch is about `3.181457256 s`; the failed run itself has no final frequency header. This is not the full 44-bit timestamp wrapping: high `1825` is far from `4095`.

### 3. Parsing preserves the bad value; pairing sees sorted order

`profiler.cpp:1759–1802` reconstructs each endpoint as `(uint64_t(time_H) << 32) | time_L` and passes it directly to marker insertion. `opTime_H/L` in that parser are used to timestamp `ZONE_TOTAL` records; they do not unwrap or repair endpoint timestamps.

`tt_metal/third_party/tracy/public/common/TracyTTDeviceData.hpp:253–275` orders markers primarily by timestamp, with chip/core/RISC/static-marker-ID tie breakers. `profiler.cpp:1588–1589,1677` collects sets per physical core/RISC. `processDeviceMarkerData` creates a stack for each such set (`:2160–2162`), pushes starts and pops ends (`:2201–2245`). An end encountered with an empty stack raises this fatal (`:2204–2210`). It does not match endpoints by execution UID.

A backdated end can therefore encounter an empty stack even when both raw FW endpoints were written. If the backdated end lands inside another active zone, the same primitive defect could instead yield an ID mismatch. The actual surrounding lane records are missing, so this report does not claim to reconstruct the precise stack history or the matching start's timestamp.

### 4. Completed drains and a close-time failure are expected together

The harness's synchronization ledger is:

| Phase | Host wait / profiler action |
| --- | --- |
| Before measured prefill | synchronize and `ReadDeviceProfiler` (`run_decoder.py:84–88`) |
| After measured prefill | synchronize, end signpost, profiler read (`:89–94`) |
| Decode correctness | three blocking replay executions plus host output reads (`:158–161`) |
| Decode warmup | fourth blocking replay, synchronize, profiler read (`:183–187`) |
| Measured decode | 16 blocking executions, each followed by profiler read (`:189–194`) |
| End of measured decode | synchronize, end signpost (`:195–196`) |
| Completion | release trace, write correctness, close mesh (`:201–205`) |

`ReadMeshDeviceProfilerResults` finishes every hardware CQ before reading (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1229–1233`), reads the buffers, processes the snapshots and waits for the mesh thread pool (`:1254–1266`). Ordinary DRAM reads snapshot control buffers, read profiler DRAM, then reset device controls (`profiler.cpp:2579–2584`). Parsing accumulates host-side marker sets; it does not run the stack validator after every ordinary read. Queue completion here is an execution-notification boundary, not proof that the worker profiler epilogue finished; the separate source-order concern below remains relevant.

Mesh close performs a final FD read (`tt_metal/distributed/mesh_device.cpp:1144–1146`). Profiler post-teardown calls `cleanup_device_profilers` (`tt_metal/impl/device/firmware/profiler_initializer.cpp:102–108`), which invokes final `dumpDeviceResults` (`profiler_state_manager.cpp:154–175`). That dump validates accumulated markers before analyses and file output (`profiler.cpp:2499–2522`). This explains both the late fatal and the missing final device CSV. Sleeping after `PERF_DECODE_END`, adding another synchronize, or moving `release_trace` does not repair a timestamp already stored in those sets.

## Buffer sizing and replay UID adjudication

The actual supported capacity knob is `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (`tt_metal/llrt/rtoptions.cpp:1347–1354`), correctly set by `tests/collect_fused_perf.sh:8`. `PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC` is a derived compiler define, not an environment variable accepting a program count.

With SUM off, constants in `tt_metal/hw/inc/hostdev/profiler_common.h:167–174` and sizing in `profiler_state_manager.cpp:62–94` give:

```text
2 uint32/marker × (2 ID + 4 guaranteed markers) × 4 bytes = 48 bytes/support unit
20000 units × 48 bytes = 960000 bytes per physical core per RISC
960000 / 4 = 240000 uint32 words per slice
```

The final compilation commands show `PROFILE_KERNEL=1` and `PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=960000`, agreeing with this configuration. Capacity is not pooled across all device cores. Actual optional markers can enlarge a chunk up to the 2048-byte L1 profiler buffer: 20000 is nominal minimum-size program support, not a guarantee of 20000 arbitrarily instrumented programs. A slice holds 468 maximum-size chunks plus remainder. For the offending TRISC lane on the ordinary one-chunk-per-program path, the 171 captured programs would require at most `171 × 2048 = 350208 bytes` per replay, below 960000. This is not a universal bound for BRISC/NCRISC, whose `quick_push()` path can emit multiple chunks per program (`kernel_profiler.hpp:623–707`); that body excludes TRISC. The bound concerns the measured per-replay interval, not the longer prefill/warmup intervals or unidentified extra device programs.

The device reservation scales by RISCs and cores per DRAM channel (`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:40–52`). Here its maximum is `960000 × 5 × 20 = 96000000 bytes/channel`, or 768000000 bytes across eight channels. Actual host backing uses the SoC descriptor's per-channel core ceiling (`tt_metal_profiler.cpp:798–816`). Blindly doubling support count increases reserved device memory substantially and changes timing.

Ordinary capacity exhaustion is secondary because:

1. `kernel_profiler.hpp:561–598` sends the complete chunk if it fits, otherwise the guaranteed 48-byte prefix, otherwise nothing. Both overflow branches set `DROPPED_ZONES`.
2. FW start/end are words 4/6 within that prefix (`:769–815`), so this truncation preserves both together or drops both together. It does not selectively drop the FW start.
3. The host checks saved dropped flags and sets `had_dropped_markers` before parsing (`profiler.cpp:1648–1662`). With that flag set, it skips orphan ends instead of taking this fatal (`:2204–2213`). Normal control resets retain the saved snapshot for processing.
4. The log contains no drop warning, and per-replay drains already exist in the exact hashed failing harness. Adding those same drains is not a new remedy.

This does not rule out unflagged corruption or a different loaded binary. It does rule out treating support count 20000, by itself, as an established insufficient-capacity diagnosis.

UID `(15456256,0,9)` means encoded host-program ID, trace ID 0, ninth execution of that trace. The encoded ID decodes to program 15094/device 0 (`tt_metal_profiler.cpp:1174–1191`); it is not an occupancy counter. `profiler.cpp:1924–1950` derives replay count from replay history, which drains do not reset. With three correctness replays and one warmup, replay 9 is measured replay 5. Host capture metadata also identifies the operation as the 157th of 171 captured programs.

Repeated runtime IDs across trace replays do not themselves create a marker-set collision: UID fields are not in the comparator. A collision additionally requires the same timestamp and static marker ID on the same lane. Start/end marker types are also absent from the comparator, so exact equal-timestamp endpoints are a real general hazard; fixed-slot parsing normally sees start first and loses the end in that case, producing a leftover start rather than this reported orphan end.

## Secondary finding: execution completion precedes the profiler epilogue

There is an additional concrete ordering gap, independent of buffer capacity:

```text
TRISC: kernel returns -> tensix_sync -> writes RUN_SYNC_MSG_DONE
       -> FW scope destructor writes END and publishes profiler end index
BRISC: waits subordinate DONE -> notifies dispatcher of program completion
       -> FW scope destructor issues profiling writes and waits for posted writes sent
```

The ordering follows `trisc.cc:219–227`, `brisc.cc:685,735–750`, and the destructor/finish path in `kernel_profiler.hpp:799–815,561–619`. The posted-write flush specifically waits for writes to be sent (`:352–364`); this is not an independently established host visibility acknowledgement. BRISC's completed-worker notification is before its profiler writes, not after them. `ReadDeviceProfiler` selects NORMAL state (`ttnn/cpp/ttnn-nanobind/device.cpp:765–769`), and an ordinary worker read does not poll `PROFILER_DONE`: the poll in `tt_metal_profiler.cpp:895–927` is guarded by dispatch-only teardown profiling (`:867–868`). The fast core-read command waits on worker completion counts (`tt_metal/impl/device/dispatch.cpp:198–216`); the corresponding dispatch-local NoC barrier (`tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1103–1135`) does not by itself prove that a worker's subsequent profiler writes have completed.

Consequently, CQ finish alone cannot establish a source-level happens-before relation from worker profiler publication to the host's control/DRAM snapshot/reset. This is a credible secondary race candidate. Actual overlap in this run is unproved: host round trips and additional instructions may provide enough incidental delay. Copying an incompletely finalized TRISC prefix would more directly miss its END; explaining this orphan END instead needs stale/partial snapshot behavior or another unobserved interleaving. This candidate does not specifically predict the observed low word `4`, so it ranks behind the clock defect. Completed replay drains do not rule it out.

The smallest stage-local discriminator is a short documented delay immediately after a blocking replay and before its profiler read, outside the measurement of device kernel durations. It is a timing experiment, not a proven readiness fence, and also shifts clock phase. Prefer comparison with a similarly sized delay after the read: both shift subsequent replay timing, while only the former gives the just-completed profiler producer extra settling time. Neither placement provides a deterministic contract repair. A real protocol repair would require a native producer/consumer readiness guarantee, outside stage scope.

| Candidate | Direct source evidence | Match to this event | Remaining requirement / smallest discriminator |
| --- | --- | --- | --- |
| Torn clock read (leading) | Live high then low without rollover verification | Specifically predicts very small low word and an end sorted too early; actual low is 4 | Matching raw FW start/neighbors; separate diagnostic raw-row capture |
| Completion/epilogue race (secondary) | Worker completion is published before profiler finalization; ordinary read lacks worker DONE poll | Explains possible unflagged snapshot errors despite blocking execution | Actual producer/read overlap and selective loss must occur; compare delay-before-read vs delay-after-read |
| Ordinary capacity overflow (weak) | Finite per-core/RISC slice with explicit drop handling | Conflicts with no warnings, guaranteed FW prefix, fatal guard, and per-replay drain | Demonstrate drops/occupancy threshold before increasing capacity |
| UID/set collision (weak) | Timestamp/static-ID set key omits UID and marker type | Reused UID alone is insufficient; equal-time fixed-slot collision usually loses END | Demonstrate equal timestamps or unintended repeated records |

## Scope-safe remediation and minimal discriminators

These are proposed follow-ups for the stage owner, not experiments performed by this investigator.

1. **First retry: preserve the workload and use a new artifact directory.** Keep b163, 20000 support units, `--no-op-info-cache`, all 16 measured replays and the existing drains; keep execution serialized. For the collection script, set `QWEN_FUSED_ARTIFACT_DIR` to a new, nonexistent model-doc directory before invoking it. A clean retry can avoid the rare sampling coincidence without any model/native edits. Accept it only after clean process close, fresh device CSV/perf report, 16 complete identical device replay sequences and no drops. Record any retries; do not turn this into unlimited retries until one passes. Success is mitigation evidence, not proof that the reader is fixed.
2. **Timing-only contrast if it recurs.** In the model-local harness, optionally insert a documented short delay after the prewindow profiler read and before the measured decode signpost, outside all timing windows. Compare small delays such as 0, 0.1 and 0.3 seconds while keeping operation count/order and drain cadence fixed. Preserve every failing timestamp and inspect its low word. The race predicts bad endpoints concentrated close to zero and, when corresponding rows are available, an approximately `-2^32` timestamp displacement. A single passing timing-shifted run is not conclusive. This is a proposed harness intervention; this investigation made no such edit.
3. **Preserve raw rows if an event-level discriminator is needed.** Existing CLI switches `--dump-device-data-mid-run --disable-device-data-push-to-tracy --no-runtime-analysis`, together with an unset `TT_METAL_PROFILER_CPP_POST_PROCESS`, can dump the collected raw device CSV after each read. Use a separate diagnostic destination and keep file dumping enabled. `tt_metal_profiler.cpp:1050–1051` calls midrun dump, and `profiler.cpp:2499–2508` skips stack validation for that dump before writing/clearing the sets. Disabling device Tracy push avoids forwarding malformed zones. Inspect the same UID/chip/physical-core/RISC/static marker ID: a present FW start with an end about `2^32` too early strongly distinguishes clock tearing from a missing start. Also inspect neighboring and nested kernel markers. This mode changes validation and may still fail in later Python processing; its raw rows are diagnostic evidence only, never an accepted replacement profile or a demonstrated fix.
4. **Capacity control only if evidence warrants it.** An environment-only support-count increase is available, but more memory and changed compilation timing confound a simple pass/fail comparison. Check the actual derived define, drop warnings and per-slice byte use. Keep replay count/drain cadence unchanged. A larger-buffer pass without drop evidence does not establish overflow. Reducing to one replay can localize a failure but cannot satisfy the required 16-replay profiling evidence.

There is no source-supported reason to expect `--no-runtime-analysis`, disabling Tracy push, another synchronization, or another ordinary drain alone to repair this abort. Final endpoint validation precedes the C++ analyses and Tracy push (`profiler.cpp:2499–2520`). `--device-trace-profiler` changes the data being measured and cannot replace the per-operation profile. Do not synthesize missing starts, silently add `2^32` to exported timestamps, or weaken acceptance checks.

## Other potential issues and evidence traps

- **Stale reports are present in the failed archive.** Its three `raw/reports` directories are earlier successful sessions, latest `04:46:24`. That latest operation CSV is byte-identical to preserved v3 layer-0 `ops.csv` (SHA256 `c027ef6b95d7902ffe3825898542a4e44f66a3d336353e9e6d8c8f06bab6690a`). None certifies the final 06:02–06:03 v4 run. Fresh directories are a necessary harness hygiene correction.
- **Native build identity remains incomplete.** Recorded HEAD, package version, build path and source line are not an archived hash/disassembly of the loaded profiler library and device firmware. If source and binary differ, re-evaluate the sampling/pairing path against the actual build before claiming definitive runtime causality.
- **BinaryNG is attribution, not proof of an arithmetic defect.** The failed zone is firmware scope, and trace correctness passed. Captured metadata identifies a BF16 tiled multiply with lhs SiLU; it does not establish that its math kernel corrupted profiler storage. No source-level model arithmetic violation was established here.
- **Unflagged marker loss/stale copies remain alternatives.** They require an additional unobserved mechanism to explain the missing start. Neither raw marker words nor the matching start survived in the failed final device report. The exact low-word rollover signature makes the clock reader the leading explanation, while this uncertainty remains explicit.

## Claim review and confidence

After drafting, claims were checked against the actual register addresses, fixed-slot scope, overflow branches, parser, timestamp comparator, pairing stack, close ordering, hashed harness provenance and final-session boundaries. A separate adversarial read confirmed that the capacity theory does not account for guaranteed-prefix handling or the no-drop fatal guard. Post-draft review tightened the per-replay size bound to the TRISC one-chunk path and retained the completion/epilogue ordering gap as a secondary finding. The clock finding survives those checks as a direct source inconsistency with a matching event signature. The missing counterpart/neighbor records prevent a definitive reconstruction of this one failure.

No model, harness, native source, device state, or pre-existing evidence was changed. Only this report and its root `AUTODEBUG.md` copy were created. No root `AUTODEBUG.md` existed when this investigation began.
