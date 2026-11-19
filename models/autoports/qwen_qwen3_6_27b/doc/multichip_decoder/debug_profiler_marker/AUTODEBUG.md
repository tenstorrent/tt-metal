# AutoDebug: incoherent Blackhole profiler timestamp at clock rollover

Inspection only, 2026-11-13. No devices executed and no source changed. **Leading diagnosis: the legacy device profiler samples the live high clock word before the low word, permitting a timestamp to jump backward at a low-word rollover.** The failing END's low word is exactly **10**, making this a specific explanation for the orphan `pp_intra` END. The source defect is established; its occurrence in this execution remains an inference because the corresponding raw START/neighbor markers were not preserved in a device CSV.

## Direct observations

- `failed.command.json` specifies TP4, layer 0, B32/S33, `--trace --profile-drain --repeats 5`, device profiling, and program support count **5000**. The current production, runner and optimized-baseline SHA256 values match the manifest (`082c82fa…`, `0095c650…`, `315187bc…` respectively), at HEAD `0915d931…`.
- The reported sequence needs correction: `failed.log:33–40` contains three complete prefill signpost pairs **and `PERF_DECODE` / `PERF_DECODE_END`**. Line 41 prints the final numerical/trace report (`pass: true`, finite, eager/replay equality). The profiler thread aborts at line 42, during subsequent final processing. This does not establish a GDN execution failure, and the printed correctness result does not establish successful collection.
- The orphan is UID `(904195, INVALID_NUM, INVALID_NUM)`, chip 3, physical core `(12,11)`, TRISC_0, `ChunkGdnDeviceOperation`, `pp_intra` at `chunk_gdn_math.hpp:990`, timestamp **3835405795338**. The two `9223372036854775808` fields are the profiler's `INVALID_NUM`, not evidence of a corrupt trace ID (`TracyTTDeviceData.hpp:154`; `profiler.cpp:1924–1940`).
- The retained sibling host log `../tracy/delivery_profile_l0_b32/.logs/tracy_ops_data.csv` associates UID 904195 with `ChunkGdnDeviceOperation` at host-relative time `10278584545 ns`, before the first `PERF_PREFILL` at `10816613369 ns`. Thus the offending invocation predates timed prefill; the fatal is delayed until final checking. These host times do not provide the missing device START timestamp.
- Earlier same-case `final_safe_profile_l0_b32.command.json` records the same HEAD, count and collection arguments, but different production (`63d2e53a…`) and runner (`82329af0…`) hashes. Its success is useful historical evidence, not an identical-source controlled contrast. Current-source B1 profiling also completed; neither result proves B32 collection integrity.

## Source evidence and causal chain

1. `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:99–100,253–257` defines high index **1**, reads `p_reg[1]` into the packed marker's high word, stores that word to L1, then reads `p_reg[0]` into the low word. On Blackhole, these are live `WALL_CLOCK_1` at register offset `0x1F4` and `WALL_CLOCK_0` at offset `0x1F0`. The latched high register is instead `WALL_CLOCK_1_AT` at offset **`0x1F8`**, index **2** (`internal/tt-1xx/blackhole/tensix.h:195–197,228–229`).
2. The intended snapshot protocol is explicit in Blackhole `c_tensix_core.h:505–509` and shared `risc_common.h:235–240`: read low, which latches high, then read the latched high. The streaming profiler implements that order with high index 2 (`internal/profiler/kernel_profiler_streaming.hpp:87,155–162`). Merely reversing the legacy reads while retaining index 1 would still use the live high register.
3. `GDN_ZONE("pp_intra")` expands to `DeviceZoneScopedN` (`chunk_gdn_math.hpp:41–44,989–997`). Its scoped START and END use this legacy timestamp writer (`kernel_profiler.hpp:730–759`). The marker therefore reaches the defective sampling path on the reported architecture.
4. A cheap integer check on the **observed** timestamp gives:

   ```text
   divmod(3835405795338, 2**32) == (893, 10)
   ```

   A concrete possible interleaving is: high read returns 893 immediately before rollover; low read returns 10 immediately afterward. The packed value is then **2^32 cycles too early**, relative to `(894 << 32) + 10`. This predicts the tiny low word in the artifact and can move an END before its own START and enclosing kernel/FW STARTs. The actual high/low read interleaving is not observable in the supplied log.
5. `profiler.cpp:1758–1802,1994–2003` reconstructs and inserts the raw timestamp unchanged. `TracyTTDeviceData.hpp:253–269` sorts markers by timestamp first. Final `processDeviceMarkerData` walks that order with a fresh stack and fatals on an END at an empty stack (`profiler.cpp:2160–2162,2201–2210`). Calibration occurs later when pushing to Tracy (`:2687`), so clock synchronization cannot repair this ordering.

This accounts for a valid source-level marker scope becoming invalid during host pairing, delayed failure after successful forwards/replays, and sensitivity to execution timing/source changes. It does not prove which timed/untimed neighbors bracket the orphan or exclude another source of marker corruption. The exact rollover-shaped timestamp makes this stronger than a generic capacity hypothesis.

## Collection controls

| Control | What the code does | Verdict for this failure |
|---|---|---|
| More frequent `ReadDeviceProfiler` | Finishes every mesh CQ, reads/resets device buffers and parses into persistent host marker sets (`tt_metal_profiler.cpp:1229–1266`; `profiler.cpp:2580–2584,2621–2623`). Existing runner drains are at lines 329, 401, 408 and 520. Without mid-run dump, pairing happens later at final dump. | Can reduce DRAM pressure or isolate collection intervals; cannot make the high/low sample coherent. More synchronization before these drains duplicates their CQ completion. |
| Higher program support count | Reserves DRAM, not a larger per-kernel L1 marker buffer. With SUM off, 5000 gives `5000 * 2 * (2+4) * 4 = 240000` bytes per core/RISC (`profiler_state_manager.cpp:62–84`). | Already raised in the failed run. No dropped-buffer warning appears. No supported reason that raising it repairs this timestamp. |
| `TT_METAL_PROFILER_MID_RUN_DUMP=1` / Tracy `--dump-device-data-mid-run` | Calls `dumpDeviceResults(true)` after reads (`tt_metal_profiler.cpp:1050–1051`). That path **skips `processDeviceMarkerData`**, writes/pushes, then clears the sets (`profiler.cpp:2499–2524`). | **Reject as an integrity workaround:** it can avoid this fatal by bypassing the exact pairing check. A completed report under this setting would not prove the marker problem solved. |

Simple optional-marker exhaustion is also a weak explanation: `profileScope` reserves both endpoints and only emits END if START was recorded (`kernel_profiler.hpp:741–759`). Normal TRISC DRAM overflow drops optional payloads as a group and sets dropped-marker status (`:563–597`), which the host reports (`profiler.cpp:1648–1662`). Neither explains the observed low-word value, and no such warning is present. Capacity exhaustion is not demonstrated by these artifacts.

Reducing `--repeats` affects later timed decode work (`run_multichip_decoder.py:403–405`); it does not remove the implicated pre-signpost GDN invocation.

## Smallest collection-only workaround and parent follow-up

**Recollect once in a fresh process with the unchanged command and new output/artifact paths**, retaining count 5000, existing drains and normal final pairing checks; explicitly leave mid-run dump disabled. Preserve this failed attempt and its provenance. A fresh clock phase can avoid the narrow rollover interleaving without touching frozen production or suppressing errors. This is a probabilistic collection workaround, **not a fix or a guaranteed pass**; no deterministic cure among the three controls above is supported.

Accept a new collection only after normal process completion, no marker/drop diagnostics, and generated device/ops reports covering the requested prefill/decode intervals and four chips. If it fails again, retain the second failure; collection-only extra drains outside capture/timing may help localize it, but should not be advertised as a clock fix. If device markers can be preserved for a failing run, inspect the same chip/core/RISC and UID's STARTs and neighboring timestamps for a `2^32`-cycle displacement. No timestamp repair should be applied to measured evidence.

Until an intact collection exists, mark this B32 device profile **failed/unavailable** and make no kernel-timing claims from its JSON, partial host logs or historical safe-source profile. Independent profiler-off latency evidence remains separate. A deterministic source repair belongs in the legacy profiler's clock sampler and lies outside this model-tests/docs-only scope.

## Review and uncertainty

The headline was rechecked against the actual Blackhole register offsets, macro expansion, raw parser, ordering comparator and final stack check. The timestamp arithmetic was run locally without importing TTNN. No hardware reproduction, binary/JIT disassembly or missing START recovery was performed; the manifest lacks profiler binary hashes. The established finding is the clock-read contract violation; attribution of this particular orphan to its rollover interleaving remains the leading, explicitly unverified runtime diagnosis.

Parent artifact-location note: after this inspection, the failed host capture was preserved at `tracy/failed_l0_b32/.logs/tracy_ops_data.csv` relative to this report. The original sibling delivery path is no longer the failing capture.
