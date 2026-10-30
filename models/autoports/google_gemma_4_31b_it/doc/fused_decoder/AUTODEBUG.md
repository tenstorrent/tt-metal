# Stage02 profiler failure: prefill exhausts the profiler DRAM buffer

## Finding

The B32 collection loses device profiler records during **both warmup and measured prefill**. The first missing host operation, `1048576`, is an ordinary untraced warmup slice, not a failed decode replay or an operation-ID wrap. `process_ops_logs.py` correctly refuses to merge an incomplete device report. The smallest harness-only intervention is to increase the existing Tracy `--op-support-count` setting before collection. Do not weaken the merge assertion, discard warmup rows, or shorten the 128-replay workload.

Inspection only: no hardware commands, reproduction, source edits, or launcher invocation. Only this report was written. Root `AUTODEBUG.md` and stage01 evidence were not changed.

## Direct observations

Artifacts are under `doc/fused_decoder/tracy/sliding_attention/`, relative to this autoport.

* B32 `collection.log:53` begins explicit warnings: `Profiler DRAM buffers were full, markers were dropped!`, `bufferEndIndex = 12000`. There are 1,100 such warning lines, in two groups before decode completion. The last is line 1154. This is direct producer-side loss evidence, not an inference from the Python assertion.
* B32 `collection.log:1157` records `PROFILE_COMPLETE sliding_attention batch=32 input=4096 decode_steps=128`. Device close completes; subsequent host/device report merging fails with `Device data missing: Op 1048576 ... device 0 (trace_id=None)`.
* A cheap CSV-only inspection decoded cached host-op messages using the same hash-cache scheme as `tools/tracy/process_ops_logs.py:444-498`, partitioned operations by signposts, and compared IDs with `cpp_device_perf_report.csv`. Results:

| Collection interval | B1 host device ops | B32 host device ops | B32 absent from device report |
|---|---:|---:|---:|
| Setup/warmup before `PERF_PREFILL` | 178 | 1706 | 683 |
| `PERF_PREFILL` through `PERF_PREFILL_END` | 141 | 1669 | 646 |
| Captured decode body after prefill | 37 | 37 | 0 |

B32 warmup IDs span `1024..1746944` in increments of 1024. The missing suffix starts at `1048576` and continues through `1746944`. Measured prefill spans `1747968..3456000`; its missing suffix starts at `2795520`. Each prefill interval retains only its first 1,023 host-op IDs in the C++ report. Presence of a row does not establish that every core's markers survived for that row. An independent raw-CSV scan found core (1,2) BRISC has only base IDs 1–1000 before the first drain, while other cores/RISCs retain later IDs through 1023. Thus 1024 is the first wholly absent operation, not the first partially damaged operation. Raw non-trace IDs are exactly base IDs 1–1023 and 1707–2729, corroborating loss before C++ report aggregation.

* B32 host metadata has one warm replay before `PERF_DECODE`, then exactly 128 replay markers inside the decode signposts. The C++ report has trace 0, sessions 1 through 129, each with 37 rows. Session 1 is warmup; sessions 2–129 correspond to the measured replays. Decode is not the missing-record interval.
* B1 has zero buffer-full warnings and a successful processed `ops.csv`/manifest. Its manifest hashes match current `tt/fused_decoder.py` (`0474e4e539be59be2ca7963c43fcddc177371ad4625e3fb68df9bbc1f3dab230`) and `tests/profile_fused_decoder.py` (`06827198ac401a8b55e1c0708c523b1c7da0a6a8100cb2ab6d63cd2c4ffd87c8`). B32 did not reach successful manifest creation; its host sequence, operation counts and compiled profiler size corroborate the current harness, but there is no B32 saved source-hash manifest.

## Causal chain and capacity ledger

1. `tests/run_fused_profile_queue.py:28-45` enables the device profiler and launches Tracy without an explicit support count. `profiler_state_manager.cpp:24,45-87` defaults to 1,000 program equivalents. With sum mode off, the per-core/per-RISC allocation is `1000 * 2 * (2 + 4) * 4 = 48,000 bytes`, or 12,000 uint32 words. Constants are in `tt_metal/hw/inc/hostdev/profiler_common.h:167-169`. B32's compile command explicitly contains `PROFILE_KERNEL=1` and `PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=48000`, confirming the effective configuration rather than assuming an environment default.
2. `tests/profile_fused_decoder.py:53-59` runs an entire warm prefill and warm decode before synchronizing and draining. Lines 60–64 run a second entire prefill before draining. Synchronization completes queued work but does not by itself increase profiler capacity or recover discarded records. The later eight-replay drain cadence at lines 74–81 cannot repair either earlier loss.
3. B32 creates many more device operations per prefill. `tt/fused_decoder.py:213-237` processes 4096 tokens as four 1024-token chunks. Prefill RoPE loops over batch elements (`153-159`) for Q and K, and cache filling does four per-request operations (`251-255`). The observed counts are `4 * (34 + 12*B) + 1`: 141 at B1 and 1669 at B32. Warm decode adds 37, yielding 178 versus 1706 before the first drain. Capacity is per participating core/RISC and counts marker words, not merely total model calls; 1,023 report rows is an observed survival boundary, not a universal 1,023-op capacity.
4. The normal profiler explicitly drops data when accumulated words exceed capacity. See the non-accumulating BRISC path in `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:524-588`: it computes device-plus-host end indices, first attempts all markers, then falls back to guaranteed markers if those still fit, and finally disables the transfer when even those cannot fit. Both loss branches set dropped timestamps. Only the separate non-dropping mode requests another region. Host `tt_metal/impl/profiler/profiler.cpp:1649-1664` reports the dropped-zone flag. Drains reset control state (`1479-1505`, `2586-2601`), explaining why later decode records can be complete despite earlier lost prefill tails.
5. `tools/tracy/process_ops_logs.py:674-725` joins by operation ID and trace ID, tries an ID-only fallback, and asserts when no candidate exists. Signposts do not exclude warmup from this merge. The first missing warmup op therefore aborts report generation before any signpost-based summary. Removing warmup from the join would merely expose the measured-prefill loss.
6. `tt_metal/impl/profiler/tt_metal_profiler.cpp:1162-1179` encodes IDs as `(base_program_id << 10) | device_id` for these device ops. Thus `1048576 = 1024 << 10`. Later, larger IDs survive after a drain; this is not a 20-bit counter wrap.

## Smallest harness-only workaround

Add the argument pair `--op-support-count 4096` to the Tracy invocation in `tests/run_fused_profile_queue.py`, before its target-module `-m`. Equivalent invocation shape:

```text
python -m tracy -r -p -v --no-web-server --op-support-count 4096 \
  -o <fresh-stage02-artifact-directory> \
  -m models.autoports.google_gemma_4_31b_it.tests.profile_fused_decoder --layer 0 --batch 32
```

The existing CLI is defined at `tools/tracy/__main__.py:65-71` and maps to `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` at lines 320–321. An environment-only override has the same effect, but the CLI pair is preferable because the queue's saved command records it. Runtime parsing is in `tt_metal/llrt/rtoptions.cpp:1322-1330`; the setting feeds both HAL reservation (`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:41-53`) and device compilation (`tt_metal/jit_build/build.cpp:371-372`). It must be set before launching the collection process.

4096 gives 196,608 bytes/49,152 words per core/RISC in this mode, 4.096 times the observed buffer. It is a conservative proposed capacity, **not a hardware-verified minimum**; marker volume and participating cores determine the true minimum. The smallest change is one argument pair, not a claim that 4096 is the smallest viable integer. Hardware validation must establish sufficiency and memory headroom.

Keep the full 4096-token prefill, existing warmup, signposts, and all 128 decode iterations. Retain the existing drains after whole prefill and every eight complete replays. This adds no synchronization inside a layer, preserving the first-FW-start to last-FW-end window, including inter-op gaps. Adding a drain only between warm prefill and warm decode is insufficient: measured prefill alone already overflows. Draining within prefill would perturb the measured whole-layer window. Reducing decode steps addresses the wrong interval.

Use a fresh artifact directory and preserve the failed capture. This also avoids the queue's existing-command/hash reuse checks and retains B1 evidence. A capacity change may rebuild profiler-enabled kernels and reserves additional DRAM; collection timing must remain warmed as in the current harness.

## Review of competing explanations and follow-up checks

* **Trace replay loss:** contradicted by the complete 129-by-37 trace rows and the failing op's `trace_id=None`. No trace/replay code changes are indicated.
* **ID truncation:** contradicted by the encoding and survival of larger IDs after drains.
* **Missing WASM copy path / pandas dtype warning:** these occur during reporting but cannot explain the earlier explicit dropped-marker warnings and two missing prefill tails.
* **Model correctness failure:** not investigated; no claim about numerical correctness follows from profiler completion.
* **Completeness-test gap:** `tests/summarize_profile.py:47-50` checks 128 sessions and equal row counts, which alone cannot detect an identical missing subset in every session. For this rerun compare each replay's exact op-ID set with the captured 37-op body, not just equal counts. Also verify all 1669 measured-prefill IDs and every warmup ID join; the collector otherwise merges warmup as well.

After the first draft, the headline chain was rechecked against actual CSV phase boundaries, runtime buffer-size defines, the device overflow branch, ID encoding, CLI forwarding and parser behavior. No hardware validation was performed. Acceptance for the proposed rerun: no dropped-marker warnings, successful strict report merge, complete prefill operation coverage, exactly 128 measured replay sessions with the complete captured op set, and unchanged whole-layer window calculation. Existing truncated prefill records cannot be reconstructed by reprocessing the current capture.
