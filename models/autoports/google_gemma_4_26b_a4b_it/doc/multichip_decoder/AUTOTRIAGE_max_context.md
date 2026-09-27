# AUTOTRIAGE: full-attention maximum context

## Diagnosis

The evidence points first to host-side lack of submissions, not an SDPA or CCL
worker deadlock. The initial long TP4 startup includes substantial, proven
per-offset SDPA JIT compilation. Compilation alone does not explain the later
interval with unchanged Inspector logs and no compiler children. A host stack and
phase/chunk progress are required before selecting a runtime fix.

No concrete producer/consumer contract violation is established by these
captures. Do not change attention kernels, fabric, context capacity, or the
nonaligned sequence contract on this evidence.

This is a source-only investigation of the existing captures; no device commands
or runtime edits were performed. Runtime SHA256 inspected:
`d3dee9c50f549b6a48e657f0f8b2745a42c436a81ebecb8d8b099e1ae34f454b`.

## Triage Evidence

- Input: `full_max_capacity.log`, `full_max_triage.txt`,
  `full_max_triage_summary.txt`, and `full_max_triage_second.txt` in this directory.
  The requested run uses layer 5, length 262143, one decode step, trace,
  shuffled cache checking, repeated input, fused tail, hybrid experts, optimized
  shared MLP, one warmed prefill sample, and full-stack DRAM reservations.
- TP1 completed before TP4 fabric initialization at 23:23:50 UTC. TP4 printed its
  successful 21,072,183,296-byte per-device reservation. The runner has no further
  phase print until the entire TP4 prefill, warm prefill, trace/decode, output
  reads, and cache reads finish. Silence after reservation does not localize the
  stall to allocation or first attention.
- First capture: `dump_op_window.py` explicitly says Inspector data may be empty
  or out of sync. Four reported IDs, 362576/362748/362836/362920, all have unknown
  names and occur on individual cores of device 1. Three other devices report
  idle. These IDs cannot establish a live operation or a stuck SDPA call.
- Of 79 first-capture callstack rows, 32 are persistent fabric routers, 32 are
  dispatch/profiler kernels, and 15 are nominal matmul/reduce/eltwise worker
  kernels. All 15 worker PCs are actually in firmware waits: BRISC lines 398/399,
  TRISC lines 135/148/152, or NCRISC `wait_for_brisc_notification`. The historical
  kernel label and sampled GO field must not be mistaken for an executing kernel
  when the PC is in firmware waiting for the next launch.
- All four prefetch RISCs are at `fetch_q_get_cmds`, line 848. The source labels
  this case “Nothing to fetch, nothing pending, nothing available, stall on
  host.” Dispatch RISCs wait for upstream CB pages. This is an upstream
  submission symptom, not evidence that a model worker cannot finish.
- The second capture at about 23:28 UTC reports no running operations. It again
  shows persistent fabric/dispatch activity rather than an SDPA worker wait.
- The full report contains binary-integrity and Ethernet NoC mismatch failures,
  although the separate summary lists those scripts as `pass`. The summary
  therefore must not be read as proof that every check found a healthy state.
  Conversely, checks against a live, non-atomic snapshot and unresolved
  operation mapping do not establish corrupted code. Ethernet links are up,
  and the reported nonposted request/ack hardware totals agree (484/484 or
  335/335); software comparison counters are zero. No route/credit deadlock is
  demonstrated.
- Inspector startup is 23:23:50 UTC, matching the TP4 phase of this run rather
  than a prior workload. Runtime-entry logging is disabled in the captured
  configuration, explaining the absent host operation mapping. Inspector files
  are current-run records, but their last event is historical, not a live PC.
- The Inspector `programs_log.yaml` inspected before any rerun has 153 distinct
  compiled `sdpa` kernel paths, with a summed 117.3265 seconds in their recorded
  compile-duration fields. The last three SDPA completions are program 4158 at
  23:26:06.092 (0.9998 s), 4166 at 23:26:07.109 (0.9811 s), and 4174 at
  23:26:08.110 (0.9682 s). This proves heavy specialization during initial TP4
  prefill; summed compile durations are not a device performance measurement.
- The parent investigator reports PID 19790 remained runnable with approximately
  13.67 GB RSS and 68–76% process CPU; by 23:28:40 there were no compiler
  children, Inspector logs had not advanced since 23:26:08, and filesystem I/O
  was largely unchanged. That later interval remains unresolved. A single high
  cumulative CPU percentage is not proof of forward progress.
- The hardware owner subsequently terminated the original run at 23:29:13 UTC
  (exit 143) after preserving both captures, and reported successful reset/list
  commands. This forced termination is not a demonstrated dispatch timeout or
  kernel fault.

## Source Evidence

### Full-attention specialization and the passing sliding contrast

`tt/multichip_decoder.py:495` issues 256 outer prefill chunks for length 262143
at a fixed 1024-token chunk size. The first chunk uses ordinary SDPA; the other
255 full-attention chunks use increasing scalar offsets through
`OptimizedAttention.prefill` (`tt/optimized_decoder.py:1392`) and
`ConfiguredChunkedPrefillAttention.__call__` (`:1080`, call at `:1108`).

`SDPAParams::chunk_start_idx`
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp:21`)
is explicitly part of the program cache key. In `sdpa_program_factory.cpp:326`,
legacy chunked attention computes `Sk = chunk_start_idx + Sq`, then derives
`Skt`, `valid_Skt`, and `k_num_chunks`. These values enter reader, writer, and
compute compile-time arguments (`:592`, `:674`, `:713`). Reusing the same query
shape therefore does not avoid a new full-attention program for each offset.
The captured distinct kernels verify this behavior rather than merely suggesting
it from source.

Sliding attention instead combines only the bounded previous tail and current
chunk (`optimized_decoder.py:1393`), then uses ordinary windowed SDPA with stable
bounded shapes. It does not compile one growing-prefix variant per scalar
offset. Full attention also has increasing prefix compute work, unlike the
bounded sliding window. These differences explain why the passing sliding
maximum test is not a reliable timeout estimate for cold full-attention prefill.

### Relevant producer/consumer ledger

| Resource | Producer | Consumer / count | Evidence |
| --- | --- | --- | --- |
| Host fetch entries | Host command queue submission | Prefetch consumes queued commands | All sampled prefetch cores wait on zero fetch-size entries at line 848. |
| Dispatch input pages | Prefetch | Dispatch | Dispatch waits for upstream pages; no model-worker completion waiter is identified. |
| Model launch | Dispatch GO / launch messages | BRISC, then NCRISC/TRISCs | All 15 sampled worker RISCs are in firmware launch waits, so historical kernel CB counts are not a valid live deadlock ledger. |
| Paged K/V | Two `paged_fill_cache` calls per outer chunk | Full SDPA reads prefix through scalar `start + physical_query_rows` | Final chunk starts 261120, has 1023 valid rows and 1024 physical rows; cache/table capacity is 262144. |
| Output chunks | One `_forward` per 1024-token group | Concat every 32 chunks, then eight groups | Current bounded assembly already avoids concat's large input-list threshold; it is not an absent fix. |

For the final nonaligned chunk, 8192 pages of 32 tokens cover the full 262144
capacity; the chunk page table covers pages 8160 through 8191. SDPA's physical
query end is 262144, and Q/K chunk rounding does not require reading beyond that
capacity. The final logical output is sliced back to 262143. The following decode
at position 262143 stays inside the requested contract. No off-by-one page-table
violation is shown by this geometry or the capture.

`tests/run_multichip_decoder.py:176` performs initial prefill and reads every TP
replica; `:181` runs a second prefill for the requested single timing sample;
`:218` onward prepares/replays decode; `:262` reads cache replicas. `TP_DONE` is
printed only after those phases. Full output/cache PCC occurs after `TP_DONE`, so
the missing print rules out the final TP1/TP4 correlation loop as the current
location. It does not rule out earlier readback/replica equality or host dispatch.

## Downstream Effects

Idle dispatch and fabric polling can follow host compilation, host computation,
or a host-side submission stall. They do not independently justify modifying
fabric teardown, sender credits, CB ownership, or SDPA loop counts. The capture
does not provide a valid current payload/route on which to base a route ledger.

The unexplained later interval must be separated from the first two minutes of
documented SDPA compilation. Calling the entire run either a proven device hang
or proven ongoing compilation would overstate the evidence.

## Proposed Fix / Focused Next Experiment

No runtime fix is selected yet. The hardware owner should perform the already
planned bounded TP4-only rerun with identical layer, length, decode position,
reservation, precision, trace, and cache-check settings. Preserve the captured
run and its JIT cache before recovery; hardware recovery remains with its owner.

Add test-harness diagnostics before restarting:

1. Enable `faulthandler.dump_traceback_later(60, repeat=True)` with an explicit
   log destination so an unresponsive host C++ call retains its Python call site.
2. Print flushed timestamps before and after setup/reservation, input upload,
   initial prefill, output readback, warmed prefill, trace capture, decode replay,
   cache readback, and mesh close.
3. Add temporary progress around every 16 outer prefill chunks, including chunk
   offset, before and after the forward call. Distinguish host submission
   progress from completion; synchronize only at diagnostic group boundaries if
   necessary, then remove these diagnostics from measured performance runs.
4. Compare progress and compiler activity across the previously observed
   specialization boundary. Most existing SDPA variants should be cached, while
   offsets never previously reached can still require compilation.

If this localizes solely to repeated specialization, the existing flexible SDPA
API (`chunk_start_idx_tensor`) is a candidate experiment: it keeps the offset in
device memory and a fixed-capacity program. It already exists in the prepared
source; adding a second equivalent C++ API is unnecessary. Adopting it would
require separate numerical, final-page, trace, and maximum-capacity validation,
and is not justified as a deadlock fix by the current capture.

If the same host call stops submitting commands with no compile children, use
the captured Python phase/call site to select the next host-runtime investigation
rather than changing attention math or silently reducing context.

## Uncertainty

- No host stack was available in the original process, and no exact Python
  phase was logged after reservation.
- Inspector logging and device sampling are not one atomic snapshot; unknown
  host op IDs and historical labels cannot establish simultaneous worker state.
- The cause of the apparent lack of host progress after 23:26:08 is not proved.
- The existing run was terminated by its hardware owner before a final result.
  On-device correctness is therefore not newly verified.
- This change is documentation only; no build or device test was required or
  run by this investigator.

## Follow-up control: unchanged runtime completed

The hardware owner ran `tests/diagnose_multichip_progress.py --tp 4` with the
same maximum length, reservation, layer, precision, cache-read, and trace flags.
`full_max_diagnostic.log` and `.json` record normal completion at 23:33:18 UTC,
with the same runtime SHA256 above. Both 256-chunk initial prefill and 256-chunk
warmed prefill completed, including the final partial logical tile. Trace decode,
replica equality, cache readback, and mesh close also completed.

The first periodic traceback was inside chunked SDPA, after which progress
continued. The second was in prefill output readback (`run_multichip_decoder.py:94`),
after which the entire warmed prefill continued. They identify expensive phases,
not stable stuck sites. Recorded warmed TP4 prefill was 11.5575 seconds and trace
decode 1.6387 milliseconds; these are host timing samples, not claimed device
speedups.

Because this control was TP4-only, its `passed` is null and PCC lists are empty.
It proves capacity execution and deterministic trace/replica behavior under the
tested settings, not TP1/TP4 numerical agreement. The hardware owner started a
paired run, `full_max_capacity_verified`, to obtain that accuracy evidence. The
original cold-run host delay remains unresolved; this successful control does not
retroactively identify a hang root cause or justify a fabric fix.

The subsequent paired `full_max_capacity_verified` run also completed normally
at 23:35:47 UTC on the unchanged runtime. Prefill/decode PCCs are
0.999960157292875 and 0.9999247235518492; minimum shuffled-cache PCC is
0.9999710211573951. Exact replica equality and deterministic trace replay passed
with the same 262144 capacity and 21,072,183,296-byte reservation. See
`AUTOFIX_max_context.md` for the experiment ledger and remaining uncertainty.
