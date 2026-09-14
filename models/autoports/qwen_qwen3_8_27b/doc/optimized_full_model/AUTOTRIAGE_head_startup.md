# AUTOTRIAGE — LM-head probe initialization stall

## Diagnosis

The strongest supported diagnosis is an initialization buffer-transfer stall on device 1: its prefetch kernel is waiting for a NoC read transaction to finish, preventing its dispatch kernel from receiving the remaining payload of an interleaved DRAM write. The source-side origin of that uncompleted read is **unresolved**. The capture supports investigating pinned host-memory upload/dispatch before host weight packing; it does not establish a defective model kernel, LM-head reader geometry, Ethernet failure, or a permanent hardware fault.

This report concerns PID `540646`, running `tests.probe_lm_head_geometry --readers 1` with output `head_8192_reader1_compatible.json`. It is separate from `AUTODEBUG_head_geometry.md`. The corrected candidate's reader-dependent padding is after model construction and has not been shown to execute.

Investigation was limited to saved triage, source, and read-only `/proc` inspection. No device access, triage invocation, reset, process signal, or implementation edit was performed by this investigator.

## Triage Evidence

- `triage_head_startup/summary.txt` reports all 22 checks passing. This means no listed check reported a violation; it does **not** mean every dispatch transfer completed.
- `triage_head_startup/tt-triage.txt`, saved at `2026-09-13 20:45:40 UTC`, contains no reported running model operation. Its `dump_running_operations.py` section says `pass`.
- Device 1, `16-2 (11,0)`, BRISC `cq_prefetch`, PC `0xc0c0`: `noc_async_read_barrier_with_trid()` at `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2430`, called by `process_relay_linear_cmd()` at `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:1754`.
- Device 1, `16-3 (11,1)`, BRISC `cq_dispatch`: `CBReader::acquire_pages()` through `wait_for_available_data_and_release_old_pages()`, called by `process_write_paged<true>()` at `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:645`. This is waiting for payload during a DRAM paged write, not merely waiting for the next command.
- Devices 0, 2, and 3 prefetch kernels are at `fetch_q_get_cmds()` line 848. Source identifies that loop as waiting for nonzero host fetch-queue input, with no pending or available work on that path. Their dispatch kernels wait at the outer command-acquisition loop (`cq_dispatch.cpp:1671`).
- All 16 listed Ethernet links are Up, with retrain count zero, RX link Up, and heartbeat true. Fabric router call stacks are in their persistent service loops. There is no demonstrated connection-credit or route-count violation from which to derive a fabric root cause.
- `head_8192_reader1_compatible.log` contains `LOAD_LAYER 0` and `LOAD_LAYER 3`, but no `MODEL_LOADED`; its last modification was `20:43:07 UTC`. The log remained 3,361 bytes when checked at `20:48:25 UTC`.

### Host observations

Read-only inspection before coordinator recovery found 70 threads. The main thread was sleeping in `futex_do_wait`; `/proc/540646/syscall` reported syscall 202 (futex). TID `540743` was running in userspace, at approximately 97.7% lifetime CPU in `ps -L`, while the other Python threads were predominantly in futex waits. Its explicit thread-stat snapshot recorded 26,653 user CPU ticks; both the thread listing and the later syscall snapshot showed it runnable. No interval CPU measurement was obtained.

The process had approximately 5.58 GiB RSS, zero swap, and an I/O snapshot of `read_bytes=3313664`, `rchar=215596615`, `write_bytes=62713856`. These are snapshots, not throughput measurements: safetensors is memory mapped, so they cannot by themselves exclude host memory work. The main-thread wait and device-1 transfer frames together make an ordinary ongoing Python weight-packing phase less likely. The busy thread cannot be identified as a completion reader without a native stack.

`head_startup_gdb.log` records attachment failure: `ptrace: Inappropriate ioctl for device`. Reading `/proc/540646/stack` and the busy thread's kernel stack returned permission denied. Consequently no Python or native host call stack is available for this run. At `20:48:25 UTC`, `/proc/540646` no longer existed; no inference about successful completion follows from that, and no second counter sample was possible.

## Source Evidence

### The candidate is beyond the observed initialization boundary

`tests/probe_lm_head_geometry.py:34` calls `build_generator(...)` before generating the activation, computing the reference, or constructing candidate weights. Reader-dependent `quantum = args.readers * 32` begins at line 62; the candidate matmul begins at line 81.

`tt/generator.py:519` constructs `QwenModel` before `QwenGenerator`. `tt/model.py:66` prints `LOAD_LAYER` **before** the corresponding `MultichipDecoder.from_state_dict` call. After the layer loop, the constructor uploads embeddings, final norm, and the head; creates the existing baseline head shards; and builds/uploads rotary tables before printing `MODEL_LOADED` at line 112. Thus `LOAD_LAYER 3` does not prove layer 3 finished, nor identify which subsequent tensor stalled. The ordinary model constructor does contain baseline DRAM-head layout conversion, but the new reader-dependent probe candidate is later and distinct.

Both `QwenModel.upload` (`tt/model.py:114`) and `MultichipDecoder`'s local upload helper (`tt/multichip_decoder.py:115`) call `ttnn.from_torch` with a device mesh and DRAM memory configuration. These provide an initialization buffer-upload path consistent with the capture.

### Transfer and synchronization ledger

| Resource or transition | Producer | Consumer / required count | Observed consequence |
| --- | --- | --- | --- |
| Pinned host payload | `issue_buffer_dispatch_command_sequence`, `tt_metal/impl/buffers/dispatch.cpp:941` | Paged device write expects `num_pages_to_write * page_size_to_write` bytes | The write path emits a separate local `RELAY_LINEAR` command from pinned source address, NoC coordinate, and byte length at lines 1012–1047. This command pairing matches the captured prefetch/dispatch frames. |
| Two prefetch scratch halves | `process_relay_linear_cmd`, `cq_prefetch.cpp:1707` | Read chunks are bounded by `scratch_db_half_size`; each half must have all its reads completed before its data is sent downstream | First read uses TRID 6; subsequent alternating reads use TRIDs 6 and 7. The captured line 1754 waits on the previous half before `write_pages_to_dispatcher` and payload release. |
| NoC read completion | `noc_read_64bit_any_len`, `cq_prefetch.cpp:1678` | One transaction per at-most-`NOC_MAX_BURST_SIZE` packet; barrier waits until the selected TRID has no outstanding read | `dataflow_api.h:2430` polls `ncrisc_noc_read_with_transaction_id_flushed`. The capture does not provide selected TRID, source address, or outstanding count. |
| Prefetch-to-dispatch payload pages | `write_pages_to_dispatcher` then `DispatchRelayInlineState::cb_writer.release_pages`, lines 1756–1759; final release at 1774 includes the command page | `CBReader::acquire_pages`, `cq_common.hpp:494`, waits until upstream page count differs from consumed count | Device 1's dispatcher cannot advance its DRAM write while prefetch withholds unretrieved payload. Its CB wait is downstream of the read barrier. |
| DRAM write bytes | `process_write_paged<true>`, `cq_dispatch.cpp:603` | `write_length = pages * page_size`; each received payload slice reduces remaining length | Captured line 645 is waiting for the next payload slice, before the transfer can finish. |

The source already assigns relay TRIDs `{6, 7}`, separates them from fetch-queue TRIDs 2–5 and exec-buffer TRID 1, and checks the per-half packet count against `NOC_MAX_TRANSACTION_ID_COUNT` (`cq_prefetch.cpp:1718–1722`). Adding those protections is therefore **not** a valid proposed fix to this prepared source. The capture is insufficient to prove source-address invalidity, pin lifetime loss, packet-accounting error, or hardware completion loss. The next boundary to inspect is the actual command and pinned-memory owner, not an assumed missing TRID split.

## Downstream Effects

Device 1's dispatch CB wait follows directly from the prefetch read wait. Idle outer dispatch loops on devices 0/2/3 and normal fabric polling are not independent evidence of a multi-device CCL deadlock. A host wait for transfer completion would explain the sleeping main thread; the native stack needed to confirm that is unavailable. In particular, the one busy userspace thread is insufficient evidence of active packing or a specific completion-loop defect.

## Proposed Fix

No source fix is justified by this capture alone. Preserve this run as an initialization-transfer failure, then let the coordinator perform bounded owned-process recovery and rerun the **same** candidate to determine whether the failure recurs.

For that rerun, enable Python faulthandler and schedule timed all-thread traceback dumps **before** `build_generator`; setting `PYTHONFAULTHANDLER=1` alone does not schedule timeout dumps. A repeated dump can identify the exact Python upload or conversion where initialization waits. If needed, add temporary before/after markers around layer 3 completion, embedding upload, norm upload, head upload, baseline head-shard creation, and rotary-table upload. Those markers should precede changing any candidate geometry.

If the same transfer stall recurs, the next targeted device capture should inspect device 1's prefetch relay command and transaction state: `noc_xy_addr`, `read_addr`, original and remaining byte lengths, `db_toggle`/selected TRID, scratch-half bounds, and read-completion counters; then compare the encoded pinned-host range and mapping lifetime with the host buffer owner. A repeat stop at line 1754 would justify pursuing that narrowly scoped transfer investigation. A successful unchanged rerun would establish recoverability of this incident, not a source-level fix or proof that the original read was valid.

All termination, reset, health checks, mesh smoke tests, and reruns belong to the coordinator. This report does not request or perform additional independent hardware work.

## Uncertainty

- There is one saved device snapshot. It identifies the sampled wait and downstream dependency, but cannot prove that the PC remained at that instruction throughout the whole wall-clock stall.
- The Python call, tensor name/size, precise relay command, transaction count, and pinned-memory mapping state were not captured.
- A pinned host-to-device write is the strongest source-path match to `RELAY_LINEAR` paired with `process_write_paged<true>`; the raw command has not independently confirmed its address or ownership.
- Host-thread roles are unproven because gdb and kernel-stack reads failed. No persistent host deadlock is diagnosed solely from futex state.
- Existing TRID separation and count checks have been verified present. They do not prove every producer/lifetime contract is correct, but they rule out claiming those changes as a new repair.
- No failure of the corrected LM-head reader-padding candidate, accuracy result, performance result, successful recovery, or successful rerun is claimed by this report.

Validation for this investigation is source/capture inspection only. The only authored artifact is this Markdown report; no build is required.
