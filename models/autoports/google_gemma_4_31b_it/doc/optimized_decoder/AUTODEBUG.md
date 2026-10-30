# AutoDebug: stage03 block16 capture L1 collision

Inspection date: 2026-10-30. Checkout HEAD: `b16d01158eb873deaa5a2bd8b4ee08b8fea947bb`.
Scope: inspection only; no device commands, reproductions, source edits, credential access, or old Gemma implementation inspection. Startup contracts were read. Proposed interventions are restricted to this run's `tt/optimized_decoder.py`, tests, and documentation. No earlier report existed at this destination.

## Findings

1. **Block16 O projection consumes 1,205,248 bytes of static dataflow storage per compute core. Its calculated end, 1,316,992, exactly matches the error.** The failure is an overlap with the live tensor allocator frontier, not proof that block16 exceeds physical L1 by itself. The same geometry completes eager decode. Block8 is the existing evidence-backed remedy; readers=3 is a legal, smaller-footprint experimental alternative. Readers=2 is illegal for this O projection.
2. **The correctness harness retains the eager sharded L1 output while recording the next decode.** This is a concrete lifetime difference from both the first eager call and the timing harness, and the leading explanation for the changed allocation frontier. Deleting this dead output before capture is the smallest diagnostic/fix. The exact owner of address 991232 is not recorded, so claiming this deletion alone is proven to make block16 pass would overstate the evidence.
3. **Capture is not exception-safe.** A capture-body exception bypasses `end_trace_capture` and reaches `close_mesh_device` with capture/bypass state still active. The runtime close path can wait for an event whose commands are diverted into the capture buffer. This explains why a host allocation rejection can be followed by an idle-device cleanup hang. No live host stack was supplied, so the precise wait reached in this run remains inferred.

## Direct observations and provenance

- `geometry_0/o_b16_correctness.log` reports prefill PCC **0.9993639768727856**, eager decode PCC **0.9995749847521892**, then program98 static dataflow end **1316992** versus L1 allocation **991232**, range `[0-0 - 7-9]`. It contains no traced PCC, traceback, or tensor/address ownership mapping.
- The sweep config in `tests/sweep_optimized_geometry.py:20-31,48-57` selects layer0, batch1 by default, sequence33, `carry=true`, default block1, O projection storage cores8/block16, attention BFP8 HiFi2, MLP BFP4 LoFi. The last completed manifest entries are block8: the failed block16 subprocess never returned to append its record. Thus the failed command is reconstructed from the sweep and log name, not a completed block16 manifest record.
- `geometry_0/o_b8_correctness.json` records four replays, deterministic output, decode PCC **0.9996362026096614**, updated-input PCC **0.9983349906636135**. Both block4 and block8 have successful correctness and timing entries. These are B1/layer0 controls, not full-stage acceptance.
- Current `optimized_decoder.py` SHA256 `d151d1af644935509e4074b032cd3b92202ee6b7627febfb035b05d4c3c1dde3` and `optimized_layout_candidate.py` SHA256 `59d2e825fe6c7e1ec638bdddb7ef75d5a1167799d653f4f2c7dbb23db38c2abb` match those completed manifest entries. The manifest does not hash the correctness harness or runtime binary; exact historical identity of those files cannot be certified.
- `triage/tt-triage.txt` shows dispatch/prefetch wait stacks and no running model operation. `triage/summary.txt` reports successful checks. A collection script's “pass” alone is not proof of every device invariant. The snapshot supports host-side failure followed by an idle dispatcher; it does not establish a matmul kernel deadlock.

## Concrete geometry and allocation ledger

References below are repository-relative. Let `F` denote `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`.

The caller (`tests/optimized_layout_candidate.py:56-76`) explicitly selects the DRAM-sharded matmul program, BF16 activation/output and the attention compute policy. Weight setup at lines15-24 width-shards the transposed O weight over DRAM banks. For layer0: K=32 heads*256=8192, N=5376; Kt=256, Nt=168. Batch1 is physically one 32-row tile. Eight storage cores give activation shards `[32,1024]` (32 K tiles), output shards `[32,672]` (21 N tiles), and `per_core_M=1`, `per_core_N=21`. Attention remains BFP8 despite `--precision bfp4` (`optimized_decoder.py:24-37,119-128`).

On the eight-bank setup consistent with this exact error, the factory derives reader compute width from **DRAM readers**, not the Python storage-core count (`F:185-218`). One reader/bank means eight readers and 21 N tiles/reader. The subblock heuristic keeps width21 (three subblocks of7); no N padding is needed. `fp32_dest_acc_en=false`, `packer_l1_acc=true`, no bias; partials and output therefore alias BF16 storage (`F:267-287,479-535`). All blocks here have more than one K iteration, so A is double-buffered and B triple-buffered (`F:295-309`). Full BF16 tiles are2048 bytes, BFP8 tiles1088 bytes including exponents and already aligned to64 bytes.

| O block / readers per bank | K iterations | Static A | Static B | Shared output/partials | Total static | End with base111744 |
|---|---:|---:|---:|---:|---:|---:|
| 4 / 1 | 64 | 16384 | 274176 | 43008 | 333568 | 445312 |
| 8 / 1 | 32 | 32768 | 548352 | 43008 | 624128 | 735872 |
| 16 / 1 | 16 | 65536 | 1096704 | 43008 | 1205248 | **1316992** |
| 16 / 3 | 16 | 65536 | 365568 | 14336 | 445440 | 557184 |
| 32 / 1 | 8 | 131072 | 2193408 | 43008 | 2367488 | 2479232 |

For block16: `2*16*2048 + 3*16*21*1088 + 21*2048 = 1205248`. The111744 base also matches the Blackhole default: memory-map end41088 plus69 KiB kernel configuration reservation (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:110-182`; `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:43,68-69`). The descriptor lists eight DRAM banks (`tt_metal/soc_descriptors/blackhole_140_arch.yaml:11-21`). The exact match strongly identifies the O factory as program98; the saved triage does not independently map that program ID to its op. Block32/one-reader exceeds the part's1536 KiB physical L1 (`dev_mem_map.h:33`), even without resident tensors, and is not a viable next step.

The DFB validator (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3071-3099`) obtains the lowest live compute-L1 address, checks the physical limit separately, then rejects `lowest < static_end`. This explains the exact “dataflow buffers” wording; there is no need to hypothesize a different runtime revision merely because the adjacent circular-buffer validator uses different wording. Cached programs still revalidate against current live allocation state (`tt_metal/impl/program/program.cpp:3396-3420`). The overlap is325760 bytes, but **that difference is an address gap, not the amount of live tensor storage that must be deleted**. A small object allocated below temporary allocations can pin the low frontier even after those temporaries die. The allocator advances this frontier when its lowest allocation is freed (`tt_metal/impl/allocator/algorithms/free_list_opt.cpp:322-332`).

The factory's multicast rectangle spans the worker grid (`F:118-132`), explaining why the diagnostic range is8x10 rather than the caller's8x1 storage shard grid. Moving storage shards to other cores is not a reliable escape from this validation. Increasing the Python `cores` value does not directly reduce the weight-reader width or its triple buffer.

## Lifetime and asynchronous boundaries

| Boundary | Relevant live state / action |
|---|---|
| First eager decode | Harness has no previous decode output; prior prefill output remains during RHS evaluation, but prefill candidate output is DRAM. |
| Eager readback | `run_optimized_decoder.py:245-247` assigns `out`, reads it to host, then synchronizes; eager output completion is observed. |
| Capture start | Lines248-249 keep `out` alive while allocating the next decode's intermediates. With carry=true, residual norms/MLP/output preserve L1 sharding (`optimized_layout_candidate.py:27-33,42-53,72-75`; decoder finish at298-329). |
| O projection | The new capture requires its activation shard, new output tensor allocation and static DFB region while old `out` is still live. Allocator placement differs even though program geometry is unchanged. |
| Normal trace end | Runtime restores dispatch state and disables bypass. Replays then execute the captured commands with explicit blocking/readback checks. |
| Failing capture | Static validation throws before the rejected program can run. End-capture and replay are skipped; the outer finally closes the mesh. |

The timing harness explicitly deletes warm decode output and prefill output before capture (`tests/time_optimized_candidate.py:78-96`). This is a useful source contrast, **not a recorded block16 passing experiment**. Capture nodes snapshot program arguments and hold program references; this is not evidence that capture inherently retains every intermediate tensor. Ordinary deallocation remains available during capture (`tt_metal/impl/allocator/allocator.cpp:321-353`; `tt_metal/impl/buffers/buffer.cpp:731-766`).

Narrow lifetime interventions to assess independently:

- In the correctness test, after eager PCC/readback and synchronization, `del out` before `begin_trace_capture`. Capture inputs, cache and eventual `traced` output must remain alive. Make the eager PCC an assertion as well: presently it is printed but only later traced/prefill PCCs are asserted.
- In `optimized_decoder.py:469-479`, release the local `q` reference after SDPA has been enqueued and before `_finish`. Q is no longer consumed, but currently remains live through the whole MLP because `_finish(...)` is evaluated before the caller returns. The same-queue ordering already supports normal temporary reuse. Prefer dropping the owned local reference over indiscriminate forced deallocation of possibly aliased inputs.
- This Q change can reduce unnecessary pressure but is **not the leading explanation for eager versus capture**, since both execute that same lifetime. A B1 D256 Q shard is `[32,256]`, only16384 bytes. Eager output is43008 bytes per shard. Neither size by itself proves a325760-byte frontier improvement; allocation addresses, holes and alias ownership matter.
- Avoid blanket deallocation in `_linear` or of method arguments: `to_memory_config` may return the same storage for matching configs, and residual/caller references can still need it. Do not move inputs inside an existing trace or discard the trace output before replay/readback.

## Capture exception causes a second failure during cleanup

`run_optimized_decoder.py:248-250,284,316-317` has no capture-state cleanup guard. Runtime `begin_mesh_trace` calls `record_begin` (`tt_metal/distributed/mesh_device.cpp:1620-1653`), which resets host dispatch counters and enables sysmem bypass (`fd_mesh_command_queue.cpp:1389-1406`). Only normal `record_end` restores those counters and disables bypass (lines1652-1674). `release_mesh_trace` only removes trace bookkeeping; it is not a substitute for ending active capture (`mesh_device.cpp:1589-1607`).

Mesh close clears command queues (`mesh_device.cpp:1093`). Their destructor can call `clear_expected_num_workers_completed` (`fd_mesh_command_queue.cpp:240-247`), which issues a completion event and waits for its readback (lines367-405). In bypass mode command reservation/writes go to the host bypass buffer (`tt_metal/impl/dispatch/system_memory_manager.cpp:529-533,585-586,603-604`), so no device event need arrive. Bypass also suppresses the fetch-queue write (`system_memory_manager.cpp:881-883`). This supplies a concrete causal path to the observed cleanup symptom and idle dispatcher. It does not prove the exact host stack in the supplied capture.

An existing sibling follows end-and-release-on-body-exception in `ttnn/api/ttnn/graph/graph_query_op_runtime.hpp:81-88`. Within test scope, guard begin/end/release as a lifecycle: on a body exception, end the partial capture to restore queue state, release it without replay, and rethrow the original error; on success, release after replay checks, including assertion-failure exits. Preserve the original exception if cleanup also fails. Do not claim `end_trace_capture` is a universal abort API or guaranteed recovery from errors inside `record_end` itself; this proposal applies to the present body-time validation failure. The timing harness has the same missing exception guard. A bounded subprocess timeout in the sweep can keep one failed cleanup from blocking all remaining candidates, but is not a correctness remedy and must preserve triage before termination under the device-use contract.

## Legal geometry remedies and remaining tests

**Use O block8, storage cores8, readers1 as the currently supported fallback.** It keeps math policy and external shapes unchanged and lowers the calculated static end to735872, below the logged991232 frontier. Saved trace/readback results support it at B1/layer0. Do not extrapolate those results to all batches, long contexts, or full attention.

**If exploring block16, readers3 is a legal shape candidate on Blackhole.** With21 weight tiles per bank, each of24 readers gets7 columns; input storage stays8 cores and output storage stays21 columns/core, with the factory's writeback partitioning connecting readers to storage. Static end drops to557184. The factory requires Blackhole and the supported read NoC for multi-reader mode (`F:148-158`), divisibility and exact width coverage (`F:195-218`). Runtime correctness, performance and allocator placement are unverified; more readers need not improve whole-layer latency.

**Do not select readers2 for O without redesigning the weight sharding/padding contract.** `21 % 2 != 0` fails the factory's explicit divisibility check. Likewise, storage core counts must produce tile-aligned width shards and compatible K/block relationships; arbitrary larger core counts are not a valid L1 remedy. The current matmul validator admits only one physical M tile (`matmul_device_operation.cpp:1583-1658`), so this path covers batches through32, not arbitrary larger batches.

The reader kernel actively uses three weight staging slots (`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:120-190`): reserve two blocks initially, read with rotating transaction IDs, wait/push one completed block while reserving the next window, then drain the last block. Block8 performs32 iterations of168 tiles; block16 performs16 iterations of336 tiles; each covers5376 weight tiles per reader. Three-reader/block16 instead performs16 iterations of112 tiles per reader, with disjoint seven-column slices of each21-column bank row. These paths explain the capacity reduction without changing K coverage. There is no evidence here of a queue-count mismatch; shrinking the triple-buffer allocation alone would violate the existing kernel strategy and is outside scope.

The later reader sweep also contains statically invalid combinations under the present weight sharding:

| Projection | Output tiles | Tiles per bank | Readers2 | Readers3 |
|---|---:|---:|---|---|
| Sliding Q | 256 | 32 | Legal | Illegal |
| Sliding K/V | 128 | 16 | Legal | Illegal |
| Full packed QK | 576 | 72 | Legal | Legal |
| O | 168 | 21 | Illegal | Legal |
| Gate/up | 672 | 84 | Legal | Legal |
| Down | 168 | 21 | Illegal | Legal |

Here “legal” means these reader/shape constraints, not tested whole-layer acceptance. Thus `o_r2`, `down_r2`, and layer0 `qkv_r3` should be filtered before dispatch (`sweep_optimized_geometry.py:36-40`). Readers are restricted to1–3 by `matmul_utilities.cpp:338-342`. A block32/readers3 candidate computes static end988288, just2944 below the failing run's frontier; that numerical comparison is insufficient margin evidence, and the frontier can change.

Follow-up controls (not executed here): first rerun the original block16 geometry with only eager-output deletion; separately assess Q lifetime; compare block8 and block16/readers3 with unchanged precision; record per-op tensor addresses/live allocation frontier to identify991232; then run B1/8/16/32, both layer kinds, required context/tail cases, changed positions/page rows, interleaved prefill and retained-output scenarios. Capture/replay determinism and PCC must pass before any timing result counts. Verify exception cleanup with a controlled body-time failure before replay. No context reduction, runtime allocator patch or removed Gemma code is needed for these remedies.

## Claim review and secondary issues

- Established: logged overlap; exact block16 static sizing under the derived eight-bank geometry; old eager output remains live; capture lacks exception cleanup; readers2 violates O weight divisibility.
- Leading but not runtime-proven: retained eager output causes the failing frontier; the exact cleanup wait described above was reached. Falsifier/control: deleting only the output leaves the same failure/frontier, or a host stack shows another close wait.
- Secondary lifetime improvement: obsolete Q survives into `_finish`; its presence alone cannot explain why the same eager call passed.
- Not established: generic trace-wide tensor retention, allocator corruption, a faulty NoC reader/barrier, low-level math corruption, or an SDPA hang. No saved evidence requires these explanations.
- Coverage gap: the failure prevented later sweep candidates from producing results. The passing block8 result is a narrow candidate check, not stage03 approval. Eager PCC is printed rather than asserted, and the failed record lacks runtime/harness hashes.

Final review: independently checked the static-buffer arithmetic, physical L1 limit, source hashes, output memory inheritance, allocator-frontier update, bypass-event cleanup path and sibling cleanup precedent. Kept tensor-address ownership and actual recovery success explicitly unproven. No device execution was used to validate proposals.
