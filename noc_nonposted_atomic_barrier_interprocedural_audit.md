# Interprocedural Audit of NoC Non-Posted Atomic Barriers

- **Date:** 2026-08-27
- **Historical snapshot:** `3f989841998b26ca8ed707eee2a28add2a2623b5` (the snapshot used by the March report)
- **Current snapshot:** `2e372a412bf3a0a9813587177309f5414ce9ab22` (`origin/main` on 2026-08-27)
- **Method:** source inspection only; no kernels or tests were run

## Executive summary

The real requirement is simple but narrower than “put a barrier near every semaphore operation” and broader than “search each kernel for `noc_semaphore_inc()`”:

> Every RISC must drain every non-posted NoC atomic that it may issue, on the same NoC, after its last possible issue and before that kernel invocation returns. `noc_async_atomic_barrier(noc_id)` and `noc_async_full_barrier(noc_id)` satisfy the requirement. Read barriers, write barriers, local semaphore waits, and a drain performed by another RISC or on another NoC do not.

The audit reached five principal conclusions:

1. All **41** entries labelled `BUG` in the March report are genuine violations at its pinned snapshot. The one dispatch entry labelled `UNCERTAIN` should remain a lifecycle/design-review item rather than be promoted to a confirmed ordinary-kernel bug without more information.
2. The March report was not interprocedural. It missed at least **33 distinct violating files** whose atomics were hidden behind `OpSignaler`, `Semaphore::up(remote)`, EDM adapters, or an Ethernet ring-gather helper. The historical snapshot therefore contained at least **74 confirmed source-level violation files**: the 41 reported files plus 33 additional files.
3. The most important miss was `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp`. Its fused path called `OpSignaler::synchronize_workers_and_signal_op()`, which issued non-posted atomic semaphore increments, and the caller returned without draining them.
4. That miss was later independently confirmed by Watcher in [#49081](https://github.com/tenstorrent/tt-metal/issues/49081), recorded as G1 in [#50886](https://github.com/tenstorrent/tt-metal/issues/50886), and fixed by [#53951](https://github.com/tenstorrent/tt-metal/pull/53951). Current `main` now ends the all-gather reader with both a write barrier and an atomic barrier.
5. The problem is not limited to old code. At the current snapshot, this interprocedural audit found **67 distinct files** with at least one hidden non-posted-atomic path lacking a source-level drain guarantee. This is a hidden-call finding count, not a complete count of every direct and indirect violation in the repository.

## Scope and terminology

This report independently reviews the findings associated with [#41056](https://github.com/tenstorrent/tt-metal/issues/41056) and the linked [March `claude_atomic_barrier_report.md`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/claude_atomic_barrier_report.md). It then extends that work across helper boundaries.

The following labels are used:

- **Confirmed violation:** there is a source path that issues a non-posted atomic and can return from `kernel_main()` without a same-RISC, same-NoC atomic or full barrier after the issue.
- **Safe:** every inspected issuing path has an adequate drain after its final possible atomic.
- **Conditionally unsafe:** a barrier exists, but a compile-time branch, role, or early return can bypass it after issuing an atomic.
- **Special/lifecycle-dependent:** the code is persistent or has a nonstandard termination contract, so ordinary kernel-return reasoning alone is insufficient.

“Confirmed” here is a source-level statement. Whether a particular template, role, or compile-time path is selected by a host program is a separate reachability question. No runtime claim in this report is inferred solely from a source path; the all-gather case is called runtime-confirmed only because Watcher independently observed it.

## The actual barrier requirement

### Why the default semaphore increment is relevant

At the historical snapshot, [`noc_semaphore_inc`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L2245-L2262) and [`noc_semaphore_inc_multicast`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L2289-L2305) both declare `template <bool posted = false>`. An unqualified call is therefore a **non-posted** atomic and consumes response-tracking state on the issuing NoC.

The experimental [`Semaphore::up(remote)`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/tt_metal/hw/inc/experimental/noc_semaphore.h#L67-L70) wrapper lowers to that default `noc_semaphore_inc`, so it has the same obligation. By contrast, `Semaphore::up(value)` only changes a local L1 semaphore and creates no NoC atomic obligation.

An explicit posted form, such as `noc_semaphore_inc<true>(...)`, does not create a non-posted response to drain. Posted calls still have their own ordering and delivery semantics, but they are outside the specific pending-non-posted-atomic failure addressed here.

### What discharges the obligation

[`noc_async_atomic_barrier(noc_idx)`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L1847-L1860) waits until the selected NoC reports that its non-posted atomics are flushed. [`noc_async_full_barrier(noc_idx)`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L1871-L1897) drains reads, writes, and atomics for the current core, so it also satisfies the requirement.

The drain must meet all of these conditions:

1. It executes on the **same issuing RISC/core context**. A barrier on a peer worker does not drain this worker's response-tracking state.
2. It targets the **same NoC** used for the atomic. This matters for helpers that accept an explicit `Noc` object or NoC ID.
3. It is ordered **after the last atomic that can execute on the path**.
4. It executes on **every exit path that may have issued an atomic**, including early returns, role-specific branches, and compile-time variants.

The barrier can live inside the issuing helper or in its caller. What matters is the whole call path, not the source file containing the low-level API spelling.

### What does not discharge it

None of the following is a substitute:

- `noc_async_write_barrier()` or `noc_async_writes_flushed()`;
- `noc_async_read_barrier()`;
- a local `noc_semaphore_wait()` or `Semaphore::wait()`;
- observing a causal action from the remote core;
- a fabric or Ethernet write barrier;
- a connection `close()` that does not itself issue an atomic or full barrier;
- an atomic barrier that occurs before the final atomic;
- an atomic barrier compiled only for an unrelated feature or role;
- a drain executed by another RISC or on another NoC.

The remote semaphore value may already have changed while the sender still has an outstanding atomic response. Therefore, a remote acknowledgement protocol is not by itself proof that the issuing RISC's non-posted-atomic tracker is empty at kernel exit.

## Audit method

The review used four source-level passes:

1. Reinspect every finding in the March report at its pinned commit, following branches and barrier placement rather than accepting the report's verdict.
2. Identify helper functions and methods that can issue non-posted atomics, then enumerate their callers.
3. For each caller, follow every relevant role, compile-time branch, and return to the end of `kernel_main()` and look for a same-NoC atomic or full barrier after the last possible issue.
4. Repeat the helper/caller audit on current `origin/main`, accounting for API migrations and newly added code.

Host-side instantiation was inspected where it was necessary to decide whether two compile-time conditions were actually coupled. It was not used to excuse a helper whose public source contract provides no drain guarantee.

## Reverification of the March report

### Result

The March report's numerical verdict is correct for the files it included:

| Reported class | Reverified result |
|---|---:|
| `BUG` | 41 confirmed |
| `UNCERTAIN` | 1 remains uncertain |
| False positive among the 41 `BUG` entries | 0 |

The common defect is that a default non-posted atomic is followed by no barrier, a read/write-only barrier, a barrier before the atomic, or a barrier hidden behind a condition that does not cover every issuing path.

### Production findings: all confirmed

| File | Inspection result |
|---|---|
| `ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp` | Sender issues a non-posted increment; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp` | Sender issues a non-posted increment; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp` | `ATOMIC_INC` command path can reach a write-only exit barrier. |
| `ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_wait_completion.cpp` | Termination loop issues non-posted increments; no barrier exists. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp` | `close_connections()` does not drain the atomic; exit is write-only. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/llama_shapes_sharded_writer.cpp` | Non-posted increment followed only by a write barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp` | Non-posted increment followed only by a write barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp` | Loop issues non-posted increments; exit is write-only. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp` | All later barriers are write-only. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp` | Two non-posted increment sites; no barrier exists. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp` | Connection close and final barrier drain writes only. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Early return is write-only; normal-path atomic barrier is conditional on `ENABLE_GLOBAL_CB`. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp` | Issuing paths have read barriers only. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp` | Issuing paths have no atomic or full barrier. |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp` | Final read and write barriers do not drain atomics. |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | Final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | Two increment sites; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | Two increment sites; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp` | Non-controller issues an atomic; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp` | Non-controller issues an atomic; final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp` | Atomic is followed by a write barrier only. |
| `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp` | The write barrier precedes the final atomic. |
| `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | The write barrier precedes the final atomic. |
| `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp` | Fabric close and final barrier do not drain the atomic. |

### Test findings: all confirmed

| File | Inspection result |
|---|---|
| `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp` | `line_sync()` hides a non-posted increment; final barrier is write-only. |
| `tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp` | Write barrier precedes the last atomic. |
| `tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver_sem.cpp` | Increment loop has no barrier. |
| `tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp` | Increment loop has no barrier. |
| `tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | Final barrier is write-only. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_mcast_receiver.cpp` | Only a read barrier exists. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp` | Only a read barrier exists, and it precedes the atomic. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/receiver_intermediate_stage.cpp` | No barrier exists. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_l1_data_forward.cpp` | Two increment sites; no barrier exists. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_receive.cpp` | No barrier exists. |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp` | Ethernet write barrier does not drain NoC atomics. |
| `tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp` | No barrier exists. |

### Example and lab findings: all confirmed

| File | Inspection result |
|---|---|
| `tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp` | No barrier exists. |
| `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp` | Two increment sites; no barrier exists. |
| `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp` | Only a read barrier exists. |
| `tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp` | Only a read barrier exists. |
| `ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp` | No barrier exists. |

### Dispatch finding: keep as special/uncertain

`tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp` issues non-posted atomics. Its full barrier is compiled for `COMPILE_FOR_IDLE_ERISC`, while worker-core builds do not show an equivalent source-level drain. However, dispatch is persistent and its shutdown/lifecycle contract differs from an ordinary finite data-movement kernel. The missing explicit guarantee deserves design review and ideally a documented assertion, but source inspection alone does not justify counting it with the 41 confirmed ordinary-kernel violations.

## False negatives in the March snapshot

### `OpSignaler`: five unsafe callers, three safe callers

The historical [`worker_sync_utils.hpp`](https://github.com/tenstorrent/tt-metal/blob/3f989841998b26ca8ed707eee2a28add2a2623b5/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp#L13-L83) issues default non-posted increments from both master and slave synchronization paths and contains no internal drain. Searching only caller files that spell `noc_semaphore_inc()` cannot see those operations.

| Historical caller | Verdict | Reason |
|---|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp` | **Violation** | Fused path calls `synchronize_workers_and_signal_op()` and returns without an atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp` | **Violation** | Signaling call has no later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/minimal_default_reader.cpp` | **Violation** | Signaling call has no later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | **Violation** | Signaling call has no later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | **Violation** | Final barrier drains writes only. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` | Safe | A later atomic barrier covers the signaling path. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp` | Safe | A later atomic barrier covers the signaling path. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/minimal_default_writer.cpp` | Safe | A later atomic barrier covers the signaling path. |

The suspicious strided reader and BMM receiver/writer named in the follow-up message are therefore genuine violations at the March snapshot.

### Remote `Semaphore::up`: 23 unsafe callers, five safe callers

The remote overload in `experimental/noc_semaphore.h` is a second large blind spot. Twenty-three historical caller files could issue its non-posted atomic and return without an adequate drain.

Production callers:

1. `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`
2. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp`
3. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
4. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
5. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp`
6. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp`
7. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_gn.cpp`
8. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp`
9. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp`

Test callers:

1. `tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp`
2. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_receiver.cpp`
3. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_sender.cpp`
4. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_sender_in1_receiver.cpp`
5. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp`
6. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel_no_issue.cpp`
7. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp`
8. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb.cpp`
9. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb_no_issue.cpp`
10. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_multi.cpp`
11. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_no_issue.cpp`
12. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel.cpp`
13. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_multi.cpp`
14. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_no_issue.cpp`

Five inspected remote-wrapper callers were safe because they drain after the helper call:

- `tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/multicast_atomic_sender_2_0.cpp`;
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp`;
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`;
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp`;
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp`.

### Other hidden helper families: six unsafe callers

| Helper family | Violating files | Why |
|---|---:|---|
| `WorkerToEdmReader` / `WorkerToEdmSender` | 3 | Payload methods issue non-posted atomics. `close()` drains only `WORKER_INITIATED`; `MESSAGE_COUNT_REACHED` does nothing. |
| `ChannelBuffer::increment_worker_semaphores()` | 1 | Helper issues atomic increments; the ERISC datamover caller returns without an atomic/full drain. |
| Ethernet ring-gather completion helper | 2 | Helper issues a remote semaphore increment; send kernels have no later atomic/full barrier. |

The affected files are:

- `ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send.cpp`;
- `tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_receiver_worker_reader.cpp`;
- `tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_sender_worker_sender.cpp`;
- `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp`;
- `tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_send.cpp`;
- `tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_send.cpp`.

For the EDM adapters, `MESSAGE_COUNT_REACHED` is not merely a theoretical template spelling: historical host tests instantiate both it and `WORKER_INITIATED`. The unsafe result applies to the message-count termination path; the worker-initiated close path is safe because it explicitly calls `noc_async_atomic_barrier()`.

### Historical false-negative total

The helper-family counts overlap in one file: `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` contains its own remote `Semaphore::up` path and also calls `OpSignaler`. After deduplication:

| Hidden family | Unsafe files | New unique files |
|---|---:|---:|
| `OpSignaler` | 5 | 5 |
| Remote `Semaphore::up` | 23 | 22 |
| EDM and ring-gather helpers | 6 | 6 |
| **Total** | 34 raw family memberships | **33 distinct files** |

Those 33 files comprise 15 non-test/production files and 18 test files. None duplicates one of the 41 confirmed files in the March report, so the historical minimum is 41 + 33 = **74 confirmed source files**, plus the separate dispatch uncertainty.

### Inspected historical helper paths that were safe

The audit also followed several families that did not add findings:

- remote circular-buffer callers in the Llama ring, main matmul ring, and prefetcher writer had covering atomic barriers on their issuing paths;
- the main matmul ring readers and the layernorm pre-allgather reader drained remote `Semaphore::up` operations;
- the inspected DeepSeek unified headers, sort exchange, TT-Train, barrier-sync tests, and routing `LocalSync` callers either used posted operations or ended issuing paths with an atomic/full barrier;
- fabric calls explicitly using posted atomics do not create this non-posted response obligation.

## Watcher confirmation and the all-gather fix

The missed historical path is straightforward:

```text
minimal_default_reader::kernel_main()
  -> OpSignaler::synchronize_workers_and_signal_op()
     -> remote Semaphore::up(...)
        -> noc_semaphore_inc(...), posted defaults to false
  -> kernel exit without atomic/full barrier
```

Watcher later reported an NCRISC exiting with pending non-posted atomics for this all-gather reader in [#49081](https://github.com/tenstorrent/tt-metal/issues/49081). The broader issue [#50886](https://github.com/tenstorrent/tt-metal/issues/50886) records it as G1. [PR #53951](https://github.com/tenstorrent/tt-metal/pull/53951), merged in commit `b5439d45d92d4875242383cf3202d795a82e7373`, added the missing drain.

At the current snapshot, the reader calls `OpSignaler` on the fused path and then ends with:

```cpp
noc_obj.async_write_barrier();
noc_obj.async_atomic_barrier();
```

See the [current all-gather reader](https://github.com/tenstorrent/tt-metal/blob/2e372a412bf3a0a9813587177309f5414ce9ab22/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp#L389-L407). This is the expected source-level fix: the atomic barrier is after all possible helper-issued atomics and is unconditional at the common kernel exit.

## Current `main` interprocedural audit

The current count below intentionally covers hidden helper/wrapper paths. It should not be interpreted as an exhaustive rescan of every kernel that directly spells a raw atomic API.

### Current `OpSignaler`

The current [`OpSignaler`](https://github.com/tenstorrent/tt-metal/blob/2e372a412bf3a0a9813587177309f5414ce9ab22/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp#L67-L127) still issues remote non-posted atomics without an internal drain. This is a valid performance-oriented design only if every caller owns and fulfills the exit contract.

Thirteen `synchronize_workers_and_signal_op()` callers were inspected. Six are unsafe:

| Current caller | Verdict | Missing path |
|---|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_neighbor_halo_reader.cpp` | **Violation** | No later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/minimal_default_reader.cpp` | **Violation** | No later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | **Violation** | No later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | **Violation** | Final barrier is write-only. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | **Violation** | No later atomic/full barrier. |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | **Conditionally unsafe** | Atomic barrier exists only under `ENABLE_GLOBAL_CB`; fused signaling is independently enabled. |

The main sender's two compile-time conditions are not equivalent: host construction sets fused reduce-scatter signaling independently from global-CB use. The barrier under `ENABLE_GLOBAL_CB` therefore cannot be treated as covering all `fuse_op_reduce_scatter` instantiations.

Seven current callers are safe:

- all-gather reader and writer;
- ring-attention all-gather reader and writer;
- strided all-gather writer;
- Quasar Metal2 receiver and sender variants, which use full barriers.

All four inspected callers of `OpSignaler::signal_op_per_core()` are safe; each has a final atomic barrier:

- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp`;
- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp`;
- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in0_sender.cpp`;
- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in1_sender_out.cpp`.

### Current remote `Semaphore` wrappers

Fifty current files contain a reachable remote `Semaphore::up` or multicast-increment path and no atomic/full barrier anywhere in the file. Twenty-nine are non-test sources and 21 are tests.

#### Non-test files with no source-level drain

Conv2d:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`

Data movement:

- `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp`

Experimental and other operations:

- `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp`
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_worker_dispatch.cpp`
- `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`

Quasar Conv2d variants:

- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/activation_reader_width_sharded_metal2.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_metal2.cpp`

Matmul:

- `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
- `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`

Normalization:

- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp`
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp`
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_gn.cpp`
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp`
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp`
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp`

SDPA:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`

#### Test files with no source-level drain

1. `tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender_2_0.cpp`
2. `tests/tt_metal/tt_metal/test_kernels/dataflow/grid_barrier.cpp`
3. `tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram_pipeline.cpp`
4. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_receiver.cpp`
5. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_receiver_in1_sender.cpp`
6. `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_sender_in1_receiver.cpp`
7. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp`
8. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_mcast_xcore_writer.cpp`
9. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_xcore_locker.cpp`
10. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_xcore_writer.cpp`
11. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_remote_cb_locker.cpp`
12. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_remote_cb_locker_no_issue.cpp`
13. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_stateful_writer_kernel.cpp`
14. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp`
15. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb.cpp`
16. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb_no_issue.cpp`
17. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_multi.cpp`
18. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_no_issue.cpp`
19. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel.cpp`
20. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_multi.cpp`
21. `tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_no_issue.cpp`

Five more current files contain an atomic/full barrier somewhere, but an issuing path bypasses it:

| File | Uncovered path |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Idle/hop early return signals and performs only a write barrier; normal-path atomic barrier is tied to `ENABLE_GLOBAL_CB`. |
| `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp` | Non-drain tilize core increments `partial_metadata_ready_sem`; barrier executes only in the drain-core branch. |
| `ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/kernels/dataflow/reader_writer_affine_exclusive_scan.cpp` | Non-coordinator paths bypass the coordinator barrier; final neighbor-ready increment can also occur after the last stage barrier. |
| `ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/dataflow/reader_writer_reduce_affine_transforms.cpp` | Non-coordinator issuers exit without the coordinator's barrier. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp` | Sender path drains, but the root output-core notification is followed only by a write barrier. |

This makes **55 unsafe files** in the current remote-`Semaphore` family: 50 with no source-level atomic/full barrier and five with uncovered paths.

### Other current hidden helpers

The following historical helper defects remain present at the current snapshot:

- three `WorkerToEdmReader`/`WorkerToEdmSender` callers are unsafe for `MESSAGE_COUNT_REACHED`;
- `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp` returns after `ChannelBuffer::increment_worker_semaphores()` without a drain;
- the two Ethernet ring-gather send tests return after their completion helper's increment without a drain.

That contributes six files, the same set listed in the historical section.

`remote_cb.pop_front()` also issues a non-posted atomic. Two current performance-test receivers lack an atomic/full barrier:

- `tests/tt_metal/tt_metal/perf_microbenchmark/10_dram_read_remote_cb_sync/kernels/receiver_l1.cpp`;
- `tests/tt_metal/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/kernels/receiver_l1.cpp`.

Other inspected current remote-CB callers either use posted operations or have a covering atomic/full barrier.

The DeepSeek `matmul_expert_compressed_dram.hpp` helper also issues raw atomics. Its micro-op and fused-MoE callers have final atomic barriers. The decoder-block caller does not, but that file already contains direct raw atomics and would be found by a direct scan; it is a helper-contract problem but not counted as an additional direct-search false negative here.

### Current hidden-path total

Two of the six unsafe `OpSignaler` files also occur in the 55-file remote-`Semaphore` set because those kernels have an independent direct wrapper use. After deduplication:

| Hidden family | Unsafe files | Added unique files |
|---|---:|---:|
| Remote `Semaphore` wrappers | 55 | 55 |
| `OpSignaler` | 6 | 4 |
| EDM and Ethernet ring-gather helpers | 6 | 6 |
| Remote circular-buffer helper | 2 | 2 |
| **Current total** | 69 raw family memberships | **67 distinct files** |

The 67 comprise 40 non-test files and 27 test files. They are not “67 regressions since March”: the set includes historical defects, API-migrated versions of old raw-call findings, and newer code. It is also not the total number of all current atomic-barrier defects, because direct raw atomic sites were outside this second-pass count.

## Persistent and special kernels

Dispatch and fabric-router kernels can be long-lived, use dedicated shutdown handshakes, or intentionally retain requests across ordinary work-loop iterations. They should not be silently declared safe merely because they are persistent, but they also should not be mixed into finite-kernel counts without establishing the lifecycle boundary at which Watcher requires the tracker to be empty.

The appropriate outcome for these cases is a separate audit that answers:

1. What event constitutes termination for each RISC?
2. Can any non-posted atomic still be pending at that event?
3. Which explicit code path drains it, and on which NoC?
4. Is the guarantee enforced by Watcher or documented only as an assumption?

Until those questions are answered, the historical dispatch finding and similar router `notify_master` paths remain design-review items rather than part of the confirmed ordinary-kernel totals.

## Recommended remediation

### Immediate code fixes

For each confirmed finite-kernel path, add a same-NoC `noc_async_atomic_barrier()` after the final possible atomic and before every return. A common unconditional exit barrier is preferable when all roles can safely execute it. Use `noc_async_full_barrier()` only when reads and writes also need draining; it is correct but potentially more expensive.

Priority should start with:

1. current `OpSignaler` readers and BMM receiver/sender variants;
2. role- or branch-specific misses where a barrier misleadingly exists elsewhere in the file;
3. active Conv2d, matmul, normalization, SDPA, and CCL wrapper callers;
4. shared helper contracts, followed by tests and examples.

### Make the contract explicit

Helpers that intentionally leave atomics outstanding should make that visible in their API contract, for example with a comment or annotation equivalent to:

```text
may_issue_nonposted_atomic(noc)
caller_must_drain_before_kernel_exit(noc)
```

Putting a barrier inside every helper is mechanically safe but can serialize hot loops and erase useful batching. Caller-owned draining is reasonable; undocumented caller-owned draining is what produced these misses.

### Add an interprocedural static check

A useful checker needs function summaries, not only text matching. Each helper summary should track at least:

- may issue a non-posted atomic, parameterized by NoC;
- definitely drains atomics, parameterized by NoC;
- may return with a pending atomic;
- role, template, and compile-time conditions governing the issue and drain.

The checker should propagate those summaries through the call graph and require the abstract pending-atomic state to be empty at every finite `kernel_main()` exit. It must recognize raw atomic APIs, `Semaphore::up(remote)`, `OpSignaler`, remote-CB methods, EDM adapters, and future wrappers. A source file containing *some* barrier must not be considered safe unless path and order prove coverage.

### Add runtime coverage

Watcher is the definitive backstop for instantiated paths. Add or retain tests that enable fused paths, master and slave roles, coordinator and non-coordinator roles, early returns, both NoCs where configurable, `MESSAGE_COUNT_REACHED`, and builds with and without feature macros such as `ENABLE_GLOBAL_CB`.

## Final assessment

The March report's individual `BUG` verdicts were sound; its search boundary was not. The governing rule attaches to the issuing RISC and NoC transaction stream, so it crosses C++ helper boundaries. Any audit that starts and ends with files containing the literal `noc_semaphore_inc()` will necessarily miss wrapper and helper callers.

The Watcher-confirmed all-gather failure demonstrates that this is not a stylistic preference or a conservative lint rule. The safe review unit is the full interprocedural path from the atomic issue to every `kernel_main()` exit, including compile-time variants and role-specific returns.
