# Atomic Barrier Bug Verification Report

Verified 42 flagged files (from `scan_atomic_barrier.py`) against correctness rules for
`noc_async_atomic_barrier()` usage in TT-Metal kernel code. Branch: `main`.

GitHub base: `https://github.com/tenstorrent/tt-metal/blob/main/`

---

## Correctness Rules Applied

The Tenstorrent NOC has **separate transaction queues** for writes and atomics.
Each queue requires its own barrier before `kernel_main()` may return.

| Barrier | What it drains |
|---|---|
| `noc_async_write_barrier()` | Write queue only |
| `noc_async_atomic_barrier()` | Atomic queue only |
| `noc_async_full_barrier()` | All queues (read + write + atomic) |
| `eth_noc_async_write_barrier()` | Write queue only — ethernet variant, insufficient for atomics |

**Operations that require an atomic barrier on every exit path:**
- `noc_semaphore_inc(...)` — non-posted (default / `<false>` template)
- `noc_semaphore_inc_multicast(...)` — non-posted
- `noc_atomic_increment(...)` / `noc_fast_atomic_increment(...)`
- `noc_multicast_atomic_increment(...)` / `noc_fast_multicast_atomic_increment(...)`

**NOT atomic (no atomic barrier needed):**
- `noc_semaphore_inc<true>(...)` — posted, does not use the atomic queue
- `noc_semaphore_set(...)` / `noc_semaphore_wait(...)` — writes and local polls

A barrier inside a helper function counts if and only if that helper is called
unconditionally on every code path before `kernel_main()` returns.

---

## Results: 41 Confirmed Bugs, 1 Uncertain

### Production — CCL / All-Gather / All-Reduce

---

**[ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp#L207)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L207](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp#L207) — non-posted, in `is_sender` branch. Exits via [`noc_async_write_barrier()` L223](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp#L223) only. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp#L187)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L187](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp#L187) — non-posted, in `is_sender` branch. Exits via [`noc_async_write_barrier()` L203](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp#L203) only. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp#L493)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(dest_noc_addr, value)` L493](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp#L493) — non-posted, on `ATOMIC_INC` command path. Kernel exits with [`noc_async_write_barrier()` L1031](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp#L1031) only. No atomic barrier anywhere in the file.

---

**[ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_wait_completion.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_wait_completion.cpp#L63)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(get_noc_addr(noc_x, noc_y, termination_addr), 1)` L63](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_wait_completion.cpp#L63) — non-posted, unconditional in termination loop. No barrier of any kind in the entire file.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp#L233)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L233](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp#L233) — non-posted. `close_connections()` helper contains no barrier. Exits via [`noc_async_write_barrier()` L248](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_rm_writer.cpp#L248) only.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/llama_shapes_sharded_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/llama_shapes_sharded_writer.cpp#L176)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L176](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/llama_shapes_sharded_writer.cpp#L176) — non-posted. Exits with `noc_async_write_barrier()` only. No atomic barrier anywhere. (This is the file partially fixed in c9b2ec2 on another branch — fix not yet merged to main.)

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp#L164)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L164](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp#L164) — non-posted. Write barrier at [`L236`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/kernels/llama_all_gather_concat_writer.cpp#L236). No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp#L159)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(...)` L159](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp#L159) — non-posted, in loop over cores. Exits via [`noc_async_write_barrier()` L168](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp#L168) only. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp#L48)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(output_semaphore_noc_addr_in_pkt, 1)` L48](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp#L48) — non-posted, in `write_data()` local path. Subsequent barriers at [L54](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp#L54), [L228](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp#L228), [L319](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/all_to_all_sender_writer.cpp#L319) are all `noc_async_write_barrier()` — insufficient for atomics.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp#L100)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1)` L100](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp#L100), [`noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1)` L147](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_receiver_reader.cpp#L147) — both non-posted. No barrier of any kind in the entire file.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp#L168)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L168](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp#L168) — non-posted. `fabric_connection.close_finish()` calls only `noc_async_write_barrier()` internally. Exit barrier at [`L180`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/rms_writer.cpp#L180) is write-only. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp#L61)**
Verdict: **BUG** (Case B — conditional barrier)
Two sub-bugs:
1. Early-return path [L82–87](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp#L82): `do_signaling()` may issue [`noc_semaphore_inc(sem_addr, 1)` L61](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp#L61) (non-privileged path), followed only by `noc_async_write_barrier()` before `return`.
2. [`noc_async_atomic_barrier()` L195](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp#L195) is inside `#ifdef ENABLE_GLOBAL_CB` — when undefined, only `noc_async_write_barrier()` runs.

---

### Production — Transformer / SDPA

---

**[ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp#L416)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L416](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp#L416), [`L562`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp#L562) — both non-posted, in `should_receive` paths. No atomic or full barrier anywhere; only `noc_async_read_barrier()` calls present (insufficient).

---

**[ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp#L297)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L297](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp#L297), [`L373`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp#L373) — both non-posted, in `should_receive` paths. No atomic or full barrier anywhere in the file.

---

### Production — Conv2d

---

**[ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp#L263)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1)` L263](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp#L263) — non-posted. File ends with `noc_async_read_barrier()` + [`noc_async_write_barrier()` L277](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp#L277) — neither drains the atomic queue.

---

**[ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp#L319)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1)` L319](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp#L319) — non-posted. Exits via [`noc_async_write_barrier()` L340](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp#L340) only. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L172)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1)` L172](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L172), [`L190`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L190) — both non-posted. Exits via [`noc_async_write_barrier()` L206](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L206) only.

---

**[ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L182)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1)` L182](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L182), [`L206`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L206) — both non-posted. Exits via [`noc_async_write_barrier()` L219](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp#L219) only.

---

### Production — Data Movement

---

**[ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp#L79)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(controller_noc_address, 1)` L79](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp#L79) — non-posted, in non-controller `else` branch. Exits via [`noc_async_write_barrier()` L88](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp#L88) only.

---

**[ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp#L78)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(controller_noc_address, 1)` L78](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp#L78) — non-posted, in non-controller `else` branch. Exits via [`noc_async_write_barrier()` L88](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp#L88) only.

---

### Production — Experimental Other

---

**[ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L164)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(all_core_barrier_noc_addrs[c], 1)` L164](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L164) — non-posted. Immediately followed by [`noc_async_write_barrier()` L166](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L166) — insufficient. No atomic barrier anywhere.

---

**[ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L147)**
Verdict: **BUG** ⚠️ (Pattern 3 — write barrier precedes atomic op)
Write barrier at [`L134`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L134) comes *before* [`noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1)` L147](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L147). Atomic op is the last NOC operation before return with no barrier of any kind after it.

---

**[ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L155)**
Verdict: **BUG** ⚠️ (Pattern 3 — write barrier precedes atomic op)
Write barrier at [`L142`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L142) comes *before* [`noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1)` L155](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L155). Same pattern as above.

---

**[ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp#L161)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(out_ready_sem_noc_addr, 1)` L161](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp#L161) — non-posted. `fabric_connection.close()` contains no atomic barrier. Exits via [`noc_async_write_barrier()` L190](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp#L190) only.

---

### Dispatch — Uncertain

---

**[tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp#L97)**
Verdict: **UNCERTAIN** (Case B — conditional barrier)
Atomic ops: [`noc_fast_atomic_increment(` L97](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp#L97), [`noc_semaphore_inc(` L195](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp#L195) — non-posted. [`noc_async_full_barrier()` L383](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp#L383) is inside `#if defined(COMPILE_FOR_IDLE_ERISC)` only. When compiled for idle erisc: correct. When compiled for worker cores: no atomic barrier exists. Code comments suggest the worker-core omission is intentional but this warrants design review.

---

### Tests

---

**[tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp#L46)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(...)` L46](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp#L46) in `line_sync()` helper, called when sync is enabled. Only [`noc_async_write_barrier()` L502](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp#L502) at end of `kernel_main()` — insufficient.

---

**[tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp#L37)**
Verdict: **BUG** ⚠️ (Pattern 3 — write barrier precedes atomic op)
[`noc_async_write_barrier()` L33](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp#L33) precedes [`noc_semaphore_inc(sem_addr, 1)` L37](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp#L37). Atomic op is the last statement before return with no barrier after it.

---

**[tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver_sem.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver_sem.cpp#L33)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(sender_sem_noc_addr, 1)` L33](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver_sem.cpp#L33) — non-posted, in loop. No barrier of any kind in the entire file.

---

**[tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp#L41)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1)` L41](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp#L41) — non-posted, in loop. No barriers of any kind in the file.

---

**[tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp#L108)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1)` L108](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp#L108), [`L128`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp#L128) — non-posted. Only [`noc_async_write_barrier()` L172](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp#L172) — insufficient.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_mcast_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_mcast_receiver.cpp#L78)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1)` L78](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_mcast_receiver.cpp#L78) — non-posted. Only [`noc_async_read_barrier()` L100](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in0_mcast_receiver.cpp#L100) present — drains reads only, not atomics.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp#L94)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1)` L94](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp#L94) — non-posted. Only [`noc_async_read_barrier()` L84](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout_in1_mcast_receiver.cpp#L84) — insufficient.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/receiver_intermediate_stage.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/receiver_intermediate_stage.cpp#L36)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L36](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/receiver_intermediate_stage.cpp#L36) — non-posted. No barrier of any kind in the file.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_l1_data_forward.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_l1_data_forward.cpp#L21)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L21](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_l1_data_forward.cpp#L21), [`L48`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_l1_data_forward.cpp#L48) — non-posted. No barrier of any kind in the file.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_receive.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_receive.cpp#L27)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L27](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_ring_gather_receive.cpp#L27) — non-posted. No barrier of any kind in the file.

---

**[tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp#L36)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(sender_semaphore_noc_addr, 1)` L36](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp#L36) — non-posted. Only [`eth_noc_async_write_barrier()` L44](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_ring_gather_receive.cpp#L44) — ethernet write barrier, insufficient for atomics.

---

**[tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp#L27)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(worker_config_sem_noc_addr, 1)` L27](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_with_reduce.cpp#L27) — non-posted. No barrier of any kind in the file.

---

### Programming Examples / Lab

---

**[tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp#L41)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(remote_sender_semaphore_noc_addr, 1)` L41](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp#L41) — non-posted. No barrier of any kind in the file.

---

**[tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp#L84)**
Verdict: **BUG**
Atomic ops: [`noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1)` L84](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp#L84), [`L99`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp#L99) — non-posted. No barriers at all in the file.

---

**[tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp#L100)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1)` L100](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp#L100) — non-posted. Only [`noc_async_read_barrier()` L129](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp#L129) — insufficient.

---

**[tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp#L158)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1)` L158](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp#L158) — non-posted. Only [`noc_async_read_barrier()` L111](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp#L111) — insufficient.

---

**[ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp#L39)**
Verdict: **BUG**
Atomic op: [`noc_semaphore_inc(receivers_ready_sem_noc_addr, 1)` L39](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp#L39) — non-posted. No barrier of any kind in the file.

---

## Summary

| Result | Count |
|---|---|
| **BUG** — confirmed | **41** |
| **UNCERTAIN** — context-dependent | **1** (`cq_dispatch_subordinate.cpp`) |
| **NOT A BUG** | **0** |
| Total verified | **42** |

---

## Fix Patterns

### Pattern 1 — Add `noc_async_atomic_barrier()` before existing write barrier
```cpp
noc_async_atomic_barrier();  // add this
noc_async_write_barrier();
```
Applies to the majority of files where a write barrier already exists at end of `kernel_main()`.

### Pattern 2 — No barrier at all: add both
```cpp
noc_async_atomic_barrier();
noc_async_write_barrier();  // if writes also pending
```
Applies to: `ccl_wait_completion.cpp`, `rms_receiver_reader.cpp`, `receiver_sem.cpp`,
`reader_bmm_tile_layout_in0_receiver.cpp`, all test_kernels matmul/erisc/socket files,
programming examples, lab example.

### Pattern 3 — Write barrier precedes atomic op: add atomic barrier after the atomic op ⚠️
The write barrier fires *before* the `noc_semaphore_inc`, so inserting before it is wrong.
Add `noc_async_atomic_barrier()` immediately after the atomic op (or at the end of `kernel_main()`):
```cpp
noc_semaphore_inc(...);
noc_async_atomic_barrier();  // add here
```
Applies to: `writer_paged_fused_update_cache_interleaved_start_id.cpp` (L134 wb / L147 atomic),
`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` (L142 wb / L155 atomic),
`loopback/kernels/sender.cpp` (L33 wb / L37 atomic).

### Pattern 4 — Move conditional barrier outside `#ifdef`
```cpp
// Before:
#ifdef ENABLE_GLOBAL_CB
    noc_async_atomic_barrier();
#endif
noc_async_write_barrier();

// After:
noc_async_atomic_barrier();
#ifdef ENABLE_GLOBAL_CB
    /* other ifdef-guarded work */
#endif
noc_async_write_barrier();
```
Applies to: `llama_all_gather_matmul_async/reader_bmm_tile_layout_in1_ring_all_gather.cpp`
(also needs atomic barrier on the early-return path at L85).

---

## Prioritization

**P0 — Production, active inference paths:**
- `ttnn/.../transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- `ttnn/.../transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`
- `ttnn/.../ccl/broadcast/device/kernels/broadcast_rm_writer.cpp`
- `ttnn/.../ccl/broadcast/device/kernels/broadcast_tile_writer.cpp`
- `ttnn/.../ccl/common/kernels/ccl_send_reader_two_input.cpp`
- `ttnn/.../ccl/common/kernels/ccl_wait_completion.cpp`
- `ttnn/.../experimental/ccl/` (7 files)
- `ttnn/.../conv/conv2d/device/kernels/` (4 files)
- `ttnn/.../data_movement/move/device/kernels/dataflow/` (2 files)
- `ttnn/.../experimental/deepseek_prefill/`, `paged_cache/`, `all_reduce_create_qkv_heads/` (4 files)

**P1 — Tests exercising real hardware paths:**
- `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp`
- `tests/tt_metal/tt_metal/data_movement/` (2 files)
- `tests/tt_metal/tt_metal/test_kernels/` (7 files)

**P2 — Examples and old benchmarks:**
- `tt_metal/programming_examples/` (4 files)
- `ttnn/examples/lab_multicast/` (1 file)
- `tests/tt_metal/tt_metal/perf_microbenchmark/old/` (2 files)

**Needs design review:**
- `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp` — UNCERTAIN; correct for
  idle erisc builds, worker-core path intentionally unguarded per code comments.
