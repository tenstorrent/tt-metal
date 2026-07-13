// SPDX-License-Identifier: Apache-2.0
// Regime-A (b): N-SLICE + STREAMING in0 ring all-gather (NO reduction). Every core owns a distinct N-sub-band
// (reads its own in1, writes its own output) and needs the FULL in0[M,K] — delivered by a ring all-gather over
// the G=8 banks: each core reads 1/G of in0 from its bank, and the shards rotate cyclically so every core sees
// all G. Unlike in0_ring_writer (reserves the whole k-slice in cb0 -> OOM at full K), cb0 here is a small
// D-slot RECYCLING ring: shard s lands in slot s%D, is forwarded to next, consumed by compute, then freed.
// Credits: recv (fwd, prev->us "shard written") + slotfree (reverse, us->prev / next->us "slot free to write").
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);       // kb
    constexpr uint32_t N_block = get_compile_time_arg_val(2);       // N_sub
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);  // G*W (full K)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt = get_compile_time_arg_val(6);
    constexpr uint32_t W = get_compile_time_arg_val(7);  // blocks per shard
    constexpr uint32_t G = get_compile_time_arg_val(8);  // ring size
    constexpr uint32_t recv_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t slotfree_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t D = get_compile_time_arg_val(11);  // cb0 ring depth (shards)
    constexpr auto in0_args = TensorAccessorArgs<12>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t out_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t m0 = get_arg_val<uint32_t>(ai++);
    const uint32_t n0 = get_arg_val<uint32_t>(ai++);
    const uint32_t ring_pos = get_arg_val<uint32_t>(ai++);
    const uint32_t next_x = get_arg_val<uint32_t>(ai++);  // ring next (cyclic)
    const uint32_t next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t prev_x = get_arg_val<uint32_t>(ai++);  // ring prev (cyclic; for slotfree back-credit)
    const uint32_t prev_y = get_arg_val<uint32_t>(ai++);

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t shard_tiles = W * in0_blk;
    constexpr uint32_t shard_bytes = shard_tiles * tile_bytes;

    const uint32_t recv_addr = get_semaphore(recv_sem_id);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_addr);
    const uint32_t slotfree_addr = get_semaphore(slotfree_sem_id);
    volatile tt_l1_ptr uint32_t* slotfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slotfree_addr);
    const uint64_t prev_slotfree = get_noc_addr(prev_x, prev_y, slotfree_addr);
    const uint64_t next_recv = get_noc_addr(next_x, next_y, recv_addr);
    const uint32_t base0 = get_write_ptr(in0_cb);  // uniform cb0 base (same on all cores)

    // phase 1: streaming ring all-gather into a D-slot recycling cb0.
    for (uint32_t s = 0; s < G; ++s) {
        cb_reserve_back(in0_cb, shard_tiles);  // wait compute freed slot (s-D)
        uint32_t slot = base0 + (s % D) * shard_bytes;
        noc_semaphore_inc(prev_slotfree, 1);  // our slot (s%D) is free -> prev may write it
        if (s == 0) {
            uint32_t p = slot;  // read own shard (index ring_pos) from DRAM
            for (uint32_t wb = 0; wb < W; ++wb) {
                uint32_t sb = ring_pos * W + wb;
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        noc_async_read_page((m0 + m) * Kt + (sb * K_block + k), in0, p);
                        p += tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(recv, s);  // prev wrote our slot for step s
        }
        if (s + 1 < G) {                              // forward this shard to next's slot (s+1)%D
            noc_semaphore_wait_min(slotfree, s + 1);  // next freed the target slot
            uint64_t dst = get_noc_addr(next_x, next_y, base0 + ((s + 1) % D) * shard_bytes);
            noc_async_write(slot, dst, shard_bytes);
            noc_async_writes_flushed();  // our read of `slot` done before prev may reuse it
            noc_semaphore_inc(next_recv, 1);
        }
        cb_push_back(in0_cb, shard_tiles);  // compute consumes this shard (W blocks)
    }
    noc_async_write_barrier();

    // phase 2: write our own output [M_block, N_block] (no reduction; each core owns distinct columns)
    for (uint32_t m = 0; m < M_block; ++m) {
        cb_wait_front(out_cb, N_block);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < N_block; ++n) {
            noc_async_write_page((m0 + m) * Nt + (n0 + n), out, r + n * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, N_block);
    }
}
