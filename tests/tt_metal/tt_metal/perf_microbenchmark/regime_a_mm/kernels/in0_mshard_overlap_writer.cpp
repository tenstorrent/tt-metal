// SPDX-License-Identifier: Apache-2.0
// Overlapped deep-K M-shard ring (Pk1, NO reduction). Streams the in0 M-shard all-gather into a D-slot recycling
// cb0 and pushes EACH shard as it arrives, so the compute overlaps the ring (hides the 8-step ring latency that
// the one-shot mshard exposes). Each shard = a deep-K M-block [Mw, Kt_local] (kb=Kt_local, K_num_blocks=1); the
// compute (M_blocks_per_core=G, IN1_RESIDENT) reuses the resident in1 across the G M-blocks. Output goes to the
// shard's rotated M-offset. Credits: recv (fwd, prev->us) + slotfree (reverse, us->prev / next->us).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);  // full M per core (= Mt/Sm)
    constexpr uint32_t K_block = get_compile_time_arg_val(1);  // Kt_local (deep-K, K_num_blocks==1)
    constexpr uint32_t N_block = get_compile_time_arg_val(2);  // N_sub
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(8);  // ring size (banks)
    constexpr uint32_t recv_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t slotfree_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t D = get_compile_time_arg_val(11);      // cb0 ring depth (shards)
    constexpr uint32_t Mw = get_compile_time_arg_val(12);     // M-rows per shard (= M_block/G)
    constexpr uint32_t N_bpc = get_compile_time_arg_val(13);  // N-sub-blocks per core
    constexpr auto in0_args = TensorAccessorArgs<14>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t out_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t m0 = get_arg_val<uint32_t>(ai++);
    const uint32_t n0 = get_arg_val<uint32_t>(ai++);
    const uint32_t ring_pos = get_arg_val<uint32_t>(ai++);
    const uint32_t next_x = get_arg_val<uint32_t>(ai++);
    const uint32_t next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t prev_x = get_arg_val<uint32_t>(ai++);
    const uint32_t prev_y = get_arg_val<uint32_t>(ai++);

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2;
    constexpr uint32_t Kt_local = K_block;
    constexpr uint32_t shard_tiles = Mw * Kt_local;
    constexpr uint32_t shard_bytes = shard_tiles * tile_bytes;
    constexpr uint32_t out_blk = Mw * N_block;

    const uint32_t recv_addr = get_semaphore(recv_sem_id);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_addr);
    const uint32_t slotfree_addr = get_semaphore(slotfree_sem_id);
    volatile tt_l1_ptr uint32_t* slotfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slotfree_addr);
    const uint64_t prev_slotfree = get_noc_addr(prev_x, prev_y, slotfree_addr);
    const uint64_t next_recv = get_noc_addr(next_x, next_y, recv_addr);
    const uint32_t base0 = get_write_ptr(in0_cb);

    // phase 1: streaming M-shard ring all-gather into a D-slot recycling cb0 (push per shard for overlap).
    for (uint32_t s = 0; s < G; ++s) {
        cb_reserve_back(in0_cb, shard_tiles);  // wait compute freed slot (s-D)
        uint32_t slot = base0 + (s % D) * shard_bytes;
        noc_semaphore_inc(prev_slotfree, 1);  // our slot free -> prev may write it
        if (s == 0) {                         // read own shard (M-rows [ring_pos*Mw : +Mw], all K)
            uint32_t p = slot;
            for (uint32_t mr = 0; mr < Mw; ++mr) {
                for (uint32_t k = 0; k < Kt_local; ++k) {
                    noc_async_read_page((m0 + ring_pos * Mw + mr) * Kt + k, in0, p);
                    p += tile_bytes;
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(recv, s);  // prev forwarded shard into our slot
        }
        if (s + 1 < G) {                              // forward to next's slot (s+1)%D
            noc_semaphore_wait_min(slotfree, s + 1);  // next freed target slot
            uint64_t dst = get_noc_addr(next_x, next_y, base0 + ((s + 1) % D) * shard_bytes);
            noc_async_write(slot, dst, shard_bytes);
            noc_async_writes_flushed();
            noc_semaphore_inc(next_recv, 1);
        }
        cb_push_back(in0_cb, shard_tiles);  // compute processes this shard as M-block s
    }
    noc_async_write_barrier();

    // phase 2: write each M-block's output to the shard's ROTATED M-offset (compute m_block s = shard (ring_pos-s)).
    for (uint32_t s = 0; s < G; ++s) {
        uint32_t g = (ring_pos + G - s) % G;  // shard index processed as compute M-block s
        uint32_t m_off = m0 + g * Mw;
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
            cb_wait_front(out_cb, out_blk);
            uint32_t r = get_read_ptr(out_cb);
            uint32_t n_off = n0 + nb * N_block;
            for (uint32_t mr = 0; mr < Mw; ++mr) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    noc_async_write_page((m_off + mr) * Nt + (n_off + n), out, r + (mr * N_block + n) * tile_bytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(out_cb, out_blk);
        }
    }
}
