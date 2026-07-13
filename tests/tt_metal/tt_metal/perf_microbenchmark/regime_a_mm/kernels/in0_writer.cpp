// SPDX-License-Identifier: Apache-2.0
// Regime-A INC2 NCRISC kernel: phase 1 feeds in0 [M_block,K_block] per K-block (interleaved, small);
// phase 2 writes the output block [M_block,N_block] to interleaved DRAM (one M-row at a time).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);
    constexpr uint32_t N_block = get_compile_time_arg_val(2);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt = get_compile_time_arg_val(6);
    constexpr uint32_t skip_in0 = get_compile_time_arg_val(7);  // ablation: skip the in0 DRAM read
    constexpr uint32_t bcast = get_compile_time_arg_val(8);     // 1 = in0 broadcast (loader mcast into cb0)
    constexpr uint32_t valid0_id = get_compile_time_arg_val(9);
    constexpr uint32_t valid1_id = get_compile_time_arg_val(10);
    constexpr uint32_t split_h = get_compile_time_arg_val(11);       // K-blocks owned by loader 0 (rest -> loader 1)
    constexpr uint32_t bstream = get_compile_time_arg_val(12);       // 1 = STREAMING broadcast receiver (small cb0)
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(13);  // loader "slot free" sem (we inc it)
    constexpr auto in0_args = TensorAccessorArgs<14>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);
    const uint32_t n0 = get_arg_val<uint32_t>(3);
    // bstream runtime args (start at 4): L, then L * {valid_sem_id, ready_sem_id, loader_x, loader_y}

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2;
    constexpr uint32_t in0_blk = M_block * K_block;

    if constexpr (bstream) {
        // Streaming broadcast receiver with L loaders: K-block k is delivered by loader (k%L). Per block:
        // reserve a slot (waits until compute freed slot k-D), credit that loader (ready+=1), wait its mcast
        // (valid[l] >= k/L+1), hand to compute. cb0 is written directly by the loader's mcast into our slot.
        constexpr uint32_t MAXL = 4;
        const uint32_t L = get_arg_val<uint32_t>(4);
        volatile tt_l1_ptr uint32_t* valid[MAXL];
        uint64_t ready_noc[MAXL];
        for (uint32_t l = 0; l < L; ++l) {
            uint32_t vid = get_arg_val<uint32_t>(5 + l * 4 + 0);
            uint32_t rid = get_arg_val<uint32_t>(5 + l * 4 + 1);
            uint32_t lx = get_arg_val<uint32_t>(5 + l * 4 + 2);
            uint32_t ly = get_arg_val<uint32_t>(5 + l * 4 + 3);
            valid[l] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(vid));
            ready_noc[l] = get_noc_addr(lx, ly, get_semaphore(rid));
        }
        uint32_t cnt[MAXL];
        for (uint32_t l = 0; l < MAXL; ++l) {
            cnt[l] = 0;
        }
        for (uint32_t k = 0; k < K_num_blocks; ++k) {
            uint32_t l = k % L;
            cb_reserve_back(in0_cb, in0_blk);
            noc_semaphore_inc(ready_noc[l], 1);
            noc_semaphore_wait_min(valid[l], ++cnt[l]);
            cb_push_back(in0_cb, in0_blk);
        }
    } else if constexpr (bcast) {
        // Phase 1 (streamed broadcast): cb0 holds the full in0; the loader mcasts block k + bumps valid to
        // k+1. Reserve the full region once, then push one in0 block each time its valid arrives so compute
        // overlaps the broadcast (no serial prologue).
        constexpr uint32_t total_in0 = M_block * Kt;
        cb_reserve_back(in0_cb, total_in0);
        volatile tt_l1_ptr uint32_t* v0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(valid0_id));
        volatile tt_l1_ptr uint32_t* v1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(valid1_id));
        for (uint32_t kb2 = 0; kb2 < K_num_blocks; ++kb2) {
            if (kb2 < split_h) {
                noc_semaphore_wait_min(v0, kb2 + 1);
            } else {
                noc_semaphore_wait_min(v1, kb2 - split_h + 1);
            }
            cb_push_back(in0_cb, in0_blk);
        }
    } else {
        // phase 1: feed in0 per K-block (m-major, K_block inner)
        for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
            const uint32_t kbase = kb * K_block;
            cb_reserve_back(in0_cb, in0_blk);
            uint32_t w = get_write_ptr(in0_cb);
            if constexpr (!skip_in0) {
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        noc_async_read_page((m0 + m) * Kt + (kbase + k), in0, w);
                        w += tile_bytes;
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(in0_cb, in0_blk);
        }
    }

    // phase 2: write output block (one M-row = N_block tiles at a time)
    for (uint32_t m = 0; m < M_block; ++m) {
        cb_wait_front(out_cb, N_block);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < N_block; ++n) {
            noc_async_write_page((m0 + m) * Nt + (n0 + n), out, r);
            r += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, N_block);
    }
}
