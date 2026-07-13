// SPDX-License-Identifier: Apache-2.0
// Regime-A K-split with in0 STORE-AND-FORWARD (unicast chain) + column reduction. Runs on the core's
// non-in1 RISC. Avoids both the 8x redundant in0 DRAM read (per-core-read) AND the mcast rect (which breaks
// bank-adjacency): in0[:,k-slice] is read from DRAM ONCE by the k-slice group's injector, then unicast
// forwarded around the group's chain (each core stores it in cb0 for its own matmul and forwards to the
// next). cb0 holds the full k-slice (Mt*Kt_local) so no ring backpressure.
//   phase 1: in0 chain  — injector reads DRAM; others wait fwd_recv; each forwards to the next (unicast).
//   phase 2: reduction  — split-K column chain (reuses compute.cpp REDUCE_K); top writes DRAM.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);
    constexpr uint32_t N_block = get_compile_time_arg_val(2);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);  // local (Kt_local/kb)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt = get_compile_time_arg_val(6);
    constexpr uint32_t fwd_sem_id = get_compile_time_arg_val(7);  // in0-chain recv semaphore
    constexpr uint32_t red_sem_id = get_compile_time_arg_val(8);  // reduction recv semaphore
    constexpr auto in0_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t out_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t m0 = get_arg_val<uint32_t>(ai++);
    const uint32_t n0 = get_arg_val<uint32_t>(ai++);
    const uint32_t k_start = get_arg_val<uint32_t>(ai++);     // this slice's first K tile (global)
    const uint32_t fwd_next_x = get_arg_val<uint32_t>(ai++);  // in0-chain next core
    const uint32_t fwd_next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t red_next_x = get_arg_val<uint32_t>(ai++);  // reduction-chain next core
    const uint32_t red_next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t is_injector = get_arg_val<uint32_t>(ai++);  // in0-chain head (reads DRAM)
    const uint32_t is_fwd_last = get_arg_val<uint32_t>(ai++);  // in0-chain tail (no forward)
    const uint32_t is_bottom = get_arg_val<uint32_t>(ai++);    // reduction bottom
    const uint32_t is_top = get_arg_val<uint32_t>(ai++);       // reduction top (writes DRAM)

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2, cb_reduce = 7;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t in0_blk_bytes = in0_blk * tile_bytes;
    constexpr uint32_t out_blk = M_block * N_block;
    const uint32_t fwd_addr = get_semaphore(fwd_sem_id);
    volatile tt_l1_ptr uint32_t* fwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_addr);
    const uint32_t red_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    const uint32_t reduce_base = get_write_ptr(cb_reduce);  // uniform cb_reduce base (fixed target)

    // phase 1: in0 store-and-forward. cb0 holds the full k-slice; reserve it once, fill/forward per block.
    cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
    const uint32_t base0 = get_write_ptr(in0_cb);
    for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
        uint32_t w = base0 + kb * in0_blk_bytes;
        if (is_injector) {
            const uint32_t kbase = kb * K_block;
            uint32_t p = w;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t k = 0; k < K_block; ++k) {
                    noc_async_read_page((m0 + m) * Kt + (k_start + kbase + k), in0, p);
                    p += tile_bytes;
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(fwd_ptr, kb + 1);  // prev forwarded block kb into our cb0[kb]
        }
        if (!is_fwd_last) {
            // pipelined forward: data then inc are NoC-ordered to the same dest, so the next core sees the
            // data before fwd_sem reaches kb+1 (no per-block barrier; cb0 is full-sized so no source reuse).
            uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, w);  // next core's cb0[kb] (uniform base)
            noc_async_write(w, dst, in0_blk_bytes);
            uint64_t nsem = get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr);
            noc_semaphore_inc(nsem, 1);
        }
        cb_push_back(in0_cb, in0_blk);  // compute consumes block kb
    }
    if (!is_fwd_last) {
        noc_async_write_barrier();  // ensure all forwards landed before reduction reuses NoC
    }

    // phase 2: split-K column reduction chain
    if (!is_bottom) {
        cb_reserve_back(cb_reduce, out_blk);
        noc_semaphore_wait(red_ptr, 1);
        noc_semaphore_set(red_ptr, 0);
        cb_push_back(cb_reduce, out_blk);
    }
    cb_wait_front(out_cb, out_blk);
    uint32_t r = get_read_ptr(out_cb);
    if (!is_top) {
        uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base);
        noc_async_write(r, dst, out_blk * tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_next_x, red_next_y, red_addr), 1);
    } else {
        for (uint32_t m = 0; m < M_block; ++m) {
            for (uint32_t n = 0; n < N_block; ++n) {
                noc_async_write_page((m0 + m) * Nt + (n0 + n), out, r + (m * N_block + n) * tile_bytes);
            }
        }
        noc_async_write_barrier();
    }
    cb_pop_front(out_cb, out_blk);
}
