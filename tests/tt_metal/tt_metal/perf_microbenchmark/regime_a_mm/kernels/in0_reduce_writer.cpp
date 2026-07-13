// SPDX-License-Identifier: Apache-2.0
// Regime-A K-split DM kernel (runs on the core's non-in1 RISC).
// phase 1: read this core's in0 K-slice [M_block, Kt_local] (interleaved) into cb0 for the matmul.
// phase 2: split-K column reduction chain (one [M_block, N_block] partial per core):
//   - if !bottom: reserve cb_reduce, wait recv_sem (prev slice sent its running sum), push cb_reduce so
//     compute's reduce_add can consume it;
//   - wait out_cb (compute's partial or accumulated sum);
//   - if !top: NoC-write out_cb -> next slice's cb_reduce base + inc its recv_sem;
//   - if top: write out_cb -> DRAM [M_block, N_block] at (m0,n0).
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
    constexpr uint32_t recv_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t skip_in0 = get_compile_time_arg_val(8);  // ablation: skip in0 k-slice read (free in0)
    constexpr auto in0_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);
    const uint32_t n0 = get_arg_val<uint32_t>(3);
    const uint32_t k_start = get_arg_val<uint32_t>(4);  // this slice's first K tile (global)
    const uint32_t next_x = get_arg_val<uint32_t>(5);   // next slice's phys coords (forward target)
    const uint32_t next_y = get_arg_val<uint32_t>(6);
    const uint32_t is_bottom = get_arg_val<uint32_t>(7);
    const uint32_t is_top = get_arg_val<uint32_t>(8);

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2, cb_reduce = 7;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t out_blk = M_block * N_block;
    const uint32_t recv_addr = get_semaphore(recv_sem_id);
    volatile tt_l1_ptr uint32_t* recv_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_addr);
    // Capture cb_reduce's base ONCE (uniform across cores). After a middle core receives+pops a block its
    // write ptr drifts, so we must forward to this fixed base (= the next core's fresh-reserve address).
    const uint32_t reduce_base = get_write_ptr(cb_reduce);

    // phase 1: feed in0 K-slice per K-block (global tile id = m*Kt + k_start + kbase + k)
    for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
        const uint32_t kbase = kb * K_block;
        cb_reserve_back(in0_cb, in0_blk);
        uint32_t w = get_write_ptr(in0_cb);
        if constexpr (!skip_in0) {
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t k = 0; k < K_block; ++k) {
                    noc_async_read_page((m0 + m) * Kt + (k_start + kbase + k), in0, w);
                    w += tile_bytes;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(in0_cb, in0_blk);
    }

    // phase 2: reduction chain
    if (!is_bottom) {
        cb_reserve_back(cb_reduce, out_blk);
        noc_semaphore_wait(recv_ptr, 1);
        noc_semaphore_set(recv_ptr, 0);
        cb_push_back(cb_reduce, out_blk);  // compute reduce_add consumes this
    }
    cb_wait_front(out_cb, out_blk);
    uint32_t r = get_read_ptr(out_cb);
    if (!is_top) {
        // forward this core's out_cb (partial or running sum) to next slice's cb_reduce base
        uint64_t dst = get_noc_addr(next_x, next_y, reduce_base);
        noc_async_write(r, dst, out_blk * tile_bytes);
        noc_async_write_barrier();
        uint64_t nsem = get_noc_addr(next_x, next_y, recv_addr);
        noc_semaphore_inc(nsem, 1);
    } else {
        // top: write final [M_block, N_block] to DRAM, one M-row at a time
        for (uint32_t m = 0; m < M_block; ++m) {
            for (uint32_t n = 0; n < N_block; ++n) {
                noc_async_write_page((m0 + m) * Nt + (n0 + n), out, r + (m * N_block + n) * tile_bytes);
            }
        }
        noc_async_write_barrier();
    }
    cb_pop_front(out_cb, out_blk);
}
