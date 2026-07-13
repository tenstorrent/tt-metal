// SPDX-License-Identifier: Apache-2.0
// Regime-A prototype INC1 reader (reader==consumer, all interleaved). Per core: feed compute.cpp for
// a single output block out[M_block, N_block] over the full K, reduced in K_block chunks.
//  - in0 [M,K] interleaved: read the core's M rows, full K. tile(m,k) id = m*Kt + k.
//  - in1 [K,N] interleaved: read K x the core's N-band. tile(k,n) id = k*Nt + n.
// Push order matches compute.cpp: per k_block push in0 block (m-major, K_block inner) then in1 block
// (k-major, N_block inner).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);
    constexpr uint32_t N_block = get_compile_time_arg_val(2);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Nt = get_compile_time_arg_val(5);  // full N width (tiles), for in1 addressing
    constexpr uint32_t Kt = get_compile_time_arg_val(6);  // full K width (tiles), for in0 addressing
    constexpr auto in0_args = TensorAccessorArgs<7>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);  // M start tile (this core)
    const uint32_t n0 = get_arg_val<uint32_t>(3);  // N-band start tile (this core)

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto in1 = TensorAccessor(in1_args, in1_addr, tile_bytes);

    constexpr uint32_t in0_cb = 0, in1_cb = 1;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t in1_blk = K_block * N_block;

    for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
        const uint32_t kbase = kb * K_block;
        // in0 block: m-major, K_block inner
        cb_reserve_back(in0_cb, in0_blk);
        uint32_t w0 = get_write_ptr(in0_cb);
        for (uint32_t m = 0; m < M_block; ++m) {
            for (uint32_t k = 0; k < K_block; ++k) {
                noc_async_read_page((m0 + m) * Kt + (kbase + k), in0, w0);
                w0 += tile_bytes;
            }
        }
        // in1 block: k-major, N_block inner
        cb_reserve_back(in1_cb, in1_blk);
        uint32_t w1 = get_write_ptr(in1_cb);
        for (uint32_t k = 0; k < K_block; ++k) {
            for (uint32_t n = 0; n < N_block; ++n) {
                noc_async_read_page((kbase + k) * Nt + (n0 + n), in1, w1);
                w1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(in0_cb, in0_blk);
        cb_push_back(in1_cb, in1_blk);
    }
}
