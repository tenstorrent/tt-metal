// SPDX-License-Identifier: Apache-2.0
// Regime-A N-sub-division reader (BRISC), fast path. P readers share a bank; this core owns a
// contiguous N-sub-band [K, N_block] of the bank shard [K, N_band] (k-major). One row = N_block tiles
// contiguous; stride N_band between rows. Uses one_packet_with_state (like the contiguous reader) with
// a single packet per row (requires N_block*tile_bytes <= 16 KB, i.e. ns <= 8 tiles). Double-buffered
// via 2 rotating TRIDs.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_block = get_compile_time_arg_val(0);
    constexpr uint32_t N_block = get_compile_time_arg_val(1);  // ns (<= 8 tiles)
    constexpr uint32_t N_band = get_compile_time_arg_val(2);   // full bank shard width (stride)
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t skip_in1 =
        get_compile_time_arg_val(5);  // ablation: feed compute without reading in1 (free in1 delivery)

    const uint32_t in1_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t sub_off = get_arg_val<uint32_t>(2);  // tiles, = p * N_block
    const uint32_t vc = get_arg_val<uint32_t>(3);

    constexpr uint32_t in1_cb = 1;
    constexpr uint32_t in1_blk = K_block * N_block;
    constexpr uint32_t row_bytes = N_block * tile_bytes;

    if constexpr (skip_in1) {  // ablation: no DRAM read; just hand compute empty blocks
        for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
            cb_reserve_back(in1_cb, in1_blk);
            cb_push_back(in1_cb, in1_blk);
        }
        return;
    }

    uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, in1_addr);
    noc_async_read_one_packet_set_state<true>(src_base, row_bytes, vc);  // one packet = one row's sub-band

    for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
        const uint32_t kbase = kb * K_block;
        cb_reserve_back(in1_cb, in1_blk);
        uint32_t w1 = get_write_ptr(in1_cb);
        const uint32_t trid = kb % 2 + 1;
        noc_async_read_set_trid(trid);
        for (uint32_t kl = 0; kl < K_block; ++kl) {
            const uint32_t off = ((kbase + kl) * N_band + sub_off) * tile_bytes;  // byte offset in bank
            noc_async_read_one_packet_with_state_with_trid(src_base, off, w1, trid);
            w1 += row_bytes;
        }
        if (kb >= 1) {
            noc_async_read_barrier_with_trid((kb - 1) % 2 + 1);
            cb_push_back(in1_cb, in1_blk);
        }
    }
    noc_async_read_barrier_with_trid((K_num_blocks - 1) % 2 + 1);
    cb_push_back(in1_cb, in1_blk);
}
