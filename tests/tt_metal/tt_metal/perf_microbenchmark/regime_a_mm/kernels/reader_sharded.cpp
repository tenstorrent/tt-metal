// SPDX-License-Identifier: Apache-2.0
// Regime-A INC2 in1 reader (BRISC, pure in1 stream for max BW). Contiguous bank read (one shard per
// bank), 16 KB packets, double-buffered via 2 rotating TRIDs so reads flow continuously across
// K-blocks. Pushes one in1 block [K_block, N_block] to cb1 per K-block. in0 + output are handled on
// NCRISC (in0_writer.cpp). Requires in1 block bytes to be a multiple of 16 KB.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_block = get_compile_time_arg_val(0);
    constexpr uint32_t N_block = get_compile_time_arg_val(1);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t skip_in1 = get_compile_time_arg_val(4);  // ablation: feed compute without reading in1

    const uint32_t in1_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t base_off = get_arg_val<uint32_t>(3);  // byte offset into bank (K-slice start); 0 = whole bank

    constexpr uint32_t in1_cb = 1;
    constexpr uint32_t in1_blk = K_block * N_block;
    constexpr uint32_t in1_blk_bytes = in1_blk * tile_bytes;
    constexpr uint32_t MAXBURST = 16384;
    constexpr uint32_t pages_per_block = in1_blk_bytes / MAXBURST;

    if constexpr (skip_in1) {  // ablation: no DRAM read; just hand compute empty blocks
        for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
            cb_reserve_back(in1_cb, in1_blk);
            cb_push_back(in1_cb, in1_blk);
        }
        return;
    }

    uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, in1_addr);
    noc_async_read_one_packet_set_state<true>(src_base, MAXBURST, vc);

    // 3 blocks in flight; push block (kb-2) after barriering its TRID. trid(b) = b%3 + 1.
    uint32_t l1_read = base_off;  // K-slice: start at k_start*N_band*tile_bytes within the bank
    for (uint32_t kb = 0; kb < K_num_blocks; ++kb) {
        cb_reserve_back(in1_cb, in1_blk);
        uint32_t w1 = get_write_ptr(in1_cb);
        const uint32_t trid = kb % 3 + 1;
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < pages_per_block; ++p) {
            noc_async_read_one_packet_with_state_with_trid(src_base, l1_read, w1, trid);
            l1_read += MAXBURST;
            w1 += MAXBURST;
        }
        if (kb >= 2) {
            noc_async_read_barrier_with_trid((kb - 2) % 3 + 1);
            cb_push_back(in1_cb, in1_blk);
        }
    }
    for (uint32_t kb = (K_num_blocks >= 2 ? K_num_blocks - 2 : 0); kb < K_num_blocks; ++kb) {
        noc_async_read_barrier_with_trid(kb % 3 + 1);
        cb_push_back(in1_cb, in1_blk);
    }
}
