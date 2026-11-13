// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 C4 compute kernel: the Dest path and a trisc3 self-loop in one kernel.
//
// Five DFBs:
//   dfb::in    (DM -> unpack)
//   dfb::out   (pack -> DM)
//   dfb::seed  (DM -> trisc3)
//   dfb::acc   (trisc3 -> trisc3, intra self-loop)
//   dfb::res   (trisc3 -> DM)
//
// Section 1, the Dest path: in -> relu -> out on unpack, math and pack. trisc3 owns none of these DFBs.
// Section 2, the self-loop: trisc3 copies each seed entry into acc, runs num_passes passes over acc that add 1
// to every word, and pushes the final entry to res. The two sections run at the same time on their own threads.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Copy one entry word by word with RISC loads and stores, adding delta to every word. src and dst are in the
// 16 B units get_read_ptr / get_write_ptr return.
inline void copy_entry_words(uint32_t src, uint32_t dst, uint32_t words, uint32_t delta) {
    const volatile tt_l1_ptr uint32_t* src_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src << 4);
    volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst << 4);
    for (uint32_t w = 0; w < words; w++) {
        dst_ptr[w] = src_ptr[w] + delta;
    }
    asm volatile("fence" ::: "memory");  // the writes land before push_back posts the credit
}

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    constexpr uint32_t num_passes = get_arg(args::num_passes);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_seed(dfb::seed);
    DataflowBuffer dfb_acc(dfb::acc);
    DataflowBuffer dfb_res(dfb::res);

    compute_kernel_hw_startup(dfb_in.get_id(), dfb_out.get_id());
    copy_init(dfb_in.get_id());
    relu_tile_init();

    // Section 1: in -> relu -> out. trisc3 owns neither DFB, so its calls here return at once.
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_out.get_id());
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }

    // Section 2: seed -> acc, num_passes times acc -> acc with +1 per word, then acc -> res. All on trisc3; the
    // waits poll the tile counters from this RISC, so an entry is in L1 or free once they return. A pass holds
    // the front entry while it reserves the next, so acc needs at least two entries.
    if constexpr (ckernel::TRISC_ID == 3) {
        const uint32_t words_per_entry = dfb_acc.get_entry_size() / sizeof(uint32_t);
        for (uint32_t e = 0; e < per_core_tile_cnt; ++e) {
            dfb_seed.wait_front(1);
            dfb_acc.reserve_back(1);
            copy_entry_words(dfb_seed.get_read_ptr(), dfb_acc.get_write_ptr(), words_per_entry, 0);
            dfb_seed.pop_front(1);
            dfb_acc.push_back(1);

            for (uint32_t p = 0; p < num_passes; ++p) {
                dfb_acc.wait_front(1);
                dfb_acc.reserve_back(1);
                copy_entry_words(dfb_acc.get_read_ptr(), dfb_acc.get_write_ptr(), words_per_entry, 1);
                dfb_acc.pop_front(1);
                dfb_acc.push_back(1);
            }

            dfb_acc.wait_front(1);
            dfb_res.reserve_back(1);
            copy_entry_words(dfb_acc.get_read_ptr(), dfb_res.get_write_ptr(), words_per_entry, 0);
            dfb_acc.pop_front(1);
            dfb_res.push_back(1);
        }
    }
}
