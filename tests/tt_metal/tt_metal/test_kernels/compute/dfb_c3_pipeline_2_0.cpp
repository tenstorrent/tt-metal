// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 C3 compute kernel: unpack and pack keep their DFBs to the DM kernels while trisc3 sits between
// them as the consumer of one intra DFB and the producer of another.
//
// Four DFB pipeline:
//   dfb::in    (DM -> unpack)
//   dfb::mid   (pack -> trisc3, intra)
//   dfb::back  (trisc3 -> unpack, intra)
//   dfb::out   (pack -> DM)
//
// Per tile: stage A runs in -> relu -> mid and stage C runs back -> relu -> out on unpack, math and pack;
// stage B has trisc3 copy the mid entry into back with RISC loads and stores. Positive input comes out unchanged.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_mid(dfb::mid);
    DataflowBuffer dfb_back(dfb::back);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb_in.get_id(), dfb_out.get_id());
    copy_init(dfb_in.get_id());
    relu_tile_init();

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // Stage A: in -> relu -> mid
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_mid.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_mid.get_id());
        dfb_in.pop_front(1);
        dfb_mid.push_back(1);
        release_dst();

        // Stage B: trisc3 copies mid -> back
        dfb_mid.wait_front(1);
        dfb_back.reserve_back(1);
        if constexpr (ckernel::TRISC_ID == 3) {
            // Both waits poll the tile counters from this RISC, so the mid entry is in L1 and the back
            // entry is free once they return. get_read_ptr / get_write_ptr are in 16 B units.
            const uint32_t words_per_entry = dfb_mid.get_entry_size() / sizeof(uint32_t);
            const volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dfb_mid.get_read_ptr() << 4);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_back.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; w++) {
                dst[w] = src[w];
            }
            asm volatile("fence" ::: "memory");  // the copy lands before push_back posts the credit
        }
        dfb_mid.pop_front(1);
        dfb_back.push_back(1);

        // Stage C: back -> relu -> out
        acquire_dst();
        dfb_back.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_back.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_out.get_id());
        dfb_back.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }
}
