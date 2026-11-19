// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly (TP8, the RESIDENT residual): the residual block across a PROGRAM boundary. Before the mhc P10 part of
// a later program: LOAD the member's block from the DRAM residual into CB_X / CB_X32 (mhc_stream.cpp read_x: reserve
// X_TILES on both views, read, push both); after it: STORE the block back (part_p16's F_X_STORE). Member BRISC.
// Mode 3 = ATTACH (PMK_TP8_RESIDENT: the block is PINNED at the same L1 address in every program of the layer, so a
// later program only re-arms the CB: reserve + push on both views, no DRAM read; the store at the end stays mode 2).
// CT at PH_CT0: [mode (1 load, 2 store, 3 attach), cb_x, cb_x32] + TensorAccessorArgs(x). RT at ph_rt0: [x address,
// rb0, c0].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace xio {
constexpr uint32_t MODE = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_X_ = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_X32_ = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t NS = 4, CPC = 16, COL_TILES = 128, X_TILES = NS * CPC, TB = 4096;
}  // namespace xio

void kernel_main() {
    using namespace xio;
    constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 3>();
    const uint32_t a = ph_rt0;
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(a + 0));
    const uint32_t rb0 = get_arg_val<uint32_t>(a + 1), c0 = get_arg_val<uint32_t>(a + 2);
    const uint32_t base = rb0 * NS * COL_TILES + c0;
    if constexpr (MODE == 3) {
        cb_reserve_back(CB_X_, X_TILES);
        cb_reserve_back(CB_X32_, X_TILES);
        cb_push_back(CB_X_, X_TILES);
        cb_push_back(CB_X32_, X_TILES);
    } else if constexpr (MODE == 1) {
        cb_reserve_back(CB_X_, X_TILES);
        cb_reserve_back(CB_X32_, X_TILES);
        const uint32_t l1 = get_write_ptr(CB_X_);
        for (uint32_t s = 0; s < NS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read_page(base + s * COL_TILES + j, x, l1 + (s * CPC + j) * TB);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_X_, X_TILES);
        cb_push_back(CB_X32_, X_TILES);
    } else {
        const uint32_t xl = get_read_ptr(CB_X_);
        for (uint32_t s = 0; s < NS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_write_page(base + s * COL_TILES + j, x, xl + (s * CPC + j) * TB);
            }
        }
        noc_async_write_barrier();
    }
}
