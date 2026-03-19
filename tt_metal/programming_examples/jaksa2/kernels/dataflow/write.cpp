// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t per_core_M = get_arg_val<uint32_t>(1);
    uint32_t per_core_N = get_arg_val<uint32_t>(2);
    uint32_t core_x = get_arg_val<uint32_t>(3);
    uint32_t core_y = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_out);

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, single_tile_size_bytes);

    SliceRange sr = {.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

    cb_wait_front(cb_out, per_core_M * per_core_N);
    DPRINT << "write - " << " - got OUT data " << ENDL();
    DPRINT << "OUT[0,0]=" << TileSlice(cb_out, 0, sr, true, false) << ENDL();
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    for (uint32_t m = 0; m < per_core_M; m++) {
        for (uint32_t n = 0; n < per_core_N; n++) {
            // DPRINT << "write - " << " - writing tile to index: " << Nt * (core_y * per_core_M + m) + core_x *
            // per_core_N + n << ENDL();
            noc_async_write_tile(Nt * (core_y * per_core_M + m) + core_x * per_core_N + n, s, l1_read_addr);
        }
    }
    cb_pop_front(cb_out, per_core_M * per_core_N);
    // DPRINT << "write - core x=" << core_x << ", y=" << core_y << " - finished!" << ENDL();
}
