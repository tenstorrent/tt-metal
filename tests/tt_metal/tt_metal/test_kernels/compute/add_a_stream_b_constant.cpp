// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"

using namespace ckernel;
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // A
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // B (single tile)
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // out

    cb_wait_front(cb_in1, 1);
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in0, 1);
        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*dst_idx*/ 0, cb_out0);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);

        // Intentionally do not pop cb_in1
        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
