// SPDX-License-Identifier: Apache-2.0
//
// Loopback compute: copies tiles from cb_in (index 0) to cb_out (index 16)
// via copy_tile — no arithmetic. Used to test unpack_to_dest behaviour.
//
// CT arg 0: per_core_tile_cnt

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;    // index 0
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;  // index 16

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(static_cast<uint32_t>(cb_in));

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        acquire_dst();

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        copy_tile(cb_in, /*src_idx=*/0, /*dst_idx=*/0);
        pack_tile(/*dst_idx=*/0, cb_out);

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
