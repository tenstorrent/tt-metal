// SPDX-License-Identifier: Apache-2.0
//
// llk_analysis one-call check (#58743, ops/blackhole/sfpu_misc): drives one misc compute API entry point per tile pair
// (a in c_0, b in c_1, the result of DEST tile 0 packed to c_16) so a host script can dump its words.
//   compile_time_args = [num_tiles, op]; op: 0 identity_tile, 1 identity_tile_uint32, 2 mask_tile (Float16_b),
//   3 mask_tile (Int32), 4 mask_posinf_tile, 5 copy_dest_values<Float16_b>, 6 <Float32>, 7 <Int32>, 8 the deprecated
//   copy_dest_values(idst_in, idst_out)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/mask.h"
#include "api/compute/copy_dest_values.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t op = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer a_cb(cb_a);
    CircularBuffer b_cb(cb_b);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_a, cb_out);
    copy_init(cb_a);
    if constexpr (op <= 1) {
        identity_tile_init();
    } else if constexpr (op <= 4) {
        mask_tile_init();
    } else {
        copy_dest_values_init();
    }

    for (uint32_t t = 0; t < num_tiles; ++t) {
        a_cb.wait_front(1);
        b_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_a, 0, 0);
        copy_tile(cb_b, 0, 1);
        if constexpr (op == 0) {
            identity_tile(0);
        } else if constexpr (op == 1) {
            identity_tile_uint32(0);
        } else if constexpr (op == 2) {
            mask_tile(0, 1, DataFormat::Float16_b);
        } else if constexpr (op == 3) {
            mask_tile(0, 1, DataFormat::Int32);
        } else if constexpr (op == 4) {
            mask_posinf_tile(0, 1);
        } else if constexpr (op == 5) {
            copy_dest_values<DataFormat::Float16_b>(1, 0);
        } else if constexpr (op == 6) {
            copy_dest_values<DataFormat::Float32>(1, 0);
        } else if constexpr (op == 7) {
            copy_dest_values<DataFormat::Int32>(1, 0);
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            copy_dest_values(1, 0);
#pragma GCC diagnostic pop
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        a_cb.pop_front(1);
        b_cb.pop_front(1);
    }
}
