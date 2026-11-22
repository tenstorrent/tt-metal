// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill message-gate tail, compute. Per output tile (R, j):
//   f = tanh(z[R] @ F[:, j]), g = tanh(z[R] @ G[:, j])   (FPU, fp32 dest; tanh_tile as ttnn.tanh)
//   out = addcmul(addcmul(h, f, h, -1), g, m, 1)          (addcmul_tile as ttnn.addcmul, fp32)
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t KZ = get_named_compile_time_arg_val("kz");
    constexpr uint32_t minus_one = get_named_compile_time_arg_val("minus_one");
    constexpr uint32_t one = get_named_compile_time_arg_val("one");
    constexpr uint32_t cb_z = 0, cb_f = 1, cb_g = 2, cb_h = 3, cb_m = 4, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_z, cb_f, cb_out);
    pack_reconfig_data_format(cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t rows = get_arg_val<uint32_t>(1 + u);
        cb_wait_front(cb_f, KZ);
        cb_wait_front(cb_g, KZ);
        for (uint32_t i = 0; i < rows; ++i) {
            cb_wait_front(cb_z, KZ);
            cb_wait_front(cb_h, 1);
            cb_wait_front(cb_m, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_f, cb_z);  // matmul: in1 -> SrcA, in0 -> SrcB
            matmul_init(cb_z, cb_f);
            for (uint32_t k = 0; k < KZ; ++k) {
                matmul_tiles(cb_z, cb_f, k, k, 1);
            }
            reconfig_data_format(cb_g, cb_z);
            matmul_init(cb_z, cb_g);
            for (uint32_t k = 0; k < KZ; ++k) {
                matmul_tiles(cb_z, cb_g, k, k, 2);
            }
            tanh_tile_init();
            tanh_tile(1);
            tanh_tile(2);
            load(cb_h, 0, 0);
            load(cb_m, 0, 3);
            addcmul_tile_init();
            addcmul_tile<DataFormat::Float32>(0, 1, 0, 4, minus_one);
            addcmul_tile<DataFormat::Float32>(4, 2, 3, 4, one);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(4, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            cb_pop_front(cb_z, KZ);
            cb_pop_front(cb_h, 1);
            cb_pop_front(cb_m, 1);
        }
        cb_pop_front(cb_f, KZ);
        cb_pop_front(cb_g, KZ);
    }
}
