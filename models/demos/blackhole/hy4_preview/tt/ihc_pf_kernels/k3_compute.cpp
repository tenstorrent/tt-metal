// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K3 compute: per column tile j: h_j = ((S0 x0 + S1 x1) + S2 x2) + S3 x3 (exact fp32 SFPU, S_s = pre_s
// replicated over the columns), packed to cb_h (fp32); A += h_j^2 (carried through the 1-page cb_acc between sections).
// CT: 0 cb_x, 1 cb_s, 2 cb_h, 3 cb_acc, 4 CT.  cb_x, cb_s, cb_acc are unpacked straight to dest (fp32 exact).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    compute_kernel_hw_startup(cb_x, cb_s, cb_h);
    CircularBuffer x(cb_x), s(cb_s), h(cb_h), acc(cb_acc);
    s.wait_front(4);
    for (uint32_t j = 0; j < CT; ++j) {
        if ((j & 1) == 0) {
            x.wait_front(8);
        }
        const uint32_t xo = (j & 1) * 4;
        if (j > 0) {
            acc.wait_front(1);
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_tile_to_dst_init_short(cb_x);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_x, xo + k, k);
        }
        reconfig_data_format_srca(cb_s);
        copy_tile_to_dst_init_short(cb_s);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_s, k, 4 + k);
        }
        mul_binary_tile_init();
        for (uint32_t k = 0; k < 4; ++k) {
            mul_binary_tile(k, 4 + k, k);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 1);  // dest 1 = h^2
        if (j > 0) {
            reconfig_data_format_srca(cb_acc);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 0, 2);
            add_binary_tile_init();
            add_binary_tile(1, 2, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (j > 0) {
            acc.pop_front(1);
        }
        h.reserve_back(1);
        acc.reserve_back(1);
        pack_tile(0, cb_h);
        pack_tile(1, cb_acc);
        h.push_back(1);
        acc.push_back(1);
        tile_regs_release();
        if ((j & 1) == 1) {
            x.pop_front(8);
        }
    }
}
