// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 TP32 prefill rs-exact tail, compute: out_c = bf16(((b0 + b1) + b2) + b3) (exact fp32 SFPU adds in the fixed order
// 0..3). CT: 0 cb_in (unpacked straight to dest), 1 cb_out, 2 NCOL.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t NCOL = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    CircularBuffer in(cb_in), out(cb_out);
    for (uint32_t c = 0; c < NCOL; ++c) {
        in.wait_front(4);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in);
        copy_tile_to_dst_init_short(cb_in);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_in, k, k);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
        in.pop_front(4);
    }
}
