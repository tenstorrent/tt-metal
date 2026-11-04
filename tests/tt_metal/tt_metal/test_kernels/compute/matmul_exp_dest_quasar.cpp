// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Route 1: matmul into Dest[0], exp in place on Dest[0] (MATH thread), pack. The result never leaves Dest.
//
// Compile-time args: 0 = in0 DFB id, 1 = in1 DFB id, 2 = out DFB id, 3 = Kt (tiles to accumulate),
// 4 = number of output tiles, 5 = DataFormat of the tiles in Dest.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr std::uint32_t in0_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t in1_id = get_compile_time_arg_val(1);
    constexpr std::uint32_t out_id = get_compile_time_arg_val(2);
    constexpr std::uint32_t Kt = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_out_tiles = get_compile_time_arg_val(4);
    constexpr DataFormat dest_format = static_cast<DataFormat>(get_compile_time_arg_val(5));

    DataflowBuffer in0(static_cast<std::uint16_t>(in0_id));
    DataflowBuffer in1(static_cast<std::uint16_t>(in1_id));
    DataflowBuffer out(static_cast<std::uint16_t>(out_id));

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_id, in1_id, out_id);
    matmul_init(in0_id, in1_id);
    sfpu_init<op::Exp, SfpuReg::Dest, dest_format>();

    for (std::uint32_t t = 0; t < num_out_tiles; ++t) {
        tile_regs_acquire();
        for (std::uint32_t kt = 0; kt < Kt; ++kt) {
            in0.wait_front(1);
            in1.wait_front(1);
            matmul_tiles(in0_id, in1_id, 0, 0, 0);  // FPU: accumulate into Dest[0]
            in0.pop_front(1);
            in1.pop_front(1);
        }
        sfpu_tile<op::Exp, SfpuReg::Dest, dest_format>(0);  // SFPU: exp(Dest[0]) in place
        tile_regs_commit();

        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_id);  // PACK: Dest[0] -> out
        out.push_back(1);
        tile_regs_release();
    }
}
