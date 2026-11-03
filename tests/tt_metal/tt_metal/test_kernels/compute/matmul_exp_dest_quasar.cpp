// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Route 1: FPU op, then SFPU op on the result in Dest, then pack. The default.
//
//   matmul (UNPACK + MATH) -> Dest[0] -> exp in place (MATH) -> Dest[0] -> pack (PACK) -> out
//
// The result never leaves Dest between the two ops. Three threads, synced by tile_regs_*. Only the
// exp line differs from the plain matmul kernel; it compiles to the same instructions as exp_tile(0).
//
// Compile-time args: 0 = in0 DFB id, 1 = in1 DFB id, 2 = out DFB id, 3 = Kt (tiles to accumulate),
// 4 = number of output tiles.

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

    DataflowBuffer in0(static_cast<std::uint16_t>(in0_id));
    DataflowBuffer in1(static_cast<std::uint16_t>(in1_id));
    DataflowBuffer out(static_cast<std::uint16_t>(out_id));

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_id, in1_id, out_id);
    matmul_init(in0_id, in1_id);
    sfpu_init<op::Exp, SfpuReg::Dest>();

    for (std::uint32_t t = 0; t < num_out_tiles; ++t) {
        tile_regs_acquire();
        for (std::uint32_t kt = 0; kt < Kt; ++kt) {
            in0.wait_front(1);
            in1.wait_front(1);
            matmul_tiles(in0_id, in1_id, 0, 0, 0);  // FPU: accumulate into Dest[0]
            in0.pop_front(1);
            in1.pop_front(1);
        }
        sfpu_tile<op::Exp, SfpuReg::Dest>(0);  // SFPU: exp(Dest[0]) in place, MATH thread
        tile_regs_commit();

        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_id);  // PACK: Dest[0] -> out
        out.push_back(1);
        tile_regs_release();
    }
}
