// SPDX-License-Identifier: Apache-2.0
// PROBE H: TT_KERNEL ("1st-world arguments") syntax on a COMPUTE kernel.
//   - CTAs are template parameters -> usable directly in `if constexpr`
//   - RTAs/CRTAs are function parameters
// genfiles synthesizes kernel_main() from this signature.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <uint32_t do_scale, uint32_t scale_bits>  // CTAs
TT_KERNEL void ttk_compute(uint32_t num_tiles) {   // RTA
    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer out(dfb::out_tiles);

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);
    copy_tile_init(dfb::in_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);
        out.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        // Real compile-time branching on a host-supplied CTA, with no macro and no
        // get_compile_time_arg_val(N) index bookkeeping.
        if constexpr (do_scale != 0) {
            mul_unary_tile(0, scale_bits);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out_tiles, 0);
        tile_regs_release();

        in.pop_front(1);
        out.push_back(1);
    }
}
