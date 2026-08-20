// SPDX-License-Identifier: Apache-2.0
// PROBE C: ScratchpadSpec on a COMPUTE kernel — private, non-FIFO L1 for a TRISC kernel.
//
// The kernel builds a per-tile scale table in its scratchpad (something a CB cannot express:
// random access, no credits, no producer/consumer) and then applies it with an SFPU
// scalar-multiply. out[i] = in[i] * (i + 1).
//
// HAZARD (documented in FINDINGS): kernel_main() is compiled three times (UNPACK/MATH/PACK) and
// all three TRISCs receive the SAME scratchpad base address. The table fill below therefore runs
// three times over the same L1. It is safe here ONLY because the writes are value-identical and
// idempotent. Anything stateful (a counter, a queue) needs an explicit UNPACK()/MATH()/PACK()
// guard; the framework provides no per-thread scratchpad and no cross-stage synchronization.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer out(dfb::out_tiles);
    Scratchpad<uint32_t> table(scratch::scale_table);

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);
    copy_tile_init(dfb::in_tiles);

    // Fill the private table. `table.size()` comes from the binding token's compile-time size.
    for (uint32_t i = 0; i < table.size(); ++i) {
        union {
            float f;
            uint32_t u;
        } c;
        c.f = static_cast<float>(i + 1);
        table[i] = c.u;
    }

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);
        out.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        mul_unary_tile(0, table[i]);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out_tiles, 0);
        tile_regs_release();

        in.pop_front(1);
        out.push_back(1);
    }
}
