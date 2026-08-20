// One source, two KernelSpecs, different per-group CTA. `tiles_per_core` is a CTA here
// (the whole point: it stays compile-time so the loop unrolls), NOT an RTA.
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_core = get_arg(args::tiles_per_core);  // CTA, per-group

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    DataflowBuffer dfb_out(dfb::out);

    binary_op_init_common(dfb::in_a, dfb::in_b, dfb::out);
    mul_tiles_init(dfb::in_a, dfb::in_b);

#pragma GCC unroll 4
    for (uint32_t i = 0; i < tiles_per_core; ++i) {
        dfb_a.wait_front(1);
        dfb_b.wait_front(1);
        tile_regs_acquire();
        mul_tiles(dfb::in_a, dfb::in_b, 0, 0, 0);
        tile_regs_commit();
        dfb_a.pop_front(1);
        dfb_b.pop_front(1);
        dfb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        dfb_out.push_back(1);
    }
}
