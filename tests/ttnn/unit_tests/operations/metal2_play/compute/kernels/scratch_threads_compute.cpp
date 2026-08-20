// SPDX-License-Identifier: Apache-2.0
// PROBE C4: prove the compute-kernel scratchpad is ONE L1 region shared by all three TRISCs.
//
// kernel_main() is compiled three times (UNPACK / MATH / PACK) and each build receives the SAME
// binding token, hence the same base address. Here the UNPACK thread stamps a sentinel; the MATH
// thread reads it back AFTER copy_tile (which is where MATH synchronizes with UNPACK, so the read
// is properly ordered) and scales by 3.0 only if it sees UNPACK's write.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

constexpr uint32_t kSentinel = 0xC0DE1234u;
constexpr uint32_t kThreeBits = 0x40400000u;  // 3.0f
constexpr uint32_t kOneBits = 0x3F800000u;    // 1.0f

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer out(dfb::out_tiles);
    Scratchpad<uint32_t> pad(scratch::scale_table);

    UNPACK((pad[0] = kSentinel));

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);
    copy_tile_init(dfb::in_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);
        out.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        // MATH reads what UNPACK wrote into the same private region.
        MATH((mul_unary_tile(0, pad[0] == kSentinel ? kThreeBits : kOneBits)));
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out_tiles, 0);
        tile_regs_release();

        in.pop_front(1);
        out.push_back(1);
    }
}
