// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) C2 compute kernel.
//
// Three DFB pipeline:
//   dfb::in    (inter, DM → TRISC)
//   dfb::self  (intra, TRISC self-loop)  ← this kernel binds both PRODUCER and CONSUMER
//   dfb::out   (inter, TRISC → DM)
//
// Per tile:
//   stage A: unpack dfb_in  → SFPU relu → pack dfb_self
//   stage B: unpack dfb_self → SFPU relu → pack dfb_out

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_self(dfb::self);
    DataflowBuffer dfb_out(dfb::out);

    // Init with the kernel's first input and final output DFBs. The per-init
    // buf_desc_id state survives across per-tile pack_tile/copy_tile calls,
    // so stage A's intermediate pack to dfb_self only works because the per-
    // tile pack_tile(0, dfb_self.get_id()) reconfigures the destination —
    // the init still needs to point at the kernel's outermost output.
    // Matches production pattern (eltwise_sfpu.cpp / eltwise_binary.cpp).
    compute_kernel_hw_startup(dfb_in.get_id(), dfb_out.get_id());
    copy_init(dfb_in.get_id());
    relu_tile_init();

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // Stage A: dfb_in → relu → dfb_self
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_self.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_self.get_id());
        dfb_in.pop_front(1);
        dfb_self.push_back(1);
        release_dst();

        // Stage B: dfb_self → relu → dfb_out
        acquire_dst();
        dfb_self.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_self.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_out.get_id());
        dfb_self.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }
}
