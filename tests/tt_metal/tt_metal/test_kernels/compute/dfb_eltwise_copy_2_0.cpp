// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) eltwise copy compute kernel.
//
// Identity operation: consume one tile from dfb::in, copy it to the math dest
// register via copy_tile, and pack it out to dfb::out. Runs per_core_tile_cnt
// times.
//
// Bindings (set by host KernelSpec):
//   dfb::in  — CONSUMER (the upstream DM producer writes here)
//   dfb::out — PRODUCER (the downstream DM consumer drains here)
//
// Used by the A1 identity-pipeline test (DM → DFB → TRISC(copy) → DFB → DM) as
// the middle Tensix stage. The relu variant in dfb_eltwise_relu_2_0.cpp is
// identical except for inserting an SFPU relu between copy and pack.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    // Configure the compute pipeline against this kernel's input/output DFBs.
    // Matches production pattern (eltwise_sfpu.cpp / eltwise_binary.cpp).
    compute_kernel_hw_startup(dfb_in.get_id(), dfb_out.get_id());
    copy_init(dfb_in.get_id());

    // One wait/pop covers share_in input tiles and one reserve/push covers share_out output tiles
    // (a whole block on a BLOCKED side, one tile otherwise). acquire/release stays per tile so
    // MATH, which has no DFB interface, runs the same trip count as PACK.
    const uint32_t share_in = dfb_in.get_consumer_share();
    const uint32_t stride_in = dfb_in.get_consumer_stride_tiles();
    const uint32_t share_out = dfb_out.get_producer_share();
    uint32_t packed = 0;  // tiles packed so far; reserve/push the output ring once per share_out
    for (uint32_t b = 0; b < per_core_tile_cnt; b += share_in) {
        dfb_in.wait_front(share_in);
        for (uint32_t j = 0; j < share_in; ++j) {
            if (packed % share_out == 0) {
                dfb_out.reserve_back(share_out);
            }
            acquire_dst();
            copy_tile(dfb_in.get_id(), j * stride_in, 0);
            pack_tile(0, dfb_out.get_id());
            release_dst();
            if (++packed % share_out == 0) {
                dfb_out.push_back(share_out);
            }
        }
        dfb_in.pop_front(share_in);
    }
}
