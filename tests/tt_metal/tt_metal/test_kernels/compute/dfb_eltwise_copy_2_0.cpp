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

    // Input ops are this hart's whole share of the input ring (a block, its part of a block, or
    // one tile); tile j of a strided share sits `stride` tiles from the bookmark. Output ops are
    // this hart's whole share of the output ring: one tile on a STRIDED ring, a whole block when
    // this producer is BLOCKED (reserve once, pack share_out tiles in order, push once). The
    // acquire/release handshake stays per tile: the MATH TRISC has no DFB interface and sees a
    // share of 1, so a per-share handshake would run different trip counts on MATH and PACK.
#ifdef ARCH_QUASAR
    const uint32_t share_in = dfb_in.get_consume_share();
    const uint32_t stride_in = dfb_in.get_consume_stride_tiles();
    const uint32_t share_out = dfb_out.get_produce_share();
#else
    const uint32_t share_in = 1;
    const uint32_t stride_in = 1;
    const uint32_t share_out = 1;
#endif
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
