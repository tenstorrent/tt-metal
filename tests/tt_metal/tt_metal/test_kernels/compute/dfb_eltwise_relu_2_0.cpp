// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) eltwise relu compute kernel.
//
// Per tile: consume one tile from dfb::in into the math dest register
// (copy_tile), apply SFPU relu_tile on dest, then pack the result out to
// dfb::out. Runs per_core_tile_cnt times.
//
// Bindings (set by host KernelSpec):
//   dfb::in  — CONSUMER (the upstream DM producer writes here)
//   dfb::out — PRODUCER (the downstream DM consumer drains here)
//
// Used by the A1 relu-pipeline test (DM → DFB → TRISC(relu) → DFB → DM) as the
// middle Tensix stage. Identical to dfb_eltwise_copy_2_0.cpp except for the
// extra relu_tile call between copy and pack.

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
    DataflowBuffer dfb_out(dfb::out);

    // Configure the compute pipeline against this kernel's input/output DFBs.
    // Matches production pattern (eltwise_sfpu.cpp / eltwise_binary.cpp).
    unary_op_init_common(dfb_in.get_id(), dfb_out.get_id());
    relu_tile_init();

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_out.get_id());
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }
}
