// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/cumsum.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Columnwise: tiles arrive in NWH order, and the cumsum chain runs down each tile column (Ht).
// ROWWISE: tiles arrive in NHW order and are transposed into and out of Dest, so the host passes Wt as Ht
// (and vice versa) and the chain runs across each tile row.
void kernel_main() {
    constexpr int onetile = 1;
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
#ifndef ROWWISE
    copy_init(dfb::in);
#else
    transpose_init(dfb::in);
#endif
    cumsum_tile_init();

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                dfb_out.reserve_back(onetile);
                tile_regs_acquire();
                dfb_in.wait_front(onetile);

#ifndef ROWWISE
                copy_tile(dfb::in, 0, 0);
#else
                transpose_init(dfb::in);
                transpose_tile(dfb::in, 0, 0);
#endif
                cumsum_tile(0, ht == 0);
#ifdef ROWWISE
                transpose_dest_init(dfb::in);
                transpose_dest(0);
#endif

                tile_regs_commit();
                tile_regs_wait();

                pack_tile(0, dfb::out);

                dfb_in.pop_front(onetile);
                tile_regs_release();
                dfb_out.push_back(onetile);
            }
        }
    }
}
