// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr int onetile = 1;

    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t Mt = get_arg(args::Mt);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);

    uint32_t compute_id = get_my_thread_id();
    uint32_t num_threads = get_num_threads();

    DataflowBuffer dfb0(dfb::src0);
    DataflowBuffer dfb1(dfb::src1);
    DataflowBuffer dfb_out(dfb::dst);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::src0, dfb::src1, dfb::dst);
    matmul_init(dfb::src0, dfb::src1);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {  // output tile of C
            if (mt_C % num_threads != compute_id) {
                continue;
            }
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    dfb0.wait_front(onetile);
                    dfb1.wait_front(onetile);
                    matmul_tiles(dfb::src0, dfb::src1, 0, 0, 0);
                    dfb0.pop_front(onetile);
                    dfb1.pop_front(onetile);
                }
                tile_regs_commit();

                dfb_out.reserve_back(onetile);
                tile_regs_wait();
                pack_tile(0, dfb::dst);
                tile_regs_release();
                dfb_out.push_back(onetile);
            }
        }
    }
    dfb_out.finish();
}
