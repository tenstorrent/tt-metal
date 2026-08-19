// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t use_matmul = get_arg(args::use_matmul);
    constexpr uint32_t num_wraps = get_arg(args::num_wraps);

    DataflowBuffer initial(dfb::initial);
    DataflowBuffer rhs(dfb::rhs);
    DataflowBuffer stage(dfb::stage);
    DataflowBuffer outbound(dfb::outbound);

    compute_kernel_hw_startup(dfb::initial, dfb::rhs, dfb::outbound);

    // Seed the empty self-loop with one current tile. The stage DFB has two
    // entries, so every subsequent update can read front and pack back through
    // this same DFB ID.
    initial.wait_front(1);
    stage.reserve_back(1);
    pack_reconfig_data_format(stage.get_id());
    reconfig_data_format_srca(initial.get_id());
    copy_tile_to_dst_init_short(initial.get_id());
    tile_regs_acquire();
    copy_tile(initial.get_id(), 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, stage.get_id());
    tile_regs_release();
    stage.push_back(1);
    initial.pop_front(1);

    rhs.wait_front(1);
    for (uint32_t wrap = 0; wrap < num_wraps; ++wrap) {
        stage.wait_front(1);
        stage.reserve_back(1);
        outbound.reserve_back(1);

        reconfig_data_format(stage.get_id(), rhs.get_id());
        if constexpr (use_matmul != 0) {
            matmul_init(stage.get_id(), rhs.get_id());
        } else {
            add_init(stage.get_id(), rhs.get_id());
        }

        tile_regs_acquire();
        if constexpr (use_matmul != 0) {
            matmul_tiles(stage.get_id(), rhs.get_id(), 0, 0, 0);
        } else {
            add_tiles(stage.get_id(), rhs.get_id(), 0, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();

        // Publish the new value into the free physical half of the same stage
        // ring and also emit a snapshot so the host can verify every wrap.
        pack_reconfig_data_format(stage.get_id());
        pack_tile(0, stage.get_id());
        pack_reconfig_data_format(outbound.get_id());
        pack_tile(0, outbound.get_id());
        tile_regs_release();

        stage.push_back(1);
        outbound.push_back(1);
        stage.pop_front(1);
    }
    rhs.pop_front(1);
}
