// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar section hand-off: every section copies one Float32 tile into DST[0] through unpack-to-dest, calls
// tile_regs_handoff(), then runs num_src_tiles add_tiles through SrcA/SrcB into DST[1] (each add overwrites, so DST[1]
// ends as the last pair's sum). num_src_tiles exceeds the two source banks: without the hand-off the unpacker stalls on
// the third src tile while math still waits for the section, and both threads hang.
// HANDOFF_EVERY_SECTION: odd sections skip the copy but still call tile_regs_handoff(); their first add goes to DST[0]
// instead, so every section packs DST[0] then DST[1] and consumes the same src tiles.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_sections = get_arg(args::num_sections);
    constexpr uint32_t num_src_tiles = get_arg(args::num_src_tiles);

    DataflowBuffer in_a(dfb::in_a);
    DataflowBuffer in_b(dfb::in_b);
    DataflowBuffer in_fp32(dfb::in_fp32);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup(dfb::in_a, dfb::in_b, dfb::out);

    for (uint32_t s = 0; s < num_sections; ++s) {
#ifdef HANDOFF_EVERY_SECTION
        const bool has_copy = (s % 2) == 0;
#else
        const bool has_copy = true;
#endif
        in_a.wait_front(num_src_tiles);
        in_b.wait_front(num_src_tiles);
        if (has_copy) {
            in_fp32.wait_front(1);
        }
        out.reserve_back(2);

        tile_regs_acquire();
        if (has_copy) {
            // Float32 into DST[0] through UNP_DEST; the unpacker is reconfigured around it like the matmul reload.
            reconfig_data_format_srca(dfb::in_a, dfb::in_fp32);
            copy_init(dfb::in_fp32);
            copy_tile(dfb::in_fp32, 0, 0);
            reconfig_data_format_srca(dfb::in_fp32, dfb::in_a);
        }
        tile_regs_handoff();
        add_init(dfb::in_a, dfb::in_b);
        for (uint32_t k = 0; k < num_src_tiles; ++k) {
            // Pair k overwrites DST[1]; in a no-copy section pair 0 fills DST[0] instead.
            add_tiles(dfb::in_a, dfb::in_b, k, k, (!has_copy && k == 0) ? 0 : 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        pack_tile(1, dfb::out);
        tile_regs_release();

        in_a.pop_front(num_src_tiles);
        in_b.pop_front(num_src_tiles);
        if (has_copy) {
            in_fp32.pop_front(1);
        }
        out.push_back(2);
    }
}
