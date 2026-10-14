// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reshuffle.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

// ci_ab instrumentation: cycles per section on each TRISC, summed and emitted once at the end.
inline uint32_t ci_wc() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t input_height = get_compile_time_arg_val(1);

    constexpr uint32_t cb_grad_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_index_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_intermed_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_24;
    constexpr uint32_t cb_chunk_count_scratch_idx = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_idx = tt::CBIndex::c_16;

    CircularBuffer cb_grad(cb_grad_idx);
    CircularBuffer cb_out_intermed(cb_out_intermed_idx);
    CircularBuffer cb_mask(cb_mask_idx);
    CircularBuffer cb_out(cb_out_idx);

    compute_kernel_hw_startup(cb_grad_idx, cb_out_idx);
    copy_init(cb_grad_idx);

    uint32_t ci_in = 0, ci_acq = 0, ci_sfpu = 0, ci_pack = 0, ci_t = 0;
    for (uint32_t i = 0; i < input_height; ++i) {
        cb_grad.wait_front(max_tiles_per_core);

        // Get chunk_count from CB using mailbox-based synchronization (issue #27979)
        uint32_t chunk_count = read_tile_value(cb_chunk_count_scratch_idx, 0, 0);

        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            cb_mask.wait_front(1);

            // Get idx_addr from CB using mailbox-based synchronization (issue #27979)
            uint32_t idx_addr = get_tile_address(cb_mask_idx, 0);
#if defined(ARCH_BLACKHOLE)
            // Workaround for tt-metal issue #11816:
            // Flush the cache, forces going to L1 to access data
            asm volatile("fence");
#endif

            ci_t = ci_wc();
            cb_out_intermed.wait_front(max_tiles_per_core);
            ci_in += ci_wc() - ci_t;

            cb_out.reserve_back(max_tiles_per_core);

            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                ci_t = ci_wc();
                tile_regs_acquire();
                tile_regs_wait();
                ci_acq += ci_wc() - ci_t;

                copy_tile(cb_grad_idx, hidden_dim, 0);

                copy_tile(cb_out_intermed_idx, hidden_dim, 1);

                ci_t = ci_wc();
                reshuffle_rows_tile_init();
                // reshuffle_rows_tile expects that tiles have a header of 16 bytes.
                // This isn't true, so we have to subtract 16 bytes from the address.
                // Check implementation of reshuffle_rows_tile in LLK for more details.
                // tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reshuffle_rows.h
                reshuffle_rows_tile(0, idx_addr - 16);
                ci_sfpu += ci_wc() - ci_t;
                ci_t = ci_wc();

                pack_tile(1, cb_out_idx, hidden_dim);  // reshuffle puts output into Tile 1 in DEST

                tile_regs_commit();
                tile_regs_release();
                ci_pack += ci_wc() - ci_t;
            }

            cb_out.push_back(max_tiles_per_core);
            cb_out_intermed.pop_front(max_tiles_per_core);

            cb_mask.pop_front(1);
        }

        cb_grad.pop_front(max_tiles_per_core);
    }
    DeviceTimestampedData("RR_IN", ci_in);
    DeviceTimestampedData("RR_ACQ", ci_acq);
    DeviceTimestampedData("RR_SFPU", ci_sfpu);
    DeviceTimestampedData("RR_PACK", ci_pack);
}
