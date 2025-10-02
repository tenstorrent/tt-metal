// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t r_tiles = get_arg_val<uint32_t>(1);

    DPRINT << "COMPUTE KERNEL START: n_tiles=" << n_tiles << ", r_tiles=" << r_tiles << ENDL();

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_interm = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t acc_reg = 0;
    constexpr uint32_t b_reg = 1;

    init_sfpu(cb_in0, cb_out0);
    add_binary_tile_init();

    for (uint32_t i = 0; i < n_tiles; i++) {
        if (i >= 14) {
            DPRINT << "COMPUTE: Processing tile i=" << i << " (critical tile)" << ENDL();
        }

        cb_wait_front(cb_in0, 1);

        // Enhanced debug logging for critical tiles
        if (i == 0 || i >= 14) {
            for (int32_t k = 0; k < 2; ++k) {
                SliceRange single_element_slice = {
                    .h0 = 0, .h1 = 1, .hs = 1,
                    .w0 = (uint8_t)k, .w1 = (uint8_t)(k + 1), .ws = 1
                };
                DPRINT << "COMPUTE in0 tile[" << i << "](" << k << "): "
                       << TSLICE(cb_in0, 0, single_element_slice) << ENDL();
            }
        }

        tile_regs_acquire();
        copy_tile(cb_in0, 0, acc_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_interm, 1);
        pack_tile(acc_reg, cb_interm);
        cb_push_back(cb_interm, 1);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);

        // Improved B tile processing with adaptive batching
        constexpr uint32_t process_batch_size = 4;

        for (uint32_t j = 0; j < r_tiles; j++) {
            if (i >= 14 && j < 2) {
                DPRINT << "COMPUTE: Processing j=" << j << " for critical tile i=" << i << ENDL();
            }

            // Wait for B tile and intermediate result
            cb_wait_front(cb_in1, 1);
            cb_wait_front(cb_interm, 1);

            tile_regs_acquire();

            copy_tile(cb_interm, 0, acc_reg);
            copy_tile(cb_in1, 0, b_reg);

            // Enhanced debug logging for critical operations
            if (i < 2 || (i >= 14 && j < 2)) {
                for (int32_t k = 0; k < 2; ++k) {
                    SliceRange single_element_slice = {
                        .h0 = 0, .h1 = 1, .hs = 1,
                        .w0 = (uint8_t)k, .w1 = (uint8_t)(k + 1), .ws = 1
                    };
                    DPRINT << "COMPUTE interm tile[" << i << "][" << j << "](" << k << "): "
                           << TSLICE(cb_interm, 0, single_element_slice) << ENDL();
                    DPRINT << "COMPUTE in1 tile[" << i << "][" << j << "](" << k << "): "
                           << TSLICE(cb_in1, 0, single_element_slice) << ENDL();
                }
            }

            add_binary_tile(acc_reg, b_reg, acc_reg);

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_interm, 1);
            pack_tile(acc_reg, cb_interm);
            cb_push_back(cb_interm, 1);
            tile_regs_release();

            // Immediate cleanup to free CB space
            cb_pop_front(cb_in1, 1);
            cb_pop_front(cb_interm, 1);

            // Enhanced synchronization for better producer-consumer balance
            if (r_tiles >= 8 && (j + 1) % process_batch_size == 0 && j < r_tiles - 1) {
                // Brief processing break for balanced flow
                tile_regs_acquire();
                tile_regs_commit();
                tile_regs_wait();
                tile_regs_release();
            }
        }

        cb_wait_front(cb_interm, 1);

        tile_regs_acquire();
        copy_tile(cb_interm, 0, acc_reg);
        tile_regs_commit();
        tile_regs_wait();

        if (i >= 14) {
            DPRINT << "COMPUTE: Final result for tile i=" << i << " before output" << ENDL();
        }

        cb_reserve_back(cb_out0, 1);
        pack_tile(acc_reg, cb_out0);
        cb_push_back(cb_out0, 1);
        tile_regs_release();

        cb_pop_front(cb_interm, 1);

        if (i >= 14) {
            DPRINT << "COMPUTE: Completed tile i=" << i << ENDL();
        }
    }

    DPRINT << "COMPUTE KERNEL COMPLETE" << ENDL();
}

}
