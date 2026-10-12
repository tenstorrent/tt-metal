// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t tap_count = 4;
constexpr auto DEST_TO_SRCB = EltwiseBinaryReuseDestType::DEST_TO_SRCB;

// grad_input = sum_s weights[s] * view_s, where view_s holds dc rows shifted by +s and weights[s] = tap[3 - s].
template <uint32_t block_ct, uint32_t Mt>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        if (item == 0 || (work - 1) / Mt != block) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            activation.wait_front(block_ct);
            const bool is_final_tap = tap + 1 == tap_count;
            const uint32_t dest_id = is_final_tap ? dfb::output : dfb::partial;
            DataflowBuffer& dest = is_final_tap ? output : partial;
            if (tap != 0) {
                partial.wait_front(block_ct);
            }
            dest.reserve_back(block_ct);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                tile_regs_acquire();
                reconfig_data_format(dfb::act_tile, dfb::weights);
                mul_bcast_rows_init(dfb::act_tile, dfb::weights);
                mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, ct, tap * block_ct + ct, 0);
                if (tap != 0) {
                    add_reuse_dest_init<DEST_TO_SRCB>(dfb::partial);
                    add_reuse_dest_tiles<DEST_TO_SRCB>(dfb::partial, ct, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dest_id);
                pack_tile(0, dest_id);
                tile_regs_release();
            }
            dest.push_back(block_ct);
            if (tap != 0) {
                partial.pop_front(block_ct);
            }
            activation.pop_front(block_ct);
        }
        DPRINT_PACK("compute: work {} (block {} mt {}) output pushed\n", work, block, work % Mt);
        if (item + 1 == wi_count || (work + 1) / Mt != block) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
