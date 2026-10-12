// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t tap_count = 4;
constexpr uint32_t fp32_one = 0x3F800000;
constexpr auto DEST_TO_SRCB = EltwiseBinaryReuseDestType::DEST_TO_SRCB;

// preact = sum_tap tap * view_tap, one tap pass over the block at a time, accumulated through partial.
template <uint32_t block_ct>
FORCE_INLINE void conv_preact(DataflowBuffer& partial, DataflowBuffer& preact) {
    for (uint32_t tap = 0; tap < tap_count; ++tap) {
        const bool is_final_tap = tap + 1 == tap_count;
        const uint32_t dest_id = is_final_tap ? dfb::preact : dfb::partial;
        DataflowBuffer& dest = is_final_tap ? preact : partial;
        if (tap != 0) {
            partial.wait_front(block_ct);
        }
        dest.reserve_back(block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            tile_regs_acquire();
            reconfig_data_format(dfb::act_tile, dfb::weights);
            mul_bcast_rows_init(dfb::act_tile, dfb::weights);
            mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, tap * block_ct + ct, tap * block_ct + ct, 0);
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
    }
}

// dc = grad_out * s * (1 + preact * (1 - s)), s = sigmoid(preact).
template <uint32_t block_ct>
FORCE_INLINE void silu_backward(DataflowBuffer& sigmoid, DataflowBuffer& dc_tile) {
    dc_tile.reserve_back(block_ct);
    for (uint32_t ct = 0; ct < block_ct; ++ct) {
        sigmoid.reserve_back(1);
        tile_regs_acquire();
        copy_init(dfb::preact);
        copy_tile(dfb::preact, ct, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dfb::sigmoid);
        pack_tile(0, dfb::sigmoid);
        tile_regs_release();
        sigmoid.push_back(1);
        sigmoid.wait_front(1);

        tile_regs_acquire();
        mul_init(dfb::preact, dfb::sigmoid);
        mul_tiles(dfb::preact, dfb::sigmoid, ct, 0, 0);
        negative_tile_init();
        negative_tile(0);
        add_reuse_dest_init<DEST_TO_SRCB>(dfb::preact);
        add_reuse_dest_tiles<DEST_TO_SRCB>(dfb::preact, ct, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, fp32_one);
        mul_reuse_dest_init<DEST_TO_SRCB>(dfb::sigmoid);
        mul_reuse_dest_tiles<DEST_TO_SRCB>(dfb::sigmoid, 0, 0);
        mul_reuse_dest_init<DEST_TO_SRCB>(dfb::grad_out);
        mul_reuse_dest_tiles<DEST_TO_SRCB>(dfb::grad_out, ct, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dfb::dc_tile);
        pack_tile(0, dfb::dc_tile);
        tile_regs_release();
        sigmoid.pop_front(1);
    }
    dc_tile.push_back(block_ct);
}

// acc[tap][ct] += dc * view_tap; the last sequence tile writes the sum to grad_tap instead of acc.
template <uint32_t block_ct>
FORCE_INLINE void accumulate_grad_tap(DataflowBuffer& acc, DataflowBuffer& grad_tap, bool first, bool last) {
    const uint32_t dest_id = last ? dfb::grad_tap : dfb::acc;
    DataflowBuffer& dest = last ? grad_tap : acc;
    if (!first) {
        acc.wait_front(tap_count * block_ct);
    }
    dest.reserve_back(tap_count * block_ct);
    for (uint32_t i = 0; i < tap_count * block_ct; ++i) {
        tile_regs_acquire();
        reconfig_data_format(dfb::dc_tile, dfb::act_tile);
        mul_init(dfb::dc_tile, dfb::act_tile);
        mul_tiles(dfb::dc_tile, dfb::act_tile, i % block_ct, i, 0);
        if (!first) {
            reconfig_data_format_srca(dfb::acc);
            add_reuse_dest_init<DEST_TO_SRCB>(dfb::acc);
            add_reuse_dest_tiles<DEST_TO_SRCB>(dfb::acc, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dest_id);
        pack_tile(0, dest_id);
        tile_regs_release();
    }
    dest.push_back(tap_count * block_ct);
    if (!first) {
        acc.pop_front(tap_count * block_ct);
    }
}

template <uint32_t block_ct, uint32_t Mt>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::dc_rm);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer grad_out(dfb::grad_out);
    DataflowBuffer preact(dfb::preact);
    DataflowBuffer sigmoid(dfb::sigmoid);
    DataflowBuffer dc_tile(dfb::dc_tile);
    DataflowBuffer acc(dfb::acc);
    DataflowBuffer grad_tap(dfb::grad_tap);

    for (uint32_t item = 0; item < wi_count; ++item) {
        weights.wait_front(tap_count * block_ct);
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t tap = 0; tap < tap_count; ++tap) {
                compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            }
            activation.wait_front(tap_count * block_ct);
            grad_out.wait_front(block_ct);
            DPRINT_PACK("compute: block {} mt {} inputs ready\n", wi_start + item, mt);

            conv_preact<block_ct>(partial, preact);
            preact.wait_front(block_ct);
            silu_backward<block_ct>(sigmoid, dc_tile);
            dc_tile.wait_front(block_ct);
            preact.pop_front(block_ct);
            DPRINT_PACK("compute: block {} mt {} dc ready\n", wi_start + item, mt);

            accumulate_grad_tap<block_ct>(acc, grad_tap, mt == 0, mt + 1 == Mt);
            compute_kernel_lib::untilize<block_ct, dfb::dc_tile, dfb::dc_rm>(1);
            activation.pop_front(tap_count * block_ct);
            grad_out.pop_front(block_ct);
            DPRINT_PACK("compute: block {} mt {} dc_rm pushed, grad_tap acc updated\n", wi_start + item, mt);
        }
        weights.pop_front(tap_count * block_ct);
    }
}
