// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/hostdevcommon/qkv_causal_conv1d_silu_bw_layout.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

using qkv_conv_bw::tap_count;
constexpr uint32_t dst_tiles = qkv_conv_bw::dest_tiles;
static_assert(dst_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);
constexpr auto DEST_TO_SRCB = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
static_assert(DST_ACCUM_MODE, "the conv accumulates its 4 taps in fp32 DST");

template <uint32_t limit>
constexpr uint32_t batch_size(uint32_t base) {
    return limit - base < dst_tiles ? limit - base : dst_tiles;
}

// out = sum_tap weights[w] * copy_tap, w = tap (preact) or 3 - tap (grad_input). ELWMUL accumulates into DST, and tap 0
// lands on the half the previous tile_regs_release left zeroed. This relies on the LLK dropping the clear flag that
// mul_tiles_bcast_rows passes on the no-dest-reuse path; if that path ever honours it, the taps stop summing.
template <uint32_t channel_tiles_per_block>
void conv(uint32_t shifted_cb, bool reversed, DataflowBuffer& out, uint32_t out_cb) {
    out.reserve_back(channel_tiles_per_block);
    reconfig_data_format(shifted_cb, dfb::weights);
    mul_bcast_rows_init(shifted_cb, dfb::weights);
    pack_reconfig_data_format(out_cb);
    for (uint32_t base = 0; base < channel_tiles_per_block; base += dst_tiles) {
        const uint32_t n = batch_size<channel_tiles_per_block>(base);
        tile_regs_acquire();
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            const uint32_t w = reversed ? tap_count - 1 - tap : tap;
            for (uint32_t i = 0; i < n; ++i) {
                mul_tiles_bcast_rows(
                    shifted_cb,
                    dfb::weights,
                    tap * channel_tiles_per_block + base + i,
                    w * channel_tiles_per_block + base + i,
                    i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
    }
    out.push_back(channel_tiles_per_block);
}

// grad_preact = grad_out * s * (1 + (preact - preact * s)), s = sigmoid(preact). FPU only after the sigmoid. The
// tiles go to the writer, which builds the shifted copies for grad_input out of them, and with tap gradients a second
// copy stays here for the partial.
template <uint32_t channel_tiles_per_block>
void silu_backward(DataflowBuffer& sigmoid, DataflowBuffer& grad_preact_out, DataflowBuffer* grad_preact_tile) {
    grad_preact_out.reserve_back(channel_tiles_per_block);
#if COMPUTE_TAP_GRADS
    grad_preact_tile->reserve_back(channel_tiles_per_block);
#endif
    for (uint32_t base = 0; base < channel_tiles_per_block; base += dst_tiles) {
        const uint32_t n = batch_size<channel_tiles_per_block>(base);
        sigmoid.reserve_back(n);
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::preact);
        copy_init(dfb::preact);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(dfb::preact, base + i, i);
        }
        sigmoid_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            sigmoid_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dfb::sigmoid);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::sigmoid);
        }
        tile_regs_release();
        sigmoid.push_back(n);
        sigmoid.wait_front(n);

        tile_regs_acquire();
        reconfig_data_format(dfb::preact, dfb::sigmoid);
        mul_init(dfb::preact, dfb::sigmoid);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(dfb::preact, dfb::sigmoid, base + i, i, i);
        }
        sub_reuse_dest_init<DEST_TO_SRCB>(dfb::preact);
        for (uint32_t i = 0; i < n; ++i) {
            sub_reuse_dest_tiles<DEST_TO_SRCB>(dfb::preact, base + i, i);
        }
        reconfig_data_format_srca(dfb::ones);
        add_reuse_dest_init<DEST_TO_SRCB>(dfb::ones);
        for (uint32_t i = 0; i < n; ++i) {
            add_reuse_dest_tiles<DEST_TO_SRCB>(dfb::ones, 0, i);
        }
        reconfig_data_format_srca(dfb::sigmoid);
        mul_reuse_dest_init<DEST_TO_SRCB>(dfb::sigmoid);
        for (uint32_t i = 0; i < n; ++i) {
            mul_reuse_dest_tiles<DEST_TO_SRCB>(dfb::sigmoid, i, i);
        }
        reconfig_data_format_srca(dfb::grad_out);  // back from the fp32 sigmoid to bf16
        mul_reuse_dest_init<DEST_TO_SRCB>(dfb::grad_out);
        for (uint32_t i = 0; i < n; ++i) {
            mul_reuse_dest_tiles<DEST_TO_SRCB>(dfb::grad_out, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dfb::grad_preact_out);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::grad_preact_out);
#if COMPUTE_TAP_GRADS
            pack_tile(i, dfb::grad_preact_tile);
#endif
        }
        tile_regs_release();
        sigmoid.pop_front(n);
    }
    grad_preact_out.push_back(channel_tiles_per_block);
#if COMPUTE_TAP_GRADS
    grad_preact_tile->push_back(channel_tiles_per_block);
#endif
}

#if COMPUTE_TAP_GRADS
// partial[tap][tile_column] += grad_preact * copy_tap. The partial lives in L1 and the packer adds each step's
// products onto them in place (pack_reconfig_l1_acc), in fp32: nothing is unpacked back into the engine, so no
// tf32 rounding. The pages are reserved once per item, written out of order by tile index, and pushed to the
// writer on the last own step.
template <uint32_t channel_tiles_per_block>
void accumulate_partial(DataflowBuffer& partial, bool first, bool last) {
    constexpr uint32_t total = tap_count * channel_tiles_per_block;
    if (first) {
        partial.reserve_back(total);
    }
    reconfig_data_format(dfb::grad_preact_tile, dfb::x_shifted_tile);
    mul_init(dfb::grad_preact_tile, dfb::x_shifted_tile);
    for (uint32_t base = 0; base < total; base += dst_tiles) {
        const uint32_t n = batch_size<total>(base);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(dfb::grad_preact_tile, dfb::x_shifted_tile, (base + i) % channel_tiles_per_block, base + i, i);
        }
        tile_regs_commit();
        pack_l1_acc_block(dfb::partial, first, n, base);
    }
    if (last) {
        partial.push_back(total);
    }
}
#endif

// grad_input = sum_tap weights[3 - tap] * grad_preact_shifted[tap], the shifted copies built by the writer.
template <uint32_t channel_tiles_per_block>
void grad_input_tile(DataflowBuffer& grad_preact_shifted, DataflowBuffer& output) {
    grad_preact_shifted.wait_front(tap_count * channel_tiles_per_block);
    conv<channel_tiles_per_block>(dfb::grad_preact_shifted_tile, true, output, dfb::output);
    grad_preact_shifted.pop_front(tap_count * channel_tiles_per_block);
}

template <uint32_t channel_tiles_per_block, uint32_t tile_rows, uint32_t row_splits>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::x_shifted_tile, dfb::weights, dfb::preact);
    DataflowBuffer x_shifted(dfb::x_shifted_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer grad_out(dfb::grad_out);
    DataflowBuffer preact(dfb::preact);
    DataflowBuffer sigmoid(dfb::sigmoid);
    DataflowBuffer grad_preact_out(dfb::grad_preact_out);
#if COMPUTE_TAP_GRADS
    DataflowBuffer grad_preact_tile(dfb::grad_preact_tile);
    DataflowBuffer* grad_preact_tile_ptr = &grad_preact_tile;
    DataflowBuffer partial(dfb::partial);
#else
    DataflowBuffer* grad_preact_tile_ptr = nullptr;
#endif
    DataflowBuffer grad_preact_shifted(dfb::grad_preact_shifted_tile);
    DataflowBuffer output(dfb::output);
    DataflowBuffer ones(dfb::ones);

    // The reader pushes the taps once per channel block (see the reader); they are popped when the block changes.
    constexpr uint32_t no_block = ~0U;
    uint32_t current_block = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t channel_block = work / row_splits;
        const uint32_t split = work % row_splits;
        const uint32_t first_tile_row = split * tile_rows / row_splits;
        const uint32_t end_tile_row = (split + 1) * tile_rows / row_splits;
        const uint32_t own_steps = end_tile_row - first_tile_row;
        const bool has_halo = end_tile_row < tile_rows;
        const uint32_t step_count = own_steps + (has_halo ? 1 : 0);

        if (channel_block != current_block) {
            if (current_block != no_block) {
                weights.pop_front(tap_count * channel_tiles_per_block);
            }
            weights.wait_front(tap_count * channel_tiles_per_block);
            current_block = channel_block;
        }
        if (item == 0) {
            ones.wait_front(1);
        }
        for (uint32_t step = 0; step < step_count; ++step) {
            x_shifted.wait_front(tap_count * channel_tiles_per_block);
            grad_out.wait_front(channel_tiles_per_block);

            conv<channel_tiles_per_block>(dfb::x_shifted_tile, false, preact, dfb::preact);
            preact.wait_front(channel_tiles_per_block);
            silu_backward<channel_tiles_per_block>(sigmoid, grad_preact_out, grad_preact_tile_ptr);
            preact.pop_front(channel_tiles_per_block);
#if COMPUTE_TAP_GRADS
            grad_preact_tile.wait_front(channel_tiles_per_block);
            if (step < own_steps) {  // the extra step's rows belong to the next core's partial
                const bool first = step == 0, last = step + 1 == own_steps;
                accumulate_partial<channel_tiles_per_block>(partial, first, last);
            }
            grad_preact_tile.pop_front(channel_tiles_per_block);
#endif
            x_shifted.pop_front(tap_count * channel_tiles_per_block);
            grad_out.pop_front(channel_tiles_per_block);

            // grad_input of tile row r needs the shifted grad_preact of r, which the writer builds once grad_preact of
            // row r + 1 exists, i.e. during step r + 1. Running it at step r + 2 gives the writer a whole step to build
            // it, instead of compute idling on the 4 shifted tiles at the end of every step.
            if (step > 1) {
                grad_input_tile<channel_tiles_per_block>(grad_preact_shifted, output);  // row first_tile_row + step - 2
            }
        }
        if (step_count > 1) {
            grad_input_tile<channel_tiles_per_block>(
                grad_preact_shifted, output);  // row first_tile_row + step_count - 2
        }
        if (!has_halo) {
            grad_input_tile<channel_tiles_per_block>(grad_preact_shifted, output);  // row end_tile_row - 1, zero tail
        }
    }
    if (current_block != no_block) {
        weights.pop_front(tap_count * channel_tiles_per_block);
    }
}
