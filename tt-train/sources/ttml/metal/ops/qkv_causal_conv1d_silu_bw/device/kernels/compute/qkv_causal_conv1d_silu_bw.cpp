// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Stages are noinline: inlined into one big kernel body the compiler merges their LLK init sequences and the
// init immediates stop being compile-time constants ("impossible constraint in asm").
constexpr uint32_t tap_count = 4;
constexpr uint32_t dst_tiles = 4;
constexpr auto DEST_TO_SRCB = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
static_assert(DST_ACCUM_MODE, "the conv accumulates its 4 taps in fp32 DST");

template <uint32_t limit>
constexpr uint32_t batch_size(uint32_t base) {
    return limit - base < dst_tiles ? limit - base : dst_tiles;
}

// mul_tiles_bcast_rows that adds onto DST instead of clearing it first (fp32 DST mode accumulates by design).
FORCE_INLINE void mul_bcast_rows_accumulate(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool clear) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, clear)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, 0)));
}

// out = sum_tap weights[w] * view_tap, w = tap (preact) or 3 - tap (grad_input), summed in DST.
template <uint32_t block_ct>
[[gnu::noinline]] void conv(uint32_t views_cb, bool reversed, DataflowBuffer& out, uint32_t out_cb) {
    out.reserve_back(block_ct);
    reconfig_data_format(views_cb, dfb::weights);
    mul_bcast_rows_init(views_cb, dfb::weights);
    pack_reconfig_data_format(out_cb);
    for (uint32_t base = 0; base < block_ct; base += dst_tiles) {
        const uint32_t n = batch_size<block_ct>(base);
        tile_regs_acquire();
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            const uint32_t w = reversed ? tap_count - 1 - tap : tap;
            for (uint32_t i = 0; i < n; ++i) {
                mul_bcast_rows_accumulate(
                    views_cb, dfb::weights, tap * block_ct + base + i, w * block_ct + base + i, i, tap == 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
    }
    out.push_back(block_ct);
}

// dc = grad_out * s * (1 + (preact - preact * s)), s = sigmoid(preact). FPU only after the sigmoid.
template <uint32_t block_ct>
[[gnu::noinline]] void silu_backward(DataflowBuffer& sigmoid, DataflowBuffer& dc_tile) {
    dc_tile.reserve_back(block_ct);
    for (uint32_t base = 0; base < block_ct; base += dst_tiles) {
        const uint32_t n = batch_size<block_ct>(base);
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
        mul_reuse_dest_init<DEST_TO_SRCB>(dfb::grad_out);
        for (uint32_t i = 0; i < n; ++i) {
            mul_reuse_dest_tiles<DEST_TO_SRCB>(dfb::grad_out, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dfb::dc_tile);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::dc_tile);
        }
        tile_regs_release();
        sigmoid.pop_front(n);
    }
    dc_tile.push_back(block_ct);
}

// acc[tap][ct] += dc * view_tap in fp32; the last own tile writes the sum to grad_tap instead of acc.
template <uint32_t block_ct, bool first, bool last>
[[gnu::noinline]] void accumulate_grad_tap(DataflowBuffer& acc, DataflowBuffer& grad_tap) {
    constexpr uint32_t total = tap_count * block_ct;
    constexpr uint32_t dest_id = last ? dfb::grad_tap : dfb::acc;
    DataflowBuffer& dest = last ? grad_tap : acc;
    if constexpr (!first) {
        acc.wait_front(total);
    }
    dest.reserve_back(total);
    for (uint32_t base = 0; base < total; base += dst_tiles) {
        const uint32_t n = batch_size<total>(base);
        tile_regs_acquire();
        reconfig_data_format(dfb::dc_tile, dfb::act_tile);
        mul_init(dfb::dc_tile, dfb::act_tile);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(dfb::dc_tile, dfb::act_tile, (base + i) % block_ct, base + i, i);
        }
        if constexpr (!first) {
            reconfig_data_format_srca(dfb::acc);
            add_reuse_dest_init<DEST_TO_SRCB>(dfb::acc);
            for (uint32_t i = 0; i < n; ++i) {
                add_reuse_dest_tiles<DEST_TO_SRCB>(dfb::acc, base + i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(dest_id);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dest_id);
        }
        tile_regs_release();
    }
    dest.push_back(total);
    if constexpr (!first) {
        acc.pop_front(total);
    }
}

template <uint32_t block_ct>
[[gnu::noinline]] void grad_input_tile(DataflowBuffer& dc_views, DataflowBuffer& output) {
    compute_kernel_lib::tilize<block_ct, dfb::dcv_rm, dfb::dcv_tile>(tap_count);
    dc_views.wait_front(tap_count * block_ct);
    conv<block_ct>(dfb::dcv_tile, true, output, dfb::output);
    dc_views.pop_front(tap_count * block_ct);
}

template <uint32_t block_ct, uint32_t Mt, uint32_t splits>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::dc_rm);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer grad_out(dfb::grad_out);
    DataflowBuffer preact(dfb::preact);
    DataflowBuffer sigmoid(dfb::sigmoid);
    DataflowBuffer dc_tile(dfb::dc_tile);
    DataflowBuffer acc(dfb::acc);
    DataflowBuffer grad_tap(dfb::grad_tap);
    DataflowBuffer dc_views(dfb::dcv_tile);
    DataflowBuffer output(dfb::output);
    DataflowBuffer ones(dfb::ones);

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t split = work % splits;
        const uint32_t m0 = split * Mt / splits;
        const uint32_t m1 = (split + 1) * Mt / splits;
        const uint32_t own = m1 - m0;
        const bool has_halo = m1 < Mt;
        const uint32_t dc_tiles = own + (has_halo ? 1 : 0);

        weights.wait_front(tap_count * block_ct);
        if (item == 0) {
            ones.wait_front(1);
        }
        for (uint32_t k = 0; k < dc_tiles; ++k) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(tap_count);
            activation.wait_front(tap_count * block_ct);
            grad_out.wait_front(block_ct);

            conv<block_ct>(dfb::act_tile, false, preact, dfb::preact);
            preact.wait_front(block_ct);
            silu_backward<block_ct>(sigmoid, dc_tile);
            dc_tile.wait_front(block_ct);
            preact.pop_front(block_ct);
            if (k < own) {
                const bool first = k == 0;
                const bool last = k + 1 == own;
                if (first && last) {
                    accumulate_grad_tap<block_ct, true, true>(acc, grad_tap);
                } else if (first) {
                    accumulate_grad_tap<block_ct, true, false>(acc, grad_tap);
                } else if (last) {
                    accumulate_grad_tap<block_ct, false, true>(acc, grad_tap);
                } else {
                    accumulate_grad_tap<block_ct, false, false>(acc, grad_tap);
                }
            }
            compute_kernel_lib::untilize<block_ct, dfb::dc_tile, dfb::dc_rm>(1);
            activation.pop_front(tap_count * block_ct);
            grad_out.pop_front(block_ct);
            DPRINT_PACK("compute: work {} mt {} dc pushed\n", work, m0 + k);

            if (k > 0) {
                grad_input_tile<block_ct>(dc_views, output);
            }
        }
        if (!has_halo) {
            grad_input_tile<block_ct>(dc_views, output);
        }
        weights.pop_front(tap_count * block_ct);
        DPRINT_PACK("compute: work {} done\n", work);
    }
}
