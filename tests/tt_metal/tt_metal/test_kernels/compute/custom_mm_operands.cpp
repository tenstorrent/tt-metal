// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/2_0/compressed_custom_mm.h"
#include "api/compute/experimental/2_0/custom_mm.h"
#include "api/compute/experimental/2_0/custom_mm_reuse_dest_srcb.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

void kernel_main() {
    constexpr std::uint32_t blocks = get_compile_time_arg_val(0);
    constexpr std::uint32_t rows = get_compile_time_arg_val(1);
    constexpr bool mixed = get_compile_time_arg_val(2);
    constexpr std::uint32_t mode = get_compile_time_arg_val(3);  // plain, compressed, DEST reuse
    constexpr std::uint32_t input_block_tiles = mode == 2 ? 5 : 3;
    constexpr std::uint32_t producer_ct = mode == 2 ? 2 : 1;
    constexpr auto weight_format = mode == 1 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
    using Activation = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{rows, 16, 1, 2}>;
    using Weight = experimental::LLKOperand<weight_format, TensorShape{16, 16, 2, 2}>;
    CircularBuffer activation(tt::CBIndex::c_0);
    CircularBuffer weight(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    // Full init must run once, before tile-register traffic. The input addresses
    // are unused until execution; only their static format/geometry matter here.
    if constexpr (mixed) {
        custom_mm_block_init<false, false, true>(tt::CBIndex::c_0, Weight(0), tt::CBIndex::c_16, producer_ct);
    } else {
        custom_mm_block_init<false, false, true>(Activation(0), Weight(0), tt::CBIndex::c_16, producer_ct);
    }
    pack_block_contiguous_init(tt::CBIndex::c_16);

    // Two BFP8 tiles, both fetching a new activation tile. This is the existing
    // compressed stream's (previous format, use-B, current format) encoding.
    alignas(16) const std::uint32_t metadata[4] = {0xfc, 0, 0, 0};
    const auto metadata_address = reinterpret_cast<std::uint32_t>(metadata);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        activation.wait_front(input_block_tiles);
        weight.wait_front(input_block_tiles);
        output.reserve_back(1);
        tile_regs_acquire();

        // Poisoned input tile zero must be skipped. Plain matmul exercises the
        // tile-index arguments; compressed matmul receives advanced base pointers.
        const auto a_base = experimental::cb_read_address(tt::CBIndex::c_0);
        const auto b_base = experimental::cb_read_address(tt::CBIndex::c_1);
        fill_tile_init();
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);  // Separate, initially zero accumulator for the second matmul.

        if constexpr (mode == 1) {
            const Weight b(experimental::cb_read_address(tt::CBIndex::c_1, 1));
            if constexpr (mixed) {
                activation.pop_front(1);
                compressed_custom_mm_block_init_short<false, false, true>(tt::CBIndex::c_0, b);
                compressed_custom_mm_block<false>(tt::CBIndex::c_0, b, metadata_address, 0, 2);
            } else {
                const Activation a(
                    a_base + experimental::tile_stride_words(DataFormat::Float16_b, TensorShape{rows, 16, 1, 2}));
                compressed_custom_mm_block_init_short<false, false, true>(a, b);
                compressed_custom_mm_block<false>(a, b, metadata_address, 0, 2);
            }
        } else {
            if constexpr (mixed) {
                custom_mm_block_init_short<false, false, true>(tt::CBIndex::c_0, Weight(b_base), producer_ct);
                custom_mm_block<false>(tt::CBIndex::c_0, Weight(b_base), 1, 1, 0, 2, producer_ct);
            } else {
                custom_mm_block_init_short<false, false, true>(Activation(a_base), Weight(b_base), producer_ct);
                custom_mm_block<false>(Activation(a_base), Weight(b_base), 1, 1, 0, 2, producer_ct);
            }
            if constexpr (mode == 2) {
                // The reuse unpacker processes two K tiles per MOP iteration.
                // Consume both producer tiles at DEST rows 0 and 32, weighted
                // by 2*I and I (input indices 2 and 4), into a disjoint accumulator.
                if (block % 2 == 0) {
                    custom_mm_reuse_dest_srcb_block_init_short(Weight(b_base), 1);
                } else {
                    custom_mm_reuse_dest_srcb_replay_init();
                    custom_mm_reuse_dest_srcb_block_init_short<false>(Weight(b_base), 1);
                }
                custom_mm_reuse_dest_srcb_block<rows>(Weight(b_base), 2, 0, 64, 2, 1, 2);
                custom_mm_reuse_dest_srcb_pack_init();
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(mode == 2 ? 4 : 0, tt::CBIndex::c_16, 1);
        tile_regs_release();
        output.push_back(1);
        activation.pop_front(mode == 1 && mixed ? input_block_tiles - 1 : input_block_tiles);
        weight.pop_front(input_block_tiles);
        if constexpr (mode == 2) {
            custom_mm_reuse_dest_srcb_pack_uninit();
        }
    }
    custom_mm_block_uninit<true>();
}
