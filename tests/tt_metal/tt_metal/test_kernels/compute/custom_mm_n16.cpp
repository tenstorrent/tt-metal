// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// custom_mm with [32, 16] weight tiles ([16, 32] when transposed), giving ct_dim [rows, 16] output tiles.
// in0 holds kt_dim * ct_dim tiles because the reader feeds both inputs the same count; only the first kt_dim are used.
// api selects the overloads: 0 = CB ids, 1 = LLKOperand in0 and in1, 2 = CB in0 with LLKOperand in1 and output.
void kernel_main() {
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(0);
    constexpr std::uint32_t ct_dim = get_compile_time_arg_val(1);
    constexpr bool transpose = get_compile_time_arg_val(2);
    constexpr bool split_acc = get_compile_time_arg_val(3);
    constexpr std::uint32_t rows = get_compile_time_arg_val(4);
    constexpr std::uint32_t api = get_compile_time_arg_val(5);
    constexpr auto layout = split_acc ? CustomMmDestLayout::DefaultSplitAcc : CustomMmDestLayout::Default;
    constexpr std::uint32_t input_tiles = kt_dim * ct_dim;
    using Activation = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{rows, 16, 1, 2}>;
    using Weight = experimental::
        LLKOperand<DataFormat::Float16_b, transpose ? TensorShape{16, 16, 1, 2} : TensorShape{16, 16, 2, 1}>;
    using Output = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{rows, 16, 1, 1}>;

    if constexpr (api == 0) {
        custom_mm_block_init<transpose, layout>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, ct_dim);
    } else if constexpr (api == 1) {
        custom_mm_block_init<transpose, layout>(Activation(0), Weight(0), tt::CBIndex::c_16, ct_dim);
    } else {
        custom_mm_block_init<transpose, layout>(tt::CBIndex::c_0, Weight(0), tt::CBIndex::c_16, ct_dim);
    }

    cb_wait_front(tt::CBIndex::c_0, input_tiles);
    cb_wait_front(tt::CBIndex::c_1, input_tiles);
    cb_reserve_back(tt::CBIndex::c_16, ct_dim);

    tile_regs_acquire();
    if constexpr (api == 0) {
        custom_mm_block<layout>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, kt_dim, ct_dim);
    } else if constexpr (api == 1) {
        const Activation a(experimental::cb_read_address(tt::CBIndex::c_0));
        const Weight b(experimental::cb_read_address(tt::CBIndex::c_1));
        custom_mm_block<layout>(a, b, 0, 0, 0, kt_dim, ct_dim);
    } else {
        const Weight b(experimental::cb_read_address(tt::CBIndex::c_1));
        custom_mm_block_init_short<transpose, layout>(tt::CBIndex::c_0, b, Output(0), ct_dim);
        custom_mm_block<layout>(tt::CBIndex::c_0, b, 0, 0, 0, kt_dim, ct_dim);
    }
    tile_regs_commit();

    tile_regs_wait();
    // The Default layouts keep output tiles 64 DEST rows apart, the packer's own tile stride.
    for (std::uint32_t i = 0; i < ct_dim; ++i) {
        pack_tile(i, tt::CBIndex::c_16);
    }
    tile_regs_release();

    cb_push_back(tt::CBIndex::c_16, ct_dim);
    cb_pop_front(tt::CBIndex::c_0, input_tiles);
    cb_pop_front(tt::CBIndex::c_1, input_tiles);

    if constexpr (api == 2) {
        custom_mm_block_uninit<layout>(Output(0));
    } else {
        custom_mm_block_uninit<layout>(tt::CBIndex::c_16);
    }
}
