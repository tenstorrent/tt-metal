// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef REDUCE_POST_EXP
#include "api/compute/eltwise_unary/exp.h"
#endif

namespace {

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_auxiliary = REDUCE_AUXILIARY_CB;
constexpr uint32_t cb_accumulator = 2;
constexpr uint32_t cb_output = 16;
constexpr uint32_t num_calls = get_compile_time_arg_val(0);
constexpr uint32_t auxiliary_tiles = get_compile_time_arg_val(1);

static_assert(num_calls >= 1);

ALWI auto make_post_op() {
#ifdef REDUCE_POST_EXP
    // The unclamped approximate exp initializer overwrites the SFPU replay buffer on Wormhole.
    // Multiple outputs must still perform their SUM collapse correctly after this callback.
    return [](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, __builtin_bit_cast(uint32_t, -0.01f));
        exp_tile_init<true, 0x3F800000, InputClamping::None>();
        exp_tile<true, false, InputClamping::None>(dst);
    };
#else
    return compute_kernel_lib::NoOp{};
#endif
}

template <uint32_t call>
ALWI auto make_accumulation() {
    if constexpr (num_calls == 1) {
        return compute_kernel_lib::NoAccumulation{};
    } else if constexpr (call + 1 == num_calls) {
        return compute_kernel_lib::Accumulate::at_last(cb_accumulator, call).with_reload(REDUCE_RELOAD_MODE);
    } else {
        return compute_kernel_lib::Accumulate::at(cb_accumulator, call).with_reload(REDUCE_RELOAD_MODE);
    }
}

template <uint32_t call = 0>
ALWI void run_reduce_calls(
    compute_kernel_lib::ReduceInputBlockShape shape,
    compute_kernel_lib::ReduceInputMemoryLayout layout,
    uint32_t input_tiles) {
    if constexpr (call < num_calls) {
        constexpr uint32_t output_cb = call + 1 == num_calls ? cb_output : cb_accumulator;
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            cb_input,
            cb_auxiliary,
            output_cb,
            REDUCE_INPUT_POLICY,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            REDUCE_FP32_MODE,
            REDUCE_ALGORITHM,
            REDUCE_WITHIN_TILE,
            REDUCE_FACTOR>(
            shape,
            layout,
            make_accumulation<call>(),
            make_post_op(),
            REDUCE_PARTIAL_MODE,
            REDUCE_OUTPUT_GROUP,
            REDUCE_AUXILIARY_OFFSET);

        constexpr bool helper_pops_input =
            REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile ||
            REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop;
        if constexpr (!helper_pops_input) {
            cb_pop_front(cb_input, input_tiles);
        }
        run_reduce_calls<call + 1>(shape, layout, input_tiles);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t cols = get_arg_val<uint32_t>(1);
    const uint32_t batches = get_arg_val<uint32_t>(2);
    const uint32_t row_stride = get_arg_val<uint32_t>(3);
    const uint32_t batch_stride = get_arg_val<uint32_t>(4);

    const uint32_t input_tiles = batch_stride * batches;
    const auto shape = compute_kernel_lib::ReduceInputBlockShape::of(rows, cols, batches);
    const auto layout = compute_kernel_lib::ReduceInputMemoryLayout::with_strides(
        row_stride == cols ? 0 : row_stride, batch_stride == rows * row_stride ? 0 : batch_stride);

    constexpr uint32_t first_output_cb = num_calls == 1 ? cb_output : cb_accumulator;
    if constexpr (cb_auxiliary == compute_kernel_lib::REDUCE_NO_AUXILIARY_CB) {
        compute_kernel_hw_startup(cb_input, first_output_cb);
    } else {
        compute_kernel_hw_startup(cb_input, cb_auxiliary, first_output_cb);
    }

    cb_reserve_back(cb_input, input_tiles * num_calls);
    cb_push_back(cb_input, input_tiles * num_calls);

    run_reduce_calls(shape, layout, input_tiles);

    if constexpr (auxiliary_tiles > 0) {
        cb_pop_front(cb_auxiliary, auxiliary_tiles);
    }
}
