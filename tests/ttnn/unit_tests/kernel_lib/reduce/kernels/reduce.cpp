// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_accumulator = 2;
constexpr uint32_t cb_output = 16;
constexpr uint32_t num_calls = get_compile_time_arg_val(0);

static_assert(num_calls >= 1);

constexpr DataFormat input_format = static_cast<DataFormat>(unpack_src_format[cb_input]);
constexpr bool uses_sfpu = is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, input_format, REDUCE_FP32_MODE>();
constexpr uint32_t expected_col_chunk =
    uses_sfpu ? (compute_kernel_lib::DEST_AUTO_LIMIT - 1) : compute_kernel_lib::DEST_AUTO_LIMIT;
static_assert(REDUCE_DIM != ckernel::ReduceDim::REDUCE_COL || REDUCE_EXPECTED_COL_CHUNK == expected_col_chunk);

#ifdef REDUCE_POST_MULTIPLIER_BITS
struct PostReduceMultiply {
    ALWI void operator()(uint32_t dst_idx) const {
        compute_kernel_lib::detail::reduce_post_mul_tile<input_format>(dst_idx, REDUCE_POST_MULTIPLIER_BITS);
    }
};
#else
using PostReduceMultiply = compute_kernel_lib::NoOp;
#endif

template <bool enable_accumulation>
ALWI auto make_accumulation(uint32_t iteration) {
    if constexpr (enable_accumulation) {
        return compute_kernel_lib::Accumulate::at(cb_accumulator, iteration);
    } else {
        return compute_kernel_lib::NoAccumulation{};
    }
}

template <uint32_t output_cb>
ALWI void run_reduce_call(
    compute_kernel_lib::ReduceInputBlockShape shape,
    compute_kernel_lib::ReduceInputMemoryLayout layout,
    uint32_t input_tiles,
    uint32_t iteration) {
    const auto accumulation = make_accumulation<(num_calls > 1)>(iteration);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        cb_input,
        cb_scaler,
        output_cb,
        REDUCE_INPUT_POLICY,
        REDUCE_RECONFIG_MODE,
        REDUCE_FP32_MODE>(shape, layout, accumulation, PostReduceMultiply{});

    constexpr bool helper_pops_input =
        REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile ||
        REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop;
    if constexpr (!helper_pops_input) {
        // The no-pop policies leave ownership with their caller. Advance to the
        // next call's block only after reduce() has finished using this one.
        cb_pop_front(cb_input, input_tiles);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t cols = get_arg_val<uint32_t>(1);
    const uint32_t batches = get_arg_val<uint32_t>(2);
    const uint32_t row_stride = get_arg_val<uint32_t>(3);

    const uint32_t input_tiles = rows * row_stride * batches;
    const auto shape = compute_kernel_lib::ReduceInputBlockShape::of(rows, cols, batches);
    const auto layout = row_stride == cols ? compute_kernel_lib::ReduceInputMemoryLayout::contiguous()
                                           : compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(row_stride);

    constexpr uint32_t first_output_cb = num_calls == 1 ? cb_output : cb_accumulator;
    compute_kernel_hw_startup(cb_input, cb_scaler, first_output_cb);

    // One sharded tensor backs a linear sequence of per-call blocks. Make the
    // complete stream visible once; ownership then follows the selected policy.
    cb_reserve_back(cb_input, input_tiles * num_calls);
    cb_push_back(cb_input, input_tiles * num_calls);

    for (uint32_t call = 0; call < num_calls - 1; ++call) {
        run_reduce_call<cb_accumulator>(shape, layout, input_tiles, call);
    }
    run_reduce_call<cb_output>(shape, layout, input_tiles, num_calls - 1);

    // reduce() deliberately keeps the scaler resident for reuse by every call.
    cb_pop_front(cb_scaler, 1);
}
