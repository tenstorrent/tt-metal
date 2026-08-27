// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#ifdef REDUCE_HELPERS_PROFILE
#include "tools/profiler/kernel_profiler.hpp"
#endif

namespace {

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_accumulator = 2;
constexpr uint32_t cb_output = 16;
constexpr uint32_t num_calls = get_compile_time_arg_val(0);
constexpr uint32_t rows = get_compile_time_arg_val(1);
constexpr uint32_t cols = get_compile_time_arg_val(2);
constexpr uint32_t batches = get_compile_time_arg_val(3);
constexpr uint32_t row_stride = get_compile_time_arg_val(4);
constexpr uint32_t valid_elements = get_compile_time_arg_val(5);
constexpr uint32_t later_valid_elements = get_compile_time_arg_val(6);

constexpr uint32_t input_tiles = rows * row_stride * batches;
constexpr auto shape = compute_kernel_lib::ReduceInputBlockShape::of(rows, cols, batches);
constexpr auto layout = row_stride == cols ? compute_kernel_lib::ReduceInputMemoryLayout::contiguous()
                                           : compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(row_stride);

static_assert(num_calls >= 1);
static_assert(rows >= 1);
static_assert(cols >= 1);
static_assert(batches >= 1);
static_assert(row_stride >= cols);
static_assert(valid_elements <= 32);
static_assert(later_valid_elements <= 32);

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
#endif

template <bool enable_accumulation>
ALWI auto make_accumulation(uint32_t iteration) {
    if constexpr (enable_accumulation) {
        return compute_kernel_lib::Accumulate::at(cb_accumulator, iteration);
    } else {
        return compute_kernel_lib::NoAccumulation{};
    }
}

template <uint32_t output_cb, uint32_t call_valid_elements>
ALWI void run_reduce_call(uint32_t iteration) {
    constexpr uint32_t call_partial_elements =
        call_valid_elements == tt::constants::TILE_WIDTH ? 0 : call_valid_elements;
    constexpr uint32_t scaler_tile_r_dim = compute_kernel_lib::get_tile_r_dim<cb_scaler>();
    constexpr uint32_t scaler_tile_c_dim = compute_kernel_lib::get_tile_c_dim<cb_scaler>();
    constexpr uint32_t reduce_axis_tiles = REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW   ? cols
                                           : REDUCE_DIM == ckernel::ReduceDim::REDUCE_COL ? rows
                                                                                          : rows * cols;
    constexpr uint32_t full_reduce_dim =
        REDUCE_DIM == ckernel::ReduceDim::REDUCE_COL ? scaler_tile_r_dim : scaler_tile_c_dim;
    constexpr uint32_t call_reduce_factor =
        REDUCE_OP != ckernel::PoolType::AVG
            ? 1
            : (REDUCE_DIM == ckernel::ReduceDim::REDUCE_SCALAR
                   ? reduce_axis_tiles * scaler_tile_r_dim * scaler_tile_c_dim
                   : (reduce_axis_tiles - 1) * full_reduce_dim +
                         (call_partial_elements != 0 ? call_partial_elements : full_reduce_dim));
    const auto accumulation = make_accumulation<(num_calls > 1)>(iteration);
#ifdef REDUCE_POST_MULTIPLIER_BITS
    const PostReduceMultiply post_reduce_op{};
#else
    const compute_kernel_lib::NoOp post_reduce_op{};
#endif

    {
#ifdef REDUCE_HELPERS_PROFILE
        DeviceZoneScopedN("reduce::call");
#endif
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            cb_input,
            cb_scaler,
            output_cb,
            REDUCE_INPUT_POLICY,
            REDUCE_RECONFIG_MODE,
            REDUCE_FP32_MODE,
            REDUCE_ALGORITHM,
            compute_kernel_lib::ReduceWithinTile::Collapse,
            call_reduce_factor>(
            shape,
            layout,
            accumulation,
            post_reduce_op,
            compute_kernel_lib::ReduceScaler::compute_managed(call_partial_elements));
    }

    constexpr bool helper_pops_input =
        REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile ||
        REDUCE_INPUT_POLICY == compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop;
    if constexpr (!helper_pops_input) {
        // The no-pop policies leave ownership with their caller. Advance to the
        // next call's block only after reduce() has finished using this one.
        cb_pop_front(cb_input, input_tiles);
    }
}

template <uint32_t output_cb>
ALWI void run_reduce_call_for_iteration(uint32_t iteration) {
    if constexpr (later_valid_elements == 0) {
        run_reduce_call<output_cb, valid_elements>(iteration);
    } else if (iteration == 0) {
        run_reduce_call<output_cb, valid_elements>(iteration);
    } else {
        run_reduce_call<output_cb, later_valid_elements>(iteration);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t first_output_cb = num_calls == 1 ? cb_output : cb_accumulator;
    compute_kernel_hw_startup(cb_input, cb_scaler, first_output_cb);
#ifdef REDUCE_HELPERS_PROFILE
    DeviceZoneScopedN("reduce::body");
#endif

    // One sharded tensor backs a linear sequence of per-call blocks. Make the
    // complete stream visible once; ownership then follows the selected policy.
    cb_reserve_back(cb_input, input_tiles * num_calls);
    cb_push_back(cb_input, input_tiles * num_calls);

    for (uint32_t call = 0; call < num_calls; ++call) {
        const bool is_last_call = call == num_calls - 1;
        if (is_last_call) {
            run_reduce_call_for_iteration<cb_output>(call);
        } else {
            run_reduce_call_for_iteration<cb_accumulator>(call);
        }
    }
}
