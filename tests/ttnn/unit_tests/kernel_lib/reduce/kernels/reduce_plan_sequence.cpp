// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
    constexpr std::uint32_t reduce_args_offset = 1;
    constexpr std::uint32_t call_count = get_compile_time_arg_val(reduce_args_offset);
    static_assert(kernel_owned_arg == 17, "The reduce args must preserve the kernel-owned prefix");
    static_assert(call_count == 2, "This sanity kernel explicitly places two planned reduce calls");

    using First = ttnn::kernel_lib::ReduceCallArgs<
        reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count>;
    using Second = ttnn::kernel_lib::ReduceCallArgs<First::next_compile_time_args_offset()>;
    static_assert(First::path == ttnn::kernel_lib::ReducePath::Tiled);
    static_assert(Second::path == ttnn::kernel_lib::ReducePath::Tiled);
    static_assert(First::has_accumulator && Second::has_accumulator);
    static_assert(First::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Intermediate);
    static_assert(First::accumulation_index == 0);
    static_assert(Second::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Final);
    static_assert(Second::accumulation_index == 1);
    static_assert(First::partial_mode == compute_kernel_lib::ReducePartialMode::None);
    static_assert(Second::partial_mode == compute_kernel_lib::ReducePartialMode::None);

    constexpr std::uint32_t startup_src_b = First::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd
                                                ? First::input_cb_id
                                                : First::auxiliary_cb_id;
    compute_kernel_hw_startup(First::input_cb_id, startup_src_b, First::output_cb_id);

    DataflowBuffer auxiliary(First::auxiliary_cb_id);

    // Call 0: create the raw running value in the accumulator CB.
    auxiliary.wait_front(First::auxiliary_tile_count);
    constexpr auto first_shape =
        compute_kernel_lib::ReduceInputBlockShape::of(First::rows, First::columns, First::batches);
    constexpr auto first_layout = First::row_stride == 0
                                      ? compute_kernel_lib::ReduceInputMemoryLayout::contiguous()
                                      : compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(First::row_stride);
    constexpr auto first_chunk =
        compute_kernel_lib::ReduceInputChunk::of(First::reduce_axis_chunk_tiles, First::output_chunk_tiles);
    compute_kernel_lib::reduce<
        First::reduce_type,
        First::reduce_dim,
        First::input_cb_id,
        First::auxiliary_cb_id,
        First::output_cb_id,
        First::input_policy,
        First::reconfig_mode,
        First::fp32_mode,
        First::algorithm,
        First::within_tile,
        First::reduce_factor>(
        first_shape,
        first_layout,
        compute_kernel_lib::Accumulate::at(First::accumulator_cb_id, First::accumulation_index)
            .with_reload(First::reload_mode),
        compute_kernel_lib::NoOp{},
        First::partial_mode,
        first_chunk);
    auxiliary.pop_front(First::auxiliary_tile_count);

    // The kernel owns sequencing. Arbitrary compute or synchronization may be placed here.

    // Call 1: reload the running value, finalize it, and write the real output.
    static_assert(Second::auxiliary_cb_id == First::auxiliary_cb_id);
    auxiliary.wait_front(Second::auxiliary_tile_count);
    constexpr auto second_shape =
        compute_kernel_lib::ReduceInputBlockShape::of(Second::rows, Second::columns, Second::batches);
    constexpr auto second_layout =
        Second::row_stride == 0 ? compute_kernel_lib::ReduceInputMemoryLayout::contiguous()
                                : compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(Second::row_stride);
    constexpr auto second_chunk =
        compute_kernel_lib::ReduceInputChunk::of(Second::reduce_axis_chunk_tiles, Second::output_chunk_tiles);
    auto final_post_scale = [](std::uint32_t dst_index) {
        if constexpr (Second::post_scale_bits != ttnn::kernel_lib::reduce_plan_args::float_one_bits) {
            constexpr DataFormat input_format = static_cast<DataFormat>(unpack_src_format[Second::input_cb_id]);
            compute_kernel_lib::detail::reduce_post_mul_tile<input_format>(dst_index, Second::post_scale_bits);
        }
    };
    compute_kernel_lib::reduce<
        Second::reduce_type,
        Second::reduce_dim,
        Second::input_cb_id,
        Second::auxiliary_cb_id,
        Second::output_cb_id,
        Second::input_policy,
        Second::reconfig_mode,
        Second::fp32_mode,
        Second::algorithm,
        Second::within_tile,
        Second::reduce_factor>(
        second_shape,
        second_layout,
        compute_kernel_lib::Accumulate::at_last(Second::accumulator_cb_id, Second::accumulation_index)
            .with_reload(Second::reload_mode),
        final_post_scale,
        Second::partial_mode,
        second_chunk);
    auxiliary.pop_front(Second::auxiliary_tile_count);
}
