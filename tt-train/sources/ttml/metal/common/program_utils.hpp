// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "metal/ttnn_all_includes.hpp"

inline uint32_t get_block_size(uint32_t num_inner, const uint32_t max_block_size = 4U) {
    for (uint32_t block_size = max_block_size; block_size > 1U; block_size--) {
        if (num_inner % block_size == 0) {  // if num_inner is divisible by block_size - choose this block_size
            return block_size;
        }
    }
    return 1U;
}

/**
 *   Number of heads H when `head_major` is [.., T, V] over B*H leading rows and `head_merged` is the same tokens
 *   laid out as [.., T, H*V] over B leading rows. Heads must start on tile boundaries unless H == 1.
 */
inline uint32_t get_num_heads_of_head_merged(
    const ttnn::Tensor& head_major, const ttnn::Tensor& head_merged, const std::string_view name) {
    const auto& major_shape = head_major.logical_shape();
    const auto& merged_shape = head_merged.logical_shape();
    TT_FATAL(
        major_shape.rank() == 4U && merged_shape.rank() == 4U,
        "{} must be 4D like the input, got {} and input {}",
        name,
        merged_shape,
        major_shape);
    const uint32_t head_dim = major_shape[-1];
    TT_FATAL(
        merged_shape[-1] % head_dim == 0,
        "{} width {} must be a whole number of heads of the input width V = {}",
        name,
        merged_shape[-1],
        head_dim);
    const uint32_t num_heads = merged_shape[-1] / head_dim;
    TT_FATAL(
        num_heads == 1U || head_dim % tt::constants::TILE_WIDTH == 0,
        "{} holds {} heads of V = {}; V must be a multiple of {} so heads start on tile boundaries",
        name,
        num_heads,
        head_dim,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        merged_shape[-2] == major_shape[-2] &&
            merged_shape[0] * merged_shape[1] * num_heads == major_shape[0] * major_shape[1],
        "{} shape {} is not the head-merged layout of input {}: expected [B, 1, T, H*V] for input [B*H, 1, T, V]",
        name,
        merged_shape,
        major_shape);
    return num_heads;
}

inline uint32_t pack_two_bfloat16_to_uint32(float value) {
    uint32_t uint32_data = std::bit_cast<uint32_t>(value);
    uint32_t casted_uint16_data = uint32_data >> 16U;
    return casted_uint16_data | (casted_uint16_data << 16U);
}

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

/**
 *   Byte-sized variant of create_circular_buffer for CBs whose total size and
 *   page size aren't naturally expressed as tile_size × num_tiles (control/scratch
 *   buffers, row-major staging, custom-aligned pages). If `page_size_bytes` is 0
 *   the whole buffer is one page.
 */
inline tt::tt_metal::CBHandle create_circular_buffer_bytes(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_bytes,
    uint32_t page_size_bytes = 0U) {
    const uint32_t page_size = page_size_bytes == 0U ? total_bytes : page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_bytes, {{cb_index, data_format}}).set_page_size(cb_index, page_size);
    return CreateCircularBuffer(program, core_ranges, cb_config);
}

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

/**
 * Create a compute kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path,
    const bool fp32_dest_acc_en) {
    return tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compile_time_args,
            .defines = defines});
}

namespace ttml::metal {

// One core's share of the work, as handed out by for_each_core_with_work.
struct CoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t index;      // position in the walk: core == {index / num_cores_y, index % num_cores_y}
    uint32_t num_units;  // rows, blocks or tiles this core processes
    uint32_t start;      // units handed to the cores before it
    bool in_group_1;     // which split_work_to_cores group the core is in; picks the per-group compute kernel
};

/**
 * Walk the cores that `tt::tt_metal::split_work_to_cores` handed work to, in the order tt-train
 * readers/writers assume (core i -> {i / num_cores_y, i % num_cores_y}), and call `fn(const CoreWork&)`
 * once per core. Ops that need the walk position (per-core seeds, reduction protocols) take it from
 * `CoreWork::index` rather than recomputing the walk; ops with one compute kernel per group pick it with
 * `CoreWork::in_group_1` rather than testing the groups again.
 */
template <typename Fn>
inline void for_each_core_with_work(
    uint32_t num_cores,
    uint32_t num_cores_y,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t num_units_per_core_group_1,
    uint32_t num_units_per_core_group_2,
    Fn&& fn) {
    uint32_t num_units_written = 0U;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const bool in_group_1 = core_group_1.contains(core);
        uint32_t num_units = 0U;
        if (in_group_1) {
            num_units = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units = num_units_per_core_group_2;
        } else {
            TT_FATAL(false, "Core {} is in neither work group", core.str());
        }
        fn(CoreWork{core, i, num_units, num_units_written, in_group_1});
        num_units_written += num_units;
    }
}

/**
 * The same core walk without the work lookup, for override_runtime_arguments where only buffer addresses
 * change and every core keeps the work it was given in create(). `fn` takes `(core)` or `(core, index)`.
 */
template <typename Fn>
inline void for_each_core(uint32_t num_cores, uint32_t num_cores_y, Fn&& fn) {
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core{i / num_cores_y, i % num_cores_y};
        if constexpr (std::is_invocable_v<Fn&, const tt::tt_metal::CoreCoord&, uint32_t>) {
            fn(core, i);
        } else {
            fn(core);
        }
    }
}

}  // namespace ttml::metal
