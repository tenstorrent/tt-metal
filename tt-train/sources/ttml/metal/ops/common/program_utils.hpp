// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

inline uint32_t get_block_size(uint32_t num_inner, const uint32_t max_block_size = 4U) {
    for (uint32_t block_size = max_block_size; block_size > 1U; block_size--) {
        if (num_inner % block_size == 0) {  // if num_inner is divisible by block_size - choose this block_size
            return block_size;
        }
    }
    return 1U;
}

inline uint32_t pack_two_bfloat16_to_uint32(float value) {
    uint32_t uint32_data = std::bit_cast<uint32_t>(value);
    uint32_t casted_uint16_data = uint32_data >> 16U;
    return casted_uint16_data | (casted_uint16_data << 16U);
}

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

inline tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

inline tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

inline tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compile_time_args,
            .defines = defines});
}
