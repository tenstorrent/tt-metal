/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt::tt_metal;

inline bool is_dram(const Tensor &input_tensor);
inline bool is_dram(const std::optional<const Tensor> input_tensor);
inline bool is_dram(const Buffer *b);

inline std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores, CoreRangeSet core_group_1, CoreRangeSet core_group_2, uint32_t offset_x, uint32_t offset_y);

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreRange core_range, uint32_t units_to_divide);

KernelID CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines);

KernelID CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines);

struct ComputeKernelArg {
    CoreRangeSet core_range;
    uint32_t num_tile_per_core_group;
    const std::vector<uint32_t> &compile_args;
};

void CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false);

struct CircularBufferArg {
    uint32_t buffer_index;
    uint32_t num_tiles;
    // TODO: change to CoreRangeSet.
    // currently using pointer because there is no default constructor for
    // CoreRangeSet
    CoreRangeSet *core_range;
    tt::DataFormat data_format;

    CircularBufferArg(uint32_t buffer_index, uint32_t num_tiles) : buffer_index(buffer_index), num_tiles(num_tiles) {
        core_range = nullptr;
        data_format = tt::DataFormat::Invalid;
    }
};

void CreateCircularBuffers(
    Program &program,
    const CoreRangeSet &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args,
    std::optional<uint32_t> l1_address = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
