/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <variant>

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt::tt_metal;

inline bool is_dram(const Tensor &tensor) { return tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> tensor) {
    return tensor.has_value() ? is_dram(tensor.value()) : true;
}
inline bool is_dram(const std::optional<std::reference_wrapper<const Tensor>> tensor) {
    return tensor.has_value() ? is_dram(tensor->get()) : true;
}
inline bool is_dram(const Buffer *buffer) { return buffer->buffer_type() == BufferType::DRAM; }

inline bool is_scalar(const Tensor &tensor) {
    const auto &shape = tensor.get_legacy_shape().without_padding();
    return (shape[0] == 1 && shape[1] == 1 && shape[2] == 1 && shape[3] == 1);
}

inline bool is_1d_tensor(const Tensor &tensor) {
    const auto &shape = tensor.get_legacy_shape().without_padding();
    return (shape[0] == 1 && shape[1] == 1 && shape[2] == 1);
}

inline bool is_same_shape(const Tensor &tensor_a, const Tensor &tensor_b) {
    const auto &tensor_a_shape = tensor_a.get_legacy_shape().without_padding();
    const auto &tensor_b_shape = tensor_b.get_legacy_shape().without_padding();
    return (tensor_a_shape == tensor_b_shape);
}

inline bool is_same_batch_shape(const Tensor &tensor_a, const Tensor &tensor_b) {
    const auto &tensor_a_shape = tensor_a.get_legacy_shape().without_padding();
    const auto &tensor_b_shape = tensor_b.get_legacy_shape().without_padding();
    return (tensor_a_shape[0] == tensor_b_shape[0] && tensor_a_shape[1] == tensor_b_shape[1]);
}

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores, CoreRangeSet core_group_1, CoreRangeSet core_group_2, uint32_t offset_x, uint32_t offset_y);

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
    CoreRange core_range, uint32_t units_to_divide);

[[maybe_unused]] KernelHandle CreateReadKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args = {},
    std::map<string, string> defines = {});

struct ComputeKernelArg {
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec;
    uint32_t num_tile_per_core_group;
    const std::vector<uint32_t> &compile_args = {};
};

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false);

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines = {},
    MathFidelity math_fidelity = MathFidelity::HiFi4,
    bool fp32_dest_acc_en = false,
    bool math_approx_mode = false);

struct CircularBufferArg {
    uint32_t buffer_index;
    uint32_t num_tiles;
    tt::DataFormat data_format;
    std::optional<std::variant<CoreCoord, CoreRange, CoreRangeSet>> core_range = std::nullopt;

    CircularBufferArg(uint32_t buffer_index, uint32_t num_tiles) : buffer_index(buffer_index), num_tiles(num_tiles) {
        data_format = tt::DataFormat::Invalid;
    }
};

[[maybe_unused]] std::vector<CBHandle> CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args);

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    CircularBufferArg arg);

}  // namespace primary
}  // namespace operations
}  // namespace tt
