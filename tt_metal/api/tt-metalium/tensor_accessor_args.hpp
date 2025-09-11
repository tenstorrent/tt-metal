// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

class TensorAccessorArgs {
public:
    TensorAccessorArgs() = default;
    explicit TensorAccessorArgs(
        const Buffer& buffer, tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);
    explicit TensorAccessorArgs(
        const Buffer* buffer, tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);
    explicit TensorAccessorArgs(
        const std::shared_ptr<Buffer>& buffer,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);
    explicit TensorAccessorArgs(
        const distributed::MeshBuffer& buffer,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);
    explicit TensorAccessorArgs(
        const distributed::MeshBuffer* buffer,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);
    explicit TensorAccessorArgs(
        const std::shared_ptr<distributed::MeshBuffer>& buffer,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);

    void append_to(std::vector<uint32_t>& compile_time_args) const;
    void append_to(std::vector<uint32_t>& compile_time_args, std::vector<uint32_t>& common_runtime_args) const;

    std::vector<uint32_t> get_compile_time_args() const;
    std::vector<uint32_t> get_common_runtime_args() const;

    static constexpr size_t MAX_NUM_DIMENSIONS = 8;

private:
    void update_args_config();

    const Buffer* buffer_ = nullptr;
    tensor_accessor::ArgsConfig args_config_ = tensor_accessor::ArgConfig::None;
};

}  // namespace tt::tt_metal
