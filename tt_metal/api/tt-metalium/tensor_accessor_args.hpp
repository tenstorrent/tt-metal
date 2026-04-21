// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

    static TensorAccessorArgs create_dram_interleaved();
    static TensorAccessorArgs create_l1_interleaved();

    template <typename ArgsVec>
    void append_to(ArgsVec& compile_time_args) const;
    template <typename ArgsVec1, typename ArgsVec2>
    void append_to(ArgsVec1& compile_time_args, ArgsVec2& common_runtime_args) const;

    std::vector<uint32_t> get_compile_time_args() const;
    std::vector<uint32_t> get_common_runtime_args() const;

    static constexpr size_t MAX_NUM_DIMENSIONS = 8;

private:
    void update_args_config();

    const Buffer* buffer_ = nullptr;
    tensor_accessor::ArgsConfig args_config_ = tensor_accessor::ArgConfig::None;
};

}  // namespace tt::tt_metal
