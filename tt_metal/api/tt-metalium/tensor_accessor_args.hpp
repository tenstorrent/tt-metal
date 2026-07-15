// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/optional_reference.hpp>

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

    explicit TensorAccessorArgs(
        const MeshTensor& tensor,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);

    // Convenience overload for optional tensor inputs. An empty
    // `optional_reference` is equivalent to constructing from a null `Buffer*`:
    // `args_config_` is forced to `None` and `append_to` emits two zero
    // compile-time args.
    explicit TensorAccessorArgs(
        ttsl::optional_reference<const MeshTensor> tensor,
        tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);

    static TensorAccessorArgs create_dram_interleaved();
    static TensorAccessorArgs create_l1_interleaved();

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
