// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

class TensorAccessorArgs {
public:
    explicit TensorAccessorArgs(
        const Buffer& buffer, tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);

    void append_args(std::vector<uint32_t>& compile_time_args) const;
    void append_args(std::vector<uint32_t>& compile_time_args, std::vector<uint32_t>& common_runtime_args) const;

    std::vector<uint32_t> get_compile_time_args() const;
    std::vector<uint32_t> get_common_runtime_args() const;

private:
    const Buffer* buffer_ = nullptr;
    tensor_accessor::ArgsConfig args_config_ = tensor_accessor::ArgConfig::None;
};

}  // namespace tt::tt_metal
