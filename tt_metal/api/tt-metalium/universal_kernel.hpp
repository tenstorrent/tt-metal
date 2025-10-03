// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>
#include <vector>
#include <map>
#include <unordered_map>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal {

struct MathConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
};

struct UniversalKernelConfig {
    ReaderDataMovementConfig reader_config;
    WriterDataMovementConfig writer_config;
    ComputeConfig compute_config;
    std::vector<uint32_t> runtime_args;
    std::vector<uint32_t> common_runtime_args;
};

class UniversalKernelConfigBuilder {
public:
    UniversalKernelConfigBuilder(MathConfig math_config = MathConfig{}) : math_config_(std::move(math_config)) {}

    UniversalKernelConfigBuilder& set_compile_time_args(
        std::vector<std::pair<std::string, uint32_t>> compile_time_args) {
        compile_time_args_ = std::move(compile_time_args);
        return *this;
    }
    UniversalKernelConfigBuilder& add_compile_time_arg(std::string name, uint32_t value) {
        compile_time_args_.emplace_back(std::move(name), value);
        return *this;
    }
    UniversalKernelConfigBuilder& set_runtime_args(std::vector<std::pair<std::string, uint32_t>> runtime_args) {
        runtime_args_ = std::move(runtime_args);
        return *this;
    }
    UniversalKernelConfigBuilder& add_runtime_arg(std::string name, uint32_t value) {
        runtime_args_.emplace_back(std::move(name), value);
        return *this;
    }
    UniversalKernelConfigBuilder& set_common_runtime_args(
        std::vector<std::pair<std::string, uint32_t>> common_runtime_args) {
        common_runtime_args_ = std::move(common_runtime_args);
        return *this;
    }
    UniversalKernelConfigBuilder& add_common_runtime_arg(std::string name, uint32_t value) {
        common_runtime_args_.emplace_back(std::move(name), value);
        return *this;
    }
    UniversalKernelConfigBuilder& set_defines(std::map<std::string, std::string> defines) {
        defines_ = std::move(defines);
        return *this;
    }
    UniversalKernelConfigBuilder& add_define(std::string name, std::string value) {
        defines_.emplace(std::move(name), std::move(value));
        return *this;
    }

    UniversalKernelConfigBuilder& add_buffer(std::string name, const Buffer& buffer);
    UniversalKernelConfigBuilder& add_buffer(std::string name, const Buffer* buffer);
    UniversalKernelConfigBuilder& add_buffer(std::string name, const std::shared_ptr<Buffer>& buffer);
    UniversalKernelConfigBuilder& add_buffer(std::string name, const distributed::MeshBuffer& buffer);
    UniversalKernelConfigBuilder& add_buffer(std::string name, const distributed::MeshBuffer* buffer);
    UniversalKernelConfigBuilder& add_buffer(std::string name, const std::shared_ptr<distributed::MeshBuffer>& buffer);

    UniversalKernelConfig build() const;

private:
    struct TensorData {
        TensorAccessorArgs accessor_args;
        uint32_t buffer_address;
        uint32_t page_size_bytes;
    };

    std::vector<std::pair<std::string, uint32_t>> compile_time_args_;
    std::vector<std::pair<std::string, uint32_t>> runtime_args_;
    std::vector<std::pair<std::string, uint32_t>> common_runtime_args_;
    std::map<std::string, std::string> defines_;
    std::vector<std::pair<std::string, TensorData>> tensor_data_;
    MathConfig math_config_;
};

}  // namespace tt::tt_metal
