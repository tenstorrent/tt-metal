// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/offline_kernel_compile.hpp>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program.hpp>

#include "impl/buffers/circular_buffer.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

#include <map>
#include <stdexcept>
#include <unordered_set>

namespace tt::tt_metal::experimental {

namespace {

using CBCompileConfig = OfflineKernelCompileParams::CBCompileConfig;

void validate_kernel_config_defines(const std::map<std::string, std::string>& defines) {
    for (const auto& [key, value] : defines) {
        if (value.find('\0') != std::string::npos) {
            throw std::invalid_argument("Define value for key '" + key + "' contains null character");
        }
    }
}

void validate_cb_compile_configs(const std::vector<CBCompileConfig>& cb_compile_configs) {
    std::unordered_set<uint8_t> seen_cb_indices;
    for (const auto& config : cb_compile_configs) {
        if (config.cb_index >= NUM_CIRCULAR_BUFFERS) {
            throw std::invalid_argument(
                "CB compile config has out-of-range cb_index: " + std::to_string(config.cb_index));
        }
        if (config.data_format == DataFormat::Invalid) {
            throw std::invalid_argument(
                "CB compile config has invalid data_format for cb_index: " + std::to_string(config.cb_index));
        }
        if (!seen_cb_indices.insert(config.cb_index).second) {
            throw std::invalid_argument("CB compile config has duplicate cb_index: " + std::to_string(config.cb_index));
        }
    }
}

}  // namespace

std::vector<OfflineKernelCompileParams::CBCompileConfig> CBCompileConfigsFromProgram(
    const Program& program, KernelHandle kernel) {
    const std::shared_ptr<Kernel> kernel_ptr = program.impl().get_kernel(kernel);

    std::map<uint8_t, CBCompileConfig> compile_config_by_cb_index;
    for (const auto& logical_cr : kernel_ptr->logical_coreranges()) {
        const std::vector<std::shared_ptr<CircularBufferImpl>> cbs_on_core_range =
            program.impl().circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core_range) {
            for (const auto buffer_index : circular_buffer->buffer_indices()) {
                const uint8_t cb_index = static_cast<uint8_t>(buffer_index);
                const CBCompileConfig candidate_config{
                    .cb_index = cb_index,
                    .data_format = circular_buffer->data_format(buffer_index),
                    .tile = circular_buffer->tile(buffer_index),
                };
                // Mirror runtime behavior: last write wins if multiple CBs overlap
                // on the same index for this kernel placement.
                compile_config_by_cb_index.insert_or_assign(cb_index, candidate_config);
            }
        }
    }

    std::vector<CBCompileConfig> compile_configs;
    compile_configs.reserve(compile_config_by_cb_index.size());
    for (const auto& [_, compile_config] : compile_config_by_cb_index) {
        compile_configs.push_back(compile_config);
    }
    return compile_configs;
}

void CompileKernelOffline(
    const std::string& file_name,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const OfflineKernelCompileParams& params) {
    (void)file_name;

    std::visit([](const auto& cfg) { validate_kernel_config_defines(cfg.defines); }, config);
    validate_cb_compile_configs(params.cb_compile_configs);
    throw std::logic_error("CompileKernelOffline is not implemented yet. Planned for Slice 4.");
}

}  // namespace tt::tt_metal::experimental
