// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace tt_metal {

enum class RotaryEmbeddingLlamaOpParallelizationStrategy { MULTI_CORE = 0, SINGLE_CORE = 1 };

operation::ProgramWithCallbacks rotary_embedding_llama_single_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, const Tensor &trans_mat, Tensor &output, DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks rotary_embedding_llama_multi_core(
    const Tensor &input, const Tensor &cos, const Tensor &sin, const Tensor &trans_mat, Tensor &output, DeviceComputeKernelConfig compute_kernel_config);

struct RotaryEmbeddingLlama {
    const uint32_t seq_len;
    const MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    RotaryEmbeddingLlamaOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor rotary_embedding_llama(
    const Tensor &input_tensor,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor trans_mat,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, cos, sin, trans_mat}))};
    operation::launch_with_autoformat(
        [output_mem_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            auto& cos = input_tensors.at(1);
            auto& sin = input_tensors.at(2);
            auto& trans_mat = input_tensors.at(3);
            TT_FATAL(input_tensor.get_legacy_shape()[-1] % TILE_WIDTH == 0, "Input X dim must be divisible into tiles");
            uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
            uint32_t B = input_tensor.get_legacy_shape()[0];
            uint32_t head_dim = input_tensor.get_legacy_shape()[-1];
            TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
            TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == head_dim, "Cos dims must match input dims");
            TT_FATAL(cos.get_legacy_shape()[-2] >= seq_len, "Cos dims must match input dims");

            TT_FATAL(trans_mat.get_legacy_shape()[0] == 1 && trans_mat.get_legacy_shape()[1] == 1, "Transformation matrix must have 1st & 2nd dim equal to 1");
            TT_FATAL(trans_mat.get_legacy_shape()[-2] == TILE_HEIGHT, "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
            TT_FATAL(trans_mat.get_legacy_shape()[-1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");

            auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            Shape input_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
            FormatParams input_format_params = {.pad_shape = input_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            Shape cos_pad_shape = AutoFormat::pad_to_tile_shape(cos.get_legacy_shape());
            FormatParams cos_format_params = {.pad_shape = cos_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            Shape sin_pad_shape = AutoFormat::pad_to_tile_shape(sin.get_legacy_shape());
            FormatParams sin_format_params = {.pad_shape = sin_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            Shape trans_mat_pad_shape = AutoFormat::pad_to_tile_shape(trans_mat.get_legacy_shape());
            FormatParams trans_mat_format_params = {.pad_shape = trans_mat_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
            return operation::run_with_autoformat(
                    RotaryEmbeddingLlama{seq_len, output_mem_config, kernel_config_val},
                    {input_tensor, cos, sin, trans_mat},
                    {input_format_params, cos_format_params, sin_format_params, trans_mat_format_params},
                    {Layout::TILE});
        }, {input_tensor, cos, sin, trans_mat}, output_tensors);
    return output_tensors.at(0);
}

}  // namespace tt_metal

}  // namespace tt
