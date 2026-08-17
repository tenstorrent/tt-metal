// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "variable_matmul_device_operation_types.hpp"
#include "variable_matmul_program_factory.hpp"

namespace ttml::metal::ops::variable_matmul::device {

struct VariableMatmulDeviceOperation {
    using operation_attributes_t = VariableMatmulParams;
    using tensor_args_t = VariableMatmulInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    using program_factory_t = std::variant<VariableMatmulProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttml::metal::ops::variable_matmul::device

namespace ttnn::prim {

ttnn::Tensor ttml_variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttml::metal::ops::variable_matmul::device::VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    ttml::metal::ops::variable_matmul::device::OffsetsRole offsets_role,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> output_tensor = std::nullopt,
    uint32_t offsets_start_index = 0,
    uint32_t expected_M_tiles = 0);

}  // namespace ttnn::prim
