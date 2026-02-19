// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "gram_matmul_device_operation_types.hpp"
#include "gram_matmul_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gram_matmul::device {

struct GramMatmulDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::gram_matmul::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::gram_matmul::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::gram_matmul::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::gram_matmul::device::tensor_return_value_t;

    using program_factory_t = std::variant<GramMatmulProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttml::metal::ops::gram_matmul::device

namespace ttnn::prim {

// Compute Gram matrix G = X @ X^T.  Takes a single input tensor X.
// The transpose is performed on-the-fly in the compute kernel (no materialized X^T).
ttml::metal::ops::gram_matmul::device::GramMatmulDeviceOperation::tensor_return_value_t ttml_gram_matmul(
    const ttnn::Tensor& input_tensor,
    const std::optional<const ttml::metal::ops::gram_matmul::device::GramMatmulConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

}  // namespace ttnn::prim
