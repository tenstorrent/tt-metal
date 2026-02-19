// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "gram_polynomial_device_operation_types.hpp"
#include "gram_polynomial_phase3_program_factory.hpp"
#include "gram_polynomial_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gram_polynomial::device {

struct GramPolynomialDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::gram_polynomial::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::gram_polynomial::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::gram_polynomial::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::gram_polynomial::device::tensor_return_value_t;

    using program_factory_t = std::variant<GramPolynomialProgramFactory>;

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

// Phase 3: X' = H @ X + a*X
struct HxPlusAxDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::gram_polynomial::device::phase3_operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::gram_polynomial::device::phase3_tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::gram_polynomial::device::phase3_spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::gram_polynomial::device::phase3_tensor_return_value_t;

    using program_factory_t = std::variant<HxPlusAxProgramFactory>;

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

}  // namespace ttml::metal::ops::gram_polynomial::device

namespace ttnn::prim {

// Phase 2: H = c*G*G + b*G.  Takes G (square) as input.
ttml::metal::ops::gram_polynomial::device::GramPolynomialDeviceOperation::tensor_return_value_t
ttml_gram_polynomial_phase2(
    const ttnn::Tensor& g_tensor,
    float b,
    float c,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

// Phase 3: X' = H @ X + a*X.  Takes H (square) and X (rectangular) as inputs.
ttml::metal::ops::gram_polynomial::device::HxPlusAxDeviceOperation::tensor_return_value_t ttml_hx_plus_ax(
    const ttnn::Tensor& h_tensor,
    const ttnn::Tensor& x_tensor,
    float a,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

}  // namespace ttnn::prim
