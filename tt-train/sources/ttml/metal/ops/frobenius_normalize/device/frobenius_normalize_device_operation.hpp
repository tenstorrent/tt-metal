// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "frobenius_normalize_device_operation_types.hpp"
#include "frobenius_normalize_program_factory.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeDeviceOperation {
    // Framework-required aliases (ttnn::device_operation::launch dispatches via these names).
    using operation_attributes_t = FrobeniusNormalizeAttributes;
    using tensor_args_t = FrobeniusNormalizeTensorArgs;
    using spec_return_value_t = FrobeniusNormalizeSpecReturn;
    using tensor_return_value_t = FrobeniusNormalizeTensorReturn;
    using program_factory_t = std::variant<FrobeniusNormalizeProgramFactory>;

    static void validate_on_program_cache_miss(
        const FrobeniusNormalizeAttributes&, const FrobeniusNormalizeTensorArgs&);

    static FrobeniusNormalizeSpecReturn compute_output_specs(
        const FrobeniusNormalizeAttributes&, const FrobeniusNormalizeTensorArgs&);

    static FrobeniusNormalizeTensorReturn create_output_tensors(
        const FrobeniusNormalizeAttributes& operation_attributes, const FrobeniusNormalizeTensorArgs&);

    static ttsl::hash::hash_t compute_program_hash(
        const FrobeniusNormalizeAttributes&, const FrobeniusNormalizeTensorArgs&);
};

}  // namespace ttml::metal::ops::frobenius_normalize::device

namespace ttnn::prim {

ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeTensorReturn ttml_frobenius_normalize(
    const ttnn::Tensor& input_tensor,
    float epsilon = 1e-7F,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
