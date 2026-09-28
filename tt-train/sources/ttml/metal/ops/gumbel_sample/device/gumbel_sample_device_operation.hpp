// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "gumbel_sample_device_operation_types.hpp"
#include "gumbel_sample_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

struct GumbelSampleDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::gumbel_sample::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::gumbel_sample::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::gumbel_sample::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::gumbel_sample::device::tensor_return_value_t;
    using program_factory_t = std::variant<GumbelSampleProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::gumbel_sample::device

namespace ttnn::prim {

ttml::metal::ops::gumbel_sample::device::GumbelSampleDeviceOperation::tensor_return_value_t ttml_gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes = {},
    const std::optional<ttnn::Tensor>& logits_mask = std::nullopt,
    const std::optional<ttnn::Tensor>& positions = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
