// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "vp_cross_entropy_bw_device_operation_types.hpp"
#include "vp_cross_entropy_bw_program_factory.hpp"

namespace ttml::metal::ops::vp_cross_entropy_bw::device {

struct VpCrossEntropyBwDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<VpCrossEntropyBwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::vp_cross_entropy_bw::device

namespace ttnn::prim {

ttnn::Tensor ttml_vp_cross_entropy_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& log_normalizer,
    const ttnn::Tensor& target,
    const ttnn::Tensor& grad,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    float scale);

}  // namespace ttnn::prim
