// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "metal/ttnn_all_includes.hpp"
#include "moe_group_device_operation_types.hpp"
#include "moe_group_program_factory.hpp"

namespace ttml::metal::ops::moe_group::device {

struct MoeGroupDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::moe_group::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::moe_group::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::moe_group::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::moe_group::device::tensor_return_value_t;
    using program_factory_t = std::variant<MoeGroupProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::moe_group::device

namespace ttnn::prim {

ttml::metal::ops::moe_group::device::MoeGroupDeviceOperation::tensor_return_value_t ttml_moe_group(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k);

}  // namespace ttnn::prim
