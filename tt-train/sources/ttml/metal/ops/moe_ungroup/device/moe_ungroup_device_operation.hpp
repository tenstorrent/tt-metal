// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "metal/ttnn_all_includes.hpp"
#include "moe_ungroup_device_operation_types.hpp"
#include "moe_ungroup_program_factory.hpp"

namespace ttml::metal::ops::moe_ungroup::device {

struct MoeUngroupDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::moe_ungroup::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::moe_ungroup::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::moe_ungroup::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::moe_ungroup::device::tensor_return_value_t;
    using program_factory_t = std::variant<MoeUngroupProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::moe_ungroup::device

namespace ttnn::prim {

ttml::metal::ops::moe_ungroup::device::MoeUngroupDeviceOperation::tensor_return_value_t ttml_moe_ungroup(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& grouped_scores,
    uint32_t e_local,
    uint32_t d,
    uint32_t b,
    uint32_t s);

}  // namespace ttnn::prim
