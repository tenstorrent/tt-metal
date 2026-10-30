// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "transpose_split_device_operation_types.hpp"
#include "transpose_split_program_factory.hpp"

namespace ttml::metal::ops::transpose_split::device {

struct TransposeSplitDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<TransposeSplitProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::transpose_split::device

namespace ttnn::prim {

ttml::metal::ops::transpose_split::device::tensor_return_value_t ttml_transpose_split(
    const ttnn::Tensor& input, uint32_t rows, uint32_t tail_rows, float tail_scale);

}  // namespace ttnn::prim
