// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "skinny_contract_device_operation_types.hpp"
#include "skinny_contract_program_factory.hpp"

namespace ttml::metal::ops::skinny_contract::device {

struct SkinnyContractDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<SkinnyContractProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::skinny_contract::device

namespace ttnn::prim {

ttnn::Tensor ttml_skinny_contract(
    const ttnn::Tensor& small_t,
    const ttnn::Tensor& big,
    float scale,
    bool transpose_out,
    bool big_transposed,
    uint32_t grid_rows);

}  // namespace ttnn::prim
