// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/device_operation.hpp>

#include "k_split_gram_matmul_device_operation_types.hpp"
#include "k_split_gram_matmul_program_factory.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

struct KSplitGramMatmulDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<KSplitGramMatmulProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t &, const tensor_args_t &);
    static void validate_on_program_cache_miss(const operation_attributes_t &, const tensor_args_t &);
    static void validate_on_program_cache_hit(const operation_attributes_t &, const tensor_args_t &);
    static spec_return_value_t compute_output_specs(const operation_attributes_t &, const tensor_args_t &);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t &, const tensor_args_t &);
};

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
