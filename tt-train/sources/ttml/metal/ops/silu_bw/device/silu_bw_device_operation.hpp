// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "silu_bw_device_operation_types.hpp"
#include "silu_bw_program_factory.hpp"

namespace ttml::metal::ops::silu_bw::device {

struct SiLUBackwardDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<SiLUBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& dL_dout_tensor,
        const std::optional<ttnn::Tensor>& preallocated_da = std::nullopt);
};

}  // namespace ttml::metal::ops::silu_bw::device

namespace ttnn::prim {
constexpr auto ttml_silu_bw = ttnn::
    register_operation<"ttnn::prim::ttml_silu_bw", ttml::metal::ops::silu_bw::device::SiLUBackwardDeviceOperation>();
}
