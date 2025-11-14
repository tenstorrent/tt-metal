// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "mean_all_cores_device_operation_types.hpp"
#include "mean_all_cores_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::examples::mean_all_cores::device {

struct MeanAllCoresDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::examples::mean_all_cores::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::examples::mean_all_cores::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::examples::mean_all_cores::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::examples::mean_all_cores::device::tensor_return_value_t;
    using program_factory_t = std::variant<MeanAllCoresProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttml::metal::ops::examples::mean_all_cores::device

namespace ttnn::prim {

constexpr auto ttml_mean_all_cores = ttnn::register_operation<
    "ttnn::prim::ttml_mean_all_cores",
    ttml::metal::ops::examples::mean_all_cores::device::MeanAllCoresDeviceOperation>();
}  // namespace ttnn::prim

