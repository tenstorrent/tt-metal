// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cyclic_sdpa_bw_device_operation_types.hpp"
#include "cyclic_sdpa_bw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

struct CyclicSDPABackwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::cyclic_sdpa_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::cyclic_sdpa_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::cyclic_sdpa_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::cyclic_sdpa_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<CyclicSDPABackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::cyclic_sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::cyclic_sdpa_bw::device::CyclicSDPABackwardDeviceOperation::tensor_return_value_t
ttml_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t rows_per_block_tiles = 1U,
    bool use_barrier = false,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt);

}  // namespace ttnn::prim
