// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "rmsnorm_bw_rows_device_operation_types.hpp"
#include "rmsnorm_bw_rows_program_factory.hpp"

namespace ttml::metal::ops::rmsnorm_bw_rows::device {

struct RMSNormBackwardRowsDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<RMSNormBackwardRowsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::rmsnorm_bw_rows::device

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_rmsnorm_bw_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& rms,
    const ttnn::Tensor& dL_dout,
    uint32_t num_slices,
    uint32_t slice_tiles,
    const std::optional<ttnn::Tensor>& residual_grad,
    bool transposed_grad);

}  // namespace ttnn::prim
