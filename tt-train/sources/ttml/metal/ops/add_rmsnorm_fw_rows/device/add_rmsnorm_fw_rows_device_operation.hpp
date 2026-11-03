// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "add_rmsnorm_fw_rows_device_operation_types.hpp"
#include "add_rmsnorm_fw_rows_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_rows::device {

struct AddRMSNormForwardRowsDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<AddRMSNormForwardRowsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::add_rmsnorm_fw_rows::device

namespace ttnn::prim {
std::vector<ttnn::Tensor> ttml_add_rmsnorm_fw_rows(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& delta,
    const ttnn::Tensor& gamma,
    uint32_t num_slices,
    uint32_t slice_tiles,
    float epsilon,
    bool delta_transposed,
    uint32_t out_tail_tiles);
}  // namespace ttnn::prim
