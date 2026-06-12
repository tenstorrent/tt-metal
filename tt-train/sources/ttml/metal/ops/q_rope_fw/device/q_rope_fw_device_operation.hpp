// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "q_rope_fw_device_operation_types.hpp"
#include "q_rope_fw_program_factory.hpp"

namespace ttml::metal::ops::q_rope_fw::device {

struct QRopeFwDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::q_rope_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::q_rope_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::q_rope_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::q_rope_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<QRopeFwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::q_rope_fw::device

namespace ttnn::prim {

ttml::metal::ops::q_rope_fw::device::QRopeFwDeviceOperation::tensor_return_value_t ttml_q_rope_fw(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    bool fp32_dest_acc_en);

}  // namespace ttnn::prim
