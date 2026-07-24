// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "mla_qkv_assemble_bw_device_operation_types.hpp"
#include "mla_qkv_assemble_bw_program_factory.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_bw::device {

struct MLAQKVAssembleBwDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::mla_qkv_assemble_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::mla_qkv_assemble_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::mla_qkv_assemble_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::mla_qkv_assemble_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<MLAQKVAssembleBwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::mla_qkv_assemble_bw::device

namespace ttnn::prim {

ttml::metal::ops::mla_qkv_assemble_bw::device::MLAQKVAssembleBwDeviceOperation::tensor_return_value_t
ttml_mla_qkv_assemble_bw(
    const ttnn::Tensor& dQ,
    const ttnn::Tensor& dK,
    const ttnn::Tensor& dV,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttnn::prim
