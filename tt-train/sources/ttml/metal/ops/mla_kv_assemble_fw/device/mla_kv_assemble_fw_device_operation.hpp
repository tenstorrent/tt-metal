// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "mla_kv_assemble_fw_device_operation_types.hpp"
#include "mla_kv_assemble_fw_program_factory.hpp"

namespace ttml::metal::ops::mla_kv_assemble_fw::device {

struct MLAKVAssembleFwDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::mla_kv_assemble_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::mla_kv_assemble_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::mla_kv_assemble_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::mla_kv_assemble_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<MLAKVAssembleFwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::mla_kv_assemble_fw::device

namespace ttnn::prim {

ttml::metal::ops::mla_kv_assemble_fw::device::MLAKVAssembleFwDeviceOperation::tensor_return_value_t
ttml_mla_kv_assemble_fw(
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttnn::prim
