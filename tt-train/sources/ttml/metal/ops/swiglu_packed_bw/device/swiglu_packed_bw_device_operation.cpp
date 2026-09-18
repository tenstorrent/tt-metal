// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/ops/common/swiglu_packed_common.hpp"
#include "swiglu_packed_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_bw::device {

namespace {
constexpr std::string_view kOp = "SwigluPackedBw";
}  // namespace

void SwigluPackedBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& dL_dh = tensor_args.dL_dh;
    swiglu_packed::check_tensor(packed, "packed", kOp);
    swiglu_packed::check_tensor(dL_dh, "dL_dh", kOp);
    swiglu_packed::check_same_device(dL_dh, packed, "dL_dh", kOp);
    swiglu_packed::validate_packed(packed, kOp);
    swiglu_packed::validate_half_of_packed(dL_dh, packed, "dL_dh", kOp);

    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        const auto& out = *tensor_args.preallocated_dL_dpacked;
        swiglu_packed::check_tensor(out, "preallocated_dL_dpacked", kOp);
        swiglu_packed::check_same_device(out, packed, "preallocated_dL_dpacked", kOp);
        swiglu_packed::validate_same_as_packed(out, packed, "preallocated_dL_dpacked", kOp);
    }
}

spec_return_value_t SwigluPackedBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        return tensor_args.preallocated_dL_dpacked->tensor_spec();
    }

    const auto& packed = tensor_args.packed;
    return tt::tt_metal::TensorSpec(
        packed.logical_shape(),
        tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
}

tensor_return_value_t SwigluPackedBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        return tensor_args.preallocated_dL_dpacked.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.packed.device());
}

ttsl::hash::hash_t SwigluPackedBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& out_memcfg = tensor_args.preallocated_dL_dpacked.has_value()
                                 ? tensor_args.preallocated_dL_dpacked->memory_config()
                                 : packed.memory_config();
    return tt::tt_metal::operation::hash_operation<SwigluPackedBwDeviceOperation>(
        args,
        packed.dtype(),
        packed.logical_shape(),
        packed.padded_shape(),
        packed.memory_config(),
        tensor_args.dL_dh.dtype(),
        tensor_args.dL_dh.logical_shape(),
        tensor_args.dL_dh.padded_shape(),
        tensor_args.dL_dh.memory_config(),
        out_memcfg);
}

}  // namespace ttml::metal::ops::swiglu_packed_bw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation::tensor_return_value_t ttml_swiglu_packed_bw(
    const ttnn::Tensor& packed, const ttnn::Tensor& dL_dh, const std::optional<ttnn::Tensor>& preallocated_dL_dpacked) {
    using Op = ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .dL_dh = dL_dh,
        .preallocated_dL_dpacked = preallocated_dL_dpacked,
    };

    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, tensor_args);
}

}  // namespace ttnn::prim
