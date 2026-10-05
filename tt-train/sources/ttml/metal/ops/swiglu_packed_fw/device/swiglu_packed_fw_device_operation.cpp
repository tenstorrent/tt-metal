// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/ops/common/swiglu_packed_common.hpp"
#include "swiglu_packed_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

namespace {
constexpr std::string_view kOp = "SwigluPackedFw";
}  // namespace

void SwigluPackedFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    swiglu_packed::check_tensor(packed, "packed", kOp);
    swiglu_packed::validate_packed(packed, kOp);

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = *tensor_args.preallocated_output;
        swiglu_packed::check_tensor(out, "preallocated_output", kOp);
        swiglu_packed::check_same_device(out, packed, "preallocated_output", kOp);
        swiglu_packed::validate_half_of_packed(out, packed, "preallocated_output", kOp);
    }
}

spec_return_value_t SwigluPackedFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const auto& packed = tensor_args.packed;
    return tt::tt_metal::TensorSpec(
        swiglu_packed::halve_last_dim(packed.logical_shape()),
        tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
}

tensor_return_value_t SwigluPackedFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.packed.device());
}

ttsl::hash::hash_t SwigluPackedFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& out_memcfg = tensor_args.preallocated_output.has_value()
                                 ? tensor_args.preallocated_output->memory_config()
                                 : packed.memory_config();
    return tt::tt_metal::operation::hash_operation<SwigluPackedFwDeviceOperation>(
        args, packed.dtype(), packed.logical_shape(), packed.padded_shape(), packed.memory_config(), out_memcfg);
}

}  // namespace ttml::metal::ops::swiglu_packed_fw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_packed_fw::device::SwigluPackedFwDeviceOperation::tensor_return_value_t ttml_swiglu_packed_fw(
    const ttnn::Tensor& packed, const std::optional<ttnn::Tensor>& preallocated_output) {
    using Op = ttml::metal::ops::swiglu_packed_fw::device::SwigluPackedFwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, tensor_args);
}

}  // namespace ttnn::prim
