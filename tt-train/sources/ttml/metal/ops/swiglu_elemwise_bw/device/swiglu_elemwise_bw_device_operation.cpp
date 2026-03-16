// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_elemwise_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "swiglu_elemwise_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

void SwigluElemwiseBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SwigluElemwiseBw requires {} on Device. Storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SwigluElemwiseBw: {} buffer is null", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SwigluElemwiseBw requires TILE layout. {} layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SwigluElemwiseBw requires BFLOAT16. {} dtype: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SwigluElemwiseBw requires INTERLEAVED. {} layout: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.linear1, "linear1");
    check_tensor(tensor_args.gate, "gate");
    check_tensor(tensor_args.dL_dprod, "dL_dprod");
    if (tensor_args.preallocated_dL_dlinear1.has_value()) {
        check_tensor(tensor_args.preallocated_dL_dlinear1.value(), "preallocated_dL_dlinear1");
    }
    if (tensor_args.preallocated_dL_dgate.has_value()) {
        check_tensor(tensor_args.preallocated_dL_dgate.value(), "preallocated_dL_dgate");
    }
}

spec_return_value_t SwigluElemwiseBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto make_spec = [&](const std::optional<ttnn::Tensor>& prealloc) -> ttnn::TensorSpec {
        if (prealloc.has_value()) {
            return prealloc->tensor_spec();
        }
        return ttnn::TensorSpec(
            tensor_args.linear1.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.linear1.dtype(), tt::tt_metal::Layout::TILE, tensor_args.linear1.memory_config()));
    };

    return {make_spec(tensor_args.preallocated_dL_dlinear1), make_spec(tensor_args.preallocated_dL_dgate)};
}

tensor_return_value_t SwigluElemwiseBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.linear1.device();

    return {
        tensor_args.preallocated_dL_dlinear1.has_value() ? tensor_args.preallocated_dL_dlinear1.value()
                                                         : create_device_tensor(specs[0], device),
        tensor_args.preallocated_dL_dgate.has_value() ? tensor_args.preallocated_dL_dgate.value()
                                                      : create_device_tensor(specs[1], device),
    };
}

ttsl::hash::hash_t SwigluElemwiseBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SwigluElemwiseBwDeviceOperation>(
        args, tensor_args.linear1.dtype(), tensor_args.linear1.logical_shape());
}

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwDeviceOperation::tensor_return_value_t
ttml_swiglu_elemwise_bw(
    const ttnn::Tensor& linear1,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& dL_dprod,
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate) {
    using Op = ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwDeviceOperation;

    auto tensor_args = Op::tensor_args_t{
        .linear1 = linear1,
        .gate = gate,
        .dL_dprod = dL_dprod,
        .preallocated_dL_dlinear1 = preallocated_dL_dlinear1,
        .preallocated_dL_dgate = preallocated_dL_dgate,
    };

    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, tensor_args);
}

}  // namespace ttnn::prim
