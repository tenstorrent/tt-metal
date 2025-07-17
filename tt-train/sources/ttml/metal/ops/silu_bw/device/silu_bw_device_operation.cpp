// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw_device_operation.hpp"

#include "silu_bw_program_factory.hpp"

namespace ttml::metal::ops::silu_bw::device {

SiLUBackwardDeviceOperation::program_factory_t SiLUBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return SiLUBackwardProgramFactory{};
}

void SiLUBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SiLUBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "SiLUBackward operation is only supported on Wormhole. Device arch: {}. Tensor name {}",
            magic_enum::enum_name(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SiLUBackward operation requires {} to be on Device. Input storage type: {}",
            name,
            static_cast<int>(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to SiLUBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SiLUBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            static_cast<int>(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SiLUBackward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            static_cast<int>(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SiLUBackward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            static_cast<int>(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_da_tensor = tensor_args.preallocated_da;

    check_tensor(input_tensor, "Input");
    check_tensor(dL_dout_tensor, "dL_dout");
    if (preallocated_da_tensor.has_value()) {
        check_tensor(preallocated_da_tensor.value(), "Preallocated dL_da");
    }
}

spec_return_value_t SiLUBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(2U);

    if (tensor_args.preallocated_da.has_value()) {
        output_specs.push_back(tensor_args.preallocated_da->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    return output_specs;
}

tensor_return_value_t SiLUBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(2U);

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_da.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_da.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    return output_tensors;
}

ttsl::hash::hash_t SiLUBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<SiLUBackwardDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<SiLUBackwardDeviceOperation::operation_attributes_t, SiLUBackwardDeviceOperation::tensor_args_t>
SiLUBackwardDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const std::optional<ttnn::Tensor>& preallocated_da) {
    return {
        operation_attributes_t{},
        tensor_args_t{
            .input = input_tensor,
            .dL_dout = dL_dout_tensor,
            .preallocated_da = preallocated_da,
        }};
}

}  // namespace ttml::metal::ops::silu_bw::device
