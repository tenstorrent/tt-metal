// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_device_operation.hpp"

#include "rmsnorm_fw_program_factory.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

RMSNormForwardDeviceOperation::program_factory_t RMSNormForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return RMSNormForwardProgramFactory{};
}

void RMSNormForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RMSNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "RMSNormForward operation is only supported on Wormhole. Device arch: {}. Tensor name {}",
            magic_enum::enum_name(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "RMSNormForward operation requires {} to be on Device. Input storage type: {}",
            name,
            static_cast<int>(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to RMSNormForward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "RMSNormForward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            static_cast<int>(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "RMSNormForward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            static_cast<int>(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "RMSNormForward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            static_cast<int>(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& preallocated_rms_tensor = tensor_args.preallocated_rms;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");
    if (preallocated_rms_tensor.has_value()) {
        check_tensor(preallocated_rms_tensor.value(), "Preallocated RMS");
    }
    if (preallocated_output_tensor.has_value()) {
        check_tensor(preallocated_output_tensor.value(), "Preallocated Output");
    }
}

spec_return_value_t RMSNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_rms.has_value()) {
            output_specs.push_back(tensor_args.preallocated_rms->tensor_spec());
        } else {
            auto shape = tensor_args.input.logical_shape();
            shape[-1] = 1U;  // RMS is a scalar per row

            output_specs.emplace_back(
                shape,
                tt::tt_metal::TensorLayout(
                    tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
        }
    }

    return output_specs;
}

tensor_return_value_t RMSNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_output.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_rms.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_rms.value());
        } else {
            output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.input.device()));
        }
    }

    return output_tensors;
}

ttsl::hash::hash_t RMSNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RMSNormForwardDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<RMSNormForwardDeviceOperation::operation_attributes_t, RMSNormForwardDeviceOperation::tensor_args_t>
RMSNormForwardDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    bool return_intermediates,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_rms,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .return_intermediates = return_intermediates,
            .epsilon = epsilon,
        },
        tensor_args_t{
            .input = input_tensor,
            .gamma = gamma_tensor,
            .preallocated_rms = preallocated_rms,
            .preallocated_output = preallocated_output,
        }};
}

}  // namespace ttml::metal::ops::rmsnorm_fw::device
