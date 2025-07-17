// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include "rmsnorm_bw_program_factory.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

RMSNormBackwardDeviceOperation::program_factory_t RMSNormBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return RMSNormBackwardProgramFactory{};
}

void RMSNormBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "RMSNormBackward operation is only supported on Wormhole. Device arch: {}. Tensor name {}",
            magic_enum::enum_name(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "RMSNormBackward operation requires {} to be on Device. Input storage type: {}",
            name,
            static_cast<int>(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to RMSNormBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "RMSNormBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            static_cast<int>(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "RMSNormBackward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            static_cast<int>(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "RMSNormBackward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            static_cast<int>(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& rms_tensor = tensor_args.rms;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_da_tensor = tensor_args.preallocated_da;
    const auto& preallocated_dgamma_components_tensor = tensor_args.preallocated_dgamma_components;

    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");
    check_tensor(rms_tensor, "RMS");
    check_tensor(dL_dout_tensor, "dL_dout");
    if (preallocated_da_tensor.has_value()) {
        check_tensor(preallocated_da_tensor.value(), "Preallocated dL_da");
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        check_tensor(preallocated_dgamma_components_tensor.value(), "Preallocated dL_dgamma_components");
    }
}

spec_return_value_t RMSNormBackwardDeviceOperation::compute_output_specs(
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

    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dgamma_components->tensor_spec());
    } else {
        // Since we cannot compute dL_dgamma in the kernel, we need to return dL_dgamma_components, which will be
        // reduced outside the kernel. The shape of dL_dgamma_components is the same as the input shape.
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
    }

    return output_specs;
}

tensor_return_value_t RMSNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(2U);

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_da.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_da.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dgamma_components.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.gamma.device()));
    }

    return output_tensors;
}

ttsl::hash::hash_t RMSNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RMSNormBackwardDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<RMSNormBackwardDeviceOperation::operation_attributes_t, RMSNormBackwardDeviceOperation::tensor_args_t>
RMSNormBackwardDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_da,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components) {
    return {
        operation_attributes_t{
            .epsilon = epsilon,
        },
        tensor_args_t{
            .input = input_tensor,
            .gamma = gamma_tensor,
            .rms = rms_tensor,
            .dL_dout = dL_dout_tensor,
            .preallocated_da = preallocated_da,
            .preallocated_dgamma_components = preallocated_dgamma_components,
        }};
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device
