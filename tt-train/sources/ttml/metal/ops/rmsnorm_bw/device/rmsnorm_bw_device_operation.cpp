// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>

#include "rmsnorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* expected_device = tensor_args.input.device();
    auto check_tensor = [expected_device](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "RMSNormBackward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to RMSNormBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "RMSNormBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "RMSNormBackward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "RMSNormBackward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));

        TT_FATAL(
            tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "RMSNormBackward operation requires {} to be in DRAM. Buffer type: {}",
            name,
            enchantum::to_string(tensor.memory_config().buffer_type()));

        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile == tt::tt_metal::Tile{} && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
            "RMSNormBackward operation requires {} to use the canonical non-transposed 32x32 tile",
            name);

        TT_FATAL(
            tensor.device() == expected_device,
            "RMSNormBackward operation requires {} to be on the same MeshDevice as Input",
            name);
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

    const auto& input_shape = input_tensor.logical_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 4U, "RMSNormBackward input must be 4D [B, N, S, C], got {}", input_shape);
    TT_FATAL(
        input_shape[0] > 0U && input_shape[1] > 0U && input_shape[2] > 0U && input_shape[3] > 0U,
        "RMSNormBackward input dimensions must be nonzero, got {}",
        input_shape);
    TT_FATAL(
        input_padded_shape.rank() == 4U && input_padded_shape[0] == input_shape[0] &&
            input_padded_shape[1] == input_shape[1],
        "RMSNormBackward input may be overpadded in height but not in leading dimensions. Logical shape: {}, padded "
        "shape: {}",
        input_shape,
        input_padded_shape);

    const uint32_t expected_padded_width =
        ((input_shape[-1] + tt::constants::TILE_WIDTH - 1U) / tt::constants::TILE_WIDTH) * tt::constants::TILE_WIDTH;
    TT_FATAL(
        input_padded_shape[-1] == expected_padded_width,
        "RMSNormBackward input must use canonical width padding {}. Got {}",
        expected_padded_width,
        input_padded_shape[-1]);
    const auto& input_alignment = input_tensor.tensor_spec().tensor_layout().get_alignment();
    TT_FATAL(
        !input_alignment.empty() && input_alignment[-1] == tt::constants::TILE_WIDTH,
        "RMSNormBackward input width alignment must be {}. Got {}",
        tt::constants::TILE_WIDTH,
        input_alignment);

    const auto expected_gamma_shape = ttnn::Shape({1U, 1U, 1U, input_shape[-1]});
    TT_FATAL(
        gamma_tensor.logical_shape() == expected_gamma_shape,
        "RMSNormBackward Gamma must have shape {}. Got {}",
        expected_gamma_shape,
        gamma_tensor.logical_shape());
    TT_FATAL(
        gamma_tensor.padded_shape()[-1] == input_padded_shape[-1],
        "RMSNormBackward Gamma padded width must match Input. Got {} and {}",
        gamma_tensor.padded_shape()[-1],
        input_padded_shape[-1]);

    const auto& input_spec = input_tensor.tensor_spec();
    auto rms_shape = input_shape;
    rms_shape[-1] = 1U;
    const auto expected_rms_spec = tt::tt_metal::TensorSpec(rms_shape, input_spec.tensor_layout());
    TT_FATAL(
        rms_tensor.tensor_spec() == expected_rms_spec,
        "RMSNormBackward RMS TensorSpec must match the input-derived RMS TensorSpec");
    TT_FATAL(
        dL_dout_tensor.tensor_spec() == input_spec, "RMSNormBackward dL_dout TensorSpec must match Input TensorSpec");
    if (preallocated_da_tensor.has_value()) {
        check_tensor(preallocated_da_tensor.value(), "Preallocated dL_da");
        TT_FATAL(
            preallocated_da_tensor->tensor_spec() == input_spec,
            "RMSNormBackward preallocated dL_da TensorSpec must match Input TensorSpec");
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        check_tensor(preallocated_dgamma_components_tensor.value(), "Preallocated dL_dgamma_components");
        TT_FATAL(
            preallocated_dgamma_components_tensor->tensor_spec() == input_spec,
            "RMSNormBackward preallocated dL_dgamma_components TensorSpec must match Input TensorSpec");
    }
}

spec_return_value_t RMSNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(2U);

    const auto& input_spec = tensor_args.input.tensor_spec();
    output_specs.push_back(input_spec);

    // Since we cannot compute dL_dgamma in the kernel, we need to return dL_dgamma_components, which will be
    // reduced outside the kernel. The shape and physical row mapping are the same as the input.
    output_specs.push_back(input_spec);

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
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dgamma_components.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], tensor_args.input.device()));
    }

    return output_tensors;
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_da,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.epsilon = epsilon};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .rms = rms_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_da = preallocated_da,
        .preallocated_dgamma_components = preallocated_dgamma_components,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
