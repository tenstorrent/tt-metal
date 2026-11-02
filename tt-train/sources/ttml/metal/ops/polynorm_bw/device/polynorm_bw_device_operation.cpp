// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

void PolyNorm3BackwardDeviceOperation::validate_on_program_cache_miss(
    const PolyNorm3BWAttributes&, const PolyNorm3BWTensorArgs& tensor_args) {
    check_device_tensor(tensor_args.input, "PolyNormBackward", "Input");
    check_device_tensor(tensor_args.dL_dout, "PolyNormBackward", "dL_dout");
    check_device_tensor(tensor_args.weight, "PolyNormBackward", "Weight");

    const auto input_shape = tensor_args.input.logical_shape().to_array_4D();
    const auto expected_packed_partials_shape = ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], 128U});

    if (tensor_args.preallocated_dL_dx.has_value()) {
        const auto& preallocated_dL_dx = tensor_args.preallocated_dL_dx.value();
        check_device_tensor(
            preallocated_dL_dx,
            "PolyNormBackward",
            "Preallocated dL_dx",
            {.buffer_type = tt::tt_metal::BufferType::DRAM});
        TT_FATAL(
            preallocated_dL_dx.logical_shape() == tensor_args.input.logical_shape(),
            "Preallocated dL_dx logical shape {} does not match expected shape {}",
            preallocated_dL_dx.logical_shape(),
            tensor_args.input.logical_shape());
        TT_FATAL(
            preallocated_dL_dx.padded_shape() == tensor_args.input.padded_shape(),
            "Preallocated dL_dx padded shape {} does not match expected shape {}",
            preallocated_dL_dx.padded_shape(),
            tensor_args.input.padded_shape());
    }
    if (tensor_args.preallocated_packed_partials.has_value()) {
        const auto& packed_partials = tensor_args.preallocated_packed_partials.value();
        check_device_tensor(
            packed_partials,
            "PolyNormBackward",
            "Preallocated packed partials",
            {.dtypes = {tt::tt_metal::DataType::FLOAT32}, .buffer_type = tt::tt_metal::BufferType::DRAM});
        TT_FATAL(
            packed_partials.logical_shape() == expected_packed_partials_shape,
            "Preallocated packed partials logical shape {} does not match expected shape {}",
            packed_partials.logical_shape(),
            expected_packed_partials_shape);
    }
}

PolyNorm3BWSpecReturn PolyNorm3BackwardDeviceOperation::compute_output_specs(
    const PolyNorm3BWAttributes&, const PolyNorm3BWTensorArgs& tensor_args) {
    PolyNorm3BWSpecReturn output_specs;
    output_specs.reserve(2U);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dL_dx->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    if (tensor_args.preallocated_packed_partials.has_value()) {
        output_specs.push_back(tensor_args.preallocated_packed_partials->tensor_spec());
    } else {
        const auto input_shape = tensor_args.input.logical_shape().to_array_4D();
        output_specs.emplace_back(
            ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], 128U}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }
    return output_specs;
}

PolyNorm3BWTensorReturn PolyNorm3BackwardDeviceOperation::create_output_tensors(
    const PolyNorm3BWAttributes& op_attrs, const PolyNorm3BWTensorArgs& tensor_args) {
    PolyNorm3BWTensorReturn output_tensors;
    output_tensors.reserve(2U);
    auto specs = compute_output_specs(op_attrs, tensor_args);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dL_dx.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(specs[0], tensor_args.input.device()));
    }

    if (tensor_args.preallocated_packed_partials.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_packed_partials.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(specs[1], tensor_args.input.device()));
    }
    return output_tensors;
}

}  // namespace ttml::metal::ops::polynorm3_bw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_bw::device::PolyNorm3BackwardDeviceOperation::tensor_return_value_t ttml_polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_dL_dx,
    const std::optional<ttnn::Tensor>& preallocated_packed_partials) {
    using OperationType = ttml::metal::ops::polynorm3_bw::device::PolyNorm3BackwardDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
    };
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .dL_dout = dL_dout_tensor,
        .weight = weight_tensor,
        .preallocated_dL_dx = preallocated_dL_dx,
        .preallocated_packed_partials = preallocated_packed_partials,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
