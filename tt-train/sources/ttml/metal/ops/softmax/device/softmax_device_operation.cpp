// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>

#include "softmax_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::softmax::device {

namespace {

bool is_canonical_tile(const tt::tt_metal::Tile& tile) {
    const auto canonical = tt::tt_metal::Tile{};
    return tile.get_tile_shape() == canonical.get_tile_shape() && tile.get_face_shape() == canonical.get_face_shape() &&
           tile.get_num_faces() == canonical.get_num_faces() && !tile.get_transpose_within_face() &&
           !tile.get_transpose_of_faces();
}

}  // namespace

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "Softmax operation requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.layout() == required_layout,
            "Tensor '{}' must have layout '{}', but got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == required_dtype,
            "Tensor '{}' must have data type '{}', but got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));

        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Tensor '{}' must be stored in DRAM, but got '{}'",
            name,
            enchantum::to_string(tensor.buffer()->buffer_type()));

        TT_FATAL(is_canonical_tile(tensor.tensor_spec().tile()), "Tensor '{}' must use the canonical 32x32 tile", name);
    };

    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_tensor(input_tensor, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    TT_FATAL(input_tensor.logical_shape().rank() == 4, "Softmax operation requires a rank-4 input tensor");
    const auto& logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();
    const auto tile_width = tt::tt_metal::Tile{}.get_width();
    const auto minimum_padded_width = ((logical_shape[-1] + tile_width - 1U) / tile_width) * tile_width;
    TT_FATAL(
        padded_shape[0] == logical_shape[0] && padded_shape[1] == logical_shape[1] &&
            padded_shape[-1] == minimum_padded_width,
        "Softmax only supports padding in the height dimension (logical shape {}, padded shape {})",
        logical_shape,
        padded_shape);
    if (preallocated_output_tensor.has_value()) {
        const auto& output_tensor = preallocated_output_tensor.value();
        check_tensor(
            output_tensor, "Preallocated Output", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
        TT_FATAL(
            output_tensor.tensor_spec() == input_tensor.tensor_spec(),
            "Preallocated softmax output must have the same tensor spec as the input");
        TT_FATAL(
            output_tensor.device() == input_tensor.device(),
            "Preallocated softmax output must be on the same device as the input");
        TT_FATAL(
            std::ranges::equal(output_tensor.device_storage().get_coords(), input_tensor.device_storage().get_coords()),
            "Preallocated softmax output must cover the same device coordinates as the input");
    }

    // Validate the dimension argument
    auto input_rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    // TT_FATAL(
    //     (args.dim < input_rank) && (-input_rank <= args.dim),
    //     "`dim` must be in the range [-input_rank, input_rank) (provided: dim = {}, rank = {}).",
    //     args.dim,
    //     input_rank);

    TT_FATAL(
        args.dim == (input_rank - 1) || args.dim == -1,
        "Currently only supports softmax over the last dimension. Got: {}",
        args.dim);
}

SoftmaxDeviceOperation::spec_return_value_t SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return tensor_args.input.tensor_spec();
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = ttnn::create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

}  // namespace ttml::metal::ops::softmax::device

namespace ttnn::prim {

ttml::metal::ops::softmax::device::SoftmaxDeviceOperation::tensor_return_value_t ttml_softmax(
    const ttnn::Tensor& input_tensor, int32_t dim, const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::softmax::device::SoftmaxDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.dim = dim};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
