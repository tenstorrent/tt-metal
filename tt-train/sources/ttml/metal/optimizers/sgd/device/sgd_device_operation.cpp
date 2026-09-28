// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "sgd_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::optimizers::sgd::device {

void SGDDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param = tensor_args.param;
    auto check_tensor = [&param](
                            const ttnn::Tensor& tensor,
                            const std::string& name,
                            const tt::tt_metal::Layout required_layout,
                            const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "SGD optimizer requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.device() == param.device(), "Tensor '{}' must be on the same mesh device as the parameter.", name);

        TT_FATAL(
            tensor.tensor_topology() == param.tensor_topology(),
            "Tensor '{}' must have the same mesh topology as the parameter. Expected {}, got {}",
            name,
            param.tensor_topology(),
            tensor.tensor_topology());

        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Tensor '{}' must be in DRAM. Got buffer type: '{}'",
            name,
            enchantum::to_string(tensor.buffer()->buffer_type()));

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

        const auto& tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_tile_shape() == tt::tt_metal::Tile::TileShape{32U, 32U} &&
                tile.get_face_shape() == tt::tt_metal::Tile::FaceShape{16U, 16U} && !tile.get_transpose_within_face() &&
                !tile.get_transpose_of_faces(),
            "Tensor '{}' must use the canonical 32x32 TILE page with 16x16 faces and no transposition; got {}",
            name,
            tile);

        // Logical shapes must match for element-for-element correspondence with the parameter;
        // padding alone cannot tell apart tensors that round up to the same tile extent.
        TT_FATAL(
            tensor.logical_shape() == param.logical_shape(),
            "Tensor '{}' must match the parameter's logical shape. Expected {}, got {}",
            name,
            param.logical_shape(),
            tensor.logical_shape());

        // Tile counts and reader/writer extents are derived solely from the parameter tensor, so any
        // smaller companion tensor would be read or written past its allocation.
        TT_FATAL(
            tensor.padded_shape() == param.padded_shape(),
            "Tensor '{}' must match the parameter's padded shape. Expected {}, got {}",
            name,
            param.padded_shape(),
            tensor.padded_shape());
    };

    const auto& grad = tensor_args.grad;
    const auto& momentum_buffer = tensor_args.momentum_buffer;
    check_tensor(param, "Parameter", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(grad, "Gradient", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    if (momentum_buffer.has_value()) {
        check_tensor(
            momentum_buffer.value(), "Momentum Buffer", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    }

    const auto momentum = args.momentum;
    const auto use_momentum = (momentum > 0.0F);
    TT_FATAL(
        momentum_buffer.has_value() == use_momentum,
        "Momentum buffer presence must match positive momentum. Got momentum {} and momentum buffer present={}",
        momentum,
        momentum_buffer.has_value());
    TT_FATAL(
        args.dampening == 0.0F || use_momentum,
        "Dampening requires positive momentum. Got dampening {} and momentum {}",
        args.dampening,
        momentum);
    TT_FATAL(
        !args.nesterov || (use_momentum && args.dampening == 0.0F),
        "Nesterov requires positive momentum and zero dampening. Got momentum {} and dampening {}",
        momentum,
        args.dampening);
}

SGDDeviceOperation::spec_return_value_t SGDDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param.tensor_spec();
}

SGDDeviceOperation::tensor_return_value_t SGDDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param;
}

std::vector<tt::tt_metal::TensorTopology> SGDDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.param.tensor_topology()};
}

ttsl::hash::hash_t SGDDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_tensor = tensor_args.param;
    const auto& param_logical_shape = param_tensor.logical_shape();
    auto nesterov = args.nesterov;
    auto momentum_initialized = tensor_args.momentum_buffer.has_value();
    auto hash = tt::tt_metal::operation::hash_operation<SGDDeviceOperation>(
        nesterov, momentum_initialized, param_tensor.dtype(), param_logical_shape);

    return hash;
}

}  // namespace ttml::metal::optimizers::sgd::device

namespace ttnn::prim {

ttml::metal::optimizers::sgd::device::SGDDeviceOperation::tensor_return_value_t sgd(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer) {
    using OperationType = ttml::metal::optimizers::sgd::device::SGDDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .lr = lr,
        .momentum = momentum,
        .dampening = dampening,
        .weight_decay = weight_decay,
        .nesterov = nesterov,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .param = param,
        .grad = grad,
        .momentum_buffer = momentum_buffer,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
