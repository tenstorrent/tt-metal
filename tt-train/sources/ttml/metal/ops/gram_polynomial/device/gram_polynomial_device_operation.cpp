// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial_device_operation.hpp"

namespace ttml::metal::ops::gram_polynomial::device {

GramPolynomialDeviceOperation::program_factory_t GramPolynomialDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GramPolynomialProgramFactory{};
}

void GramPolynomialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must be allocated on device");
    TT_FATAL(input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Input tensor must be in DRAM");
    TT_FATAL(input.layout() == tt::tt_metal::Layout::TILE, "Input tensor must have TILE layout");
    TT_FATAL(input.dtype() == ttnn::DataType::BFLOAT16, "Input tensor must be BFLOAT16");
    TT_FATAL(
        input.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
        "Input tensor must use INTERLEAVED memory layout");
    const auto rank = input.logical_shape().rank();
    TT_FATAL(rank == 2 || rank == 4, "Input tensor must be 2D [M, M] or 4D [1, 1, M, M]");
    if (rank == 4) {
        TT_FATAL(
            input.logical_shape()[0] == 1 && input.logical_shape()[1] == 1,
            "Batch dimensions must be 1, got [{}, {}]",
            input.logical_shape()[0],
            input.logical_shape()[1]);
    }
    TT_FATAL(
        input.logical_shape()[-2] == input.logical_shape()[-1],
        "Input G must be square, got [{}, {}]",
        input.logical_shape()[-2],
        input.logical_shape()[-1]);
    const uint32_t M_tiles = input.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    TT_FATAL(M_tiles % 2 == 0, "M dimension ({} tiles) must be even for K-split", M_tiles);
    const auto device_grid = input.device()->compute_with_storage_grid_size();
    const uint32_t grid_dim = static_cast<uint32_t>(std::min(device_grid.x - 1, device_grid.y));
    TT_FATAL(grid_dim >= 3, "Device grid too small for gram polynomial (need at least 4x3 compute grid)");

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output = tensor_args.preallocated_output.value();
        const uint32_t M = input.logical_shape()[-2];
        TT_FATAL(output.storage_type() == tt::tt_metal::StorageType::DEVICE, "Preallocated output must be on device");
        TT_FATAL(output.buffer() != nullptr, "Preallocated output must be allocated on device");
        TT_FATAL(
            output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Preallocated output must be in DRAM");
        TT_FATAL(output.layout() == tt::tt_metal::Layout::TILE, "Preallocated output must have TILE layout");
        TT_FATAL(output.dtype() == ttnn::DataType::BFLOAT16, "Preallocated output must be BFLOAT16");
        TT_FATAL(
            output.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Preallocated output must use INTERLEAVED memory layout");
        TT_FATAL(
            output.logical_shape()[-2] == M && output.logical_shape()[-1] == M,
            "Preallocated output shape must be [{}, {}], got [{}, {}]",
            M,
            M,
            output.logical_shape()[-2],
            output.logical_shape()[-1]);
    }
}

GramPolynomialDeviceOperation::spec_return_value_t GramPolynomialDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const uint32_t M = tensor_args.input_tensor.logical_shape()[-2];
    auto shape = ttnn::Shape({1, 1, M, M});
    return ttnn::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
}

GramPolynomialDeviceOperation::tensor_return_value_t GramPolynomialDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t GramPolynomialDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<GramPolynomialDeviceOperation>(
        operation_attributes.b,
        operation_attributes.c,
        operation_attributes.output_mode,
        operation_attributes.math_fidelity,
        tensor_args.input_tensor.logical_shape());
}

}  // namespace ttml::metal::ops::gram_polynomial::device
