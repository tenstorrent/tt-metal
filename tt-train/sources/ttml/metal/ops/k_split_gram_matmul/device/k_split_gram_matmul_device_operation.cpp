// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "k_split_gram_matmul_device_operation.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

KSplitGramMatmulDeviceOperation::program_factory_t KSplitGramMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KSplitGramMatmulProgramFactory{};
}

void KSplitGramMatmulDeviceOperation::validate_on_program_cache_miss(
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
    TT_FATAL(rank == 2 || rank == 4, "Input tensor must be 2D [M, K] or 4D [1, 1, M, K]");
    if (rank == 4) {
        TT_FATAL(
            input.logical_shape()[0] == 1 && input.logical_shape()[1] == 1,
            "Batch dimensions must be 1, got [{}, {}]",
            input.logical_shape()[0],
            input.logical_shape()[1]);
    }
    const uint32_t K_tiles = input.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(K_tiles % 2 == 0, "K dimension ({} tiles) must be even for K-split", K_tiles);
}

KSplitGramMatmulDeviceOperation::spec_return_value_t KSplitGramMatmulDeviceOperation::compute_output_specs(
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

KSplitGramMatmulDeviceOperation::tensor_return_value_t KSplitGramMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t KSplitGramMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<KSplitGramMatmulDeviceOperation>(
        operation_attributes.output_mode, operation_attributes.math_fidelity, tensor_args.input_tensor.logical_shape());
}

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
