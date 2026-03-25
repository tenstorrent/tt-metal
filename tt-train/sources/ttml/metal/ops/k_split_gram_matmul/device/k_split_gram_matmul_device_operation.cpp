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
    uint32_t K_tiles = tensor_args.input_tensor.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(K_tiles % 2 == 0, "K dimension ({} tiles) must be even for K-split", K_tiles);
}

void KSplitGramMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {
}

KSplitGramMatmulDeviceOperation::spec_return_value_t KSplitGramMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    uint32_t M = tensor_args.input_tensor.logical_shape()[-2];
    auto shape = ttnn::Shape({1, 1, M, M});
    return ttnn::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
}

KSplitGramMatmulDeviceOperation::tensor_return_value_t KSplitGramMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
