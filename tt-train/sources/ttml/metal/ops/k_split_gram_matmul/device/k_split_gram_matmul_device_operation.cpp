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
    const operation_attributes_t&, const tensor_args_t& ta) {
    uint32_t K_tiles = ta.input_tensor.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(K_tiles % 2 == 0, "K dimension ({} tiles) must be even for K-split", K_tiles);
}

void KSplitGramMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {
}

KSplitGramMatmulDeviceOperation::spec_return_value_t KSplitGramMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& ta) {
    // Output: [1, 1, padded_M, (grid_dim+1)*Mpc*TILE_W] — extra column for helper partials
    auto* device = ta.input_tensor.device();
    auto device_grid = device->compute_with_storage_grid_size();
    uint32_t grid_dim = static_cast<uint32_t>(std::min({device_grid.x, device_grid.y, (std::size_t)10}));
    uint32_t M_tiles = ta.input_tensor.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    uint32_t Mpc = tt::round_up(M_tiles, grid_dim) / grid_dim;
    uint32_t padded_M = grid_dim * Mpc * tt::constants::TILE_HEIGHT;
    uint32_t padded_N = (grid_dim + 1) * Mpc * tt::constants::TILE_WIDTH;
    auto shape = ttnn::Shape({1, 1, padded_M, padded_N});
    return ttnn::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
}

KSplitGramMatmulDeviceOperation::tensor_return_value_t KSplitGramMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& oa, const tensor_args_t& ta) {
    return create_device_tensor(compute_output_specs(oa, ta), ta.input_tensor.device());
}

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
