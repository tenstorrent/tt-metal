// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "gram_matmul_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttml::metal::ops::gram_matmul::device {

GramMatmulDeviceOperation::program_factory_t GramMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return GramMatmulProgramFactory{};
}

void GramMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void GramMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "gram_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "gram_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "gram_matmul inputs must be allocated in device buffers");

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "gram_matmul requires TILE layout for both inputs");

    // DType constraint: only BFLOAT16 is supported for both inputs
    TT_FATAL(
        act_tensor.dtype() == DataType::BFLOAT16 && weight_tensor.dtype() == DataType::BFLOAT16,
        "gram_matmul supports only BFLOAT16 for inputs, got act={} weight={}",
        act_tensor.dtype(),
        weight_tensor.dtype());

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "gram_matmul expects rank >= 2 tensors");

    // Allow upper-dim broadcasting on activation (LHS): activation may have arbitrary upper dims
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(w_logical[i] == 1, "gram_matmul weight must have 1 in all dims < -2");
    }

    const uint32_t M = a_logical[-2];
    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "gram_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "gram_matmul dimensions must be positive");

    // Gram matrix constraints:
    // - Input A is wide: cols >= rows (K >= M)
    // - Weight B is the transpose: rows >= cols (K_w >= N)
    // - Output is always square: M == N
    TT_FATAL(K >= M, "gram_matmul expects input A to have cols >= rows (K={} >= M={})", K, M);
    TT_FATAL(K_w >= N, "gram_matmul expects weight B to have rows >= cols (K={} >= N={})", K_w, N);
    TT_FATAL(M == N, "gram_matmul output must be square (M={} must equal N={})", M, N);

    // Tile alignment checks (implicitly guaranteed by TILE layout, but assert inner two dims are tile-aligned)
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "gram_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0, "gram_matmul weight must be tile-aligned");

    // Config constraints
    if (config.has_value()) {
        const auto& cfg = config.value();
        TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");
        TT_FATAL(cfg.subblock_h > 0 && cfg.subblock_w > 0, "Subblock sizes must be > 0");
        TT_FATAL(
            (cfg.M_block_size % cfg.subblock_h) == 0,
            "M_block_size ({}) must be divisible by subblock_h ({})",
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            (cfg.N_block_size % cfg.subblock_w) == 0,
            "N_block_size ({}) must be divisible by subblock_w ({})",
            cfg.N_block_size,
            cfg.subblock_w);

        // Grid must be at least 2x2
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        // Additional grid checks are performed when creating the program
        auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size must be <= device grid size");

        const uint32_t max_dest_volume = get_dest_reg_count(operation_attributes.compute_kernel_config);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
    }
}

GramMatmulDeviceOperation::spec_return_value_t GramMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& in0_input_tensor = tensor_args.input_tensor;
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();

    // Output is always square: [M, M]
    uint32_t M = in0_input_tensor_shape[-2];
    ttnn::Shape output_shape(in0_input_tensor_shape);
    output_shape[-1] = M;

    const auto& memory_config = operation_attributes.output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = operation_attributes.output_dtype.value_or(in0_input_tensor.dtype());

    return ttnn::TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), memory_config));
}

GramMatmulDeviceOperation::tensor_return_value_t GramMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttml::metal::ops::gram_matmul::device

namespace ttnn::prim {

ttml::metal::ops::gram_matmul::device::GramMatmulDeviceOperation::tensor_return_value_t ttml_gram_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const ttml::metal::ops::gram_matmul::device::GramMatmulConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttml::metal::ops::gram_matmul::device::GramMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor});
}

}  // namespace ttnn::prim
