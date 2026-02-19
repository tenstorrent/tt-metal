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
    const auto& input = tensor_args.input_tensor;
    const auto& config = operation_attributes.config;

    // Basic device/storage checks
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "gram_matmul input must be on device");
    TT_FATAL(input.buffer() != nullptr, "gram_matmul input must be allocated in a device buffer");

    // Layout
    TT_FATAL(input.layout() == Layout::TILE, "gram_matmul requires TILE layout");

    // DType constraint: only BFLOAT16
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "gram_matmul supports only BFLOAT16, got {}", input.dtype());

    // Shape constraints
    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() >= 2, "gram_matmul expects rank >= 2 tensor");

    const uint32_t M = shape[-2];
    const uint32_t K = shape[-1];

    TT_FATAL(M > 0 && K > 0, "gram_matmul dimensions must be positive");
    TT_FATAL(K >= M, "gram_matmul expects cols >= rows (K={} >= M={})", K, M);

    // Tile alignment
    const auto& padded = input.padded_shape();
    TT_FATAL(padded[-2] % TILE_HEIGHT == 0 && padded[-1] % TILE_WIDTH == 0, "gram_matmul input must be tile-aligned");

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

        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        auto device_grid = input.device()->compute_with_storage_grid_size();
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
    const auto& input = tensor_args.input_tensor;
    const auto& input_shape = input.logical_shape();

    // Output is square: [..., M, M]
    uint32_t M = input_shape[-2];
    ttnn::Shape output_shape(input_shape);
    output_shape[-1] = M;

    const auto& memory_config = operation_attributes.output_mem_config.value_or(input.memory_config());
    auto dtype = operation_attributes.output_dtype.value_or(input.dtype());

    return ttnn::TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), memory_config));
}

GramMatmulDeviceOperation::tensor_return_value_t GramMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttml::metal::ops::gram_matmul::device

namespace ttnn::prim {

ttml::metal::ops::gram_matmul::device::GramMatmulDeviceOperation::tensor_return_value_t ttml_gram_matmul(
    const Tensor& input_tensor,
    const std::optional<const ttml::metal::ops::gram_matmul::device::GramMatmulConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& output) {
    using OperationType = ttml::metal::ops::gram_matmul::device::GramMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .output_tensor = output});
}

}  // namespace ttnn::prim
