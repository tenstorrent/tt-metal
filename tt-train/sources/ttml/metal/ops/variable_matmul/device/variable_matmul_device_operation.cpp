// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "variable_matmul_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttml::metal::ops::variable_matmul::device {

void VariableMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "variable_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "variable_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "variable_matmul inputs must be allocated in device buffers");

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "variable_matmul requires TILE layout for activation and weight");

    // DType constraints
    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "variable_matmul supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "variable_matmul expects rank >= 2 tensors");

    const uint32_t M = a_logical[-2];
    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "variable_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "variable_matmul dimensions must be positive");

    // Variable-M constraint
    TT_FATAL(
        M <= operation_attributes.max_M,
        "variable_matmul actual M ({}) exceeds max_M ({})",
        M,
        operation_attributes.max_M);
    TT_FATAL(
        M % TILE_HEIGHT == 0, "variable_matmul actual M ({}) must be a multiple of TILE_HEIGHT ({})", M, TILE_HEIGHT);
    TT_FATAL(
        operation_attributes.max_M % TILE_HEIGHT == 0,
        "variable_matmul max_M ({}) must be a multiple of TILE_HEIGHT ({})",
        operation_attributes.max_M,
        TILE_HEIGHT);

    // Tile alignment checks
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul weight must be tile-aligned");

    // Config constraints
    const auto& cfg = config;
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

    auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        cfg.compute_with_storage_grid_size.x <= device_grid.x && cfg.compute_with_storage_grid_size.y <= device_grid.y,
        "compute_with_storage_grid_size must be <= device grid size");

    const uint32_t max_dest_volume = get_dest_reg_count(operation_attributes.compute_kernel_config);
    TT_FATAL(cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
}

VariableMatmulDeviceOperation::spec_return_value_t VariableMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    const uint32_t N = in1.logical_shape()[-1];

    const auto& memory_config = in0.memory_config();
    auto dtype = in0.dtype();

    ttnn::Shape output_shape(in0.logical_shape());
    output_shape[-1] = N;
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

VariableMatmulDeviceOperation::tensor_return_value_t VariableMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return create_device_tensor(output_spec, device);
}

ttsl::hash::hash_t VariableMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Two-program caching: transpose_core_grid is determined by actual_M vs N.
    // This gives at most 2 cached programs: one for actual_M <= N, one for actual_M > N.
    // Each program uses an effective_max_M appropriate for its transpose layout.
    const auto& w = tensor_args.weight_tensor;
    const auto& a = tensor_args.input_tensor;
    uint32_t actual_M = a.physical_volume() / a.padded_shape()[-1];
    uint32_t N = w.logical_shape()[-1];
    bool transpose_core_grid = actual_M > N;

    // effective_max_M: for non-transposed (M <= N), cap at N since M never exceeds N in this variant.
    // For transposed (M > N), use the full max_M.
    uint32_t effective_max_M =
        transpose_core_grid ? operation_attributes.max_M : std::min(operation_attributes.max_M, N);

    return ttsl::hash::hash_objects_with_default_seed(
        effective_max_M,
        transpose_core_grid,
        operation_attributes.config.M_block_size,
        operation_attributes.config.K_block_size,
        operation_attributes.config.N_block_size,
        operation_attributes.config.subblock_h,
        operation_attributes.config.subblock_w,
        operation_attributes.config.compute_with_storage_grid_size.x,
        operation_attributes.config.compute_with_storage_grid_size.y,
        operation_attributes.compute_kernel_config,
        a.dtype(),
        w.dtype(),
        w.logical_shape());
}

}  // namespace ttml::metal::ops::variable_matmul::device

namespace ttnn::prim {

ttnn::Tensor ttml_variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    uint32_t max_M,
    const ttml::metal::ops::variable_matmul::device::VariableMatmulConfig& config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttml::metal::ops::variable_matmul::device::VariableMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .max_M = max_M, .config = config, .compute_kernel_config = kernel_config_val},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor});
}

}  // namespace ttnn::prim
