// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
AllGatherWeightMinimalMatmulAsyncDeviceOperation::program_factory_t
AllGatherWeightMinimalMatmulAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllGatherWeightMinimalMatmulAsyncProgramFactory{};
}

void AllGatherWeightMinimalMatmulAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& w = tensor_args.weight_tensor;

    TT_FATAL(
        x.storage_type() == ttnn::StorageType::DEVICE && w.storage_type() == ttnn::StorageType::DEVICE,
        "Input and weight must be device tensors, got storage types {} and {}. Move them to the device first.",
        x.storage_type(),
        w.storage_type());
    TT_FATAL(
        x.device() == w.device(),
        "Input and weight must be on the same device: the input is on device {} and the weight on device {}.",
        x.device()->id(),
        w.device()->id());
    TT_FATAL(
        x.buffer() != nullptr && w.buffer() != nullptr,
        "Input and weight must be allocated device tensors: the input buffer is {} and the weight buffer is {}.",
        x.buffer() != nullptr ? "allocated" : "null",
        w.buffer() != nullptr ? "allocated" : "null");
    TT_FATAL(
        x.layout() == Layout::TILE && w.layout() == Layout::TILE,
        "Input and weight must be in TILE layout, got {} and {}. Convert them with ttnn::to_layout first.",
        x.layout(),
        w.layout());

    auto dtype_supported = [](DataType dt) { return dt == DataType::BFLOAT16 || dt == DataType::FLOAT32; };
    TT_FATAL(
        dtype_supported(x.dtype()) && dtype_supported(w.dtype()),
        "Unsupported dtypes: input {} and weight {}. The op supports BFLOAT16 and FLOAT32.",
        x.dtype(),
        w.dtype());

    const auto& x_shape = x.logical_shape();
    const auto& w_shape = w.logical_shape();
    TT_FATAL(
        x_shape.rank() >= 2 && w_shape.rank() >= 2,
        "Input and weight must have rank >= 2, got ranks {} and {}.",
        x_shape.rank(),
        w_shape.rank());
    for (int i = 0; i < static_cast<int>(w_shape.rank()) - 2; ++i) {
        TT_FATAL(
            w_shape[i] == 1,
            "The weight must be a matrix: every dim before the last two must be 1, but dim {} is {} (shape {}).",
            i,
            w_shape[i],
            w_shape);
    }
    TT_FATAL(
        x_shape[-1] == w_shape[-1],
        "inner dimensions must match: input K={} vs weight [N, K] K={}",
        x_shape[-1],
        w_shape[-1]);
    TT_FATAL(
        x_shape[-2] > 0 && x_shape[-1] > 0 && w_shape[-2] > 0,
        "Matmul dims must be positive, got M={}, K={}, N={}.",
        x_shape[-2],
        x_shape[-1],
        w_shape[-2]);

    TT_FATAL(attrs.ring_size >= 1, "ring_size must be >= 1, got {}.", attrs.ring_size);
    TT_FATAL(
        attrs.ring_index < attrs.ring_size,
        "ring_index ({}) must be < ring_size ({}).",
        attrs.ring_index,
        attrs.ring_size);
    TT_FATAL(
        attrs.topology == ttnn::ccl::Topology::Ring || attrs.topology == ttnn::ccl::Topology::Linear,
        "Unsupported topology {}: the op supports Ring and Linear.",
        attrs.topology);
    const uint32_t N_tiles = w.padded_shape()[-2] / TILE_HEIGHT;
    TT_FATAL(
        w_shape[-2] % TILE_HEIGHT == 0 && N_tiles % attrs.ring_size == 0,
        "weight N={} must be a multiple of 32 * ring_size ({}) so every device owns a tile-aligned stripe",
        w_shape[-2],
        attrs.ring_size);

    if (attrs.config.has_value()) {
        const auto& cfg = attrs.config.value();
        TT_FATAL(
            cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0,
            "Block sizes must be > 0, got M_block_size={}, K_block_size={}, N_block_size={}.",
            cfg.M_block_size,
            cfg.K_block_size,
            cfg.N_block_size);
        TT_FATAL(
            cfg.subblock_h > 0 && cfg.subblock_w > 0,
            "Subblock sizes must be > 0, got subblock_h={}, subblock_w={}.",
            cfg.subblock_h,
            cfg.subblock_w);
        TT_FATAL(
            cfg.M_block_size % cfg.subblock_h == 0,
            "M_block_size ({}) must be divisible by subblock_h ({}).",
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            cfg.N_block_size % cfg.subblock_w == 0,
            "N_block_size ({}) must be divisible by subblock_w ({}).",
            cfg.N_block_size,
            cfg.subblock_w);
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be at least 2x2, got {}x{}.",
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y);
        const auto device_grid = x.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size ({}x{}) must fit in the device grid ({}x{}).",
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y,
            device_grid.x,
            device_grid.y);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= get_dest_reg_count(attrs.compute_kernel_config),
            "Subblock {}x{} needs {} destination registers, but this compute_kernel_config has {}. Use a smaller "
            "subblock_h or subblock_w.",
            cfg.subblock_h,
            cfg.subblock_w,
            cfg.subblock_h * cfg.subblock_w,
            get_dest_reg_count(attrs.compute_kernel_config));
    }
}

AllGatherWeightMinimalMatmulAsyncDeviceOperation::spec_return_value_t
AllGatherWeightMinimalMatmulAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    ttnn::Shape output_shape(x.logical_shape());
    output_shape[-1] = tensor_args.weight_tensor.logical_shape()[-2];
    const auto& memory_config = attrs.output_mem_config.value_or(x.memory_config());
    const auto dtype = attrs.output_dtype.value_or(x.dtype());
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

AllGatherWeightMinimalMatmulAsyncDeviceOperation::tensor_return_value_t
AllGatherWeightMinimalMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async

namespace ttnn::prim {
ttnn::Tensor ttml_all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType =
        ttml::metal::ops::all_gather_weight_minimal_matmul_async::AllGatherWeightMinimalMatmulAsyncDeviceOperation;
    const auto arch = input_tensor.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true, true);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val,
            .ring_size = ring_size,
            .ring_index = ring_index,
            .topology = topology},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor});
}
}  // namespace ttnn::prim
