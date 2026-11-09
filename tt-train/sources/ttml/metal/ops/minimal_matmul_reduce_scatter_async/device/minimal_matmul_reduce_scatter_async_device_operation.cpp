// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_reduce_scatter_async_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "metal/common/tensor_validation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
namespace {
constexpr std::string_view kOp = "minimal_matmul_reduce_scatter_async";

void check_leading_dims_are_one(const ttnn::Shape& shape, std::string_view name) {
    for (int i = 0; i < static_cast<int>(shape.rank()) - 2; ++i) {
        TT_FATAL(
            shape[i] == 1,
            "{}: every dim of {} before the last two must be 1 (the rows are scattered as one [M, N] matrix), but dim "
            "{} is {} (shape {}).",
            kOp,
            name,
            i,
            shape[i],
            shape);
    }
}

void validate_config(
    const MinimalMatmulConfig& cfg, const ttnn::CoreCoord& device_grid, const ttnn::DeviceComputeKernelConfig& ckc) {
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
    TT_FATAL(
        cfg.compute_with_storage_grid_size.x <= device_grid.x && cfg.compute_with_storage_grid_size.y <= device_grid.y,
        "compute_with_storage_grid_size ({}x{}) must fit in the device grid ({}x{}).",
        cfg.compute_with_storage_grid_size.x,
        cfg.compute_with_storage_grid_size.y,
        device_grid.x,
        device_grid.y);
    TT_FATAL(
        cfg.subblock_h * cfg.subblock_w <= get_dest_reg_count(ckc),
        "Subblock {}x{} needs {} destination registers, but this compute_kernel_config has {}. Use a smaller "
        "subblock_h or subblock_w.",
        cfg.subblock_h,
        cfg.subblock_w,
        cfg.subblock_h * cfg.subblock_w,
        get_dest_reg_count(ckc));
}
}  // namespace

MinimalMatmulReduceScatterAsyncDeviceOperation::program_factory_t
MinimalMatmulReduceScatterAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MinimalMatmulReduceScatterAsyncProgramFactory{};
}

void MinimalMatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& w = tensor_args.weight_tensor;
    const auto& buffer = tensor_args.persistent_buffer;
    const DeviceTensorRequirements req{.dtypes = {DataType::BFLOAT16}, .memory_layout = std::nullopt};
    check_device_tensor(x, kOp, "input", req);
    check_device_tensor(w, kOp, "weight", req);
    check_device_tensor(
        buffer,
        kOp,
        "persistent_buffer",
        {.dtypes = {DataType::BFLOAT16},
         .memory_layout = TensorMemoryLayout::INTERLEAVED,
         .buffer_type = BufferType::DRAM});
    check_same_device(w, x, kOp, "weight", "input");
    check_same_device(buffer, x, kOp, "persistent_buffer", "input");

    const auto& x_shape = x.logical_shape();
    const auto& w_shape = w.logical_shape();
    const auto& buffer_shape = buffer.logical_shape();
    TT_FATAL(
        x_shape.rank() >= 2 && w_shape.rank() >= 2 && buffer_shape.rank() >= 2,
        "Input, weight and persistent_buffer must have rank >= 2, got ranks {}, {} and {}.",
        x_shape.rank(),
        w_shape.rank(),
        buffer_shape.rank());
    check_leading_dims_are_one(x_shape, "input");
    check_leading_dims_are_one(w_shape, "weight");
    check_leading_dims_are_one(buffer_shape, "persistent_buffer");

    const int n_dim = attrs.transpose_b ? -2 : -1;
    const int k_dim = attrs.transpose_b ? -1 : -2;
    const uint32_t M = x_shape[attrs.transpose_a ? -1 : -2];
    const uint32_t K = x_shape[attrs.transpose_a ? -2 : -1];
    const uint32_t N = w_shape[n_dim];
    TT_FATAL(
        K == w_shape[k_dim],
        "inner dimensions must match: input K={} vs weight K={} (transpose_a={}, transpose_b={})",
        K,
        w_shape[k_dim],
        attrs.transpose_a,
        attrs.transpose_b);
    TT_FATAL(M > 0 && K > 0 && N > 0, "Matmul dims must be positive, got M={}, K={}, N={}.", M, K, N);

    TT_FATAL(
        M % (attrs.ring_size * TILE_HEIGHT) == 0,
        "M ({}) must be a multiple of ring_size * {} = {}, so every device keeps a tile-aligned slice of rows",
        M,
        TILE_HEIGHT,
        attrs.ring_size * TILE_HEIGHT);
    TT_FATAL(N % TILE_WIDTH == 0, "N ({}) must be a multiple of {}", N, TILE_WIDTH);
    TT_FATAL(
        buffer_shape[-2] == M && buffer_shape[-1] == N,
        "persistent_buffer must hold the full [M, N] = [{}, {}] product, got [{}, {}]",
        M,
        N,
        buffer_shape[-2],
        buffer_shape[-1]);
    const DataType out_dtype = attrs.output_dtype.value_or(x.dtype());
    TT_FATAL(
        out_dtype == DataType::BFLOAT16 && buffer.dtype() == out_dtype,
        "The running sums travel in the output dtype, which must be BFLOAT16 like persistent_buffer; got output {} "
        "and persistent_buffer {}",
        out_dtype,
        buffer.dtype());

    const ttnn::CoreCoord device_grid = x.device()->compute_with_storage_grid_size();

    if (attrs.config.has_value()) {
        validate_config(attrs.config.value(), device_grid, attrs.compute_kernel_config);
    }
}

MinimalMatmulReduceScatterAsyncDeviceOperation::spec_return_value_t
MinimalMatmulReduceScatterAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    ttnn::Shape output_shape(x.logical_shape());
    output_shape[-2] = x.logical_shape()[attrs.transpose_a ? -1 : -2] / attrs.ring_size;
    output_shape[-1] = tensor_args.weight_tensor.logical_shape()[attrs.transpose_b ? -2 : -1];
    const auto& memory_config = attrs.output_mem_config.value_or(x.memory_config());
    const auto dtype = attrs.output_dtype.value_or(x.dtype());
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

MinimalMatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
MinimalMatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async

namespace ttnn::prim {
ttnn::Tensor ttml_minimal_matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_buffer,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool transpose_a,
    bool transpose_b) {
    using OperationType =
        ttml::metal::ops::minimal_matmul_reduce_scatter_async::MinimalMatmulReduceScatterAsyncDeviceOperation;
    auto* device = input_tensor.device();
    const auto& mesh_shape = device->shape();
    TT_FATAL(
        !cluster_axis.has_value() || cluster_axis.value() < mesh_shape.dims(),
        "cluster_axis ({}) must be < the mesh rank ({})",
        cluster_axis.value_or(0),
        mesh_shape.dims());
    const uint32_t ring_size =
        cluster_axis.has_value() ? mesh_shape[cluster_axis.value()] : static_cast<uint32_t>(device->num_devices());
    const auto arch = device->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true, true);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val,
            .ring_size = ring_size,
            .cluster_axis = cluster_axis,
            .transpose_a = transpose_a,
            .transpose_b = transpose_b},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .weight_tensor = weight_tensor, .persistent_buffer = persistent_buffer});
}
}  // namespace ttnn::prim
