// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_weight_minimal_matmul_async_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "metal/common/tensor_validation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
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

    const auto& pwb = tensor_args.persistent_weight_buffer;
    constexpr std::string_view kOp = "all_gather_weight_minimal_matmul_async";
    const DeviceTensorRequirements req{
        .dtypes = {DataType::BFLOAT16, DataType::FLOAT32}, .memory_layout = std::nullopt};
    check_device_tensor(x, kOp, "input", req);
    check_device_tensor(w, kOp, "weight", req);
    check_device_tensor(pwb, kOp, "persistent_weight_buffer", req);
    check_same_device(w, x, kOp, "weight", "input");
    check_same_device(pwb, x, kOp, "persistent_weight_buffer", "input");

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
    const int n_dim = attrs.transpose_b ? -2 : -1;
    const int k_dim = attrs.transpose_b ? -1 : -2;
    TT_FATAL(
        x_shape[-1] == w_shape[k_dim],
        "inner dimensions must match: input K={} vs weight K={} (transpose_b={})",
        x_shape[-1],
        w_shape[k_dim],
        attrs.transpose_b);
    TT_FATAL(
        x_shape[-2] > 0 && x_shape[-1] > 0 && w_shape[n_dim] > 0,
        "Matmul dims must be positive, got M={}, K={}, N={}.",
        x_shape[-2],
        x_shape[-1],
        w_shape[n_dim]);

    TT_FATAL(
        attrs.topology == ttnn::ccl::Topology::Ring || attrs.topology == ttnn::ccl::Topology::Linear,
        "Unsupported topology {}: the op supports Ring and Linear.",
        attrs.topology);

    TT_FATAL(
        pwb.dtype() == w.dtype(),
        "persistent_weight_buffer dtype ({}) must match the weight dtype ({}).",
        pwb.dtype(),
        w.dtype());
    const auto& pwb_shape = pwb.logical_shape();
    TT_FATAL(
        pwb_shape[k_dim] == w_shape[k_dim] && pwb_shape[n_dim] == w_shape[n_dim] * attrs.ring_size,
        "persistent_weight_buffer must be the full weight (N = {}, K = {}) for a weight shard [{}, {}] on a "
        "ring of {} (transpose_b={}), got [{}, {}]",
        w_shape[n_dim] * attrs.ring_size,
        w_shape[k_dim],
        w_shape[-2],
        w_shape[-1],
        attrs.ring_size,
        attrs.transpose_b,
        pwb_shape[-2],
        pwb_shape[-1]);
    TT_FATAL(w_shape[n_dim] % TILE_WIDTH == 0, "weight shard N ({}) must be tile-aligned", w_shape[n_dim]);
    TT_FATAL(
        attrs.ring_size >= 2,
        "The fabric gather needs a ring of at least 2 devices, got ring_size={}.",
        attrs.ring_size);
    TT_FATAL(
        attrs.semaphores.size() == attrs.ring_size,
        "multi_device_global_semaphore must hold one semaphore per ring device ({}), got {}",
        attrs.ring_size,
        attrs.semaphores.size());
    const ttnn::CoreCoord device_grid = x.device()->compute_with_storage_grid_size();
    const ttnn::CoreRange all_cores({0, 0}, {device_grid.x - 1, device_grid.y - 1});
    for (const auto& sem : attrs.semaphores) {
        TT_FATAL(
            std::get<0>(sem.attribute_values()).contains(all_cores),
            "multi_device_global_semaphore must be created on the whole compute grid {}",
            all_cores.str());
    }
    TT_FATAL(attrs.num_links >= 1, "num_links must be >= 1, got {}.", attrs.num_links);
    TT_FATAL(
        attrs.num_workers_per_link >= 1 && attrs.num_workers_per_link <= 255,
        "num_workers_per_link must be in [1, 255], got {}",
        attrs.num_workers_per_link);
    TT_FATAL(
        attrs.num_buffers_per_channel >= 1 && attrs.num_buffers_per_channel <= 255,
        "num_buffers_per_channel must be in [1, 255], got {}",
        attrs.num_buffers_per_channel);
    TT_FATAL(
        attrs.topology != ttnn::ccl::Topology::Ring ||
            (attrs.cluster_axis.has_value() && ttnn::ccl::is_axis_wrap_wired(*x.device(), attrs.cluster_axis.value())),
        "topology=Ring needs cluster_axis set and its wrap-around link wired on this mesh; use Topology.Linear");

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
    output_shape[-1] = tensor_args.persistent_weight_buffer.logical_shape()[attrs.transpose_b ? -2 : -1];
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
    const ttnn::Tensor& persistent_weight_buffer,
    const std::vector<ttnn::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    bool transpose_b) {
    using OperationType =
        ttml::metal::ops::all_gather_weight_minimal_matmul_async::AllGatherWeightMinimalMatmulAsyncDeviceOperation;
    const auto& mesh_shape = input_tensor.device()->shape();
    TT_FATAL(
        !cluster_axis.has_value() || cluster_axis.value() < mesh_shape.dims(),
        "cluster_axis ({}) must be < the mesh rank ({})",
        cluster_axis.value_or(0),
        mesh_shape.dims());
    const uint32_t ring_size = cluster_axis.has_value() ? mesh_shape[cluster_axis.value()]
                                                        : static_cast<uint32_t>(input_tensor.device()->num_devices());
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
            .topology = topology,
            .cluster_axis = cluster_axis,
            .num_links = num_links,
            .num_workers_per_link = num_workers_per_link,
            .num_buffers_per_channel = num_buffers_per_channel,
            .semaphores = multi_device_global_semaphore,
            .transpose_b = transpose_b},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .persistent_weight_buffer = persistent_weight_buffer});
}
}  // namespace ttnn::prim
