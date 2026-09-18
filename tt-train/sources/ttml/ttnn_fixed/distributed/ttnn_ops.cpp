// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/all_gather_matmul_sp_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/matmul_reduce_scatter_sp_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ttnn_fixed::distributed {

namespace {

// Convert FabricConfig to FabricType (local implementation to avoid internal header dependency)
tt::tt_fabric::FabricType get_fabric_type_from_config(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return tt::tt_fabric::FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return tt::tt_fabric::FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return tt::tt_fabric::FabricType::TORUS_XY;
        default: return tt::tt_fabric::FabricType::MESH;
    }
}

// Helper function to determine if a cluster axis is a ring (has wraparound connections)
// cluster_axis: 0 = Y axis (rows, N-S direction), 1 = X axis (columns, E-W direction)
bool is_cluster_axis_ring(uint32_t cluster_axis) {
    auto fabric_config = tt::tt_fabric::GetFabricConfig();
    auto fabric_type = get_fabric_type_from_config(fabric_config);

    if (cluster_axis == 0) {
        // Y axis (rows) - check for TORUS_Y
        return tt::tt_fabric::has_flag(fabric_type, tt::tt_fabric::FabricType::TORUS_Y);
    } else if (cluster_axis == 1) {
        // X axis (columns) - check for TORUS_X
        return tt::tt_fabric::has_flag(fabric_type, tt::tt_fabric::FabricType::TORUS_X);
    }
    return false;
}

// Get the appropriate CCL topology based on cluster axis ring status
// The options are Linear, Ring, based on where the fabric was initialized with
// wrap around connections along the axis.
ttnn::ccl::Topology get_topology(const std::optional<uint32_t>& cluster_axis) {
    if (!cluster_axis.has_value()) {
        auto* mesh_device = &ttml::autograd::ctx().get_device();
        const auto& mesh_shape = mesh_device->shape();

        TT_FATAL(
            mesh_shape.is_line_topology(),
            "cluster_axis must be specified for non-line mesh topologies. "
            "Mesh shape {} has multiple non-trivial dimensions.",
            mesh_shape);

        // Find the only non-trivial axis (dimension > 1)
        for (size_t i = 0; i < mesh_shape.dims(); ++i) {
            if (mesh_shape[i] > 1) {
                return is_cluster_axis_ring(i) ? ttnn::ccl::Topology::Ring : ttnn::ccl::Topology::Linear;
            }
        }
        // All dimensions are 1 (single device case) - use Linear
        return ttnn::ccl::Topology::Linear;
    }

    // cluster_axis is specified - check if that axis has ring connectivity
    return is_cluster_axis_ring(cluster_axis.value()) ? ttnn::ccl::Topology::Ring : ttnn::ccl::Topology::Linear;
}

// A collective runs on the sub-device that belongs to the command queue it is issued on: the CCL
// sub-device for the second queue, the compute sub-device (the op default) for the first. A sub-device is
// owned by the queue that last launched on it, so a collective issued from the compute queue must stay on
// the compute sub-device with the compute around it, and one issued from the second queue must not.
bool on_ccl_queue() {
    return *ttnn::core::get_current_command_queue_id_for_thread() != 0U;
}

std::optional<tt::tt_metal::SubDeviceId> collective_sub_device_id() {
    if (!on_ccl_queue()) {
        return std::nullopt;
    }
    auto& ctx = ttml::autograd::ctx();
    TT_FATAL(
        ctx.has_ccl_sub_device(),
        "a collective was issued on the second command queue but no CCL sub-device is enabled "
        "(AutoContext::enable_ccl_sub_device)");
    return ctx.ccl_sub_device_id();
}

}  // namespace

ttnn::Tensor all_gather(
    const ttnn::Tensor& tensor,
    const int dim,
    const std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::Tensor>& persistent_output) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    // Use cluster_axis overload for 2D mesh
    // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
    auto gathered = ttnn::experimental::all_gather_async(
        tensor,
        persistent_output,
        dim,
        ccl_resources.get_all_gather_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        topology,
        /* subdevice_id */ collective_sub_device_id(),
        cluster_axis,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());
    // A caller that hands over a persistent output relies on the result landing there.
    TT_FATAL(
        !persistent_output.has_value() || &gathered.mesh_buffer() == &persistent_output->mesh_buffer(),
        "all_gather did not write into the persistent output for shape {} dim {}: the op took a path that "
        "allocates its own result (typically the composite fallback for a shard that is not tile-aligned)",
        tensor.logical_shape(),
        dim);
    return gathered;
}

ttnn::Tensor all_reduce(const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }

    auto shape = tensor.logical_shape();
    if (shape.rank() != 4U) {
        throw std::logic_error("All reduce supports only 4D tensors");
    }

    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto all_reduce_barrier_semaphores = ccl_resources.get_all_reduce_barrier_semaphores();
    auto all_gather_semaphores = ccl_resources.get_all_gather_semaphore();
    auto reduce_scatter_semaphores = ccl_resources.get_reduce_scatter_semaphores();

    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    if (cluster_axis.has_value()) {
        // Use cluster_axis overload for 2D mesh
        // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
        return ttnn::experimental::all_reduce_async(
            tensor,
            cluster_axis,
            *mesh_device,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            reduction_common::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            topology,
            std::optional<size_t>(num_links),
            /* worker_subdevice_id_opt */ std::nullopt);
    } else {
        // Use original overload for 1D mesh
        return ttnn::experimental::all_reduce_async(
            tensor,
            num_devices,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            reduction_common::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            topology,
            /* num_preferred_links */ num_links);
    }
}

ttnn::Tensor reduce_scatter(
    const ttnn::Tensor& tensor,
    const int dim,
    const std::optional<uint32_t> cluster_axis,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_buffers) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    TT_FATAL(
        persistent_buffers.has_value() || !on_ccl_queue(),
        "reduce_scatter on the second command queue needs its buffers passed in (reduce_scatter_buffers): the "
        "temporaries the op allocates itself are freed by the host while the collective may still be using them");

    // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
    auto scattered = ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        persistent_buffers,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        topology,
        /* subdevice_id */ collective_sub_device_id(),
        /* cluster_axis */ cluster_axis);
    // A caller that hands over the buffers relies on the result landing in them (the composite fallback for
    // shapes the direct kernels cannot take allocates its own).
    TT_FATAL(
        !persistent_buffers.has_value() || &scattered.mesh_buffer() == &persistent_buffers->at(1).mesh_buffer(),
        "reduce_scatter did not write into the persistent output for shape {} dim {}",
        tensor.logical_shape(),
        dim);
    return scattered;
}

std::vector<ttnn::Tensor> reduce_scatter_buffers(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto& mesh_device = ttml::autograd::ctx().get_device();
    const auto& mesh_shape = mesh_device.shape();
    const uint32_t axis_size = cluster_axis.has_value() ? mesh_shape[*cluster_axis] : mesh_device.num_devices();
    const auto logical_shape = tensor.logical_shape();
    const int normalized_dim = dim < 0 ? static_cast<int>(logical_shape.rank()) + dim : dim;
    TT_FATAL(
        normalized_dim > 0 && logical_shape[normalized_dim] % axis_size == 0,
        "reduce_scatter_buffers: dim {} of {} is not scattered {} ways",
        dim,
        logical_shape,
        axis_size);
    // The topology the op will really run (a 2-device ring is demoted to a line before the staging layout is
    // chosen; the same helper reduce_scatter_minimal_async uses).
    const auto topology = ::ttnn::ccl::get_usable_topology(tensor, get_topology(cluster_axis), cluster_axis);

    auto output_shape = logical_shape;
    output_shape[normalized_dim] /= axis_size;
    auto output = ttnn::empty(output_shape, tensor.dtype(), tensor.layout(), &mesh_device, tensor.memory_config());

    if (topology == ttnn::ccl::Topology::Ring) {
        // Contiguous ring path: chunk-paged intermediate plus the smaller penult intermediate, sized by the
        // op's own helper so they match what it validates against.
        auto staging = ttnn::experimental::reduce_scatter_minimal_async_create_intermediate_buffer(
            tensor, normalized_dim, topology, cluster_axis, /* compute_kernel_config */ std::nullopt);
        TT_FATAL(staging.size() == 2, "expected {{intermediate, penult}} staging buffers, got {}", staging.size());
        return {staging[0], output, staging[1]};
    }
    // Line: an input-shaped tiled intermediate with dim 0 doubled (one half per direction), same dtype,
    // layout and memory config as the input (ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs).
    auto intermediate_shape = tensor.padded_shape();
    intermediate_shape[0] *= 2;
    auto intermediate =
        ttnn::empty(intermediate_shape, tensor.dtype(), tensor.layout(), &mesh_device, tensor.memory_config());
    return {intermediate, output};
}

ttnn::Tensor mesh_partition(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    // ttnn::mesh_partition already returns the input unchanged when the axis size is 1,
    // so no single-device guard is needed here.
    return ttnn::mesh_partition(tensor, dim, cluster_axis, /* memory_config */ std::nullopt);
}

ttnn::Tensor ring_shift(
    const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis, const RingShiftDirection direction) {
    auto& ctx = ttml::autograd::ctx();
    auto& socket_manager = ctx.get_socket_manager();
    auto distributed_ctx = ctx.get_distributed_context();
    auto mesh_device_ptr = ctx.get_device_ptr();
    const auto mesh_shape = mesh_device_ptr->shape();

    TT_FATAL(
        (cluster_axis.has_value() && cluster_axis.value() < mesh_shape.dims() && cluster_axis.value() >= 0) ||
            (!cluster_axis.has_value() &&
             (tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D ||
              tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D_RING)),
        "cluster_axis must be either >= 0 and < {} for 2D mesh or nullopt for 1D mesh and linear topology",
        mesh_shape.dims());

    const uint32_t cluster_axis_value = cluster_axis.has_value() ? cluster_axis.value() : 0;
    const uint32_t ring_size = mesh_shape[cluster_axis_value];
    TT_FATAL(ring_size % 2 == 0, "ring_shift requires an even number of devices in the ring, got {}", ring_size);

    if (ring_size <= 1U) {
        return tensor;
    }

    auto output_tensor = ttnn::empty_like(tensor);

    const uint32_t num_devices = mesh_shape.mesh_size();

    // Build connections for even->odd and odd->even transfers separately
    // This two-phase approach avoids deadlock since send is blocking
    const auto send_recv_core = tt::tt_metal::CoreCoord(0, 0);
    std::vector<tt::tt_metal::distributed::SocketConnection> even_to_odd_connections;
    std::vector<tt::tt_metal::distributed::SocketConnection> odd_to_even_connections;
    even_to_odd_connections.reserve(num_devices / 2);
    odd_to_even_connections.reserve(num_devices / 2);

    const bool forward = (direction == RingShiftDirection::Forward);
    for (const auto& sender_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const uint32_t idx = sender_coord[cluster_axis_value];
        const uint32_t target_idx = forward ? (idx + 1) % ring_size : (idx + ring_size - 1) % ring_size;

        tt::tt_fabric::MeshCoordinate recv_coord = sender_coord;
        recv_coord[cluster_axis_value] = target_idx;

        auto& target_connections = (idx % 2U == 0U) ? even_to_odd_connections : odd_to_even_connections;
        target_connections.emplace_back(
            tt::tt_metal::distributed::MeshCoreCoord{sender_coord, send_recv_core},
            tt::tt_metal::distributed::MeshCoreCoord{recv_coord, send_recv_core});
    }

    // For intra-mesh, we use same distributed context and rank (same host)
    const core::distributed::InterHostParameters inter_host_params{distributed_ctx, distributed_ctx->rank()};

    tt::tt_metal::distributed::Synchronize(*mesh_device_ptr, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    // Phase 1: Even positions send, odd positions receive
    const core::distributed::IntraMeshParameters even_to_odd_params{even_to_odd_connections};
    socket_manager.send(tensor, inter_host_params, even_to_odd_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, even_to_odd_params);

    // Phase 2: Odd positions send, even positions receive
    const core::distributed::IntraMeshParameters odd_to_even_params{odd_to_even_connections};
    socket_manager.send(tensor, inter_host_params, odd_to_even_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, odd_to_even_params);

    tt::tt_metal::distributed::Synchronize(*mesh_device_ptr, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    return output_tensor;
}

// ---- Sequence-parallel linears: the implementation switch and the two fusable sequences ----

namespace {

SPLinearImpl g_sp_linear_impl = SPLinearImpl::Fused;
std::optional<SPLinearImpl> g_sp_linear_backward_impl;  // nullopt: follows the forward

// The matmul exactly as tt-train issues it today, which is what makes Composed bit-identical to the
// unfused modules: the forward `a @ W^T (+ bias)` is linear_op's 4-D ttnn::linear on the full compute
// grid, the dgrad `g @ W` is ttnn_linear_backward's 2-D ttnn_fixed::matmul on the row-flattened
// activation. transpose_b tells the two apart because tt-train stores weights as [N, K]: the forward
// always transposes and the dgrad never does.
ttnn::Tensor unfused_matmul(
    const ttnn::Tensor& a, const ttnn::Tensor& w, bool transpose_b, const std::optional<ttnn::Tensor>& bias) {
    if (transpose_b) {
        const auto grid_size = a.device()->compute_with_storage_grid_size();
        auto core_grid = std::make_optional<ttnn::CoreGrid>(grid_size.x, grid_size.y);
        return ttnn::linear(
            a,
            w,
            bias,
            /* transpose_a */ false,
            /* transpose_b */ true,
            /* memory_config */ std::nullopt,
            /* dtype */ std::nullopt,
            /* program_config */ std::nullopt,
            /* activation */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
            /* core_grid */ core_grid);
    }
    TT_FATAL(!bias.has_value(), "unfused_matmul: a bias is only supported with transpose_b (the forward linear)");
    const auto& shape = a.logical_shape();
    TT_FATAL(shape.rank() == 4, "unfused_matmul: expected a rank-4 activation, got {}", shape);
    const auto rows = static_cast<uint32_t>(a.logical_volume() / shape[-1]);
    auto mm = ttnn_fixed::matmul(
        ttnn::reshape(a, ttnn::Shape({rows, shape[-1]})), w, /* transpose_a */ false, /* transpose_b */ false);
    return ttnn::reshape(mm, ttnn::Shape({shape[0], shape[1], shape[2], mm.logical_shape()[-1]}));
}

std::pair<ttnn::Tensor, ttnn::Tensor> all_gather_matmul_composed(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t cluster_axis,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& bias) {
    auto gathered = ttml::ttnn_fixed::distributed::all_gather(x, /* dim */ 2, cluster_axis);
    auto mm = unfused_matmul(gathered, w, transpose_b, bias);
    return {std::move(gathered), std::move(mm)};
}

ttnn::Tensor matmul_reduce_scatter_composed(
    const ttnn::Tensor& x, const ttnn::Tensor& w, uint32_t cluster_axis, bool transpose_b) {
    return ttml::ttnn_fixed::distributed::reduce_scatter(
        unfused_matmul(x, w, transpose_b, std::nullopt), /* dim */ 2, cluster_axis);
}

// The fused ttnn ops of issue #52944: the collective overlaps the matmul, one (batch, sequence slice) at a
// time. Semaphores, links and topology as the standalone all_gather / reduce_scatter above; the matmul with
// linear_op's compute config (HiFi4, fp32 accumulation), which the ops would otherwise lower to HiFi2.
// `ccl_core_rows` is passed as the ops' own default constant only because C++ positional arguments leave no other
// way to reach `compute_kernel_config`; `num_workers_per_link` is derived by the ops (nullopt).
std::pair<ttnn::Tensor, ttnn::Tensor> all_gather_matmul_fused(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t cluster_axis,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& bias) {
    auto& ctx = ttml::autograd::ctx();
    auto& ccl_resources = ctx.get_ccl_resources();
    const uint32_t num_links = ttnn::operations::ccl::common::get_num_links(ctx.get_device(), cluster_axis);
    auto outputs = ttnn::experimental::all_gather_matmul_sp_async(
        x,
        w,
        cluster_axis,
        ccl_resources.get_all_gather_semaphore(),
        ccl_resources.get_barrier_semaphore(),
        transpose_b,
        bias,
        num_links,
        get_topology(cluster_axis),
        ttnn::experimental::kDefaultAllGatherMatmulSpCclCoreRows,
        /* num_workers_per_link */ std::nullopt,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
    TT_FATAL(
        outputs.size() == 2,
        "all_gather_matmul_sp_async returned {} tensors, expected {{gathered, mm}}",
        outputs.size());
    return {std::move(outputs[0]), std::move(outputs[1])};
}

ttnn::Tensor matmul_reduce_scatter_fused(
    const ttnn::Tensor& x, const ttnn::Tensor& w, uint32_t cluster_axis, bool transpose_b) {
    auto& ctx = ttml::autograd::ctx();
    auto& ccl_resources = ctx.get_ccl_resources();
    const uint32_t num_links = ttnn::operations::ccl::common::get_num_links(ctx.get_device(), cluster_axis);
    return ttnn::experimental::matmul_reduce_scatter_sp_async(
        x,
        w,
        cluster_axis,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        transpose_b,
        num_links,
        get_topology(cluster_axis),
        ttnn::experimental::kDefaultMatmulReduceScatterSpCclCoreRows,
        /* num_workers_per_link */ std::nullopt,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());
}

// Measurement only (SPLinearImpl::NoComm): the collective's output is allocated but never computed, so the step
// runs every matmul at its real shape and no communication at all -- the ideal the other implementations chase.
std::pair<ttnn::Tensor, ttnn::Tensor> all_gather_matmul_nocomm(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t cluster_axis,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& bias) {
    auto& device = ttml::autograd::ctx().get_device();
    const uint32_t ranks = device.shape()[cluster_axis];
    const auto& s = x.logical_shape();
    auto gathered =
        ttml::core::empty(ttnn::Shape({s[0], s[1], s[2] * ranks, s[3]}), &device, x.memory_config());
    auto mm = unfused_matmul(gathered, w, transpose_b, bias);
    return {std::move(gathered), std::move(mm)};
}

ttnn::Tensor matmul_reduce_scatter_nocomm(
    const ttnn::Tensor& x, const ttnn::Tensor& w, uint32_t cluster_axis, bool transpose_b) {
    auto& device = ttml::autograd::ctx().get_device();
    const uint32_t ranks = device.shape()[cluster_axis];
    auto mm = unfused_matmul(x, w, transpose_b, std::nullopt);
    const auto& s = mm.logical_shape();
    return ttml::core::empty(ttnn::Shape({s[0], s[1], s[2] / ranks, s[3]}), &device, mm.memory_config());
}

}  // namespace

ttnn::Tensor sp_linear_matmul(
    const ttnn::Tensor& a, const ttnn::Tensor& w, bool transpose_b, const std::optional<ttnn::Tensor>& bias) {
    return unfused_matmul(a, w, transpose_b, bias);
}

void set_sp_linear_impl(SPLinearImpl impl) {
    g_sp_linear_impl = impl;
    g_sp_linear_backward_impl = std::nullopt;
}

SPLinearImpl get_sp_linear_impl() {
    return g_sp_linear_impl;
}

void set_sp_linear_backward_impl(std::optional<SPLinearImpl> impl) {
    g_sp_linear_backward_impl = impl;
}

SPLinearImpl get_sp_linear_backward_impl() {
    return g_sp_linear_backward_impl.value_or(g_sp_linear_impl);
}

SPLinearImpl get_sp_linear_impl(SPLinearSite site) {
    return site == SPLinearSite::Backward ? get_sp_linear_backward_impl() : g_sp_linear_impl;
}

std::pair<ttnn::Tensor, ttnn::Tensor> all_gather_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t cluster_axis,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& bias,
    SPLinearSite site) {
    switch (get_sp_linear_impl(site)) {
        case SPLinearImpl::Fused: return all_gather_matmul_fused(x, w, cluster_axis, transpose_b, bias);
        case SPLinearImpl::NoComm: return all_gather_matmul_nocomm(x, w, cluster_axis, transpose_b, bias);
        case SPLinearImpl::Composed: break;
    }
    return all_gather_matmul_composed(x, w, cluster_axis, transpose_b, bias);
}

ttnn::Tensor matmul_reduce_scatter(
    const ttnn::Tensor& x, const ttnn::Tensor& w, uint32_t cluster_axis, bool transpose_b, SPLinearSite site) {
    switch (get_sp_linear_impl(site)) {
        case SPLinearImpl::Fused: return matmul_reduce_scatter_fused(x, w, cluster_axis, transpose_b);
        case SPLinearImpl::NoComm: return matmul_reduce_scatter_nocomm(x, w, cluster_axis, transpose_b);
        case SPLinearImpl::Composed: break;
    }
    return matmul_reduce_scatter_composed(x, w, cluster_axis, transpose_b);
}

}  // namespace ttml::ttnn_fixed::distributed
