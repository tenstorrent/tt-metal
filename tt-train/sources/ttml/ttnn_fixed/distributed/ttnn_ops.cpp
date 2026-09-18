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
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"

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

}  // namespace

namespace {

// A collective runs on the sub-device that belongs to the command queue it is issued on: the CCL
// sub-device for the second queue, the compute sub-device (the default) for the first. Two queues
// launching programs on the same sub-device interleave on the same cores and corrupt each other, so
// a tensor-parallel collective issued from the compute queue must stay with the compute around it.
bool on_ccl_queue() {
    return *ttnn::core::get_current_command_queue_id_for_thread() != 0U;
}

std::optional<tt::tt_metal::SubDeviceId> collective_sub_device_id() {
    auto& ctx = ttml::autograd::ctx();
    return on_ccl_queue() ? ctx.ccl_sub_device_id() : std::nullopt;
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
    auto ag_result = ttnn::experimental::all_gather_async(
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
    // A caller that hands over a persistent output relies on the result landing there (it may already be
    // referenced elsewhere, and on the CCL queue a fresh buffer could alias memory compute is using).
    TT_FATAL(
        !persistent_output.has_value() || &ag_result.mesh_buffer() == &persistent_output->mesh_buffer(),
        "all_gather did not write into the persistent output for shape {} dim {}: the op took a path that "
        "allocates its own result (typically the composite fallback for a shard that is not tile-aligned)",
        tensor.logical_shape(),
        dim);
    return ag_result;
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

namespace {

// A shard of `dim_size` split `axis_size` ways is a whole number of 32-element tiles. Only then does
// reduce_scatter_minimal_async take its direct kernel path; the composite fallback ignores persistent
// buffers.
bool shard_is_tile_aligned(uint32_t dim_size, uint32_t axis_size) {
    return axis_size > 0 && dim_size % axis_size == 0 && (dim_size / axis_size) % 32 == 0;
}

}  // namespace

ttnn::Tensor reduce_scatter(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    // Reuse the op's staging buffers (see CCLResources) and hand it a freshly allocated output, so the
    // only per-call allocation is the result the caller keeps. The contiguous ring path is the only one
    // with a persistent staging layout; every other configuration lets the op allocate as before --
    // except on the CCL queue, where the op's own temporaries would be freed by the host while compute
    // on the other queue can be handed their addresses.
    std::optional<std::vector<ttnn::Tensor>> persistent_buffers;
    const auto& mesh_shape = mesh_device.shape();
    const uint32_t axis_size = cluster_axis.has_value() ? mesh_shape[*cluster_axis] : mesh_device.num_devices();
    const auto logical_shape = tensor.logical_shape();
    const int normalized_dim = dim < 0 ? static_cast<int>(logical_shape.rank()) + dim : dim;
    if (topology == ttnn::ccl::Topology::Ring && axis_size > 2 && normalized_dim > 0 &&
        tensor.layout() == ttnn::Layout::TILE && shard_is_tile_aligned(logical_shape[normalized_dim], axis_size)) {
        const auto& staging =
            ccl_resources.get_reduce_scatter_staging_buffers(tensor, normalized_dim, cluster_axis, topology);
        if (staging.size() == 2) {
            auto output_shape = logical_shape;
            output_shape[normalized_dim] /= axis_size;
            auto output =
                ttnn::empty(output_shape, tensor.dtype(), tensor.layout(), &mesh_device, tensor.memory_config());
            persistent_buffers = std::vector<ttnn::Tensor>{staging[0], output, staging[1]};
        }
    }
    TT_FATAL(
        persistent_buffers.has_value() || !on_ccl_queue(),
        "reduce_scatter on the CCL queue needs persistent staging buffers (ring topology, more than two devices, "
        "tile-aligned shard on a dim > 0); shape {} dim {} over {} devices has none",
        logical_shape,
        normalized_dim,
        axis_size);

    // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
    return ttnn::experimental::reduce_scatter_minimal_async(
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

}  // namespace ttml::ttnn_fixed::distributed
