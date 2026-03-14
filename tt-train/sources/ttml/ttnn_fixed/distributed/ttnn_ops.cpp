// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
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

// Repair output tensor topology after a collective.
// CCL ops don't propagate TensorTopology, so we must rewrite it to match
// the semantic result of the collective (e.g. all_gather output is replicated
// along the gather axis, reduce_scatter output is sharded, etc.).
tt::tt_metal::Tensor repair_topology(
    const tt::tt_metal::Tensor& input,
    tt::tt_metal::Tensor output,
    const std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::TensorTopology& new_topology) {
    return output.with_tensor_topology(new_topology);
}

// Build a topology where the given cluster_axis is replicated (output of all_gather / all_reduce).
tt::tt_metal::TensorTopology make_replicated_on_axis(
    const tt::tt_metal::Tensor& input, const std::optional<uint32_t> cluster_axis) {
    auto topo = input.tensor_topology();
    auto placements = topo.placements();

    if (cluster_axis.has_value()) {
        uint32_t axis = cluster_axis.value();
        if (axis < placements.size()) {
            placements[axis] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    } else {
        for (auto& p : placements) {
            p = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }

    return tt::tt_metal::TensorTopology(topo.distribution_shape(), std::move(placements), topo.mesh_coords());
}

// Build a topology where the given cluster_axis is sharded on `dim` (output of reduce_scatter / scatter).
tt::tt_metal::TensorTopology make_sharded_on_axis(
    const tt::tt_metal::Tensor& input, int dim, const std::optional<uint32_t> cluster_axis) {
    auto topo = input.tensor_topology();
    auto placements = topo.placements();

    if (cluster_axis.has_value()) {
        uint32_t axis = cluster_axis.value();
        if (axis < placements.size()) {
            placements[axis] = tt::tt_metal::distributed::MeshMapperConfig::Shard{dim};
        }
    } else {
        if (!placements.empty()) {
            placements[0] = tt::tt_metal::distributed::MeshMapperConfig::Shard{dim};
        }
    }

    return tt::tt_metal::TensorTopology(topo.distribution_shape(), std::move(placements), topo.mesh_coords());
}

}  // namespace

tt::tt_metal::Tensor all_gather(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
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
    auto result = ttnn::experimental::all_gather_async(
        tensor,
        /* persistent_output_buffer */ std::nullopt,
        dim,
        ccl_resources.get_all_gather_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        topology,
        /* subdevice_id */ std::nullopt,
        cluster_axis,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());

    // all_gather produces a replicated tensor along the gather axis
    auto new_topo = make_replicated_on_axis(tensor, cluster_axis);
    return repair_topology(tensor, std::move(result), cluster_axis, new_topo);
}

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor, const std::optional<uint32_t> cluster_axis) {
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

    tt::tt_metal::Tensor result;
    if (cluster_axis.has_value()) {
        // Use cluster_axis overload for 2D mesh
        // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
        result = ttnn::experimental::all_reduce_async(
            tensor,
            cluster_axis,
            *mesh_device,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            topology,
            std::optional<size_t>(num_links),
            /* worker_subdevice_id_opt */ std::nullopt);
    } else {
        // Use original overload for 1D mesh
        result = ttnn::experimental::all_reduce_async(
            tensor,
            num_devices,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            topology,
            /* num_preferred_links */ num_links);
    }

    // all_reduce produces a replicated tensor along the reduce axis
    auto new_topo = make_replicated_on_axis(tensor, cluster_axis);
    return repair_topology(tensor, std::move(result), cluster_axis, new_topo);
}

tt::tt_metal::Tensor reduce_scatter(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
    auto result = ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        /* persistent_output_buffers */ std::nullopt,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        topology,
        /* subdevice_id */ std::nullopt,
        /* cluster_axis */ cluster_axis);

    // reduce_scatter produces a tensor sharded on `dim` along the scatter axis
    auto new_topo = make_sharded_on_axis(tensor, dim, cluster_axis);
    return repair_topology(tensor, std::move(result), cluster_axis, new_topo);
}

tt::tt_metal::Tensor ring_shift(
    const tt::tt_metal::Tensor& tensor,
    const std::optional<uint32_t> cluster_axis,
    const RingShiftDirection direction) {
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

    tt::tt_metal::distributed::Synchronize(
        mesh_device_ptr.get(), std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    // Phase 1: Even positions send, odd positions receive
    const core::distributed::IntraMeshParameters even_to_odd_params{even_to_odd_connections};
    socket_manager.send(tensor, inter_host_params, even_to_odd_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, even_to_odd_params);

    // Phase 2: Odd positions send, even positions receive
    const core::distributed::IntraMeshParameters odd_to_even_params{odd_to_even_connections};
    socket_manager.send(tensor, inter_host_params, odd_to_even_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, odd_to_even_params);

    tt::tt_metal::distributed::Synchronize(
        mesh_device_ptr.get(), std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    return output_tensor;
}

}  // namespace ttml::ttnn_fixed::distributed
