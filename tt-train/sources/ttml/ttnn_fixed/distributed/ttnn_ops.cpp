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
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/creation/creation.hpp"
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
[[maybe_unused]] ttnn::ccl::Topology get_topology(const std::optional<uint32_t>& cluster_axis) {
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

// TODO(nuked-op ccl): the TTNN CCL ops (all_gather_async, all_reduce_async, reduce_scatter_minimal_async,
// mesh_partition) were removed for agent evaluation. These wrappers keep their signatures and pass the
// input through so tt-train builds; they are NOT correct on a multi-device mesh.
ttnn::Tensor all_gather(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    if (mesh_device->num_devices() == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    // TODO(nuked-op all_gather_async): restore real call
    (void)dim;
    (void)cluster_axis;
    return tensor;
}

ttnn::Tensor all_reduce(const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    if (mesh_device->num_devices() == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }
    if (tensor.logical_shape().rank() != 4U) {
        throw std::logic_error("All reduce supports only 4D tensors");
    }
    // TODO(nuked-op all_reduce_async): restore real call
    (void)cluster_axis;
    return tensor;
}

ttnn::Tensor reduce_scatter(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    // TODO(nuked-op reduce_scatter_minimal_async): restore real call
    (void)dim;
    (void)cluster_axis;
    return tensor;
}

ttnn::Tensor mesh_partition(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    // TODO(nuked-op mesh_partition): restore real call
    (void)dim;
    (void)cluster_axis;
    return tensor;
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
