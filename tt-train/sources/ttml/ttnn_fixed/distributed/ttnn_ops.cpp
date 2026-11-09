// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <deque>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
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

ttnn::Tensor all_gather(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
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
    return ttnn::experimental::all_gather_async(
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

ttnn::Tensor reduce_scatter(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ cluster_axis);

    // Determine topology based on cluster axis configuration (Ring if torus, Linear otherwise)
    auto topology = get_topology(cluster_axis);

    // Note: Pass topology (not hardcoded Ring) - Ring only works with proper TORUS fabric config
    return ttnn::experimental::reduce_scatter_minimal_async(
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
}

ttnn::Tensor mesh_partition(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    // ttnn::mesh_partition already returns the input unchanged when the axis size is 1,
    // so no single-device guard is needed here.
    return ttnn::mesh_partition(tensor, dim, cluster_axis, /* memory_config */ std::nullopt);
}

namespace {

// Socket connections of one ring shift. `ring` holds every device -> its neighbour; `even_to_odd` / `odd_to_even`
// split them by the parity of the sender's ring position.
struct RingShiftConnections {
    core::distributed::IntraMeshParameters ring;
    core::distributed::IntraMeshParameters even_to_odd;
    core::distributed::IntraMeshParameters odd_to_even;
};

// Built once per (mesh shape, axis, direction): ring attention shifts with the same few settings every step.
const RingShiftConnections& get_ring_shift_connections(
    const tt::tt_metal::distributed::MeshShape& mesh_shape, const uint32_t cluster_axis, const bool forward) {
    struct Entry {
        tt::tt_metal::distributed::MeshShape mesh_shape;
        uint32_t cluster_axis;
        bool forward;
        RingShiftConnections connections;
    };
    static std::deque<Entry> cache;  // a deque keeps the returned references valid as it grows
    for (const auto& entry : cache) {
        if (entry.mesh_shape == mesh_shape && entry.cluster_axis == cluster_axis && entry.forward == forward) {
            return entry.connections;
        }
    }

    const uint32_t ring_size = mesh_shape[cluster_axis];
    const auto send_recv_core = tt::tt_metal::CoreCoord(0, 0);
    RingShiftConnections connections;
    for (const auto& sender_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const uint32_t idx = sender_coord[cluster_axis];
        const uint32_t target_idx = forward ? (idx + 1) % ring_size : (idx + ring_size - 1) % ring_size;

        tt::tt_fabric::MeshCoordinate recv_coord = sender_coord;
        recv_coord[cluster_axis] = target_idx;

        const tt::tt_metal::distributed::SocketConnection connection(
            tt::tt_metal::distributed::MeshCoreCoord{sender_coord, send_recv_core},
            tt::tt_metal::distributed::MeshCoreCoord{recv_coord, send_recv_core});
        connections.ring.connections.push_back(connection);
        auto& parity_connections = (idx % 2U == 0U) ? connections.even_to_odd : connections.odd_to_even;
        parity_connections.connections.push_back(connection);
    }
    cache.push_back(Entry{mesh_shape, cluster_axis, forward, std::move(connections)});
    return cache.back().connections;
}

// Bytes one device sends for `tensor`.
uint64_t per_device_bytes(const ttnn::Tensor& tensor) {
    return tensor.buffer()->aligned_size();
}

}  // namespace

std::vector<ttnn::Tensor> ring_shift(
    const std::vector<ttnn::Tensor>& tensors,
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
        return tensors;
    }

    const auto& connections =
        get_ring_shift_connections(mesh_shape, cluster_axis_value, direction == RingShiftDirection::Forward);

    std::vector<ttnn::Tensor> outputs;
    outputs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        outputs.push_back(ttnn::empty_like(tensor));
    }

    // For intra-mesh, we use same distributed context and rank (same host)
    const core::distributed::InterHostParameters inter_host_params{distributed_ctx, distributed_ctx->rank()};

    // No host synchronization: each send/recv waits only on its peer, through the socket's flow-control state on
    // device. Every device issues the same socket ops in the same order, and the tensors move in groups of two
    // kinds:
    // - A round, over the ring socket: every device sends its tensors of the round, then receives them. The round's
    //   bytes fit in the receiver's FIFO, so a send waits only for its receiver to drain earlier groups, never for a
    //   recv of its own round, and both parities transfer at once.
    // - A tensor too large for a round, in two phases over the parity sockets: even positions send while odd ones
    //   receive, then odd send while even receive. Here a send can wait for its receiver's recv in the same phase,
    //   which is the receiver's next socket op.
    // So a blocked op only ever waits on an earlier group, or, within a phase, on the receiver's next op. Re-check
    // this if the sends and recvs below are reordered or the round budget is raised. Half the FIFO leaves room for
    // the block alignment of each op.
    const uint64_t round_budget = core::distributed::SocketManager::intra_mesh_fifo_size() / 2U;
    size_t begin = 0;
    while (begin < tensors.size()) {
        uint64_t round_bytes = per_device_bytes(tensors[begin]);
        if (round_bytes > round_budget) {
            // Phase 1: Even positions send, odd positions receive
            socket_manager.send(tensors[begin], inter_host_params, connections.even_to_odd);
            outputs[begin] = socket_manager.recv(outputs[begin], inter_host_params, connections.even_to_odd);
            // Phase 2: Odd positions send, even positions receive
            socket_manager.send(tensors[begin], inter_host_params, connections.odd_to_even);
            outputs[begin] = socket_manager.recv(outputs[begin], inter_host_params, connections.odd_to_even);
            ++begin;
            continue;
        }
        size_t end = begin + 1;
        while (end < tensors.size() && round_bytes + per_device_bytes(tensors[end]) <= round_budget) {
            round_bytes += per_device_bytes(tensors[end]);
            ++end;
        }
        for (size_t i = begin; i < end; ++i) {
            socket_manager.send(tensors[i], inter_host_params, connections.ring);
        }
        for (size_t i = begin; i < end; ++i) {
            outputs[i] = socket_manager.recv(outputs[i], inter_host_params, connections.ring);
        }
        begin = end;
    }

    return outputs;
}

ttnn::Tensor ring_shift(
    const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis, const RingShiftDirection direction) {
    return ring_shift(std::vector<ttnn::Tensor>{tensor}, cluster_axis, direction).front();
}

}  // namespace ttml::ttnn_fixed::distributed
