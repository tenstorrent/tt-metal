// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/ops/ring_shift_fused/ring_shift_fused.hpp"
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

// The ring's geometry for a shift: the receiver of every chip and the
// number of sender cores (one per fabric link) a direct transport uses.
struct RingPlan {
    uint32_t cluster_axis{};
    uint32_t ring_size{};
    uint32_t cores_per_pair{1U};
    std::function<tt::tt_fabric::MeshCoordinate(const tt::tt_fabric::MeshCoordinate&)> receiver_of;
};

RingPlan plan_ring(
    const std::optional<uint32_t> cluster_axis,
    const RingShiftDirection direction,
    const RingShiftTransport transport,
    const uint32_t connections) {
    auto& ctx = ttml::autograd::ctx();
    auto mesh_device_ptr = ctx.get_device_ptr();
    const auto mesh_shape = mesh_device_ptr->shape();

    TT_FATAL(
        (cluster_axis.has_value() && cluster_axis.value() < mesh_shape.dims() && cluster_axis.value() >= 0) ||
            (!cluster_axis.has_value() &&
             (tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D ||
              tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D_RING)),
        "cluster_axis must be either >= 0 and < {} for 2D mesh or nullopt for 1D mesh and linear topology",
        mesh_shape.dims());

    RingPlan plan;
    plan.cluster_axis = cluster_axis.has_value() ? cluster_axis.value() : 0;
    plan.ring_size = mesh_shape[plan.cluster_axis];
    const bool forward = (direction == RingShiftDirection::Forward);
    const uint32_t axis = plan.cluster_axis;
    const uint32_t ring_size = plan.ring_size;
    plan.receiver_of = [axis, ring_size, forward](const tt::tt_fabric::MeshCoordinate& sender_coord) {
        const uint32_t idx = sender_coord[axis];
        const uint32_t target_idx = forward ? (idx + 1) % ring_size : (idx + ring_size - 1) % ring_size;
        tt::tt_fabric::MeshCoordinate recv_coord = sender_coord;
        recv_coord[axis] = target_idx;
        return recv_coord;
    };

    // Direct: one sender core per fabric link between the two chips, so the
    // links run in parallel. The count is the minimum over the pairs, which
    // on a homogeneous ring is the same everywhere. Fifo: one core, as ever.
    if (transport != RingShiftTransport::Fifo && plan.ring_size > 1U) {
        uint32_t links = std::numeric_limits<uint32_t>::max();
        for (const auto& sender_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
            const auto count = static_cast<uint32_t>(
                tt::tt_fabric::get_forwarding_link_indices(
                    mesh_device_ptr->get_fabric_node_id(sender_coord),
                    mesh_device_ptr->get_fabric_node_id(plan.receiver_of(sender_coord)))
                    .size());
            links = std::min(links, count);
        }
        TT_FATAL(links >= 1U, "ring_shift: no fabric link between some ring neighbours");
        plan.cores_per_pair = connections == 0U ? links : std::min(connections, links);
    }
    return plan;
}

tt::tt_metal::distributed::SocketConnection make_connection(
    const RingPlan& plan,
    const tt::tt_fabric::MeshCoordinate& sender_coord,
    const uint32_t c,
    const RingShiftTransport transport) {
    // Senders in worker row 0, receivers in row 1: the socket runtime
    // refuses a core that appears in two connections of one socket,
    // and (0, 0) on both ends is what the Fifo path always used.
    const auto sender_core =
        transport != RingShiftTransport::Fifo ? tt::tt_metal::CoreCoord(c, 0) : tt::tt_metal::CoreCoord(0, 0);
    const auto receiver_core =
        transport != RingShiftTransport::Fifo ? tt::tt_metal::CoreCoord(c, 1) : tt::tt_metal::CoreCoord(0, 0);
    return tt::tt_metal::distributed::SocketConnection(
        tt::tt_metal::distributed::MeshCoreCoord{sender_coord, sender_core},
        tt::tt_metal::distributed::MeshCoreCoord{plan.receiver_of(sender_coord), receiver_core});
}

// One tensor in two phases: even chips send while odd chips receive, then
// the reverse. Separate send and receive launches on one command queue
// cannot do better: a chip's send waits for its neighbour's receive, which
// sits behind that neighbour's own send. The Fifo path also needs this
// order because its send blocks.
ttnn::Tensor ring_shift_two_phase(
    const ttnn::Tensor& tensor, const RingPlan& plan, const RingShiftTransport transport) {
    auto& ctx = ttml::autograd::ctx();
    auto& socket_manager = ctx.get_socket_manager();
    auto distributed_ctx = ctx.get_distributed_context();
    auto mesh_device_ptr = ctx.get_device_ptr();
    const auto mesh_shape = mesh_device_ptr->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();

    auto output_tensor = ttnn::empty_like(tensor);

    std::vector<tt::tt_metal::distributed::SocketConnection> even_to_odd_connections;
    std::vector<tt::tt_metal::distributed::SocketConnection> odd_to_even_connections;
    even_to_odd_connections.reserve(num_devices / 2 * plan.cores_per_pair);
    odd_to_even_connections.reserve(num_devices / 2 * plan.cores_per_pair);
    for (const auto& sender_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const uint32_t idx = sender_coord[plan.cluster_axis];
        auto& target_connections = (idx % 2U == 0U) ? even_to_odd_connections : odd_to_even_connections;
        for (uint32_t c = 0; c < plan.cores_per_pair; ++c) {
            target_connections.push_back(make_connection(plan, sender_coord, c, transport));
        }
    }

    // For intra-mesh, we use same distributed context and rank (same host)
    const core::distributed::InterHostParameters inter_host_params{distributed_ctx, distributed_ctx->rank()};
    const core::distributed::IntraMeshParameters even_to_odd_params{even_to_odd_connections};
    const core::distributed::IntraMeshParameters odd_to_even_params{odd_to_even_connections};

    if (transport == RingShiftTransport::DirectTwoPhase) {
        // Every chip is a sender in one phase and a receiver in the other, and
        // its two programs sit in its own command queue in that order, so the
        // ordering the Fifo path's host synchronisations enforce is already
        // there; none are issued.
        socket_manager.send_direct(tensor, inter_host_params, even_to_odd_params);
        output_tensor = socket_manager.recv_direct(output_tensor, inter_host_params, even_to_odd_params);
        socket_manager.send_direct(tensor, inter_host_params, odd_to_even_params);
        output_tensor = socket_manager.recv_direct(output_tensor, inter_host_params, odd_to_even_params);
        return output_tensor;
    }

    tt::tt_metal::distributed::Synchronize(*mesh_device_ptr, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    // Phase 1: Even positions send, odd positions receive
    socket_manager.send(tensor, inter_host_params, even_to_odd_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, even_to_odd_params);

    // Phase 2: Odd positions send, even positions receive
    socket_manager.send(tensor, inter_host_params, odd_to_even_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, odd_to_even_params);

    tt::tt_metal::distributed::Synchronize(*mesh_device_ptr, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    return output_tensor;
}

}  // namespace

std::vector<ttnn::Tensor> ring_shift_many(
    const std::vector<ttnn::Tensor>& tensors,
    const std::optional<uint32_t> cluster_axis,
    const RingShiftDirection direction,
    const RingShiftTransport transport,
    const uint32_t connections) {
    if (tensors.empty()) {
        return {};
    }
    const RingPlan plan = plan_ring(cluster_axis, direction, transport, connections);
    TT_FATAL(plan.ring_size % 2 == 0, "ring_shift requires an even number of devices in the ring, got {}", plan.ring_size);
    if (plan.ring_size <= 1U) {
        return tensors;
    }

    if (transport != RingShiftTransport::Direct) {
        std::vector<ttnn::Tensor> outputs;
        outputs.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            outputs.push_back(ring_shift_two_phase(tensor, plan, transport));
        }
        return outputs;
    }

    // Direct: every tensor in one launch. On a single ring every chip sends
    // to one neighbour and receives from the other at once, so every link
    // is busy: one launch per shift. On a mesh of several rings side by side
    // (the 2x4 loudbox mesh) that launch loses the fabric handshake's
    // answers and hangs, while each half of the ring alone is fine, so there
    // the shift is two launches, even chips sending and then odd chips --
    // the two-phase shift's order, at its bandwidth, with all tensors in
    // each launch. The connections are the two-phase shift's two sockets in
    // both cases; a chip that is both a sender and a receiver of one socket
    // hangs as well.
    auto& ctx = ttml::autograd::ctx();
    auto& socket_manager = ctx.get_socket_manager();
    auto distributed_ctx = ctx.get_distributed_context();
    const auto mesh_shape = ctx.get_device_ptr()->shape();
    const bool single_ring = mesh_shape.mesh_size() == plan.ring_size;
    static bool logged = false;
    if (!logged) {
        logged = true;
        if (single_ring) {
            log_info(
                tt::LogAlways, "ring shift: one launch per shift, every chip sending and receiving at once ({} chips)",
                plan.ring_size);
        } else {
            log_info(
                tt::LogAlways,
                "ring shift: two launches per shift (even chips send, then odd) -- a mesh of {} rings of {}, where a "
                "full ring at once hangs the fabric handshake",
                mesh_shape.mesh_size() / plan.ring_size, plan.ring_size);
        }
    }
    std::vector<tt::tt_metal::distributed::SocketConnection> even_to_odd;
    std::vector<tt::tt_metal::distributed::SocketConnection> odd_to_even;
    for (const auto& sender_coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        auto& target = (sender_coord[plan.cluster_axis] % 2U == 0U) ? even_to_odd : odd_to_even;
        for (uint32_t c = 0; c < plan.cores_per_pair; ++c) {
            target.push_back(make_connection(plan, sender_coord, c, transport));
        }
    }
    const core::distributed::InterHostParameters inter_host_params{distributed_ctx, distributed_ctx->rank()};
    std::vector<tt::tt_metal::distributed::MeshSocket> sends;
    std::vector<tt::tt_metal::distributed::MeshSocket> recvs;
    for (const auto* connections_of_phase : {&even_to_odd, &odd_to_even}) {
        const core::distributed::IntraMeshParameters intra_mesh_params{*connections_of_phase};
        const auto [send_socket, recv_socket] = socket_manager.direct_socket_pair(inter_host_params, intra_mesh_params);
        sends.push_back(send_socket);
        recvs.push_back(recv_socket);
    }
    if (single_ring) {
        return ttml::metal::ring_shift_fused(tensors, sends, recvs);
    }
    // The second launch writes into the first's outputs: a chip receives in
    // one of the two only, so the tensors come back whole.
    std::vector<ttnn::Tensor> outputs = ttml::metal::ring_shift_fused(tensors, {sends[0]}, {recvs[0]});
    return ttml::metal::ring_shift_fused(tensors, {sends[1]}, {recvs[1]}, outputs);
}

ttnn::Tensor ring_shift(
    const ttnn::Tensor& tensor,
    const std::optional<uint32_t> cluster_axis,
    const RingShiftDirection direction,
    const RingShiftTransport transport,
    const uint32_t connections) {
    return ring_shift_many({tensor}, cluster_axis, direction, transport, connections)[0];
}

}  // namespace ttml::ttnn_fixed::distributed
