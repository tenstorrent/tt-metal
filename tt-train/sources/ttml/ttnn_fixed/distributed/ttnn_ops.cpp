// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/distributed.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, /* cluster_axis */ cluster_axis);

    // Use cluster_axis overload for 2D mesh
    return ttnn::experimental::all_gather_async(
        tensor,
        /* persistent_output_buffer */ std::nullopt,
        dim,
        ccl_resources.get_all_gather_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear,
        /* subdevice_id */ std::nullopt,
        cluster_axis,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());
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

    if (cluster_axis.has_value()) {
        // Use cluster_axis overload for 2D mesh
        return ttnn::experimental::all_reduce_async(
            tensor,
            cluster_axis,
            *mesh_device,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            ttnn::ccl::Topology::Linear,
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
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            /* topology */ ttnn::ccl::Topology::Linear,
            /* num_preferred_links */ num_links);
    }
}

tt::tt_metal::Tensor reduce_scatter(
    const tt::tt_metal::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ cluster_axis);
    return ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        /* persistent_output_buffers */ std::nullopt,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear,
        /* subdevice_id */ std::nullopt,
        /* cluster_axis */ cluster_axis);
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

    // Phase 1: Even positions send, odd positions receive
    const core::distributed::IntraMeshParameters even_to_odd_params{even_to_odd_connections};
    socket_manager.send(tensor, inter_host_params, even_to_odd_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, even_to_odd_params);

    // Phase 2: Odd positions send, even positions receive
    const core::distributed::IntraMeshParameters odd_to_even_params{odd_to_even_connections};
    socket_manager.send(tensor, inter_host_params, odd_to_even_params);
    output_tensor = socket_manager.recv(output_tensor, inter_host_params, odd_to_even_params);

    return output_tensor;
}

}  // namespace ttml::ttnn_fixed::distributed
