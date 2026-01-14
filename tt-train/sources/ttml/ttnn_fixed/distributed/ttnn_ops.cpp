// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_socket.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"

using tt::tt_metal::distributed::MeshCoordinateRange;

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, /* cluster_axis */ std::nullopt);

    return ttnn::experimental::all_gather_async(
        tensor,
        dim,
        ccl_resources.get_all_gather_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear,
        /* subdevice_id */ std::nullopt,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());
}

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis) {
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

    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, /* cluster_axis */ std::nullopt);

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

tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, /* cluster_axis */ std::nullopt);
    return ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        /* persistent_output_buffers */ std::nullopt,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear);
}

tt::tt_metal::Tensor ring_shift(
    const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis, bool forward) {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_ptr = ttml::autograd::ctx().get_device_ptr();
    const auto mesh_shape = mesh_device_ptr->shape();

    TT_FATAL(
        (cluster_axis.has_value() && cluster_axis.value() < mesh_shape.dims() && cluster_axis.value() >= 0) ||
            (!cluster_axis.has_value() && tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D ||
             tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_1D_RING),
        "cluster_axis must be either > 0 and < {} for 2D mesh, got {} or nullopt for 1D mesh and linear topology, the "
        "actual "
        "fabric config is {}",
        mesh_shape.dims() - 1,
        cluster_axis.value(),
        tt::tt_fabric::GetFabricConfig());

    const uint32_t cluster_axis_value = cluster_axis.has_value() ? cluster_axis.value() : 1;
    const uint32_t ring_size = mesh_shape[cluster_axis_value];
    TT_FATAL(ring_size % 2 == 0, "ring_shift requires an even number of devices in the ring, got {}", ring_size);

    if (ring_size <= 1U) {
        return tensor;
    }

    std::vector<std::pair<tt::tt_metal::CoreCoord, tt::tt_metal::CoreCoord>> send_recv_logical_coord = {
        {tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(0, 1)},
    };

    // we need to perform the ring shift in two stages as send is blocking and will hang
    // if the tensor is larger than the socket fifo size.
    // TBD: create a fused ring shift ccl.
    std::vector<tt::tt_metal::distributed::SocketConnection> even_to_odd_connections;
    std::vector<tt::tt_metal::distributed::SocketConnection> odd_to_even_connections;
    even_to_odd_connections.reserve(send_recv_logical_coord.size() * mesh_device_ptr->num_devices() / 2);
    odd_to_even_connections.reserve(send_recv_logical_coord.size() * mesh_device_ptr->num_devices() / 2);

    for (auto sender_coord : MeshCoordinateRange(mesh_shape)) {
        const uint32_t idx = sender_coord[cluster_axis_value];
        const uint32_t target_idx = forward ? (idx + 1) % ring_size : (idx + ring_size - 1) % ring_size;

        tt::tt_fabric::MeshCoordinate recv_coord = sender_coord;
        recv_coord[cluster_axis_value] = target_idx;

        std::vector<tt::tt_metal::distributed::SocketConnection> connections;
        connections.reserve(send_recv_logical_coord.size());
        for (auto [sender_core, recv_core] : send_recv_logical_coord) {
            auto connection = tt::tt_metal::distributed::SocketConnection{
                tt::tt_metal::distributed::MeshCoreCoord{sender_coord, sender_core},
                tt::tt_metal::distributed::MeshCoreCoord{recv_coord, recv_core}};
            connections.push_back(connection);
        }
        if (idx % 2U == 0U) {
            std::copy(connections.begin(), connections.end(), std::back_inserter(even_to_odd_connections));
        } else {
            std::copy(connections.begin(), connections.end(), std::back_inserter(odd_to_even_connections));
        }
    }

    tt::tt_metal::distributed::SocketMemoryConfig socket_mem_config{};
    socket_mem_config.socket_storage_type = ttnn::BufferType::L1;
    socket_mem_config.fifo_size = 128U * 1024U;  // 128KB FIFO

    auto output_tensor = ttnn::empty_like(tensor);

    auto send_through_sockets = [&](const std::vector<tt::tt_metal::distributed::SocketConnection>& connections,
                                    const tt::tt_metal::Tensor& input_tensor,
                                    tt::tt_metal::Tensor& output_tensor) {
        tt::tt_metal::distributed::SocketConfig config{};
        config.socket_mem_config = socket_mem_config;
        config.socket_connection_config = connections;

        auto [send_socket, recv_socket] =
            tt::tt_metal::distributed::MeshSocket::create_socket_pair(mesh_device_ptr, mesh_device_ptr, config);

        ttnn::experimental::send_async(tensor, send_socket);
        ttnn::experimental::recv_async(output_tensor, recv_socket);
        tt::tt_metal::distributed::Synchronize(
            mesh_device_ptr.get(), std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    };

    // Phase 1: Even → Odd (even sends, odd receives)
    send_through_sockets(even_to_odd_connections, tensor, output_tensor);
    // Phase 2: Odd → Even (odd sends, even receives)
    send_through_sockets(odd_to_even_connections, tensor, output_tensor);

    return output_tensor;
}

}  // namespace ttml::ttnn_fixed::distributed
