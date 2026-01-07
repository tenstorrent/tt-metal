// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Basically, this is clone of the microbenchmark program `run_unicast_once.cpp`
// Check it for more details.

#include <functional>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "common.hpp"
#include "fabric_benchmark_units.hpp"

#include <cstdint>
#include <vector>
#include <chrono>
#include <utility>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Check number of devices
    auto num_devices = GetNumAvailableDevices();
    TT_FATAL(
        num_devices > 1, "Currently {} devices are available. Cannot test number of devices under two.", num_devices);

    // ------------ Setup MeshDevice ------------
    // Initialize test mesh_desc
    MeshDescriptor mesh_desc{};

    // Set system mesh shape as default
    mesh_desc.mesh_shape = distributed::SystemMesh::instance().shape();

    // Extract core mesh_desc
    auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
    bool is_n300_or_t3k_cluster = cluster_type == ClusterType::T3K or cluster_type == ClusterType::N300;
    auto core_type =
        (mesh_desc.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

    // Create a mesh device
    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::FABRIC_2D);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(mesh_desc.mesh_shape),
        mesh_desc.l1_small_size,
        mesh_desc.trace_region_size,
        mesh_desc.num_cqs,
        core_type,
        {},
        mesh_desc.worker_l1_size);

    // ------------ Setup Fabric ------------
    // Initialize fabric test descriptor
    FabricTestDescriptor fabric_desc{};

    // Get control plance instance
    const auto& control_plane = MetalContext::instance().get_control_plane();

    // print all fabric packet spec info
    log_info(tt::LogTest, "fabric packet header size bytes : {}", tt_fabric::get_tt_fabric_packet_header_size_bytes());
    log_info(tt::LogTest, "fabric max payload size bytes : {}", tt_fabric::get_tt_fabric_max_payload_size_bytes());
    log_info(
        tt::LogTest, "fabric channel buffer size bytes : {}", tt_fabric::get_tt_fabric_channel_buffer_size_bytes());

    // Set fabric node ids. One is composed of {MeshId, ChipId}
    // MeshId is same for both src and dst because it is intra-routing test.
    tt_fabric::FabricNodeId src_fabric_node{tt_fabric::MeshId{fabric_desc.mesh_id}, fabric_desc.src_chip};
    tt_fabric::FabricNodeId dst_fabric_node{tt_fabric::MeshId{fabric_desc.mesh_id}, fabric_desc.dst_chip};

    auto src_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node);
    auto dst_phy_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node);

    auto src_dev = tt::tt_metal::detail::GetActiveDevice(src_phy_id);
    auto dst_dev = tt::tt_metal::detail::GetActiveDevice(dst_phy_id);
    TT_FATAL(src_dev && dst_dev, "Both devices should be valid.");

    distributed::MeshCoordinate src_mesh_coord = extract_coord_of_phy_id(mesh_device, src_phy_id);
    distributed::MeshCoordinate dst_mesh_coord = extract_coord_of_phy_id(mesh_device, dst_phy_id);

    // ------------ Setup MeshBuffer ------------
    const uint32_t uint_size = sizeof(uint32_t);
    constexpr uint32_t page_size = uint_size * tt::constants::TILE_HW;
    const uint32_t buffer_size = 1u << 20;  // 1MiB

    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig global_config{.size = buffer_size};
    auto src_buf = distributed::MeshBuffer::create(global_config, dram_local_config, mesh_device.get());
    auto dst_buf = distributed::MeshBuffer::create(global_config, dram_local_config, mesh_device.get());

    distributed::DeviceLocalBufferConfig device_perf_local_config{
        .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig device_perf_global_config{.size = page_size};
    auto device_perf_buf =
        distributed::MeshBuffer::create(device_perf_global_config, device_perf_local_config, mesh_device.get());

    const auto num_words = buffer_size / uint_size;
    auto tx_send_data = make_src_data(num_words);
    std::vector<uint32_t> zeros(num_words, 0u);
    std::vector<uint32_t> perf_zeros(num_words, 0xDEADBEEF);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // blocking write data to buffers.
    distributed::WriteShard(cq, src_buf, tx_send_data, src_mesh_coord, /*blocking=*/true);
    distributed::WriteShard(cq, device_perf_buf, perf_zeros, src_mesh_coord, /*blocking=*/true);
    distributed::WriteShard(cq, dst_buf, zeros, dst_mesh_coord, /*blocking=*/true);

    auto rx_core_range_set = CoreRange(fabric_desc.receiver_core, fabric_desc.receiver_core);
    static std::optional<tt::tt_metal::GlobalSemaphore> global_sema_a;
    static std::optional<tt::tt_metal::GlobalSemaphore> global_sema_b;
    if (!global_sema_a) {
        global_sema_a = CreateGlobalSemaphore(mesh_device.get(), rx_core_range_set, /*init_val*/ 0);
    }
    if (!global_sema_b) {
        global_sema_b = CreateGlobalSemaphore(mesh_device.get(), rx_core_range_set, /*init_val*/ 0);
    }

    static uint32_t sem_sel = 0;
    auto& cur_global_sema = (sem_sel++ & 1) ? *global_sema_b : *global_sema_a;

    // ------------ Run each benchmarks ------------
    execute_default_intra_mesh_routing_bench(
        fabric_desc,
        cur_global_sema,
        buffer_size,
        page_size,
        mesh_device,
        src_fabric_node,
        dst_fabric_node,
        src_buf,
        dst_buf,
        tx_send_data);

    execute_incremental_packet_size_bench(
        fabric_desc,
        cur_global_sema,
        page_size,
        mesh_device,
        src_fabric_node,
        dst_fabric_node,
        src_buf,
        dst_buf,
        device_perf_buf);

    // Teardown mesh device
    mesh_device->close();
    mesh_device.reset();
}
