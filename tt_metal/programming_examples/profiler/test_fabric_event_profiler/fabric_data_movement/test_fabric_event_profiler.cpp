// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "umd/device/tt_core_coordinates.h"

namespace tt::tt_fabric {
namespace fabric_router_tests {
std::random_device rd;  // Non-deterministic seed source
std::mt19937 global_rng(rd());

constexpr int RETCODE_INCOMPATIBLE_DEVICE = 95;
constexpr int RETCODE_SUCCESS = 0;

struct WorkerMemMap {
    uint32_t packet_header_address;
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap generate_worker_mem_map(tt_metal::IDevice* device, Topology topology) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t packet_header_address = base_addr;
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;

    uint32_t packet_payload_size_bytes = (topology == Topology::Mesh) ? 2048 : 4096;

    return {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        TEST_RESULTS_SIZE_BYTES};
}

std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

int RunFabricProfilerUnicastTest(BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // Find a device num_hops away in specified direction.
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    chip_id_t src_physical_device_id;
    chip_id_t dst_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    fabric_hops[direction] = num_hops;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    std::vector<chan_id_t> eth_chans;
    chan_id_t edm_port;

    const auto& fabric_context = control_plane->get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const auto& edm_config = fabric_context.get_fabric_router_config();

    uint32_t is_2d_fabric = topology == Topology::Linear || topology == Topology::Ring;
    TT_ASSERT(!is_2d_fabric);

    // Find a device with enough neighbours in the specified directions
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        tt::log_warning(tt::LogTest, "Cannot perform test because neighbors aren't available");
        return RETCODE_INCOMPATIBLE_DEVICE;
    }
    mesh_shape = control_plane->get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    dst_physical_device_id = physical_end_device_ids_by_dir[direction][num_hops - 1];
    dst_fabric_node_id = end_fabric_node_ids_by_dir[direction][num_hops - 1];

    // get a port to connect to
    eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);
    if (eth_chans.size() == 0) {
        tt::log_warning(tt::LogTest, "Cannot perform test because ethernet channels don't exist");
        return RETCODE_INCOMPATIBLE_DEVICE;
    }

    // Pick any port, for now pick the 1st one in the set
    edm_port = *eth_chans.begin();

    auto edm_direction = control_plane->get_eth_chan_direction(src_fabric_node_id, edm_port);
    CoreCoord edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        src_physical_device_id, edm_port);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* mcast_mode */,
        topology == Topology::Mesh,
        fabric_config == tt_metal::FabricConfig::FABRIC_2D_DYNAMIC};

    std::map<string, string> defines = {};

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tt_metal/programming_examples/profiler/test_fabric_event_profiler/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id,
        num_hops};

    // append the EDM connection rt args
    const auto sender_channel = topology == Topology::Mesh ? edm_direction : 0;
    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = edm_eth_core.x,
        .edm_noc_y = edm_eth_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
        .persistent_fabric = true,
        .edm_direction = edm_direction};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    tt_metal::DumpDeviceProfileResults(sender_device, sender_program);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    TT_ASSERT(sender_status[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);
    TT_ASSERT(receiver_status[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    TT_ASSERT(sender_bytes == receiver_bytes);

    return RETCODE_SUCCESS;
}
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric

int main() {
    tt::tt_fabric::fabric_router_tests::BaseFabricFixture fixture;
    fixture.SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_1D);
    int ret_code = tt::tt_fabric::fabric_router_tests::RunFabricProfilerUnicastTest(&fixture);
    fixture.TearDown();
    return ret_code;
}
