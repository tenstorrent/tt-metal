// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "hostdevcommon/fabric_common.h"
#include <vector>
#include "tt_metal/fabric/fabric_context.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_fabric::fabric_router_tests {

struct WorkerMemMap {
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t notification_mailbox_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap generate_worker_mem_map(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, Topology /*topology*/) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;
    uint32_t notification_mailbox_address = test_results_address + TEST_RESULTS_SIZE_BYTES;

    uint32_t packet_payload_size_bytes = get_tt_fabric_max_payload_size_bytes();

    return {
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        notification_mailbox_address,
        TEST_RESULTS_SIZE_BYTES};
}

void RunTestUnicastSmoke(BaseFabricFixture* fixture) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = fixture->get_devices();

    // Need exactly 2 devices for smoke test
    if (devices.size() != 2) {
        GTEST_SKIP() << "Smoke test requires exactly 2 devices";
    }

    // Use first two devices for simple smoke test
    auto sender_device = devices[0];
    auto receiver_device = devices[1];

    auto src_physical_device_id = sender_device->get_devices()[0]->id();
    auto dst_physical_device_id = receiver_device->get_devices()[0]->id();

    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    // Get fabric context and topology
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // Get available links between devices
    auto eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id);
    if (eth_chans.empty()) {
        GTEST_SKIP() << "No fabric connection available between device 0 and device 1";
    }
    auto edm_port = *eth_chans.begin();

    // Simple test parameters for smoke test
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 5;  // Small number for smoke test
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0, /* use_dram_dst */
        topology == Topology::Mesh,
        0, /* is_chip_multicast */
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        1, /* num_hops - simple for smoke */
        1, /* fwd_range */
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    append_worker_to_fabric_edm_sender_rt_args(
        edm_port, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create receiver program
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Run programs
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate results
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device->get_devices()[0],
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

TEST_F(Fabric2DFixture, TestUnicastConnAPI2DSmoke) { RunTestUnicastSmoke(this); }
TEST_F(Fabric1DFixture, TestUnicastConnAPI1DSmoke) { RunTestUnicastSmoke(this); }

}  // namespace tt::tt_fabric::fabric_router_tests
