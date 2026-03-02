// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end smoke test that verifies the TT_METAL_FABRIC_OPT_LEVEL override
// is correctly threaded through to the fabric router kernel compilation.
// Sets the opt level via rtoptions setter and runs a simple 1D unicast.

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
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <umd/device/types/core_coordinates.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric::fabric_router_tests {

TEST_F(Fabric1DFixture, FabricOptLevelTest_UnicastSmokeWithOsOverride) {
    // Override the fabric kernel opt level via rtoptions setter
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto original_opt_level = rtoptions.get_fabric_kernel_opt_level();
    rtoptions.set_fabric_kernel_opt_level(tt::tt_metal::KernelBuildOptLevel::Os);

    // Verify it was set
    ASSERT_TRUE(rtoptions.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(rtoptions.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::Os);

    // Run a simple unicast smoke test to verify fabric works with the override
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = get_devices();

    if (devices.size() < 2) {
        rtoptions.set_fabric_kernel_opt_level(original_opt_level);
        GTEST_SKIP() << "Smoke test requires at least 2 devices";
    }

    auto sender_device = devices[0];
    auto receiver_device = devices[1];

    auto src_physical_device_id = sender_device->get_devices()[0]->id();
    auto dst_physical_device_id = receiver_device->get_devices()[0]->id();

    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    auto eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id);
    if (eth_chans.empty()) {
        rtoptions.set_fabric_kernel_opt_level(original_opt_level);
        GTEST_SKIP() << "No fabric connection available between device 0 and device 1";
    }
    auto edm_port = *eth_chans.begin();

    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 5;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0,
        topology == Topology::Mesh,
        0,
        0};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

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
        1,
        1,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    append_worker_to_fabric_edm_sender_rt_args(
        edm_port, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

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

    RunProgramNonblocking(receiver_device, receiver_program);
    RunProgramNonblocking(sender_device, sender_program);
    WaitForSingleProgramDone(sender_device, sender_program);
    WaitForSingleProgramDone(receiver_device, receiver_program);

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

    // Restore original opt level
    rtoptions.set_fabric_kernel_opt_level(original_opt_level);
}

}  // namespace tt::tt_fabric::fabric_router_tests
