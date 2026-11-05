// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <tt-umd/types/core_coordinates.hpp>

namespace tt::tt_fabric::fabric_router_tests {

void RunTestUnicastSmoke(BaseFabricFixture* fixture) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = fixture->get_devices();

    // Need exactly 2 devices for smoke test
    if (devices.size() != 2) {
        GTEST_SKIP() << "Smoke test requires exactly 2 devices";
    }

    // Use first two devices for simple smoke test
    auto sender_device = devices[0];
    auto receiver_device = devices[1];

    auto src_physical_device_id = sender_device->get_device_ids()[0];
    auto dst_physical_device_id = receiver_device->get_device_ids()[0];

    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    const FabricTestWorker sender = BaseFabricFixture::get_fabric_test_worker(sender_device, {0, 0});
    const FabricTestWorker receiver = BaseFabricFixture::get_fabric_test_worker(receiver_device, {1, 0});

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
    auto worker_mem_map = fixture->generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 5;  // Small number for smoke test
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

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
    auto sender_program = tt_metal::CreateProgram(*sender_device);
    auto sender_kernel = BaseFabricFixture::create_fabric_test_worker_kernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        sender,
        compile_time_args,
        defines);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver.noc_core.x,
        receiver.noc_core.y,
        1, /* num_hops - simple for smoke */
        1, /* fwd_range */
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    auto worker_teardown_semaphore_id =
        tt_metal::CreateSemaphore(sender_program, sender.logical, 0, sender.core_type);
    auto worker_buffer_index_semaphore_id =
        tt_metal::CreateSemaphore(sender_program, sender.logical, 0, sender.core_type);
    append_worker_to_fabric_edm_sender_rt_args(
        edm_port, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender.logical, sender_runtime_args);

    // Create receiver program
    auto receiver_program = tt_metal::CreateProgram(*receiver_device);
    auto receiver_kernel = BaseFabricFixture::create_fabric_test_worker_kernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        receiver,
        compile_time_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver.logical, receiver_runtime_args);

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        sender_program.impl().compile(&tt_metal::slow_dispatch::physical_device_from_unit_mesh(*sender_device));
        receiver_program.impl().compile(&tt_metal::slow_dispatch::physical_device_from_unit_mesh(*receiver_device));
        GTEST_SKIP() << "Mock: kernels compiled. MockChip reads and writes are no-ops, so nothing launches.";
    }

    // Run programs
    fixture->RunProgramNonblocking(receiver_device, std::move(receiver_program));
    tt_metal::LaunchProgram(*sender_device, std::move(sender_program));
    fixture->WaitForSingleProgramDone(receiver_device);

    // Validate results
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::slow_dispatch::ReadFromL1(
        *sender_device,
        sender.logical,
        worker_mem_map.test_results_address - worker_mem_map.kernel_view_offset,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        sender.core_type);

    tt_metal::slow_dispatch::ReadFromL1(
        *receiver_device,
        receiver.logical,
        worker_mem_map.test_results_address - worker_mem_map.kernel_view_offset,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        receiver.core_type);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
    EXPECT_EQ(sender_bytes, uint64_t{worker_mem_map.packet_payload_size_bytes} * num_packets);

    // On Keraunos, as chippy's compat harness checks: after the traffic, closing the devices terminates every master
    // router. Unless a router did not terminate, the devices are then re-opened, so the fixture's other tests find
    // them as before.
    if (BaseFabricFixture::workers_on_ethernet_cores(sender_device)) {
        constexpr auto kTerminateTimeout = std::chrono::minutes(10);
        const auto fabric_config = tt::tt_fabric::GetFabricConfig();
        const auto routers = BaseFabricFixture::get_master_routers();
        BaseFabricFixture::DoTearDownTestSuite();
        for (const auto& router : routers) {
            EXPECT_EQ(
                BaseFabricFixture::wait_for_router_terminated(router, kTerminateTimeout),
                static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED))
                << "chip " << router.chip << " router " << router.logical.str();
        }
        if (::testing::Test::HasFailure()) {
            return;
        }
        BaseFabricFixture::DoSetUpTestSuite(fabric_config);
    }
}

TEST_F(Fabric2DFixture, TestUnicastConnAPI2DSmoke) { RunTestUnicastSmoke(this); }
TEST_F(Fabric1DFixture, TestUnicastConnAPI1DSmoke) { RunTestUnicastSmoke(this); }

}  // namespace tt::tt_fabric::fabric_router_tests
