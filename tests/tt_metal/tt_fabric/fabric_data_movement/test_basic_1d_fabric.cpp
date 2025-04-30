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

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "fabric_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/system_memory_manager.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "umd/device/tt_core_coordinates.h"

static const char* SENDER_KERNEL_PATH =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp";
static const char* RECEIVER_KERNEL_PATH =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp";
namespace tt::tt_fabric {
namespace fabric_router_tests {
struct TestParameters {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = 4096;
    uint32_t num_packets = 10;
    uint32_t num_hops = 1;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    bool is_mcast = false;
};
struct ProgramInfo {
    tt::tt_metal::IDevice* device;
    tt::tt_metal::Program& program;
    CoreCoord core;
};

void LaunchAndSyncPhase(Fabric1DFixture* fixture, std::vector<ProgramInfo>& infos) {
    for (auto& info : infos) {
        fixture->RunProgramNonblocking(info.device, info.program);
    }
    for (auto& info : infos) {
        fixture->WaitForSingleProgramDone(info.device, info.program);
    }
}

void ValidationPhase(std::vector<ProgramInfo>& infos, TestParameters& params) {
    uint64_t prev_bytes;
    for (uint32_t i = 0; i < infos.size(); i++) {
        auto& info = infos[i];
        std::vector<uint32_t> status;
        tt_metal::detail::ReadFromDeviceL1(
            info.device,
            info.core,
            params.test_results_address,
            params.test_results_size_bytes,
            status,
            CoreType::WORKER);

        EXPECT_EQ(status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t next_bytes = ((uint64_t)status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | status[TT_FABRIC_WORD_CNT_INDEX];
        if (i > 0) {
            EXPECT_EQ(prev_bytes, next_bytes);
        }
        prev_bytes = next_bytes;
    }
}

std::tuple<chip_id_t, std::vector<chip_id_t>, uint32_t, CoreCoord> GetConnectedDeviceInfo(
    Fabric1DFixture* fixture,
    CoreCoord sender_logical_core,
    CoreCoord receiver_logical_core,
    bool is_mcast = false,
    bool get_active_eth_core = false) {
    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    chip_id_t src_physical_device_id = 0, left_physical_device_id = 0, right_physical_device_id = 0;
    std::vector<chip_id_t> dest_chip_ids;
    CoreCoord edm_eth_core = {0, 0};
    if (!is_mcast) {
        std::pair<mesh_id_t, chip_id_t> src_mesh_chip_id;
        std::pair<mesh_id_t, chip_id_t> dst_mesh_chip_id;
        chip_id_t not_used_1;
        chip_id_t not_used_2;
        // Find a device with a neighbour in the East direction
        bool connection_found = fixture->find_device_with_neighbor_in_direction(
            src_mesh_chip_id, dst_mesh_chip_id, not_used_1, not_used_2, RoutingDirection::E);
        if (!connection_found) {
            return std::make_tuple(0, dest_chip_ids, 0, CoreCoord{0, 0});
        }

        src_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(src_mesh_chip_id);
        left_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);

        if (get_active_eth_core) {
            std::set<chan_id_t> eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(
                src_mesh_chip_id.first, src_mesh_chip_id.second, RoutingDirection::E);
            if (eth_chans.size() == 0) {
                return std::make_tuple(0, dest_chip_ids, 0, CoreCoord{0, 0});
            }
            auto edm_port = *(eth_chans.begin());
            edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                src_physical_device_id, edm_port);
        }
        dest_chip_ids.push_back(left_physical_device_id);
    } else {
        // use control plane to find a mesh with 3 devices
        auto user_meshes = control_plane->get_user_physical_mesh_ids();
        std::optional<mesh_id_t> mesh_id;
        for (const auto& mesh : user_meshes) {
            auto mesh_shape = control_plane->get_physical_mesh_shape(mesh);
            if (mesh_shape.mesh_size() >= 3) {
                mesh_id = mesh;
                break;
            }
        }
        if (!mesh_id.has_value()) {
            return std::make_tuple(0, dest_chip_ids, 0, CoreCoord{0, 0});
        }

        // for this test, logical chip id 1 is the sender, 0 is the left receiver and 1 is the right receiver
        auto src_physical_device_id =
            control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), 1));
        auto left_physical_device_id =
            control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), 0));
        auto right_physical_device_id =
            control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), 2));
        dest_chip_ids.push_back(left_physical_device_id);
        dest_chip_ids.push_back(right_physical_device_id);
    }

    auto sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto receiver_l_device = DevicePool::instance().get_active_device(left_physical_device_id);
    auto receiver_r_device = DevicePool::instance().get_active_device(right_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_l_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
        receiver_virtual_core.x, receiver_virtual_core.y);  //

    // return tuple
    return std::make_tuple(src_physical_device_id, dest_chip_ids, receiver_noc_encoding, edm_eth_core);
}

void CreateRx(
    TestParameters& params, tt::tt_metal::Program& receiver_program, std::vector<uint32_t>& compile_time_args) {
    std::vector<uint32_t> receiver_runtime_args = {
        params.packet_payload_size_bytes, params.num_packets, params.time_seed};
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        RECEIVER_KERNEL_PATH,
        {params.receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, params.receiver_logical_core, receiver_runtime_args);
}

void CreateTx(
    TestParameters& params,
    tt::tt_metal::Program& sender_program,
    std::vector<uint32_t>& compile_time_args,
    uint32_t receiver_noc_encoding,
    chip_id_t src_chip_id,
    std::vector<chip_id_t>& dest_chip_ids) {
    std::vector<uint32_t> sender_runtime_args = {
        params.packet_header_address,
        params.source_l1_buffer_address,
        params.packet_payload_size_bytes,
        params.num_packets,
        receiver_noc_encoding,
        params.time_seed};
    sender_runtime_args.push_back(params.num_hops);
    if (dest_chip_ids.size() > 1) {
        /* mcast_bwd_hops for mcast */
        sender_runtime_args.push_back(params.num_hops);
    }
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        SENDER_KERNEL_PATH,
        {params.sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});
    for (auto dest_chip_id : dest_chip_ids) {
        append_fabric_connection_rt_args(
            src_chip_id, dest_chip_id, 0, sender_program, {params.sender_logical_core}, sender_runtime_args);
    }
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, params.sender_logical_core, sender_runtime_args);
}

void ExecuteFabricTest(Fabric1DFixture* fixture, TestParameters params) {
    auto [src_physical_device_id, dst_chip_ids, receiver_noc_encoding, edm_eth_core] = GetConnectedDeviceInfo(
        fixture, params.sender_logical_core, params.receiver_logical_core, params.is_mcast, false);
    if (src_physical_device_id == 0 && dst_chip_ids.empty() && receiver_noc_encoding == 0) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    std::vector<uint32_t> compile_time_args = {
        params.test_results_address,
        params.test_results_size_bytes,
        params.target_address,
        params.is_mcast ? 1 : 0,
    };

    auto sender_program = tt_metal::CreateProgram();
    auto receiver_program = tt_metal::CreateProgram();
    CreateTx(params, sender_program, compile_time_args, receiver_noc_encoding, src_physical_device_id, dst_chip_ids);
    CreateRx(params, receiver_program, compile_time_args);

    std::vector<ProgramInfo> program_infos;
    auto sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    program_infos.emplace_back(sender_device, sender_program, params.sender_logical_core);

    // [0] is left receiver, [1] is right receiver for mcast
    for (auto dest_chip_id : dst_chip_ids) {
        auto receiver_device = DevicePool::instance().get_active_device(dest_chip_id);
        program_infos.emplace_back(receiver_device, receiver_program, params.receiver_logical_core);
    }
    LaunchAndSyncPhase(fixture, program_infos);
    ValidationPhase(program_infos, params);
}

TEST_F(Fabric1DFixture, TestUnicastRaw) {
    // test parameters (default)
    TestParameters params;

    auto [src_physical_device_id, dst_physical_device_ids, receiver_noc_encoding, edm_eth_core] =
        GetConnectedDeviceInfo(this, params.sender_logical_core, params.receiver_logical_core, false, true);
    if (src_physical_device_id == 0 && dst_physical_device_ids.size() == 0 && receiver_noc_encoding == 0) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    auto sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto receiver_device = DevicePool::instance().get_active_device(dst_physical_device_ids[0]);

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        params.test_results_address, params.test_results_size_bytes, params.target_address, 0 /* mcast_mode */
    };

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        SENDER_KERNEL_PATH,
        {params.sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> sender_runtime_args = {
        params.packet_header_address,
        params.source_l1_buffer_address,
        params.packet_payload_size_bytes,
        params.num_packets,
        receiver_noc_encoding,
        params.time_seed,
        params.num_hops};

    // append the EDM connection rt args
    const auto edm_config = get_1d_fabric_config();

    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = edm_eth_core.x,
        .edm_noc_y = edm_eth_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[0],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[0],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[0],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[0],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[0],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[0],
        .persistent_fabric = true};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(sender_program, params.sender_logical_core, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, params.sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, params.sender_logical_core, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, params.sender_logical_core, sender_runtime_args);

    auto receiver_program = tt_metal::CreateProgram();
    CreateRx(params, receiver_program, compile_time_args);

    std::vector<ProgramInfo> program_infos;
    program_infos.emplace_back(sender_device, sender_program, params.sender_logical_core);
    program_infos.emplace_back(receiver_device, receiver_program, params.receiver_logical_core);

    LaunchAndSyncPhase(this, program_infos);
    ValidationPhase(program_infos, params);
}

TEST_F(Fabric1DFixture, TestUnicastConnAPI) {
    // Test parameters for unicast (default).
    TestParameters params;
    ExecuteFabricTest(this, params);
}

TEST_F(Fabric1DFixture, TestMCastConnAPI) {
    // Test parameters for mcast.
    TestParameters params = {
        .num_packets = 100,  // mcast test uses more packets
        .is_mcast = true};
    ExecuteFabricTest(this, params);
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
