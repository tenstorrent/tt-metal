// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/allocator.hpp>
#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/fabric_mux_config.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric {
namespace fabric_router_tests {
namespace fabric_mux_tests {

const std::string mux_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
const std::string sender_kernel_src =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_sender_client.cpp";
const std::string receiver_kernel_src =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_receiver_client.cpp";

const uint32_t test_results_size_bytes = 128;
const uint32_t noc_address_padding_bytes = 16;

using FabricMuxFixture = Fabric1DFixture;

struct TestConfig {
    uint32_t num_sender_clients = 0;
    uint32_t num_packets = 0;
    uint32_t num_credits = 0;
    uint32_t num_return_credits_per_packet = 0;
    uint32_t packet_payload_size_bytes = 0;
    uint8_t num_full_size_channels = 0;
    uint8_t num_header_only_channels = 0;
    uint8_t num_buffers_full_size_channel = 0;
    uint8_t num_buffers_header_only_channel = 0;
};

void run_two_chip_test_variant(FabricMuxFixture* fixture, TestConfig test_config) {
    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    std::pair<mesh_id_t, chip_id_t> mesh_chip_id_0;
    std::pair<mesh_id_t, chip_id_t> mesh_chip_id_1;
    chip_id_t device_id_0;
    chip_id_t device_id_1;

    // Find a device with a neighbour in the East direction (N300 has E<>W chips only)
    bool connection_found = fixture->find_device_with_neighbor_in_direction(
        mesh_chip_id_0, mesh_chip_id_1, device_id_0, device_id_1, RoutingDirection::E);
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    std::vector<tt::tt_metal::IDevice*> devices = {
        DevicePool::instance().get_active_device(device_id_0), DevicePool::instance().get_active_device(device_id_1)};
    std::vector<tt::tt_metal::IDevice*> dest_devices = devices;
    std::reverse(dest_devices.begin(), dest_devices.end());

    std::vector<tt::tt_metal::Program> program_handles(devices.size());

    std::vector<CoreCoord> worker_logical_cores;
    auto grid_size = devices[0]->compute_with_storage_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            worker_logical_cores.push_back(CoreCoord({i, j}));
        }
    }

    // there will be one receiver client per sender client
    // sender clients on device_id_0 send payloads to receiver clients on device_id_1
    // receiver clients on device_id_1 send credits back to sender clients on device_id_0
    // sender clients on device_id_1 send payloads to receiver clients on device_id_0
    // receiver clients on device_id_0 send credits back to sender clients on device_id_1

    // test config
    const uint32_t num_sender_clients = test_config.num_sender_clients;
    const uint32_t num_packets = test_config.num_packets;
    const uint32_t num_credits = test_config.num_credits;
    const uint32_t num_return_credits_per_packet = test_config.num_return_credits_per_packet;
    const uint32_t packet_payload_size_bytes = test_config.packet_payload_size_bytes;
    const uint8_t num_full_size_channels = test_config.num_full_size_channels;
    const uint8_t num_header_only_channels = test_config.num_header_only_channels;
    const uint8_t num_buffers_full_size_channel = test_config.num_buffers_full_size_channel;
    const uint8_t num_buffers_header_only_channel = test_config.num_buffers_header_only_channel;

    // mux config
    const size_t buffer_size_bytes_full_size_channel =
        sizeof(tt::tt_fabric::PacketHeader) + packet_payload_size_bytes;  // packet header + 4K payload

    const uint32_t l1_unreserved_base_address =
        devices[0]->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    uint32_t num_hops = 1;  // the sender and reciever devices are adjacent for this test

    // memory map for senders and receivers
    const uint32_t test_results_address = l1_unreserved_base_address;

    // address to load the mux status into
    const uint32_t local_mux_status_address = test_results_address + test_results_size_bytes;
    const uint32_t local_flow_control_address = local_mux_status_address + noc_address_padding_bytes;
    const uint32_t local_teardown_address = local_flow_control_address + noc_address_padding_bytes;
    const uint32_t local_buffer_index_address = local_teardown_address + noc_address_padding_bytes;
    const uint32_t credit_handshake_address = local_buffer_index_address + noc_address_padding_bytes;
    const uint32_t packet_header_buffer_address = credit_handshake_address + noc_address_padding_bytes;

    // address at which the sender will write packets to in the receiver's L1
    const uint32_t base_l1_target_address = packet_header_buffer_address + 1024;

    // address at which the payload will reside in the sender's L1
    const uint32_t payload_buffer_address = base_l1_target_address;

    std::vector<uint32_t> worker_zero_vec(1, 0);
    auto zero_out_worker_address = [&](tt::tt_metal::IDevice* device, CoreCoord& logical_core, uint32_t address) {
        tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, address, worker_zero_vec);
    };

    std::vector<CoreCoord> mux_logical_cores;
    std::vector<size_t> mux_termiation_signal_addresses;
    std::vector<std::vector<CoreCoord>> sender_logical_cores;
    std::vector<std::vector<CoreCoord>> receiver_logical_cores;

    for (auto i = 0; i < devices.size(); i++) {
        program_handles[i] = tt_metal::CreateProgram();
        // use logical core (0,0) for the mux kernel
        CoreCoord mux_logical_core = worker_logical_cores[0];
        CoreCoord mux_virtual_core = devices[i]->worker_core_from_logical_core(mux_logical_core);
        mux_logical_cores.push_back(mux_logical_core);

        auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            num_buffers_full_size_channel,
            num_buffers_header_only_channel,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);
        mux_termiation_signal_addresses.push_back(mux_kernel_config.get_termination_signal_address());

        std::vector<uint32_t> mux_ct_args = mux_kernel_config.get_fabric_mux_compile_time_args();

        std::vector<uint32_t> mux_rt_args;
        append_fabric_connection_rt_args(
            devices[i]->id(),
            dest_devices[i]->id(),
            0 /* link_idx (routing plane) */,
            program_handles[i],
            {mux_logical_core},
            mux_rt_args);

        auto mux_kernel = tt_metal::CreateKernel(
            program_handles[i],
            mux_kernel_src,
            {mux_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_ct_args});
        tt_metal::SetRuntimeArgs(program_handles[i], mux_kernel, mux_logical_core, mux_rt_args);

        std::vector<uint32_t> mux_zero_vec((mux_kernel_config.get_num_bytes_to_clear() / sizeof(uint32_t)), 0);
        tt::tt_metal::detail::WriteToDeviceL1(
            devices[i], mux_logical_core, mux_kernel_config.get_start_address_to_clear(), mux_zero_vec);

        std::vector<CoreCoord> sender_cores = {};
        std::vector<CoreCoord> receiver_cores = {};

        for (uint32_t sender_id = 0; sender_id < num_sender_clients; sender_id++) {
            // skip the 1st worker core for the mux kernel
            CoreCoord sender_logical_core = worker_logical_cores[1 + sender_id];
            CoreCoord sender_virtual_core = devices[i]->worker_core_from_logical_core(sender_logical_core);
            sender_cores.push_back(sender_logical_core);

            // receiver workers on this device will also connect to the mux kerel to send credits back
            CoreCoord receiver_logical_core = worker_logical_cores[worker_logical_cores.size() - 1 - sender_id];
            CoreCoord receiver_virtual_core = devices[i]->worker_core_from_logical_core(receiver_logical_core);
            receiver_cores.push_back(receiver_logical_core);

            std::vector<uint32_t> sender_ct_args = {
                mux_virtual_core.x,
                mux_virtual_core.y,
                num_buffers_full_size_channel,
                buffer_size_bytes_full_size_channel,
                mux_kernel_config.get_channel_base_address(
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, sender_id),
                mux_kernel_config.get_connection_info_address(
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, sender_id),
                mux_kernel_config.get_connection_handshake_address(
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, sender_id),
                mux_kernel_config.get_flow_control_address(
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, sender_id),
                mux_kernel_config.get_buffer_index_address(
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, sender_id),
                mux_kernel_config.get_status_address()};

            std::vector<uint32_t> sender_rt_args = {
                num_packets,
                num_credits,
                packet_payload_size_bytes,
                time_seed,
                num_hops,
                local_mux_status_address,
                local_flow_control_address,
                local_teardown_address,
                local_buffer_index_address,
                test_results_address,
                test_results_size_bytes,
                base_l1_target_address,
                credit_handshake_address,
                packet_header_buffer_address,
                payload_buffer_address};

            // sender id will be used to populate the packet payload which will be verified on the receiver
            sender_rt_args.push_back(sender_id);

            // virtual coordinates will be the same for the receiver device
            // hence, we can use the noc encoding derived using current device
            sender_rt_args.push_back(tt_metal::MetalContext::instance().hal().noc_xy_encoding(
                receiver_virtual_core.x, receiver_virtual_core.y));

            auto sender_kernel = tt_metal::CreateKernel(
                program_handles[i],
                sender_kernel_src,
                {sender_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = sender_ct_args});
            tt_metal::SetRuntimeArgs(program_handles[i], sender_kernel, sender_logical_core, sender_rt_args);

            zero_out_worker_address(devices[i], sender_logical_core, local_flow_control_address);
            zero_out_worker_address(devices[i], sender_logical_core, local_teardown_address);
            zero_out_worker_address(devices[i], sender_logical_core, local_buffer_index_address);

            // reuse the sender ids as reciever ids since each device has the same number of senders and receivers
            std::vector<uint32_t> receiver_ct_args = {
                mux_virtual_core.x,
                mux_virtual_core.y,
                num_buffers_header_only_channel,
                sizeof(tt::tt_fabric::PacketHeader),
                mux_kernel_config.get_channel_base_address(
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL, sender_id),
                mux_kernel_config.get_connection_info_address(
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL, sender_id),
                mux_kernel_config.get_connection_handshake_address(
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL, sender_id),
                mux_kernel_config.get_flow_control_address(
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL, sender_id),
                mux_kernel_config.get_buffer_index_address(
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL, sender_id),
                mux_kernel_config.get_status_address()};

            std::vector<uint32_t> receiver_rt_args = {
                num_packets,
                num_credits,
                num_return_credits_per_packet,
                packet_payload_size_bytes,
                time_seed,
                num_hops,
                local_mux_status_address,
                local_flow_control_address,
                local_teardown_address,
                local_buffer_index_address,
                test_results_address,
                test_results_size_bytes,
                base_l1_target_address,
                credit_handshake_address,
                packet_header_buffer_address};

            // sender id will be used to populate the packet payload which will be verified on the receiver
            receiver_rt_args.push_back(sender_id);

            // virtual coordinates will be the same for the receiver device
            // hence, we can use the noc encoding derived using current device
            receiver_rt_args.push_back(
                tt_metal::MetalContext::instance().hal().noc_xy_encoding(sender_virtual_core.x, sender_virtual_core.y));

            auto receiver_kernel = tt_metal::CreateKernel(
                program_handles[i],
                receiver_kernel_src,
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = receiver_ct_args});
            tt_metal::SetRuntimeArgs(program_handles[i], receiver_kernel, receiver_logical_core, receiver_rt_args);

            zero_out_worker_address(devices[i], receiver_logical_core, local_flow_control_address);
            zero_out_worker_address(devices[i], receiver_logical_core, local_teardown_address);
            zero_out_worker_address(devices[i], receiver_logical_core, local_buffer_index_address);
        }

        sender_logical_cores.push_back(sender_cores);
        receiver_logical_cores.push_back(receiver_cores);
    }

    log_info(LogTest, "Running programs");
    for (auto i = 0; i < devices.size(); i++) {
        fixture->RunProgramNonblocking(devices[i], program_handles[i]);
    }

    log_info(LogTest, "Waiting for senders to complete");
    for (auto i = 0; i < devices.size(); i++) {
        for (auto j = 0; j < sender_logical_cores[i].size(); j++) {
            std::vector<uint32_t> sender_status(1, 0);
            while ((sender_status[0] & 0xFFFF) == 0) {
                tt_metal::detail::ReadFromDeviceL1(
                    devices[i], sender_logical_cores[i][j], test_results_address, 4, sender_status);
            }
        }
    }

    log_info(LogTest, "Senders done, waiting for receivers to complete");
    for (auto i = 0; i < devices.size(); i++) {
        for (auto j = 0; j < receiver_logical_cores[i].size(); j++) {
            std::vector<uint32_t> receiver_status(1, 0);
            while ((receiver_status[0] & 0xFFFF) == 0) {
                tt_metal::detail::ReadFromDeviceL1(
                    devices[i], receiver_logical_cores[i][j], test_results_address, 4, receiver_status);
            }
        }
    }

    log_info(LogTest, "Receivers done, terminating mux kernel");
    std::vector<uint32_t> mux_termiation_signal(1, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    for (auto i = 0; i < devices.size(); i++) {
        tt::tt_metal::detail::WriteToDeviceL1(
            devices[i], mux_logical_cores[i], mux_termiation_signal_addresses[i], mux_termiation_signal);
    }

    log_info(LogTest, "Waiting for programs");
    for (auto i = 0; i < devices.size(); i++) {
        fixture->WaitForSingleProgramDone(devices[i], program_handles[i]);
    }

    log_info(LogTest, "Programs done, validating results");
    for (auto i = 0; i < devices.size(); i++) {
        for (auto j = 0; j < sender_logical_cores[i].size(); j++) {
            std::vector<uint32_t> sender_status;
            tt_metal::detail::ReadFromDeviceL1(
                devices[i], sender_logical_cores[i][j], test_results_address, test_results_size_bytes, sender_status);
            EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        }
    }

    for (auto i = 0; i < devices.size(); i++) {
        for (auto j = 0; j < receiver_logical_cores[i].size(); j++) {
            std::vector<uint32_t> receiver_status;
            tt_metal::detail::ReadFromDeviceL1(
                devices[i],
                receiver_logical_cores[i][j],
                test_results_address,
                test_results_size_bytes,
                receiver_status);
            log_info(
                LogTest,
                "[Device: Phys: {}] Receiver id: {}, status: {}, num packets processed: {}",
                devices[i]->id(),
                j,
                tt_fabric_status_to_string(receiver_status[TT_FABRIC_STATUS_INDEX]),
                receiver_status[TX_TEST_IDX_NPKT]);
            EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        }
    }
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant1) {
    TestConfig test_config = {
        .num_sender_clients = 1,
        .num_packets = 1000,
        .num_credits = 1,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 1,
        .num_header_only_channels = 1,
        .num_buffers_full_size_channel = 1,
        .num_buffers_header_only_channel = 1,
    };
    run_two_chip_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant2) {
    TestConfig test_config = {
        .num_sender_clients = 1,
        .num_packets = 1000,
        .num_credits = 1,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 4,
        .num_header_only_channels = 4,
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
    };
    run_two_chip_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant3) {
    TestConfig test_config = {
        .num_sender_clients = 2,
        .num_packets = 1000,
        .num_credits = 1,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 4,
        .num_header_only_channels = 4,
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
    };
    run_two_chip_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant4) {
    TestConfig test_config = {
        .num_sender_clients = 2,
        .num_packets = 1000,
        .num_credits = 16,
        .num_return_credits_per_packet = 4,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 4,
        .num_header_only_channels = 4,
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
    };
    run_two_chip_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant5) {
    TestConfig test_config = {
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 8,
        .num_header_only_channels = 8,
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
    };
    run_two_chip_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant6) {
    TestConfig test_config = {
        .num_sender_clients = 8,
        .num_packets = 50000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .num_full_size_channels = 8,
        .num_header_only_channels = 8,
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
    };
    run_two_chip_test_variant(this, test_config);
}

}  // namespace fabric_mux_tests
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
