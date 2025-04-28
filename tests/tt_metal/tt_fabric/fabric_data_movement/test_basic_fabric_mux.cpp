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
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/allocator.hpp>
#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/fabric_mux_config.hpp"

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

TEST_F(FabricMuxFixture, TestFabricMux) {
    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    std::pair<mesh_id_t, chip_id_t> mesh_chip_id_0;
    std::pair<mesh_id_t, chip_id_t> mesh_chip_id_1;
    chip_id_t device_id_0;
    chip_id_t device_id_1;

    // Find a device with a neighbour in the East direction (N300 has E<>W chips only)
    bool connection_found = find_device_with_neighbor_in_direction(
        mesh_chip_id_0, mesh_chip_id_1, device_id_0, device_id_1, RoutingDirection::E);
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    std::vector<tt_metal::IDevice*> devices = {
        DevicePool::instance().get_active_device(device_id_0), DevicePool::instance().get_active_device(device_id_1)};
    std::vector<tt_metal::IDevice*> dest_devices = devices;
    std::reverse(dest_devices.begin(), dest_devices.end());

    std::vector<tt::tt_metal::Program> program_handles(devices.size());

    std::vector<CoreCoord> worker_logical_cores;
    auto grid_size = devices[0]->compute_with_storage_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            worker_logical_cores.push_back(CoreCoord({i, j}));
        }
    }

    // sender clients on device_id_0 send payloads to receiver clients on device_id_1
    // receiver clients on device_id_1 send credits back to sender clients on device_id_0
    // sender clients on device_id_1 send payloads to receiver clients on device_id_0
    // receiver clients on device_id_0 send credits back to sender clients on device_id_1

    // there will be one receiver client per sender client
    const uint32_t num_sender_clients = 8;

    const uint32_t l1_unreserved_base_address =
        devices[0]->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    // mux config
    const uint8_t num_full_size_channels = 8;
    const uint8_t num_header_only_channels = 8;
    const uint8_t num_buffers_full_size_channel = 8;
    const uint8_t num_buffers_header_only_channel = 8;
    const size_t buffer_size_bytes_full_size_channel =
        sizeof(tt::tt_fabric::PacketHeader) + 4096;  // packet header + 4K payload
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    // test params
    const uint32_t num_packets = 5000;
    const uint32_t num_credits = 16;
    const uint32_t packet_payload_size_bytes = 4096;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    uint32_t num_hops = 1;  // the sender and reciever devices are adjacent for this test

    // memory map for senders and receivers
    const uint32_t test_results_address = l1_unreserved_base_address;

    // address which will be used for flow control b/w sender and fabric mux kernel
    const uint32_t local_flow_control_address = test_results_address + test_results_size_bytes;

    // address which will be used for connection teardown signalling
    const uint32_t local_teardown_address = local_flow_control_address + noc_address_padding_bytes;

    // address to sync buffer index
    const uint32_t local_buffer_index_address = local_teardown_address + noc_address_padding_bytes;

    // address at which the sender will write packets to in the receiver's L1
    // receiver will use the same address to send the credits back to the sender
    const uint32_t base_l1_target_address = local_buffer_index_address + noc_address_padding_bytes;
    const uint32_t credit_handshake_address = base_l1_target_address;

    // address for packet headers' buffer
    const uint32_t packet_header_buffer_address = credit_handshake_address + noc_address_padding_bytes;

    // address at which the payload will reside in the sender's L1
    const uint32_t payload_buffer_address = packet_header_buffer_address + 1024;

    for (auto i = 0; i < devices.size(); i++) {
        program_handles[i] = tt_metal::CreateProgram();
        // use logical core (0,0) for the mux kernel
        CoreCoord mux_logical_core = worker_logical_cores[0];
        CoreCoord mux_virtual_core = devices[i]->worker_core_from_logical_core(mux_logical_core);

        // TODO: pass mux logical/phys encoding
        // TODO: pass the phys chip id for the next chip in the sequence so that the mux can connect with the edm

        auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            num_buffers_full_size_channel,
            num_buffers_header_only_channel,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);

        std::vector<uint32_t> mux_ct_args = mux_kernel_config.get_fabric_mux_compile_time_args();
        std::vector<uint32_t> mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args();
        auto mux_kernel = tt_metal::CreateKernel(
            program_handles[i],
            mux_kernel_src,
            {mux_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_ct_args});
        tt_metal::SetRuntimeArgs(program_handles[i], mux_kernel, mux_logical_core, mux_rt_args);

        // TODO: zero out any addresses if needed

        for (uint32_t sender_id = 0; sender_id < num_sender_clients; sender_id++) {
            // skip the 1st worker core for the mux kernel
            CoreCoord sender_logical_core = worker_logical_cores[1 + sender_id];
            CoreCoord sender_virtual_core = devices[i]->worker_core_from_logical_core(sender_logical_core);

            // receiver workers on this device will also connect to the mux kerel to send credits back
            CoreCoord receiver_logical_core = worker_logical_cores[worker_logical_cores.size() - 1 - sender_id];
            CoreCoord receiver_virtual_core = devices[i]->worker_core_from_logical_core(receiver_logical_core);

            std::vector<uint32_t> sender_ct_args = {
                mux_virtual_core.x,
                mux_virtual_core.y,
                num_buffers_full_size_channel,
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
                local_flow_control_address,
                local_teardown_address,
                local_buffer_index_address};

            std::vector<uint32_t> sender_rt_args = {
                num_packets,
                num_credits,
                packet_payload_size_bytes,
                time_seed,
                num_hops,
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

            // TODO: zero out the addresses

            // reuse the sender ids as reciever ids since each device has the same number of senders and receivers
            std::vector<uint32_t> receiver_ct_args = {
                mux_virtual_core.x,
                mux_virtual_core.y,
                num_buffers_header_only_channel,
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
                local_flow_control_address,
                local_teardown_address,
                local_buffer_index_address};

            std::vector<uint32_t> receiver_rt_args = {
                num_packets,
                num_credits,
                packet_payload_size_bytes,
                time_seed,
                num_hops,
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

            // TODO: zero out the addresses
        }
    }

    log_info(LogTest, "running programs");
    for (auto i = 0; i < devices.size(); i++) {
        this->RunProgramNonblocking(devices[i], program_handles[i]);
    }

    log_info(LogTest, "waiting for programs");
    for (auto i = 0; i < devices.size(); i++) {
        this->WaitForSingleProgramDone(devices[i], program_handles[i]);
    }

    log_info(LogTest, "programs done");
    // collect tx and rx results
}

}  // namespace fabric_mux_tests
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
