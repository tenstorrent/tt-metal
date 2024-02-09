// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/common/core_descriptor.hpp"

using namespace tt::tt_metal;

namespace remote_tests {

TEST_F(CommandQueueMultiDeviceFixture, TestCommandReachesRemoteDevice) {
    GTEST_SKIP() << "Test incorrect. Writing to a buffer, but nothing blocking to ensure that the write has been received";
    for (unsigned int id = 0; id < devices_.size(); id++) {
        auto device = devices_.at(id);
        if (device->is_mmio_capable()) {
            continue;
        }

        CommandQueue &remote_cq = detail::GetCommandQueue(device);

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        uint8_t num_hw_cqs = device->num_hw_cqs();
        ASSERT_EQ(num_hw_cqs, 1) << "Expected this test to be run with single HW CQ test suite!";
        uint8_t cq_id = 0;

        uint32_t num_pages = 1;
        uint32_t page_size = 2048;
        uint32_t buff_size = num_pages * page_size;
        Buffer bufa(device, buff_size, page_size, BufferType::DRAM);

        std::vector<uint32_t> src(buff_size / sizeof(uint32_t), 0);
        for (uint32_t i = 0; i < src.size(); i++) {
            src.at(i) = i;
        }

        EnqueueWriteBuffer(remote_cq, bufa, src.data(), false);

        uint32_t num_bytes_in_cmd_header = DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t);

        std::vector<uint32_t> cq_header(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
        tt::Cluster::instance().read_sysmem(cq_header.data(), num_bytes_in_cmd_header, CQ_START, mmio_device_id, channel);

        tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(num_hw_cqs).issue_queue_reader_core(device->id(), channel, cq_id);
        CoreCoord issue_q_physical_core = tt::get_physical_core_coordinate(issue_q_reader_location, CoreType::WORKER);
        std::vector<uint32_t> issue_q_reader_header(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
        tt::Cluster::instance().read_core(issue_q_reader_header.data(), num_bytes_in_cmd_header, tt_cxy_pair(issue_q_reader_location.chip, issue_q_physical_core), L1_UNRESERVED_BASE);
        EXPECT_EQ(cq_header, issue_q_reader_header) << "Remote issue queue reader did not read in expected command!";

        tt_cxy_pair src_eth_router_location = tt::Cluster::instance().get_eth_core_for_dispatch_core(issue_q_reader_location, EthRouterMode::FD_SRC, device->id());
        CoreCoord physical_src_eth_router = tt::get_physical_core_coordinate(src_eth_router_location, CoreType::ETH);
        std::vector<uint32_t> source_router_header(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
        tt::Cluster::instance().read_core(source_router_header.data(), num_bytes_in_cmd_header, tt_cxy_pair(src_eth_router_location.chip, physical_src_eth_router), eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE);
        EXPECT_EQ(cq_header, source_router_header) << "Ethernet router on local chip (src) did not receive expected command!";

        // Need ethernet cores ot get out of FD mode to do host read backs from remote device
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

        tt_cxy_pair remote_processor_location = dispatch_core_manager::get(num_hw_cqs).remote_processor_core(device->id(), channel, cq_id);
        CoreCoord remote_processor_physical_core = tt::get_physical_core_coordinate(remote_processor_location, CoreType::WORKER);

        tt_cxy_pair dst_eth_router_location = tt::Cluster::instance().get_eth_core_for_dispatch_core(remote_processor_location, EthRouterMode::FD_DST, mmio_device_id);
        CoreCoord physical_dst_eth_router = tt::get_physical_core_coordinate(dst_eth_router_location, CoreType::ETH);

        std::vector<uint32_t> dst_router_header(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
        tt::Cluster::instance().read_core(dst_router_header.data(), num_bytes_in_cmd_header, tt_cxy_pair(dst_eth_router_location.chip, physical_dst_eth_router), eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE);
        EXPECT_EQ(cq_header, dst_router_header) << "Ethernet router on remote chip (dst) did not receive expected command!";

        std::vector<uint32_t> remote_cmd_processor_header(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
        tt::Cluster::instance().read_core(remote_cmd_processor_header.data(), num_bytes_in_cmd_header, tt_cxy_pair(remote_processor_location.chip, remote_processor_physical_core), L1_UNRESERVED_BASE);
        EXPECT_EQ(cq_header, remote_cmd_processor_header) << "Remote command processor on remote chip did not receive expected command!";

        std::vector<uint32_t> remote_cmd_processor_data(buff_size / sizeof(uint32_t));
        tt::Cluster::instance().read_core(remote_cmd_processor_data.data(), buff_size, tt_cxy_pair(remote_processor_location.chip, remote_processor_physical_core), L1_UNRESERVED_BASE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
        EXPECT_EQ(src, remote_cmd_processor_data) << "Remote command processor on remote chip did not receive expected data!";
    }
}

}
