// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher polling the eth link training counter.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

static void RunTest(MeshWatcherFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    CoreCoord logical_core, virtual_core;
    if (device->get_active_ethernet_cores(true).empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }

    for (auto core : device->get_active_ethernet_cores(true)) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device->id(), core)) {
            logical_core = core;
            break;
        }
    }

    virtual_core = device->ethernet_core_from_logical_core(logical_core);
    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_eth_link_check.cpp",
        logical_core,
        EthernetConfig{.noc = tt_metal::NOC::NOC_0});

    fixture->RunProgram(mesh_device, workload);

    auto x = MetalContext::instance().watcher_server()->exception_message();
    // read out link status from L1
    auto link_up_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);
    uint32_t link_status_rd;
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        &link_status_rd, sizeof(uint32_t), tt_cxy_pair(device->id(), virtual_core.x, virtual_core.y), link_up_addr);

    if (x.empty()) {
        EXPECT_NE(link_status_rd, 0);
    } else {
        EXPECT_EQ(link_status_rd, 0);
    }
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherEthLinkCheck) {
    // Eth link retraining only supported on WH for now, this test is also dispatch-agnostic so just pick one.
    if (this->slow_dispatch_ || this->arch_ != tt::ARCH::WORMHOLE_B0 || this->devices_.size() == 1) {
        log_info(LogTest, "Test only runs on fast dispatch + multi-chip WH, skipping...");
        GTEST_SKIP();
    }

    // Just try forcing an eth retrain on Device 0
    auto mesh_device = this->devices_[0];
    auto device = mesh_device->get_devices()[0];
    vector<uint32_t> reset_val = {0x1};
    uint32_t retrain_force_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_FORCE);
    for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device->id(), eth_core)) {
            continue;
        }
        // Only force a retrain on odd-numbered eth cores
        if (eth_core.y % 2) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                device->id(), virtual_core, reset_val, retrain_force_addr);
        }
    }

    // Just wait a few seconds to let the link retrain
    std::this_thread::sleep_for(std::chrono::seconds(5));
    vector<std::string> expected_strings;
    for (const CoreCoord &eth_core : device->get_active_ethernet_cores()) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device->id(), eth_core)) {
            continue;
        }
        CoreCoord virtual_core = device->ethernet_core_from_logical_core(eth_core);
        expected_strings.push_back(fmt::format(
            "\tDevice {} Ethernet Core {} retraining events: {}",
            device->id(),
            virtual_core,
            (eth_core.y % 2) ? 1 : 0));
    }

    // Close devices/context to trigger watcher check on teardown.
    DebugToolsMeshFixture::TearDown();  // NOLINT(bugprone-parent-virtual-call) Call parent teardown so we don't disable
                                    // watcher
    MetalContext::instance().teardown();
    EXPECT_TRUE(FileContainsAllStrings(this->log_file_name, expected_strings));
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherDetectLinkUp) {
    if (this->arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP()
            << "Enable this test on BH when base FW updated to flush data cache and invalidate instruction cache";
    }
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
            RunTest(fixture, mesh_device);
        },
        this->devices_[0]);
}
