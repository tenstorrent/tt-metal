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

static void RunTest(WatcherFixture* fixture, IDevice* device) {
}

TEST_F(WatcherFixture, ActiveEthTestWatcherEthLinkCheck) {
    // Eth link retraining only supported on WH for now, this test is also dispatch-agnostic so just pick one.
    if (this->slow_dispatch_ || this->arch_ != tt::ARCH::WORMHOLE_B0 || this->devices_.size() == 1) {
        log_info(LogTest, "Test only runs on fast dispatch + multi-chip WH, skipping...");
        GTEST_SKIP();
    }

    // Just try forcing an eth retrain on Device 0
    IDevice* device = this->devices_[0];
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
            tt::llrt::write_hex_vec_to_core(device->id(), virtual_core, reset_val, retrain_force_addr);
        }
    }

    // Just wait a few seconds to let the link retrain
    std::this_thread::sleep_for(std::chrono::seconds(5));
    vector<string> expected_strings;
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

    // Close devices to trigger watcher check on teardown.
    for (IDevice* device : this->devices_) {
        tt::tt_metal::CloseDevice(device);
    }
    EXPECT_TRUE(FileContainsAllStrings(this->log_file_name, expected_strings));
}
