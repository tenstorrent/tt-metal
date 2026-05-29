// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include "impl/dispatch/command_queue_common.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

class SimulatorFixture : public MeshDeviceFixture {
protected:
    void SetUp() override {
        // Check if simulator mode is enabled
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
            GTEST_SKIP()
                << "Simulator mode not enabled. Set TT_METAL_SIMULATOR environment variable to run simulator tests.";
        }

        // Call parent SetUp to initialize devices
        MeshDeviceFixture::SetUp();
    }
};

TEST_F(SimulatorFixture, SimulatorDeviceInitialization) {
    // Verify that all devices are properly initialized in simulator mode
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = devices_.at(id);

        // Check that device is valid
        EXPECT_NE(mesh_device, nullptr);

        // Verify device is accessible
        EXPECT_NO_THROW({});

        // Test that we can access the allocator
        EXPECT_NE(mesh_device->allocator(), nullptr);

        // Verify we can get base addresses
        EXPECT_GT(mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1), 0);
        EXPECT_GT(mesh_device->allocator()->get_base_allocator_addr(HalMemType::DRAM), 0);
    }
}

TEST_F(SimulatorFixture, QuasarStaticTlbReadWrite) {
    auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP();
    }

    const auto& hal = MetalContext::instance().hal();
    const uint64_t scratch_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    constexpr uint32_t value32 = 0xDEADBEEF;

    for (unsigned int id = 0; id < num_devices_; id++) {
        const auto mesh_device = devices_.at(id);
        const ChipId chip_id = mesh_device->get_devices()[0]->id();
        const auto& sdesc = cluster.get_soc_desc(chip_id);

        const std::vector<tt::umd::CoreCoord> tensix_cores =
            sdesc.get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED);
        ASSERT_FALSE(tensix_cores.empty());
        const tt::umd::CoreCoord tensix = tensix_cores.front();
        const tt_cxy_pair target(chip_id, tensix.x, tensix.y);

        ASSERT_TRUE(cluster.get_tlb_data(target).has_value());

        tt::umd::TlbWindow* window = cluster.get_static_tlb_window(target);
        ASSERT_NE(window, nullptr);

        window->write32(scratch_addr, value32);
        EXPECT_EQ(window->read32(scratch_addr), value32);

        std::array<uint32_t, 16> tx;
        std::iota(tx.begin(), tx.end(), 0x12345678);
        std::array<uint32_t, 16> rx{};
        window->write_block(scratch_addr, tx.data(), tx.size() * sizeof(uint32_t));
        window->read_block(scratch_addr, rx.data(), rx.size() * sizeof(uint32_t));
        EXPECT_EQ(tx, rx);
    }
}
}  // namespace tt::tt_metal
