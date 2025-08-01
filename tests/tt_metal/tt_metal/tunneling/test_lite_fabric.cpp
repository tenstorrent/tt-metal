// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>

#include "context/metal_context.hpp"
#include "data_types.hpp"
#include "fabric_lite.hpp"
#include "hal_types.hpp"
#include "kernel.hpp"
#include "kernel_types.hpp"
#include "rtoptions.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster.hpp"
#include "tt_memory.h"
#include "build.hpp"
#include "lite_fabric_host.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal.hpp"

#define CHECK_TEST_REQS()                                                                       \
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) { \
        GTEST_SKIP() << "Blackhole only";                                                       \
    }                                                                                           \
    if (tt::tt_metal::GetNumAvailableDevices() != 2) {                                          \
        GTEST_SKIP() << "2 Devices are required";                                               \
    }                                                                                           \
    if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {                                          \
        GTEST_SKIP() << "TT_METAL_SLOW_DISPATCH_MODE required";                                 \
    }                                                                                           \
    if (!std::getenv("TT_METAL_CLEAR_L1")) {                                                    \
        GTEST_SKIP() << "TT_METAL_CLEAR_L1 required";                                           \
    }

TEST(Tunneling, LiteFabricBuild) {
    CHECK_TEST_REQS();
    std::string root_dir = std::getenv("TT_METAL_HOME");
    std::string lite_fabric_dir = fmt::format("{}/{}", root_dir, "lite_fabric");

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    lite_fabric::CompileLiteFabric(cluster, root_dir, lite_fabric_dir);
    lite_fabric::LinkLiteFabric(root_dir, lite_fabric_dir);

    const std::string k_FabricLitePath = fmt::format("{}/lite_fabric/lite_fabric.elf", root_dir);
    ll_api::memory bin = ll_api::memory(k_FabricLitePath, ll_api::memory::Loading::DISCRETE);

    uint32_t text_start = bin.get_text_addr();
    uint32_t bin_size = tt::align(bin.get_text_size() + 1, 16);

    ASSERT_GE(bin_size, 0);
    ASSERT_GE(text_start, 0);

    uint32_t lite_fabric_config_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG);

    tt_cxy_pair logical_eth_core{0, 0, 0};
    tt_cxy_pair virtual_eth_core =
        cluster->get_virtual_coordinate_from_logical_coordinates(logical_eth_core, CoreType::ETH);
    constexpr uint32_t k_TestAddr = 0x20000;

    log_info(tt::LogTest, "Test Eth Core logical={}, virtual={}", logical_eth_core.str(), virtual_eth_core.str());

    lite_fabric::SetResetState(cluster, virtual_eth_core, true);
    lite_fabric::SetPC(cluster, virtual_eth_core, 0xFFB00000 | 0x14008, text_start);

    {
        std::vector<uint32_t> twoZeroes{0, 0};
        cluster->write_core((void*)&twoZeroes, twoZeroes.size() * sizeof(uint32_t), virtual_eth_core, k_TestAddr);
    }

    bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        auto local_init_addr = hal.get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LOCAL_L1_INIT_SCRATCH);
        uint64_t relo_addr = hal.relocate_dev_addr(addr, local_init_addr);

        cluster->write_core(&*mem_ptr, len_words * sizeof(uint32_t), virtual_eth_core, relo_addr);
    });

    cluster->l1_barrier(0);

    lite_fabric::SetResetState(cluster, virtual_eth_core, false);

    // good enough
    std::this_thread::sleep_for(std::chrono::seconds(1));

    lite_fabric::SetResetState(cluster, virtual_eth_core, true);

    {
        std::vector<uint32_t> readback(2);
        cluster->read_core(readback.data(), 2 * sizeof(uint32_t), virtual_eth_core, 0x20000);
        ASSERT_EQ(readback[0], 0xdeadbeef);
        ASSERT_EQ(readback[1], 0xcafecafe);
    }
}

TEST(Tunneling, LiteFabricInitWithMetal) {
    CHECK_TEST_REQS();

    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    auto desc = lite_fabric::GetSystemDescriptor2Devices(devices, 0, 1);

    auto lite_fabric = lite_fabric::LaunchLiteFabricWithMetal(devices, desc);

    lite_fabric::TerminateLiteFabric(tt::tt_metal::MetalContext::instance().get_cluster(), desc);

    tt::tt_metal::detail::WaitProgramDone(devices[0], *lite_fabric);

    tt::tt_metal::detail::CloseDevices(devices);
}
