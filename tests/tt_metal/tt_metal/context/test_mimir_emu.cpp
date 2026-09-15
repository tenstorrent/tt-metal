// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint64_t kSmcSramOffset = 0x00060000;

TEST(MimirEmu, SmcSramRoundTripThroughMetalCluster) {
    if (std::getenv("TT_METAL_EMU_SERVER") == nullptr || std::getenv("TT_METAL_EMU_SOC_DESC") == nullptr) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::GRENDEL);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    // write_core/read_core resolve the pair they are given as TRANSLATED coordinates.
    const auto smc_cores = cluster.get_soc_desc(chip_id).get_cores(tt::CoreType::SMC, CoordSystem::TRANSLATED);
    ASSERT_EQ(smc_cores.size(), 1);
    const auto& smc = smc_cores.front();

    constexpr uint32_t written = 0xC0FFEE01;
    uint32_t read_back = 0;
    const tt_cxy_pair target(chip_id, smc.x, smc.y);
    cluster.write_core(&written, sizeof(written), target, kSmcSramOffset);
    cluster.read_core(&read_back, sizeof(read_back), target, kSmcSramOffset);

    EXPECT_EQ(read_back, written);
}

}  // namespace
}  // namespace tt::tt_metal
