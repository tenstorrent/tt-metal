// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Backend checks. These open a Cluster directly and use the host NOC tag, so they stay out of the
// contract suite. A PCIe/ATT backend would replace this file; the contract tests would not change.
#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>

#include "dev_mem_map.h"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "mimir_contract.hpp"

namespace tt::tt_metal {
namespace {
using namespace mimir_contract;

bool backend_transport_configured() {
    const bool emu = std::getenv("TT_METAL_EMU_SERVER") != nullptr && std::getenv("TT_METAL_EMU_SOC_DESC") != nullptr;
    const bool jtag = std::getenv("TT_METAL_GRENDEL_JTAG_SERVER") != nullptr &&
                      std::getenv("TT_METAL_GRENDEL_JTAG_SOC_DESC") != nullptr;
    return emu || jtag;
}

class MimirCceSramRoundTripThroughMetalCluster : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirCceSramRoundTripThroughMetalCluster, RoundTrip) {
    if (!backend_transport_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const uint32_t mimir_cce_index = GetParam();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const uint32_t num_mimir_cces = num_mimir_cces_from_soc(soc_desc);
    if (mimir_cce_index >= num_mimir_cces) {
        GTEST_SKIP() << "Configured descriptor exposes only " << num_mimir_cces << " CCEs.";
    }
    const CoreCoord mimir_cce = translated_dram_core(soc_desc, mimir_cce_index);
    constexpr uint64_t address = MEM_MIMIR_CCE_L1_NOC_OFFSET + kMimirCceSramTestOffset;
    const uint32_t written = 0xC0FFEE01 + mimir_cce_index;
    uint32_t read_back = 0;

    cluster.write_core(&written, sizeof(written), {chip_id, mimir_cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {chip_id, mimir_cce}, address);

    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, MimirCceSramChannelsDoNotAliasThroughMetalCluster) {
    if (!backend_transport_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const uint32_t num_mimir_cces = num_mimir_cces_from_soc(soc_desc);
    ASSERT_GE(num_mimir_cces, 2u);

    // Every channel is written before any is read: an aliasing window only shows up once a later
    // channel has landed on top of an earlier one.
    constexpr uint64_t address = MEM_MIMIR_CCE_L1_NOC_OFFSET + kMimirCceSramTestOffset;
    for (uint32_t mimir_cce_index = 0; mimir_cce_index < num_mimir_cces; ++mimir_cce_index) {
        const uint32_t written = 0xC0FFEE01 + mimir_cce_index;
        cluster.write_core(&written, sizeof(written), {chip_id, translated_dram_core(soc_desc, mimir_cce_index)}, address);
    }
    for (uint32_t mimir_cce_index = 0; mimir_cce_index < num_mimir_cces; ++mimir_cce_index) {
        uint32_t read_back = 0;
        cluster.read_core(&read_back, sizeof(read_back), {chip_id, translated_dram_core(soc_desc, mimir_cce_index)}, address);
        EXPECT_EQ(read_back, 0xC0FFEE01 + mimir_cce_index) << "CCE" << mimir_cce_index;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, MimirCceSramRoundTripThroughMetalCluster, ::testing::ValuesIn(mimir_cce_indices()), mimir_cce_param_name);

}  // namespace
}  // namespace tt::tt_metal
