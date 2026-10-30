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

class CceSramRoundTripThroughMetalCluster : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramRoundTripThroughMetalCluster, RoundTrip) {
    if (!backend_transport_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const uint32_t cce_index = GetParam();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const CoreCoord cce = translated_dram_core(soc_desc, cce_index);
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    const uint32_t written = 0xC0FFEE01 + cce_index;
    // A host-only process inherits this word from the last one, so poison it first or a dropped write passes on a
    // rerun.
    const uint32_t poison = ~written;
    uint32_t read_back = 0;

    cluster.write_core(&poison, sizeof(poison), {chip_id, cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {chip_id, cce}, address);
    ASSERT_EQ(read_back, poison);

    cluster.write_core(&written, sizeof(written), {chip_id, cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {chip_id, cce}, address);

    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, CceSramChannelsDoNotAliasThroughMetalCluster) {
    if (!backend_transport_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const uint32_t num_cces = num_cces_from_soc(soc_desc);
    ASSERT_GE(num_cces, 2u);

    // Every channel is written before any is read: an aliasing window only shows up once a later
    // channel has landed on top of an earlier one.
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const uint32_t written = 0xC0FFEE01 + cce_index;
        cluster.write_core(&written, sizeof(written), {chip_id, translated_dram_core(soc_desc, cce_index)}, address);
    }
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        uint32_t read_back = 0;
        cluster.read_core(&read_back, sizeof(read_back), {chip_id, translated_dram_core(soc_desc, cce_index)}, address);
        EXPECT_EQ(read_back, 0xC0FFEE01 + cce_index) << "CCE" << cce_index;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, CceSramRoundTripThroughMetalCluster, ::testing::ValuesIn(cce_indices()), cce_param_name);

}  // namespace
}  // namespace tt::tt_metal
