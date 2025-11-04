// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

// Existing fixture from bench_unicast_addrgen_main.cpp
struct Fixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

TEST(AddrgenUnicast, BasicWrite) {
    Fixture fixture;
    fixture.setup();

    // Hardcoded parameters - minimal test case
    tt::tt_fabric::bench::PerfParams p{
        .mesh_id = 0,
        .src_chip = 0,
        .dst_chip = 1,
        .use_dram_dst = false,
        .tensor_bytes = 16384,  // 4 pages
        .page_size = 4096,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .trace_iters = 1  // minimal trace
    };

    // Call existing benchmark function (returns PerfPoint with timing)
    auto result = tt::tt_fabric::bench::run_unicast_once(&fixture, p);

    // Test passes if no assertion failures occurred in run_unicast_once
    // (it already validates data with ADD_FAILURE)
    EXPECT_GT(result.bytes, 0u);

    fixture.teardown();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
