// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

// Existing fixture from original test
struct Fixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

// Parameterized test for different API variants
class AddrgenApiVariantTest : public ::testing::TestWithParam<tt::tt_fabric::bench::AddrgenApiVariant> {
protected:
    Fixture fixture;

    void SetUp() override { fixture.setup(); }

    void TearDown() override { fixture.teardown(); }
};

TEST_P(AddrgenApiVariantTest, Write) {
    auto api_variant = GetParam();

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
        .trace_iters = 1,
        .api_variant = api_variant  // Test parameter
    };

    // Call existing benchmark function
    auto result = tt::tt_fabric::bench::run_unicast_once(&fixture, p);

    // Test passes if no assertion failures occurred
    EXPECT_GT(result.bytes, 0u);
}

// Instantiate with all variants
INSTANTIATE_TEST_SUITE_P(
    AddrgenOverloads,
    AddrgenApiVariantTest,
    ::testing::Values(
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWrite,
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteWithState,
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteSetState),
    [](const ::testing::TestParamInfo<AddrgenApiVariantTest::ParamType>& info) {
        switch (info.param) {
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWrite: return "UnicastWrite";
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteWithState: return "UnicastWriteWithState";
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteSetState: return "UnicastWriteSetState";
            default: return "UnknownVariant";
        }
    });

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
