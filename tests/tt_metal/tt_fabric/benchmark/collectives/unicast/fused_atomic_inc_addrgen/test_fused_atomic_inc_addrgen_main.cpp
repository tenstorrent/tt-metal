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

// Single test for fused atomic inc with addrgen
class FusedAtomicIncAddrgenTest : public ::testing::Test {
protected:
    Fixture fixture;

    void SetUp() override { fixture.setup(); }

    void TearDown() override { fixture.teardown(); }
};

TEST_F(FusedAtomicIncAddrgenTest, FusedAtomicIncWrite) {
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
        .api_variant = tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWrite  // Base variant
    };

    // Call benchmark function (reusing infrastructure)
    auto result = tt::tt_fabric::bench::run_unicast_once(&fixture, p);

    // Test passes if no assertion failures occurred
    EXPECT_GT(result.bytes, 0u);
}

TEST_F(FusedAtomicIncAddrgenTest, FusedAtomicIncWriteWithState) {
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
        .api_variant = tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteWithState  // _with_state variant
    };

    // Call benchmark function (reusing infrastructure)
    auto result = tt::tt_fabric::bench::run_unicast_once(&fixture, p);

    // Test passes if no assertion failures occurred
    EXPECT_GT(result.bytes, 0u);
}

TEST_F(FusedAtomicIncAddrgenTest, FusedAtomicIncWriteSetState) {
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
        .api_variant = tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteSetState  // _set_state variant
    };

    // Call benchmark function (reusing infrastructure)
    auto result = tt::tt_fabric::bench::run_unicast_once(&fixture, p);

    // Test passes if no assertion failures occurred
    EXPECT_GT(result.bytes, 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
