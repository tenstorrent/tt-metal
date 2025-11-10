// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

// Forward declaration of test runners
namespace tt::tt_fabric::test {
void run_addrgen_write_test(tt::tt_fabric::bench::HelpersFixture* fixture, const tt::tt_fabric::bench::PerfParams& p);
void run_multicast_write_test(tt::tt_fabric::bench::HelpersFixture* fixture, const tt::tt_fabric::bench::PerfParams& p);
}

// Fixture for mesh device setup
struct Fixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

// Parameterized test for all addrgen API variants
class AddrgenApiVariantTest : public ::testing::TestWithParam<tt::tt_fabric::bench::AddrgenApiVariant> {
protected:
    Fixture fixture;

    void SetUp() override { fixture.setup(); }

    void TearDown() override { fixture.teardown(); }
};

TEST_P(AddrgenApiVariantTest, Write) {
    auto api_variant = GetParam();

    // Different parameters for multicast vs others
    bool is_multicast = (api_variant == tt::tt_fabric::bench::AddrgenApiVariant::MulticastWrite);

    tt::tt_fabric::bench::PerfParams p{
        .mesh_id = 0,
        .src_chip = is_multicast ? 2 : 0,  // 0:2 for multicast, 0:0 for others
        .dst_chip = is_multicast ? 0 : 1,  // 0:0 for multicast, 0:1 for others
        .use_dram_dst = false,
        .tensor_bytes = 4096,  // 2 pages
        .page_size = 2048,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .trace_iters = 1,
        .api_variant = api_variant,
        .mesh_rows = is_multicast ? 2 : 0,  // 2x2 for multicast
        .mesh_cols = is_multicast ? 2 : 0};

    // Dispatch to appropriate runner
    if (is_multicast) {
        tt::tt_fabric::test::run_multicast_write_test(&fixture, p);
    } else {
        tt::tt_fabric::test::run_addrgen_write_test(&fixture, p);
    }
}

// Instantiate with all variants (3 unicast + 3 scatter + 3 fused atomic inc + 1 multicast)
INSTANTIATE_TEST_SUITE_P(
    AddrgenOverloads,
    AddrgenApiVariantTest,
    ::testing::Values(
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWrite,
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteWithState,
        tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteSetState,
        tt::tt_fabric::bench::AddrgenApiVariant::ScatterWrite,
        tt::tt_fabric::bench::AddrgenApiVariant::ScatterWriteWithState,
        tt::tt_fabric::bench::AddrgenApiVariant::ScatterWriteSetState,
        tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWrite,
        tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteWithState,
        tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteSetState,
        tt::tt_fabric::bench::AddrgenApiVariant::MulticastWrite),
    [](const ::testing::TestParamInfo<AddrgenApiVariantTest::ParamType>& info) {
        switch (info.param) {
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWrite: return "UnicastWrite";
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteWithState: return "UnicastWriteWithState";
            case tt::tt_fabric::bench::AddrgenApiVariant::UnicastWriteSetState: return "UnicastWriteSetState";
            case tt::tt_fabric::bench::AddrgenApiVariant::ScatterWrite: return "ScatterWrite";
            case tt::tt_fabric::bench::AddrgenApiVariant::ScatterWriteWithState: return "ScatterWriteWithState";
            case tt::tt_fabric::bench::AddrgenApiVariant::ScatterWriteSetState: return "ScatterWriteSetState";
            case tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWrite: return "FusedAtomicIncWrite";
            case tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteWithState:
                return "FusedAtomicIncWriteWithState";
            case tt::tt_fabric::bench::AddrgenApiVariant::FusedAtomicIncWriteSetState:
                return "FusedAtomicIncWriteSetState";
            case tt::tt_fabric::bench::AddrgenApiVariant::MulticastWrite: return "MulticastWrite";
            default: return "UnknownVariant";
        }
    });

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
