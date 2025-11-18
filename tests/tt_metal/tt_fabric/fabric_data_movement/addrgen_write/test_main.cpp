// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_common.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

// Forward declaration of test runners
namespace tt::tt_fabric::test {
void run_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p);
void run_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p);
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

// Parameterized test for all 6 addrgen API variants
class AddrgenApiVariantTest : public ::testing::TestWithParam<tt::tt_fabric::test::AddrgenApiVariant> {
protected:
    Fixture fixture;

    void SetUp() override { fixture.setup(); }

    void TearDown() override { fixture.teardown(); }
};

TEST_P(AddrgenApiVariantTest, Write) {
    auto api_variant = GetParam();
    bool is_multicast =
        (api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState);

    // Hardcoded parameters - minimal test case
    tt::tt_fabric::test::AddrgenTestParams p{
        .mesh_id = 0,
        .src_chip = is_multicast ? 2 : 0,
        .dst_chip = is_multicast ? 0 : 1,
        .use_dram_dst = false,
        .tensor_bytes = 16384,
        .page_size = 2048,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .api_variant = api_variant,  // Test parameter
        .mesh_rows = is_multicast ? 2 : 0,
        .mesh_cols = is_multicast ? 2 : 0};

    // Run appropriate test
    if (is_multicast) {
        tt::tt_fabric::test::run_multicast_write_test(&fixture, p);
    } else {
        tt::tt_fabric::test::run_unicast_write_test(&fixture, p);
    }
}

// Instantiate with all 15 variants (3 unicast + 3 fused atomic inc + 3 multicast + 3 multicast scatter + 3 scatter)
INSTANTIATE_TEST_SUITE_P(
    AddrgenOverloads,
    AddrgenApiVariantTest,
    ::testing::Values(
        tt::tt_fabric::test::AddrgenApiVariant::UnicastWrite,
        tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteSetState,
        tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWrite,
        tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteSetState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastWrite,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWrite,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState,
        tt::tt_fabric::test::AddrgenApiVariant::ScatterWrite,
        tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithState,
        tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetState),
    [](const ::testing::TestParamInfo<AddrgenApiVariantTest::ParamType>& info) {
        switch (info.param) {
            case tt::tt_fabric::test::AddrgenApiVariant::UnicastWrite: return "UnicastWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteWithState: return "UnicastWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteSetState: return "UnicastWriteSetState";
            case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWrite: return "FusedAtomicIncWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteWithState:
                return "FusedAtomicIncWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteSetState:
                return "FusedAtomicIncWriteSetState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastWrite: return "MulticastWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithState: return "MulticastWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetState: return "MulticastWriteSetState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWrite: return "MulticastScatterWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithState:
                return "MulticastScatterWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetState:
                return "MulticastScatterWriteSetState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite:
                return "MulticastFusedAtomicIncWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState:
                return "MulticastFusedAtomicIncWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState:
                return "MulticastFusedAtomicIncWriteSetState";
            case tt::tt_fabric::test::AddrgenApiVariant::ScatterWrite: return "ScatterWrite";
            case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithState: return "ScatterWriteWithState";
            case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetState: return "ScatterWriteSetState";
            default: return "UnknownVariant";
        }
    });

// Test large page auto-packetization
TEST(AddrgenLargePage, UnicastWriteLargePage) {
    // Setup fixture
    Fixture fixture;
    fixture.setup();

    // Test parameters with large page size
    tt::tt_fabric::test::AddrgenTestParams p{
        .mesh_id = 0,
        .src_chip = 0,
        .dst_chip = 1,
        .use_dram_dst = false,
        .tensor_bytes = 40000,
        .page_size = 10000,
        .sender_core = {0, 0},
        .receiver_core = {0, 0},
        .api_variant = tt::tt_fabric::test::AddrgenApiVariant::UnicastWrite,
        .mesh_rows = 0,
        .mesh_cols = 0};

    // Run test
    tt::tt_fabric::test::run_unicast_write_test(&fixture, p);

    // Teardown
    fixture.teardown();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
