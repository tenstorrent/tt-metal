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

// Comprehensive parameterized test for all API variants, page sizes, and destinations
// Uses std::tuple<api_variant, page_size, use_dram_dst> as parameter type (required by ::testing::Combine)
class AddrgenComprehensiveTest : public ::testing::TestWithParam<std::tuple<tt::tt_fabric::test::AddrgenApiVariant, uint32_t, bool>> {
protected:
    inline static Fixture* fixture = nullptr;

    static void SetUpTestSuite() {
        fixture = new Fixture();
        fixture->setup();
    }

    static void TearDownTestSuite() {
        fixture->teardown();
        delete fixture;
        fixture = nullptr;
    }
};

TEST_P(AddrgenComprehensiveTest, Write) {
    auto [api_variant, page_size, use_dram_dst] = GetParam();

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

    // Calculate tensor_bytes based on page_size (8 pages total)
    uint32_t num_pages = 8;
    uint32_t tensor_bytes = num_pages * page_size;

    tt::tt_fabric::test::AddrgenTestParams p{
        .mesh_id = 0,
        .src_chip = is_multicast ? 2 : 0,
        .dst_chip = is_multicast ? 0 : 1,
        .use_dram_dst = use_dram_dst,
        .tensor_bytes = tensor_bytes,
        .page_size = page_size,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .api_variant = api_variant,
        .mesh_rows = is_multicast ? 2 : 0,
        .mesh_cols = is_multicast ? 2 : 0};

    // Run appropriate test
    if (is_multicast) {
        tt::tt_fabric::test::run_multicast_write_test(fixture, p);
    } else {
        tt::tt_fabric::test::run_unicast_write_test(fixture, p);
    }
}

// Helper function to get variant name as string
static std::string GetVariantName(tt::tt_fabric::test::AddrgenApiVariant variant) {
    switch (variant) {
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
}

// Instantiate comprehensive test suite with all 18 variants × 3 page sizes × 2 destinations = 108 test cases
INSTANTIATE_TEST_SUITE_P(
    AllVariantsAndSizes,
    AddrgenComprehensiveTest,
    ::testing::Combine(
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
        ::testing::Values(100, 112, 2048, 10000, 10100, 99999),  // Page sizes: Aligned and unaligned
        ::testing::Bool()                                        // Destination: false=L1, true=DRAM
        ),
    [](const ::testing::TestParamInfo<AddrgenComprehensiveTest::ParamType>& info) {
        auto api_variant = std::get<0>(info.param);
        auto page_size = std::get<1>(info.param);
        auto use_dram_dst = std::get<2>(info.param);
        std::string name = GetVariantName(api_variant);
        name += "_" + std::to_string(page_size) + "B";
        name += use_dram_dst ? "_DRAM" : "_L1";
        return name;
    });

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
