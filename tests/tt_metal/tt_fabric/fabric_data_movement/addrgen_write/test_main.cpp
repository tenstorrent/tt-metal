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
void run_linear_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p);
void run_linear_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p);
}

// Fixture for 2D mesh device setup (used for mesh/unicast/multicast tests)
struct Fixture2D : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture2D() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

// Fixture for 1D linear device setup (used for linear fabric tests)
struct Fixture1D : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture1D() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1, .trace_region_size = 1u << 20, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

// Comprehensive parameterized test for 2D API variants (unicast, multicast, etc.)
// Uses std::tuple<api_variant, page_size, use_dram_dst> as parameter type (required by ::testing::Combine)
class AddrgenComprehensiveTest : public ::testing::TestWithParam<std::tuple<tt::tt_fabric::test::AddrgenApiVariant, uint32_t, bool>> {
protected:
    inline static Fixture2D* fixture = nullptr;

    static void SetUpTestSuite() {
        fixture = new Fixture2D();
        fixture->setup();
    }

    static void TearDownTestSuite() {
        fixture->teardown();
        delete fixture;
        fixture = nullptr;
    }
};

// Separate test class for 1D linear fabric tests
class AddrgenLinear1DTest
    : public ::testing::TestWithParam<std::tuple<tt::tt_fabric::test::AddrgenApiVariant, uint32_t, bool>> {
protected:
    inline static Fixture1D* fixture = nullptr;

    static void SetUpTestSuite() {
        fixture = new Fixture1D();
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
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithStateConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetStateConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithStateConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetStateConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithStateConnMgr ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetStateConnMgr);

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

    // Run appropriate test (2D fabric: unicast or multicast)
    if (is_multicast) {
        tt::tt_fabric::test::run_multicast_write_test(fixture, p);
    } else {
        tt::tt_fabric::test::run_unicast_write_test(fixture, p);
    }
}

// Linear 1D test - uses separate fixture with FABRIC_1D configuration
TEST_P(AddrgenLinear1DTest, LinearUnicastWrite) {
    auto [api_variant, page_size, use_dram_dst] = GetParam();

    // Check if this is a multicast variant
    bool is_linear_multicast =
        (api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteSetState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteSetState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWrite ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteWithState ||
         api_variant == tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteSetState);

    // Calculate tensor_bytes based on page_size (8 pages total)
    uint32_t num_pages = 8;
    uint32_t tensor_bytes = num_pages * page_size;

    tt::tt_fabric::test::AddrgenTestParams p{
        .mesh_id = 0,
        .src_chip = 0,
        .dst_chip = 1,
        .use_dram_dst = use_dram_dst,
        .tensor_bytes = tensor_bytes,
        .page_size = page_size,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .api_variant = api_variant,
        .mesh_rows = is_linear_multicast ? 2 : 0,  // Use mesh_rows to indicate number of receivers
        .mesh_cols = 0};

    // Run appropriate test
    if (is_linear_multicast) {
        tt::tt_fabric::test::run_linear_multicast_write_test(fixture, p);
    } else {
        tt::tt_fabric::test::run_linear_unicast_write_test(fixture, p);
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
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteConnMgr:
            return "MulticastScatterWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithStateConnMgr:
            return "MulticastScatterWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetStateConnMgr:
            return "MulticastScatterWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite:
            return "MulticastFusedAtomicIncWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState:
            return "MulticastFusedAtomicIncWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState:
            return "MulticastFusedAtomicIncWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteConnMgr:
            return "MulticastFusedAtomicIncWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithStateConnMgr:
            return "MulticastFusedAtomicIncWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetStateConnMgr:
            return "MulticastFusedAtomicIncWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteConnMgr: return "MulticastWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithStateConnMgr:
            return "MulticastWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetStateConnMgr:
            return "MulticastWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWrite: return "ScatterWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithState: return "ScatterWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetState: return "ScatterWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteConnMgr: return "UnicastWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteWithStateConnMgr:
            return "UnicastWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteSetStateConnMgr: return "UnicastWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteConnMgr: return "FusedAtomicIncWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteWithStateConnMgr:
            return "FusedAtomicIncWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteSetStateConnMgr:
            return "FusedAtomicIncWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteConnMgr: return "ScatterWriteConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithStateConnMgr:
            return "ScatterWriteWithStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetStateConnMgr: return "ScatterWriteSetStateConnMgr";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWrite: return "LinearUnicastWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWriteWithState: return "LinearUnicastWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWriteSetState: return "LinearUnicastWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWrite: return "LinearScatterWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWriteWithState: return "LinearScatterWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWriteSetState: return "LinearScatterWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWrite: return "LinearFusedAtomicIncWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWriteWithState:
            return "LinearFusedAtomicIncWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWriteSetState:
            return "LinearFusedAtomicIncWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWrite: return "LinearMulticastWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteWithState:
            return "LinearMulticastWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteSetState:
            return "LinearMulticastWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWrite: return "LinearMulticastScatterWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteWithState:
            return "LinearMulticastScatterWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteSetState:
            return "LinearMulticastScatterWriteSetState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWrite:
            return "LinearMulticastFusedAtomicIncWrite";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteWithState:
            return "LinearMulticastFusedAtomicIncWriteWithState";
        case tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteSetState:
            return "LinearMulticastFusedAtomicIncWriteSetState";
        default: return "UnknownVariant";
    }
}

// Instantiate 2D fabric test suite (unicast, multicast variants)
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
            tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastScatterWriteSetStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWrite,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastFusedAtomicIncWriteSetStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::MulticastWriteSetStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWrite,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::UnicastWriteSetStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::FusedAtomicIncWriteSetStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteWithStateConnMgr,
            tt::tt_fabric::test::AddrgenApiVariant::ScatterWriteSetStateConnMgr),
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

// Instantiate 1D linear fabric test suite (LinearUnicastWrite variants)
// Uses separate Fixture1D with FABRIC_1D configuration
INSTANTIATE_TEST_SUITE_P(
    Linear1D,
    AddrgenLinear1DTest,
    ::testing::Combine(
        ::testing::Values(
            tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearUnicastWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearScatterWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearFusedAtomicIncWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastScatterWriteSetState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWrite,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteWithState,
            tt::tt_fabric::test::AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteSetState),
        ::testing::Values(100, 112, 2048, 10000, 10100, 99999),  // Page sizes: Aligned and unaligned
        ::testing::Bool()                                        // Destination: false=L1, true=DRAM
        ),
    [](const ::testing::TestParamInfo<AddrgenLinear1DTest::ParamType>& info) {
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
