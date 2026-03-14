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

using tt::tt_fabric::test::FabricTestVariant;
using tt::tt_fabric::test::CastMode;
using tt::tt_fabric::test::WriteOp;
using tt::tt_fabric::test::StateMode;
using tt::tt_fabric::test::ConnectionMode;

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
// Uses std::tuple<FabricTestVariant, page_size, use_dram_dst> as parameter type (required by ::testing::Combine)
class AddrgenComprehensiveTest : public ::testing::TestWithParam<std::tuple<FabricTestVariant, uint32_t, bool>> {
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
    : public ::testing::TestWithParam<std::tuple<FabricTestVariant, uint32_t, bool>> {
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
    auto [variant, page_size, use_dram_dst] = GetParam();

    bool is_multicast = variant.is_multicast();

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
        .variant = variant,
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
    auto [variant, page_size, use_dram_dst] = GetParam();

    // Check if this is a multicast variant
    bool is_linear_multicast = variant.is_multicast();

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
        .variant = variant,
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
static std::string GetVariantName(const FabricTestVariant& v) {
    return tt::tt_fabric::test::to_string(v);
}

// Generate all valid 2D test variants
static auto Make2DVariants() {
    std::vector<FabricTestVariant> variants;
    for (auto cast : {CastMode::Unicast, CastMode::Multicast}) {
        for (auto op : {WriteOp::Write, WriteOp::Scatter, WriteOp::FusedAtomicInc}) {
            for (auto state : {StateMode::Stateless, StateMode::WithState, StateMode::SetState}) {
                for (auto conn : {ConnectionMode::Direct, ConnectionMode::ConnMgr}) {
                    variants.push_back({cast, op, state, conn});
                }
            }
        }
    }
    return variants;
}

// Instantiate 2D fabric test suite (unicast, multicast variants)
INSTANTIATE_TEST_SUITE_P(
    AllVariantsAndSizes,
    AddrgenComprehensiveTest,
    ::testing::Combine(
        ::testing::ValuesIn(Make2DVariants()),
        ::testing::Values(100, 112, 2048, 10000, 10100, 99999),
        ::testing::Bool()),
    [](const ::testing::TestParamInfo<AddrgenComprehensiveTest::ParamType>& info) {
        auto variant = std::get<0>(info.param);
        auto page_size = std::get<1>(info.param);
        auto use_dram_dst = std::get<2>(info.param);
        std::string name = GetVariantName(variant);
        name += "_" + std::to_string(page_size) + "B";
        name += use_dram_dst ? "_DRAM" : "_L1";
        return name;
    });

static auto Make1DVariants() {
    std::vector<FabricTestVariant> variants;
    // Linear unicast: all ops x all states x Direct
    for (auto op : {WriteOp::Write, WriteOp::Scatter, WriteOp::FusedAtomicInc}) {
        for (auto state : {StateMode::Stateless, StateMode::WithState, StateMode::SetState}) {
            variants.push_back({CastMode::LinearUnicast, op, state, ConnectionMode::Direct});
        }
    }
    // Linear unicast ConnMgr (only Write/Stateless for now)
    variants.push_back({CastMode::LinearUnicast, WriteOp::Write, StateMode::Stateless, ConnectionMode::ConnMgr});
    // Linear multicast: all ops x all states x Direct
    for (auto op : {WriteOp::Write, WriteOp::Scatter, WriteOp::FusedAtomicInc}) {
        for (auto state : {StateMode::Stateless, StateMode::WithState, StateMode::SetState}) {
            variants.push_back({CastMode::LinearMulticast, op, state, ConnectionMode::Direct});
        }
    }
    return variants;
}

// Instantiate 1D linear fabric test suite
// Uses separate Fixture1D with FABRIC_1D configuration
INSTANTIATE_TEST_SUITE_P(
    Linear1D,
    AddrgenLinear1DTest,
    ::testing::Combine(
        ::testing::ValuesIn(Make1DVariants()),
        ::testing::Values(100, 112, 2048, 10000, 10100, 99999),
        ::testing::Bool()),
    [](const ::testing::TestParamInfo<AddrgenLinear1DTest::ParamType>& info) {
        auto variant = std::get<0>(info.param);
        auto page_size = std::get<1>(info.param);
        auto use_dram_dst = std::get<2>(info.param);
        std::string name = GetVariantName(variant);
        name += "_" + std::to_string(page_size) + "B";
        name += use_dram_dst ? "_DRAM" : "_L1";
        return name;
    });
