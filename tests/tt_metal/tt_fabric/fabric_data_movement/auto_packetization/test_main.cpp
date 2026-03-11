// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

// Forward declaration of test runners (implementation in plan 07: unicast_runner.cpp, multicast_runner.cpp)
namespace tt::tt_fabric::test {
void run_raw_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const RawTestParams& p);
void run_raw_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const RawTestParams& p);
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

// ---------------------------------------------------------------------------
// Compile-only test: creates a Program, adds the sender kernels, and calls
// detail::CompileProgram to verify the device-side headers compile with the
// device toolchain. Does NOT run on hardware.
//
// This is the key compile probe for plans 02-06. After those plans modify the
// API headers, running this test (--gtest_filter=*CompileOnly*) validates that
// the headers compile correctly.
// ---------------------------------------------------------------------------
TEST_F(Fixture2D, CompileOnlyKernels) {
    auto device = this->mesh_device_->get_devices()[0];
    tt::tt_metal::Program program;
    auto core = CoreCoord{0, 0};

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default});

    // Compile-only: validates headers compile with device toolchain
    tt::tt_metal::detail::CompileProgram(device, program);
}

// ---------------------------------------------------------------------------
// Parameterized test for raw-size auto-packetization correctness (plan 07).
// Uses std::tuple<api_variant, page_size, use_dram_dst> as parameter type.
// ---------------------------------------------------------------------------
class RawPacketizationTest
    : public ::testing::TestWithParam<
          std::tuple<tt::tt_fabric::test::RawApiVariant, uint32_t, bool>> {
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

TEST_P(RawPacketizationTest, ChunkingCorrectness) {
    auto [api_variant, page_size, use_dram_dst] = GetParam();

    bool is_multicast =
        (api_variant == tt::tt_fabric::test::RawApiVariant::MulticastWrite ||
         api_variant == tt::tt_fabric::test::RawApiVariant::MulticastWriteConnMgr ||
         api_variant == tt::tt_fabric::test::RawApiVariant::LinearMulticastWrite ||
         api_variant == tt::tt_fabric::test::RawApiVariant::LinearSparseMulticastWrite);

    tt::tt_fabric::test::RawTestParams p{
        .mesh_id = 0,
        .src_chip = is_multicast ? 2 : 0,
        .dst_chip = is_multicast ? 0 : 1,
        .use_dram_dst = use_dram_dst,
        .tensor_bytes = 8 * page_size,
        .page_size = page_size,
        .sender_core = {0, 0},
        .receiver_core = {1, 0},
        .api_variant = api_variant,
        .mesh_rows = is_multicast ? 2 : 0,
        .mesh_cols = is_multicast ? 2 : 0};

    if (is_multicast) {
        tt::tt_fabric::test::run_raw_multicast_write_test(fixture, p);
    } else {
        tt::tt_fabric::test::run_raw_unicast_write_test(fixture, p);
    }
}

// Helper function to get variant name as string (for test naming)
static std::string GetRawVariantName(tt::tt_fabric::test::RawApiVariant variant) {
    switch (variant) {
        case tt::tt_fabric::test::RawApiVariant::UnicastWrite: return "UnicastWrite";
        case tt::tt_fabric::test::RawApiVariant::UnicastWriteConnMgr: return "UnicastWriteConnMgr";
        case tt::tt_fabric::test::RawApiVariant::MulticastWrite: return "MulticastWrite";
        case tt::tt_fabric::test::RawApiVariant::MulticastWriteConnMgr: return "MulticastWriteConnMgr";
        case tt::tt_fabric::test::RawApiVariant::LinearUnicastWrite: return "LinearUnicastWrite";
        case tt::tt_fabric::test::RawApiVariant::LinearMulticastWrite: return "LinearMulticastWrite";
        case tt::tt_fabric::test::RawApiVariant::LinearSparseMulticastWrite: return "LinearSparseMulticastWrite";
        default: return "UnknownVariant";
    }
}

// Instantiate 2D fabric test suite (stub: initial page sizes, L1 only)
INSTANTIATE_TEST_SUITE_P(
    AllRawVariantsAndSizes,
    RawPacketizationTest,
    ::testing::Combine(
        ::testing::Values(
            tt::tt_fabric::test::RawApiVariant::UnicastWrite,
            tt::tt_fabric::test::RawApiVariant::MulticastWrite),
        ::testing::Values(uint32_t{4096}),  // Initial stub - real sizes depend on MAX (runtime)
        ::testing::Values(false)            // L1 only for initial stub
        ),
    [](const ::testing::TestParamInfo<RawPacketizationTest::ParamType>& info) {
        auto api_variant = std::get<0>(info.param);
        auto page_size = std::get<1>(info.param);
        auto use_dram_dst = std::get<2>(info.param);
        std::string name = GetRawVariantName(api_variant);
        name += "_" + std::to_string(page_size) + "B";
        name += use_dram_dst ? "_DRAM" : "_L1";
        return name;
    });

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
