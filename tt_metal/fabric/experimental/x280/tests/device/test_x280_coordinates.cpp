// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_metal_transport.hpp"
#include "x280/device/x280_runtime.hpp"

#include <gtest/gtest.h>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <array>
#include <memory>
#include <stdexcept>

namespace {

using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::NOC;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshShape;

class X280Coordinates : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
        ASSERT_EQ(mesh_->num_devices(), 2u);
        transport_ = x280::X280MetalTransport::bind(*mesh_);
    }

    void TearDown() override {
        transport_.reset();
        if (mesh_ && mesh_->is_initialized()) {
            mesh_->close();
        }
    }

    std::shared_ptr<MeshDevice> mesh_;
    std::shared_ptr<x280::X280MetalTransport> transport_;
};

CoreCoord resolve(const MeshDevice& scope, const CoreCoord& logical, CoreType type) {
    const auto t = x280::X280Runtime::virtual_core_from_logical_core(scope, logical, type, NOC::NOC_0);
    return CoreCoord(t.x(), t.y());
}

TEST_F(X280Coordinates, LogicalL2cpuOrderWithoutMetalPatch) {
    EXPECT_THROW(mesh_->virtual_core_from_logical_core({0, 0}, CoreType::L2CPU), std::runtime_error);
    constexpr std::array<std::size_t, 4> expected_y{3, 5, 7, 9};
    for (std::size_t y = 0; y < expected_y.size(); ++y) {
        EXPECT_EQ(resolve(*mesh_, {0, y}, CoreType::L2CPU), CoreCoord(8, expected_y[y]));
    }
    EXPECT_THROW(resolve(*mesh_, {1, 0}, CoreType::L2CPU), std::runtime_error);
    EXPECT_THROW(resolve(*mesh_, {0, 4}, CoreType::L2CPU), std::runtime_error);
}

TEST_F(X280Coordinates, ExistingCoreTypesRetainMetalSemantics) {
    for (const auto type : {CoreType::TENSIX, CoreType::WORKER, CoreType::ETH, CoreType::DRAM}) {
        EXPECT_EQ(resolve(*mesh_, {0, 0}, type), mesh_->virtual_core_from_logical_core({0, 0}, type));
    }
    EXPECT_THROW(resolve(*mesh_, {0, 0}, CoreType::PCIE), std::runtime_error);
}

// The resolver agrees with Metal's type-specific calls.
TEST_F(X280Coordinates, TypedResolutionMatchesTypeSpecificCalls) {
    auto unit = mesh_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    const auto* device = unit->get_devices().front();
    const auto eth_cores = device->get_active_ethernet_cores(true);
    ASSERT_FALSE(eth_cores.empty()) << "the mock descriptor must link its chips by Ethernet";
    for (const auto& eth : eth_cores) {
        EXPECT_EQ(resolve(*unit, eth, CoreType::ETH), device->ethernet_core_from_logical_core(eth));
    }
    for (const CoreCoord tensix : {CoreCoord{0, 0}, CoreCoord{6, 3}, CoreCoord{7, 3}}) {
        EXPECT_EQ(resolve(*unit, tensix, CoreType::TENSIX), unit->worker_core_from_logical_core(tensix));
    }
}

TEST_F(X280Coordinates, PhysicalIdentityIsNotALogicalIndex) {
    for (int chip : mesh_->get_device_ids()) {
        EXPECT_EQ(transport_->virtual_core_from_noc0(chip, {8, 9}), CoreCoord(8, 9));
        EXPECT_EQ(transport_->virtual_core_from_noc0(chip, {8, 5}), CoreCoord(8, 5));
        EXPECT_THROW(transport_->virtual_core_from_noc0(chip, {1, 2}), std::runtime_error);
    }
    EXPECT_THROW(transport_->virtual_core_from_noc0(999, {8, 3}), std::runtime_error);
}

TEST_F(X280Coordinates, ClosedScopeIsRejected) {
    auto scope = mesh_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    scope->close();
    EXPECT_THROW(resolve(*scope, {0, 0}, CoreType::L2CPU), std::runtime_error);
    EXPECT_THROW(resolve(*scope, {0, 0}, CoreType::ETH), std::runtime_error);
}

TEST_F(X280Coordinates, HarvestedLogicalCoordinatesCompact) {
    constexpr std::array<std::size_t, 3> expected_y{3, 7, 9};
    for (std::size_t y = 0; y < expected_y.size(); ++y) {
        EXPECT_EQ(resolve(*mesh_, {0, y}, CoreType::L2CPU), CoreCoord(8, expected_y[y]));
    }
    EXPECT_THROW(resolve(*mesh_, {0, 3}, CoreType::L2CPU), std::runtime_error);
    for (int chip : mesh_->get_device_ids()) {
        EXPECT_EQ(transport_->virtual_core_from_noc0(chip, {8, 9}), CoreCoord(8, 9));
        EXPECT_THROW(transport_->virtual_core_from_noc0(chip, {8, 5}), std::runtime_error);
    }
}

TEST_F(X280Coordinates, DisagreementRequiresNarrowerScope) {
    EXPECT_EQ(resolve(*mesh_, {0, 0}, CoreType::L2CPU), CoreCoord(8, 3));
    EXPECT_THROW(resolve(*mesh_, {0, 1}, CoreType::L2CPU), std::runtime_error);
    EXPECT_THROW(resolve(*mesh_, {0, 3}, CoreType::L2CPU), std::runtime_error);
    for (std::size_t column = 0; column < 2; ++column) {
        auto scope = mesh_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, column));
        const auto ids = scope->get_device_ids();
        ASSERT_EQ(ids.size(), 1u);
        const std::size_t expected_y = ids.front() == 0 ? 5 : 7;
        EXPECT_EQ(resolve(*scope, {0, 1}, CoreType::L2CPU), CoreCoord(8, expected_y));
        const auto on_chip = x280::X280Runtime::virtual_core_from_logical_core(
            *mesh_, MeshCoordinate(0, column), {0, 1}, CoreType::L2CPU, NOC::NOC_0);
        EXPECT_EQ(CoreCoord(on_chip.x(), on_chip.y()), CoreCoord(8, expected_y));
    }
}

}  // namespace
