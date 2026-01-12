// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "mesh_dispatch_fixture.hpp"
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

class TwoMeshDeviceFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices != 2) {
            GTEST_SKIP() << "TwoDeviceFixture can only be run on machines with two devices";
        }

        MeshDispatchFixture::SetUp();
    }
};

class N300MeshDeviceFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1) {
            MeshDispatchFixture::SetUp();
        } else {
            GTEST_SKIP() << "This suite can only be run on N300";
        }
    }
};

class TwoDeviceBlackholeFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ == tt::ARCH::BLACKHOLE && num_devices == 2 && num_pci_devices >= 1) {
            MeshDispatchFixture::SetUp();
        } else {
            GTEST_SKIP() << "This suite can only be run on two chip Blackhole systems";
        }
    }
};

class MeshDeviceFixtureBase : public ::testing::Test {
public:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() {
        TT_FATAL(mesh_device_, "MeshDevice not initialized in {}", __FUNCTION__);
        return mesh_device_;
    }

protected:
    using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;
    using MeshDeviceConfig = ::tt::tt_metal::distributed::MeshDeviceConfig;
    using MeshShape = ::tt::tt_metal::distributed::MeshShape;

    struct Config {
        // If specified, the fixture will open a mesh device with the specified shape and offset.
        // Otherwise, SystemMesh shape with zero offset will be used.
        std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape;
        std::optional<tt::tt_metal::distributed::MeshCoordinate> mesh_offset;

        // If specified, the associated tests will run only if the machine architecture matches the specified
        // architecture.
        std::optional<tt::ARCH> arch;

        int num_cqs = 1;
        uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
        uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
        uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
        tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED;
        tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED;
    };

    explicit MeshDeviceFixtureBase(const Config& fixture_config) : config_(fixture_config) {}

    void SetUp() override {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }

        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (config_.arch.has_value() && *config_.arch != arch) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with architecture {} that does not match the requested "
                "architecture {}",
                arch,
                *config_.arch);
        }

        const auto system_mesh_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
        if (config_.mesh_shape.has_value() && config_.mesh_shape->mesh_size() > system_mesh_shape.mesh_size()) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with SystemMesh {} that is smaller than the requested "
                "mesh "
                "shape {}",
                system_mesh_shape,
                *config_.mesh_shape);
        }

        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        bool is_n300_or_t3k_cluster =
            cluster_type == tt::tt_metal::ClusterType::T3K or cluster_type == tt::tt_metal::ClusterType::N300;
        auto core_type =
            (config_.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

        if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            tt_fabric::SetFabricConfig(
                config_.fabric_config,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                std::nullopt,
                config_.fabric_tensix_config,
                config_.fabric_udm_mode);
        }
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(config_.mesh_shape.value_or(system_mesh_shape), config_.mesh_offset),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            core_type,
            {},
            config_.worker_l1_size);
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }
        mesh_device_->close();
        mesh_device_.reset();
        if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
        }
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;

private:
    Config config_;
};

// Fixtures that determine the mesh device type automatically.
// The associated test will be run if the topology is supported.
class GenericMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    GenericMeshDeviceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 1}) {}
};

class GenericMultiCQMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    GenericMultiCQMeshDeviceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 2}) {}
};

// Fixtures that specify the mesh device type explicitly.
// The associated test will be run if the cluster topology matches
// what is specified.
class MeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}}) {}
};

class MeshDevice4x8Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice4x8Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{4, 8}}) {}
};

class MultiCQMeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MultiCQMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 2}) {}
};

class MeshDevice2x4Fabric1DFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric1DFixture() :
        MeshDeviceFixtureBase(
            Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_1D}) {}
};

class GenericMeshDeviceFabric2DFixture : public MeshDeviceFixtureBase {
protected:
    GenericMeshDeviceFabric2DFixture() :
        MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_2D}) {}
};

class MeshDevice2x4Fabric2DFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric2DFixture() :
        MeshDeviceFixtureBase(
            Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_2D}) {}
};

class MeshDevice1x4Fabric2DUDMFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice1x4Fabric2DUDMFixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{1, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED}) {}

    void SetUp() override {
        // When Fabric is enabled, it requires all devices in the system to be active.
        // For Blackhole P150_X8 systems (8 independent chips), opening 4 devices leaves 4 inactive, causing a fatal error.
        // For T3K (4 N300 boards = 8 chips), opening 4 MMIO devices automatically activates all 8 chips, so it works.
        // Skip the test only on Blackhole systems with more than 4 devices.
        const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        const size_t num_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices();
        const size_t requested_devices = 4;  // 1x4 mesh

        if (cluster_type == tt::tt_metal::ClusterType::P150_X8 && num_devices > requested_devices) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice1x4Fabric2DUDMFixture test on P150_X8: "
                "System has {} independent Blackhole devices but test only requests {}. "
                "Fabric requires all devices to be active.",
                num_devices,
                requested_devices);
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

class MeshDevice2x4Fabric2DUDMFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric2DUDMFixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{2, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED}) {}
};

}  // namespace tt::tt_metal
