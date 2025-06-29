// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "dispatch_fixture.hpp"
#include "umd/device/types/arch.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

class TwoDeviceFixture : public DispatchFixture<TwoDeviceFixture> {
public:
    static bool WillSkip() {
        if (!IsSlowDispatch()) {
            return true;
        }
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires Slow Dispatch and >= 2 Devices"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<TwoDeviceFixture>::DoSetUpTestSuiteWithNumberOfDevices(2);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<TwoDeviceFixture>::DoTearDownTestSuite();
    }
};

class N300DeviceFixture : public DispatchFixture<N300DeviceFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (!(arch == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1)) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires Slow Dispatch and N300"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<N300DeviceFixture>::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<N300DeviceFixture>::DoTearDownTestSuite();
    }
};

template <typename IMPL>
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

    enum class MeshDeviceType {
        // N150/P150 devices opened as 1x1 meshes.
        N150,
        P150,
        N300,
        P300,
        T3000,
        TG,
    };

    struct Config {
        // If empty, the mesh device type will be deduced automatically based on the connected devices.
        // Otherwise, the associated tests will run only if the connected cluster corresponds to one of the
        // specified mesh device types.
        std::unordered_set<MeshDeviceType> mesh_device_types;
        int num_cqs = 1;
        uint32_t trace_region_size = 0;
        uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
        FabricConfig fabric_config = FabricConfig::DISABLED;
    };

    inline static std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    inline static Config config_;

    static void DoSetUpTestSuite(const Config& fixture_config) {
        config_ = fixture_config;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(LogTest, "Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.");
            return;
        }

        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        const auto mesh_device_type = derive_mesh_device_type(num_devices, arch);
        if (!mesh_device_type) {
            log_info(
                LogTest,
                "Skipping MeshDevice test suite on a machine with an unsupported number of devices {}.",
                num_devices);
            return;
        }

        if (!config_.mesh_device_types.empty() &&
            config_.mesh_device_types.find(*mesh_device_type) == config_.mesh_device_types.end()) {
            std::vector<std::string> requested_device_types;
            std::transform(
                config_.mesh_device_types.begin(),
                config_.mesh_device_types.end(),
                std::back_inserter(requested_device_types),
                [](const auto t) { return mesh_device_type_to_string(t); });
            log_info(
                LogTest,
                "Skipping MeshDevice test suite on a {} machine that does not match any of the configured mesh device "
                "types {}",
                mesh_device_type_to_string(*mesh_device_type),
                boost::algorithm::join(requested_device_types, ", "));
            return;
        }

        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        bool is_n300_or_t3k_cluster = cluster_type == tt::ClusterType::T3K or cluster_type == tt::ClusterType::N300;
        auto core_type =
            (config_.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

        if (config_.fabric_config != FabricConfig::DISABLED) {
            tt::tt_metal::detail::SetFabricConfig(config_.fabric_config);
        }
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(get_mesh_shape(*mesh_device_type)),
            0,
            config_.trace_region_size,
            config_.num_cqs,
            core_type,
            {},
            config_.worker_l1_size);
    }

    static void DoTearDownTestSuite() {
        if (!mesh_device_) {
            return;
        }
        mesh_device_->close();
        mesh_device_.reset();
        if (config_.fabric_config != FabricConfig::DISABLED) {
            tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
        }
    }

    static void SetUpTestSuite() { DoSetUpTestSuite(config_); }
    static void TearDownTestSuite() { DoTearDownTestSuite(); }

    void SetUp() override {}

    void TearDown() override {}

private:
    // Returns the mesh shape for a given mesh device type.
    static MeshShape get_mesh_shape(MeshDeviceType mesh_device_type) {
        switch (mesh_device_type) {
            case MeshDeviceType::N150:
            case MeshDeviceType::P150: return MeshShape(1, 1);
            case MeshDeviceType::N300:
            case MeshDeviceType::P300: return MeshShape(2, 1);
            case MeshDeviceType::T3000: return MeshShape(2, 4);
            case MeshDeviceType::TG: return MeshShape(4, 8);
            default: TT_FATAL(false, "Querying shape for unspecified Mesh Type.");
        }
    }

    // Manual string conversion for MeshDeviceType enum
    static std::string mesh_device_type_to_string(MeshDeviceType mesh_device_type) {
        switch (mesh_device_type) {
            case MeshDeviceType::N150: return "N150";
            case MeshDeviceType::P150: return "P150";
            case MeshDeviceType::N300: return "N300";
            case MeshDeviceType::P300: return "P300";
            case MeshDeviceType::T3000: return "T3000";
            case MeshDeviceType::TG: return "TG";
            default: return "Unknown";
        }
    }

    // Determines the mesh device type based on the number of devices.
    static std::optional<MeshDeviceType> derive_mesh_device_type(size_t num_devices, tt::ARCH arch) {
        switch (num_devices) {
            case 1: {
                switch (arch) {
                    case tt::ARCH::WORMHOLE_B0: return MeshDeviceType::N150;
                    case tt::ARCH::BLACKHOLE: return MeshDeviceType::P150;
                    default: return std::nullopt;
                }
            }
            case 2: {
                switch (arch) {
                    case tt::ARCH::WORMHOLE_B0: return MeshDeviceType::N300;
                    case tt::ARCH::BLACKHOLE: return MeshDeviceType::P300;
                    default: return std::nullopt;
                }
            }
            case 8: {
                switch (arch) {
                    case tt::ARCH::WORMHOLE_B0: return MeshDeviceType::T3000;
                    default: return std::nullopt;
                }
            }
            case 32: {
                switch (arch) {
                    case tt::ARCH::WORMHOLE_B0: return MeshDeviceType::TG;
                    default: return std::nullopt;
                }
            }
            default: return std::nullopt;
        }
    }
};

// Fixtures that determine the mesh device type automatically.
// The associated test will be run if the topology is supported.
class GenericMeshDeviceFixture : public MeshDeviceFixtureBase<GenericMeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<GenericMeshDeviceFixture>::DoSetUpTestSuite(Config{.num_cqs = 1});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<GenericMeshDeviceFixture>::DoTearDownTestSuite(); }
};

class GenericMultiCQMeshDeviceFixture : public MeshDeviceFixtureBase<GenericMultiCQMeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<GenericMultiCQMeshDeviceFixture>::DoSetUpTestSuite(Config{.num_cqs = 2});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<GenericMultiCQMeshDeviceFixture>::DoTearDownTestSuite(); }
};

// Fixtures that specify the mesh device type explicitly.
// The associated test will be run if the cluster topology matches
// what is specified.
class T3000MeshDeviceFixture : public MeshDeviceFixtureBase<T3000MeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<T3000MeshDeviceFixture>::DoSetUpTestSuite(
            Config{.mesh_device_types = {MeshDeviceType::T3000}});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<T3000MeshDeviceFixture>::DoTearDownTestSuite(); }
};

class TGMeshDeviceFixture : public MeshDeviceFixtureBase<TGMeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<TGMeshDeviceFixture>::DoSetUpTestSuite(Config{.mesh_device_types = {MeshDeviceType::TG}});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<TGMeshDeviceFixture>::DoTearDownTestSuite(); }
};

class T3000MultiCQMeshDeviceFixture : public MeshDeviceFixtureBase<T3000MultiCQMeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<T3000MultiCQMeshDeviceFixture>::DoSetUpTestSuite(
            Config{.mesh_device_types = {MeshDeviceType::T3000}, .num_cqs = 2});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<T3000MultiCQMeshDeviceFixture>::DoTearDownTestSuite(); }
};

class TGMultiCQMeshDeviceFixture : public MeshDeviceFixtureBase<TGMultiCQMeshDeviceFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<TGMultiCQMeshDeviceFixture>::DoSetUpTestSuite(
            Config{.mesh_device_types = {MeshDeviceType::TG}, .num_cqs = 2});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<TGMultiCQMeshDeviceFixture>::DoTearDownTestSuite(); }
};

class T3000MeshDevice1DFabricFixture : public MeshDeviceFixtureBase<T3000MeshDevice1DFabricFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<T3000MeshDevice1DFabricFixture>::DoSetUpTestSuite(Config{
            .mesh_device_types = {MeshDeviceType::T3000}, .num_cqs = 1, .fabric_config = FabricConfig::FABRIC_1D});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<T3000MeshDevice1DFabricFixture>::DoTearDownTestSuite(); }
};

class T3000MeshDevice2DFabricFixture : public MeshDeviceFixtureBase<T3000MeshDevice2DFabricFixture> {
protected:
    static void SetUpTestSuite() {
        MeshDeviceFixtureBase<T3000MeshDevice2DFabricFixture>::DoSetUpTestSuite(Config{
            .mesh_device_types = {MeshDeviceType::T3000},
            .num_cqs = 1,
            .fabric_config = FabricConfig::FABRIC_2D_DYNAMIC});
    }

    static void TearDownTestSuite() { MeshDeviceFixtureBase<T3000MeshDevice2DFabricFixture>::DoTearDownTestSuite(); }
};

}  // namespace tt::tt_metal
