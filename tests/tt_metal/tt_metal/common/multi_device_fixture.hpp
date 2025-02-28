// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "dispatch_fixture.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

class MultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override { this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()); }
};

class TwoDeviceFixture : public MultiDeviceFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }

        MultiDeviceFixture::SetUp();

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        if (num_devices == 2) {
            std::vector<chip_id_t> ids;
            for (chip_id_t id = 0; id < num_devices; id++) {
                ids.push_back(id);
            }

            const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
            tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
            this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        } else {
            GTEST_SKIP() << "TwoDeviceFixture can only be run on machines with two devices";
        }
    }
};

class N300DeviceFixture : public MultiDeviceFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }

        MultiDeviceFixture::SetUp();

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1) {
            std::vector<chip_id_t> ids;
            for (chip_id_t id = 0; id < num_devices; id++) {
                ids.push_back(id);
            }

            const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
            tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
            this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        } else {
            GTEST_SKIP();
        }
    }
};

class MeshDeviceFixtureBase : public ::testing::Test {
protected:
    using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;
    using MeshDeviceConfig = ::tt::tt_metal::distributed::MeshDeviceConfig;
    using MeshShape = ::tt::tt_metal::distributed::MeshShape;

    enum class MeshDeviceType {
        N300,
        T3000,
    };

    struct Config {
        // If unset, the mesh device type will be deduced automatically based on the connected devices.
        // The associated test will be run if the connected cluster corresponds to a supported topology.
        std::optional<MeshDeviceType> mesh_device_type;
        int num_cqs = 1;
        uint32_t trace_region_size = 0;
    };

    MeshDeviceFixtureBase(const Config& fixture_config) : config_(fixture_config) {}

    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }

        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping MeshDevice test suite on a non-wormhole machine.";
        }

        const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        const auto mesh_device_type = derive_mesh_device_type(num_devices);
        if (!mesh_device_type) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with an unsupported number of devices {}.", num_devices);
        }

        if (config_.mesh_device_type.has_value() && *config_.mesh_device_type != *mesh_device_type) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a {} machine that does not match the configured mesh device type {}",
                magic_enum::enum_name(*mesh_device_type),
                magic_enum::enum_name(*config_.mesh_device_type));
        }
        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        auto core_type = (config_.num_cqs >= 2) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig{.mesh_shape = get_mesh_shape(*mesh_device_type)},
            0,
            config_.trace_region_size,
            config_.num_cqs,
            core_type);
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }
        mesh_device_->close();
        mesh_device_.reset();
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;

private:
    // Returns the mesh shape for a given mesh device type.
    MeshShape get_mesh_shape(MeshDeviceType mesh_device_type) {
        switch (mesh_device_type) {
            case MeshDeviceType::N300: return MeshShape(2, 1);
            case MeshDeviceType::T3000: return MeshShape(2, 4);
            default: TT_FATAL(false, "Querying shape for unspecified Mesh Type.");
        }
    }

    // Determines the mesh device type based on the number of devices.
    std::optional<MeshDeviceType> derive_mesh_device_type(size_t num_devices) {
        switch (num_devices) {
            case 2: return MeshDeviceType::N300;
            case 8: return MeshDeviceType::T3000;
            default: return std::nullopt;
        }
    }

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

class GenericMeshDeviceTraceFixture : public MeshDeviceFixtureBase {
protected:
    GenericMeshDeviceTraceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 1, .trace_region_size = (64 << 20)}) {}
};

// Fixtures that specify the mesh device type explicitly.
// The associated test will be run if the cluster topology matches
// what is specified.
class N300MeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    N300MeshDeviceFixture() : MeshDeviceFixtureBase(Config{.mesh_device_type = MeshDeviceType::N300}) {}
};

class T3000MeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    T3000MeshDeviceFixture() : MeshDeviceFixtureBase(Config{.mesh_device_type = MeshDeviceType::T3000}) {}
};

class N300MultiCQMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    N300MultiCQMeshDeviceFixture() :
        MeshDeviceFixtureBase(Config{.mesh_device_type = MeshDeviceType::N300, .num_cqs = 2}) {}
};

class T3000MultiCQMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    T3000MultiCQMeshDeviceFixture() :
        MeshDeviceFixtureBase(Config{.mesh_device_type = MeshDeviceType::T3000, .num_cqs = 2}) {}
};

}  // namespace tt::tt_metal
