// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// #22835: These Fixtures will be removed once tests are fully migrated, and replaced by
// UnitMeshMultiCQSingleDeviceFixtures
class MultiCommandQueueSingleDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (this->num_cqs_ != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        const chip_id_t device_id = 0;
        const DispatchCoreType dispatch_core_type = this->get_dispatch_core_type();
        this->create_device(device_id, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    }

    void TearDown() override {
        if (this->device_ != nullptr) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            return false;
        }
        return true;
    }

    DispatchCoreType get_dispatch_core_type() {
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in SetUp()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }
        return dispatch_core_type;
    }

    void create_device(
        const chip_id_t device_id,
        const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        const DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER) {
        this->device_ = tt::tt_metal::CreateDevice(
            device_id, this->num_cqs_, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_type);
    }

    tt::tt_metal::IDevice* device_ = nullptr;
    tt::ARCH arch_;
    uint8_t num_cqs_;
};

class UnitMeshMultiCQSingleDeviceFixture : public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (this->num_cqs_ != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }
        log_info(tt::LogTest, "here1");
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        log_info(tt::LogTest, "here2");
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        log_info(tt::LogTest, "here3");
        auto test =
            (enable_remote_chip or
             tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB);
        log_info(tt::LogTest, "here4");
        const chip_id_t device_id =
            test ? *tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids().begin()
                 : *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        log_info(tt::LogTest, "Using device {} for UnitMeshMultiCQSingleDeviceFixture", device_id);
        this->create_device(device_id, DEFAULT_TRACE_REGION_SIZE);
    }

    void TearDown() override { device_.reset(); }

    bool validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            return false;
        }
        return true;
    }

    DispatchCoreType get_dispatch_core_type() {
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in SetUp()");
                dispatch_core_type = DispatchCoreType::ETH;
            }
        }
        return dispatch_core_type;
    }

    void create_device(const chip_id_t device_id, const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        std::vector<chip_id_t> chip_id = {device_id};

        auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
            chip_id, DEFAULT_L1_SMALL_SIZE, trace_region_size, 2, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            log_info(tt::LogTest, "Reserved device: {}", id);
        }

        this->device_ = reserved_devices[device_id];
    }

    std::shared_ptr<distributed::MeshDevice> device_;
    tt::ARCH arch_;
    uint8_t num_cqs_;
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshMultiCQSingleDeviceProgramFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceBufferFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceEventFixture : public UnitMeshMultiCQSingleDeviceFixture {};

class UnitMeshMultiCQSingleDeviceTraceFixture : public UnitMeshMultiCQSingleDeviceFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }

        this->num_cqs_ = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        this->create_device(0 /* device_id */, trace_region_size);
    }
};
class UnitMeshMultiCQMultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        log_info(tt::LogTest, "DEBUG: Starting UnitMeshMultiCQMultiDeviceFixture SetUp");

        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }

        auto num_cqs = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        log_info(tt::LogTest, "DEBUG: Number of CQs: {}", num_cqs);
        if (num_cqs != 2) {
            log_info(tt::LogTest, "This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
            GTEST_SKIP();
        }

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        log_info(tt::LogTest, "DEBUG: Got dispatch core config");

        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        log_info(tt::LogTest, "DEBUG: MMIO device ID: {}", mmio_device_id);

        std::vector<chip_id_t> chip_ids;
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        log_info(tt::LogTest, "DEBUG: enable_remote_chip: {}", enable_remote_chip ? "true" : "false");

        // Get all available chip IDs first
        auto all_mmio_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids();
        auto all_user_exposed_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();

        log_info(tt::LogTest, "DEBUG: Available MMIO chip IDs count: {}", all_mmio_chip_ids.size());
        for (auto id : all_mmio_chip_ids) {
            log_info(tt::LogTest, "DEBUG: MMIO chip ID: {}", id);
        }

        log_info(tt::LogTest, "DEBUG: Available user exposed chip IDs count: {}", all_user_exposed_chip_ids.size());
        for (auto id : all_user_exposed_chip_ids) {
            log_info(tt::LogTest, "DEBUG: User exposed chip ID: {}", id);
        }

        if (enable_remote_chip or
            tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            log_info(tt::LogTest, "DEBUG: Using remote chip path");

            // Use all user exposed chip IDs without filtering
            for (chip_id_t id : all_user_exposed_chip_ids) {
                log_info(tt::LogTest, "DEBUG: Adding user exposed chip ID: {}", id);
                chip_ids.push_back(id);
            }

            // Fallback: if no user exposed chips, try MMIO chips
            if (chip_ids.empty()) {
                log_warning(tt::LogTest, "DEBUG: No user exposed chips available, falling back to MMIO chips");
                for (chip_id_t id : all_mmio_chip_ids) {
                    log_info(tt::LogTest, "DEBUG: Adding MMIO chip ID as fallback: {}", id);
                    chip_ids.push_back(id);
                }
            }
        } else {
            log_info(tt::LogTest, "DEBUG: Using local chip path");

            // First try with just MMIO device
            chip_ids.push_back(mmio_device_id);

            // Validate that the MMIO device is actually available
            if (all_mmio_chip_ids.find(mmio_device_id) == all_mmio_chip_ids.end()) {
                log_warning(
                    tt::LogTest, "DEBUG: Selected MMIO device {} not found in available MMIO chips", mmio_device_id);
                chip_ids.clear();

                // Use first available MMIO chip as fallback
                if (!all_mmio_chip_ids.empty()) {
                    auto fallback_id = *all_mmio_chip_ids.begin();
                    log_info(tt::LogTest, "DEBUG: Using fallback MMIO chip ID: {}", fallback_id);
                    chip_ids.push_back(fallback_id);
                }
            }
        }

        if (chip_ids.empty()) {
            log_error(tt::LogTest, "DEBUG: No valid chip IDs available for mesh creation!");
            GTEST_SKIP() << "No valid chip IDs available for testing";
        }

        log_info(tt::LogTest, "DEBUG: Final chip IDs to create: {}", chip_ids.size());
        for (size_t i = 0; i < chip_ids.size(); i++) {
            log_info(tt::LogTest, "DEBUG: chip_ids[{}] = {}", i, chip_ids[i]);
        }

        log_info(tt::LogTest, "DEBUG: About to call MeshDevice::create_unit_meshes");
        try {
            auto reserved_devices = distributed::MeshDevice::create_unit_meshes(
                chip_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2, dispatch_core_config);
            log_info(
                tt::LogTest,
                "DEBUG: MeshDevice::create_unit_meshes completed, created {} devices",
                reserved_devices.size());

            if (reserved_devices.empty()) {
                log_error(tt::LogTest, "DEBUG: No devices were created!");
                GTEST_SKIP() << "No devices were created by create_unit_meshes";
            }

            // Log all created device IDs
            for (const auto& [device_id, device] : reserved_devices) {
                log_info(tt::LogTest, "DEBUG: Reserved device created with ID: {}", device_id);
            }

            if (enable_remote_chip) {
                log_info(tt::LogTest, "DEBUG: Processing remote chip path");
                const auto tunnels =
                    tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
                log_info(tt::LogTest, "DEBUG: Got {} tunnels from MMIO device", tunnels.size());

                for (const auto& tunnel : tunnels) {
                    log_info(tt::LogTest, "DEBUG: Processing tunnel with {} chips", tunnel.size());
                    for (const auto chip_id : tunnel) {
                        log_info(tt::LogTest, "DEBUG: Checking tunnel chip ID: {}", chip_id);
                        if (reserved_devices.find(chip_id) != reserved_devices.end()) {
                            log_info(
                                tt::LogTest, "DEBUG: Found chip {} in reserved_devices, adding to devices_", chip_id);
                            devices_.push_back(reserved_devices.at(chip_id));
                        } else {
                            log_warning(tt::LogTest, "DEBUG: Chip {} NOT found in reserved_devices", chip_id);
                        }
                    }
                    break;
                }
            } else {
                log_info(tt::LogTest, "DEBUG: Processing local chip path, looking for MMIO device: {}", mmio_device_id);
                if (reserved_devices.find(mmio_device_id) != reserved_devices.end()) {
                    log_info(tt::LogTest, "DEBUG: Found MMIO device {} in reserved_devices", mmio_device_id);
                    devices_.push_back(reserved_devices.at(mmio_device_id));
                } else {
                    log_error(tt::LogTest, "DEBUG: MMIO device {} NOT found in reserved_devices!", mmio_device_id);
                    // Log what devices are actually available
                    for (const auto& [device_id, device] : reserved_devices) {
                        log_error(tt::LogTest, "DEBUG: Available device ID: {}", device_id);
                    }

                    // Try to use any available device as fallback
                    if (!reserved_devices.empty()) {
                        auto fallback_device_id = reserved_devices.begin()->first;
                        log_warning(tt::LogTest, "DEBUG: Using fallback device ID: {}", fallback_device_id);
                        devices_.push_back(reserved_devices.begin()->second);
                    } else {
                        GTEST_SKIP() << "No devices available in reserved_devices map";
                    }
                }
            }

            if (devices_.empty()) {
                log_error(tt::LogTest, "DEBUG: No devices were added to devices_ vector!");
                GTEST_SKIP() << "No devices available for testing";
            }

        } catch (const std::exception& e) {
            log_error(tt::LogTest, "DEBUG: Exception in SetUp: {}", e.what());
            GTEST_SKIP() << "Exception in SetUp: " << e.what();
        }

        log_info(tt::LogTest, "DEBUG: SetUp completed successfully, total devices: {}", devices_.size());
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device.reset();
        }
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> devices_;
    distributed::MeshCoordinate zero_coord_ = distributed::MeshCoordinate::zero_coordinate(2);
    distributed::MeshCoordinateRange device_range_ = distributed::MeshCoordinateRange(zero_coord_, zero_coord_);
};

class UnitMeshMultiCQMultiDeviceBufferFixture : public UnitMeshMultiCQMultiDeviceFixture {};

class UnitMeshMultiCQMultiDeviceEventFixture : public UnitMeshMultiCQMultiDeviceFixture {};

class DISABLED_MultiCQMultiDeviceOnFabricFixture : public UnitMeshMultiCQMultiDeviceFixture,
                                                   public ::testing::WithParamInterface<tt::tt_fabric::FabricConfig> {
private:
    // Save the result to reduce UMD calls
    inline static bool should_skip_ = false;

protected:
    void SetUp() override {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Dispatch on Fabric tests only applicable on Wormhole B0";
        }
        // Skip for TG as it's still being implemented
        if (tt::tt_metal::IsGalaxyCluster()) {
            GTEST_SKIP();
        }
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_fabric::SetFabricConfig(GetParam(), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        UnitMeshMultiCQMultiDeviceFixture::SetUp();
    }

    void TearDown() override {
        UnitMeshMultiCQMultiDeviceFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

}  // namespace tt::tt_metal
