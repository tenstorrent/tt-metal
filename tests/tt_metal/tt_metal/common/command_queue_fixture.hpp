// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"

namespace tt::tt_metal {

class CommandQueueFixture : public DispatchFixture {
protected:
    tt::tt_metal::IDevice* device_;
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_device();
    }

    void TearDown() override {
        if (!this->IsSlowDispatch()) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

    void validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }
    }

    void create_device(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const chip_id_t device_id = 0;
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        this->device_ =
            tt::tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
    }
};

class CommandQueueEventFixture : public CommandQueueFixture {};

class CommandQueueBufferFixture : public CommandQueueFixture {};

class CommandQueueProgramFixture : public CommandQueueFixture {};

class CommandQueueTraceFixture : public CommandQueueFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }

    void CreateDevice(const size_t trace_region_size) { this->create_device(trace_region_size); }
};

class CommandQueueSingleCardFixture : virtual public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    void validate_dispatch_mode() {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }
    }

    void create_devices(const std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        const chip_id_t mmio_device_id = 0;
        std::vector<chip_id_t> chip_ids;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (unsigned int id = 0; id < tt::tt_metal::GetNumAvailableDevices(); id++) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            chip_ids, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_config);
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip) {
            for (const auto& [id, device] : this->reserved_devices_) {
                this->devices_.push_back(device);
            }
        } else {
            for (const auto& chip_id : chip_ids) {
                this->devices_.push_back(this->reserved_devices_.at(chip_id));
            }
        }
    }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
};

class CommandQueueSingleCardBufferFixture : public CommandQueueSingleCardFixture {};

class CommandQueueSingleCardTraceFixture : virtual public CommandQueueSingleCardFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices(90000000);
    }
};

class CommandQueueSingleCardProgramFixture : virtual public CommandQueueSingleCardFixture {};

class CommandQueueMultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = false;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            this->slow_dispatch_ = true;
            GTEST_SKIP();
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2) {
            GTEST_SKIP();
        }

        std::vector<chip_id_t> chip_ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            chip_ids.push_back(id);
        }

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            chip_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    size_t num_devices_;
};

class CommandQueueMultiDeviceProgramFixture : public CommandQueueMultiDeviceFixture {};

class CommandQueueMultiDeviceBufferFixture : public CommandQueueMultiDeviceFixture {};

}  // namespace tt::tt_metal
