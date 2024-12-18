// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "dispatch_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

class DeviceFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Some CI machines have lots of cards, running all tests on all cards is slow
        // Coverage for multidevices is decent if we just confirm 2 work
        this->num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::GRAYSKULL && num_devices_ > 2) {
            this->num_devices_ = 2;
        }

        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        this->create_devices(ids);
    }

    void validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(
                tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }
    }

    void create_devices(const std::vector<chip_id_t>& device_ids) {
        const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
        tt::DevicePool::initialize(
            device_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        this->num_devices_ = this->devices_.size();
    }

    size_t num_devices_;
};

class DeviceSingleCardFixture : public DispatchFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    virtual void validate_dispatch_mode() {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(
                tt::LogTest, "This suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }
    }

    void create_devices() {
        const chip_id_t mmio_device_id = 0;
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id});
        this->device_ = this->reserved_devices_.at(mmio_device_id);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        this->num_devices_ = this->reserved_devices_.size();
    }

    tt::tt_metal::Device* device_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    size_t num_devices_;
};

class DeviceSingleCardBufferFixture : public DeviceSingleCardFixture {};

class BlackholeSingleCardFixture : public DeviceSingleCardFixture {
protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        this->create_devices();
    }
};

class DeviceSingleCardFastSlowDispatchFixture : public DeviceSingleCardFixture {
   protected:
    void validate_dispatch_mode() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            this->slow_dispatch_ = false;
        }
    }
};
