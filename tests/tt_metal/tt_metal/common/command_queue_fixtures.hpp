// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

class CommandQueueSingleCardFixture : virtual public DispatchFixture {
   protected:
    void SetUp() override {
        this->validate_dispatch_mode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        this->create_devices();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    void validate_dispatch_mode() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
    }

    void create_devices(const std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        const chip_id_t mmio_device_id = 0;
        this->reserved_devices_ = tt::tt_metal::detail::CreateDevices(
            {mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_type);
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        if (enable_remote_chip) {
            for (const auto &[id, device] : this->reserved_devices_) {
                this->devices_.push_back(device);
            }
        } else {
            this->devices_.push_back(this->reserved_devices_.at(mmio_device_id));
        }

        this->num_devices_ = this->reserved_devices_.size();
    }

    std::vector<tt::tt_metal::Device *> devices_;
    std::map<chip_id_t, tt::tt_metal::Device *> reserved_devices_;
    size_t num_devices_;
};
