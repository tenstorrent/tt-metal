// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "host_api.hpp"
#include "dispatch_fixture.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/llrt/llrt.hpp"

class MultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override { this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()); }
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
