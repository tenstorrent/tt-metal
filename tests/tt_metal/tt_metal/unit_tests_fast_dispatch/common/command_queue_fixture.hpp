// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

class CommandQueueFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    tt::tt_metal::Device* device_;
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const int device_id = 0;

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        this->device_ = tt::tt_metal::CreateDevice(device_id, {.dispatch_core_type = dispatch_core_type});
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
            tt::tt_metal::CloseDevice(this->device_);
        }
    }
};


class CommandQueueMultiDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 2 ) {
            GTEST_SKIP();
        }
        std::vector<chip_id_t> chip_ids;
        for (unsigned int id = 0; id < num_devices_; id++) {
            chip_ids.push_back(id);
        }

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        reserved_devices_ = tt::tt_metal::detail::CreateDevices(chip_ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        for (const auto &[id, device] : reserved_devices_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::Device*> devices_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class CommandQueueSingleCardFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
        const chip_id_t mmio_device_id = 0;
        reserved_devices_ = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
        if (enable_remote_chip) {
            for (const auto &[id, device] : reserved_devices_) {
                devices_.push_back(device);
            }
        } else {
            devices_.push_back(reserved_devices_.at(mmio_device_id));
        }

        num_devices_ = reserved_devices_.size();
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(reserved_devices_); }

    std::vector<tt::tt_metal::Device*> devices_;
    std::map<chip_id_t, tt::tt_metal::Device*> reserved_devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class SingleDeviceTraceFixture: public ::testing::Test {
protected:
    tt::tt_metal::Device* device_;
    tt::ARCH arch_;

    void Setup(const size_t buffer_size, const uint8_t num_hw_cqs = 1) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        if (num_hw_cqs > 1) {
            // Running multi-CQ test. User must set this explicitly.
            auto num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
            if (num_cqs == nullptr or strcmp(num_cqs, "2")) {
                TT_THROW("This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2");
                GTEST_SKIP();
            }
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id, {.trace_region_size = buffer_size});
    }

    void TearDown() override {
        if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

};
