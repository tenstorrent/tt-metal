#pragma once

#include <gtest/gtest.h>
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"

inline
bool is_multi_device_gs_machine(const tt::ARCH& arch, const size_t num_devices) {
    return arch == tt::ARCH::GRAYSKULL && num_devices > 1;
}
class MultiDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        if (arch_ != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        num_devices_ = tt::tt_metal::Device::detect_num_available_devices(); // FIXME: Is this too greedy?

        if (is_multi_device_gs_machine(arch_, num_devices_)) {
            GTEST_SKIP();
        }
        for (unsigned int id = 0; id < num_devices_; id++) {
            devices_.push_back(tt::tt_metal::CreateDevice(arch_, id));
            tt::tt_metal::InitializeDevice(devices_.at(id));
        }
    }

    void TearDown() override {
        if (not is_multi_device_gs_machine(arch_, num_devices_)) {
            for (unsigned int id = 0; id < num_devices_; id++) {
                tt::tt_metal::CloseDevice(devices_.at(id));
            }
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};
