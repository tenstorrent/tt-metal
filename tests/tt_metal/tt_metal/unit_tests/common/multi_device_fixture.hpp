#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"

namespace unit_tests {
class MultiDeviceFixture {
   public:
    MultiDeviceFixture() {
        auto arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        if (arch_ != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        num_devices_ = tt::tt_metal::Device::detect_num_available_devices(); // FIXME: Is this too greedy?

        for (unsigned int id = 0; id < num_devices_; id++) {
            devices_.push_back(tt::tt_metal::CreateDevice(arch_, id));
            tt::tt_metal::InitializeDevice(devices_.at(id));
        }
    }

    ~MultiDeviceFixture() {
        for (unsigned int id = 0; id < num_devices_; id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

   protected:
    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};
}  // namespace unit_tests
