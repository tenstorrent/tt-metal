#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"

namespace unit_tests {
class SingleDeviceFixture {
   public:
    SingleDeviceFixture() {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int pci_express_slot = 0;

        if (arch_ != tt::ARCH::GRAYSKULL) {
            // Once this test is uplifted to use fast dispatch, this can be removed.
            char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
            putenv(env);
        }
        device_ = tt::tt_metal::CreateDevice(arch_, pci_express_slot);
        tt::tt_metal::InitializeDevice(device_);
    }

    ~SingleDeviceFixture() { tt::tt_metal::CloseDevice(device_); }

   protected:
    tt::tt_metal::Device* device_;
    tt::ARCH arch_;
};
}  // namespace unit_tests
