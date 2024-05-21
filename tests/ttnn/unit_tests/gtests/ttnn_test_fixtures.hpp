#include <math.h>

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

namespace ttnn {

class TTNNFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    size_t num_devices_;

    void SetUp() override {
        std::srand(0);
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    }

    void TearDown() override { tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false); }
};
}  // namespace ttnn
