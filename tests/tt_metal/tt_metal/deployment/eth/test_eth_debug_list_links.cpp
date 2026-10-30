// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

TEST_F(MeshDispatchFixture, TensixDeploymentDebugEthernet00ListLinks) {
    print_detected_devices();
    ASSERT_TRUE(ensure_links(devices_));

    for (const auto& mesh_device : devices_) {
        auto* const chip = mesh_device->get_devices()[0];

        for (const auto& core : chip->ethernet_cores()) {
            char t = eth_core_connects_within_cluster(chip, core) ? '=' : ' ';
            string macs = read_eth_mac_str(chip, core);
            log_info(
                tt::LogTest,
                "{} {} ({})\t{} {}",
                chip->id(),
                get_eth_unique_id(chip, core),
                get_eth_phys_id(chip, core),
                t,
                macs);
        }
    }
}

}  // namespace tt::tt_metal
