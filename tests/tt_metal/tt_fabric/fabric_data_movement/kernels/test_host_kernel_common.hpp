// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common parameters that are used in both the testbenches and the kernels that they call.

#pragma once

namespace tt::tt_fabric::fabric_router_tests {

// Enum used to specify the fabric packet type used in the test
enum FabricPacketType {
    CHIP_UNICAST = 0,
    CHIP_MULTICAST = 1,
    CHIP_SPARSE_MULTICAST = 2,
    COUNT = CHIP_SPARSE_MULTICAST + 1
};

// Enum used to specify the NoC packet type used in the test
enum TestNocPacketType {
    TEST_NOC_UNICAST_WRITE = 0,
    TEST_NOC_UNICAST_INLINE_WRITE = 1,
    TEST_NOC_UNICAST_ATOMIC_INC = 2,
    TEST_NOC_FUSED_UNICAST_ATOMIC_INC = 3,
    TEST_NOC_UNICAST_SCATTER_WRITE = 4,
    TEST_NOC_MULTICAST_WRITE = 5,       // mcast has bug
    TEST_NOC_MULTICAST_ATOMIC_INC = 6,  // mcast has bug
    TEST_NOC_UNICAST_READ = 7,
    TEST_NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC = 8,
    TEST_COUNT = TEST_NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC + 1
};

}  // namespace tt::tt_fabric::fabric_router_tests
