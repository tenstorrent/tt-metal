// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    FABRIC_PACKET_TYPE_COUNT = CHIP_SPARSE_MULTICAST + 1
};

// Enum used to specify the NoC packet type used in the test
enum NocPacketType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_UNICAST_INLINE_WRITE = 1,
    NOC_UNICAST_ATOMIC_INC = 2,
    NOC_FUSED_UNICAST_ATOMIC_INC = 3,
    NOC_UNICAST_SCATTER_WRITE = 4,
    NOC_MULTICAST_WRITE = 5,       // mcast has bug
    NOC_MULTICAST_ATOMIC_INC = 6,  // mcast has bug
    NOC_UNICAST_READ = 7,          // read wont be supported without UDM mode
    NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC = 8,
    NOC_PACKET_TYPE_COUNT = NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC + 1
};

}  // namespace tt::tt_fabric::fabric_router_tests
