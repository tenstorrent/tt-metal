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

}  // namespace tt::tt_fabric::fabric_router_tests
