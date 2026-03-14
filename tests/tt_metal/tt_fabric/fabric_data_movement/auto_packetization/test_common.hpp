// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight test header for auto-packetization silicon tests.
// Uses FabricTestVariant from runner_common.hpp for family/operation selection.

#pragma once

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <tt-metalium/core_coord.hpp>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/runner_common.hpp"

namespace tt::tt_fabric::test {

// Minimal test configuration for raw-size silicon tests.
struct RawTestParams {
    uint32_t mesh_id;
    ChipId src_chip;
    ChipId dst_chip;
    uint32_t tensor_bytes;
    CoreCoord sender_core;
    CoreCoord receiver_core;
    FabricTestVariant variant;
    bool use_dram_dst = false;
};

}  // namespace tt::tt_fabric::test
