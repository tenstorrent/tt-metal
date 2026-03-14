// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/runner_common.hpp"

namespace tt::tt_fabric::test {

using ChipId = tt::tt_metal::distributed::ChipId;

// ---- Reusable defaults for addrgen tests ----
inline constexpr uint32_t kDefaultMeshId = 0;
inline constexpr ChipId kDefaultSrcChip = 0;
inline constexpr ChipId kDefaultDstChip = 1;
inline constexpr bool kDefaultUseDramDst = false;
inline constexpr uint32_t kDefaultTensorBytes = 1u << 20;
inline constexpr uint32_t kDefaultPageSize = 4096;
inline constexpr tt::tt_metal::CoreCoord kDefaultCore = {0, 0};
inline constexpr uint32_t kDefaultMeshRows = 0;
inline constexpr uint32_t kDefaultMeshCols = 0;

// Test parameters for addrgen write correctness tests.
// Uses FabricTestVariant (CastMode × WriteOp × StateMode × ConnectionMode)
// instead of a flat enum.
struct AddrgenTestParams {
    uint32_t mesh_id = kDefaultMeshId;
    ChipId src_chip = kDefaultSrcChip;
    ChipId dst_chip = kDefaultDstChip;
    bool use_dram_dst = kDefaultUseDramDst;
    uint32_t tensor_bytes = kDefaultTensorBytes;
    uint32_t page_size = kDefaultPageSize;
    tt::tt_metal::CoreCoord sender_core = kDefaultCore;
    tt::tt_metal::CoreCoord receiver_core = kDefaultCore;
    FabricTestVariant variant = {CastMode::Unicast, WriteOp::Write};
    uint32_t mesh_rows = kDefaultMeshRows;
    uint32_t mesh_cols = kDefaultMeshCols;
};

}  // namespace tt::tt_fabric::test
