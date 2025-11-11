// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_fabric::test {

// Import ChipId from distributed namespace
using ChipId = tt::tt_metal::distributed::ChipId;

// API variants for addrgen overload testing
enum class AddrgenApiVariant {
    UnicastWrite,                  // fabric_unicast_noc_unicast_write
    UnicastWriteWithState,         // fabric_unicast_noc_unicast_write_with_state
    UnicastWriteSetState,          // fabric_unicast_noc_unicast_write_set_state + _with_state
    FusedAtomicIncWrite,           // fabric_unicast_noc_fused_unicast_with_atomic_inc
    FusedAtomicIncWriteWithState,  // fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state
    FusedAtomicIncWriteSetState,   // fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state + _with_state
    MulticastWrite                 // fabric_multicast_noc_unicast_write (2x2 mesh multicast)
};

// Test parameters for addrgen write correctness tests
struct AddrgenTestParams {
    uint32_t mesh_id;
    ChipId src_chip;
    ChipId dst_chip;
    bool use_dram_dst;
    uint32_t tensor_bytes;
    uint32_t page_size;
    tt::tt_metal::CoreCoord sender_core;
    tt::tt_metal::CoreCoord receiver_core;
    AddrgenApiVariant api_variant;  // Which API variant to test
    uint32_t mesh_rows = 0;         // mesh rectangle rows for multicast (0 = use full mesh)
    uint32_t mesh_cols = 0;         // mesh rectangle cols for multicast (0 = use full mesh)
};

}  // namespace tt::tt_fabric::test
