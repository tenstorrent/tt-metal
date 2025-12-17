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
enum class AddrgenApiVariant : std::uint8_t {
    UnicastWrite,                           // fabric_unicast_noc_unicast_write
    UnicastWriteWithState,                  // fabric_unicast_noc_unicast_write_with_state
    UnicastWriteSetState,                   // fabric_unicast_noc_unicast_write_set_state + _with_state
    FusedAtomicIncWrite,                    // fabric_unicast_noc_fused_unicast_with_atomic_inc
    FusedAtomicIncWriteWithState,           // fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state
    FusedAtomicIncWriteSetState,            // fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state + _with_state
    MulticastWrite,                         // fabric_multicast_noc_unicast_write
    MulticastWriteWithState,                // fabric_multicast_noc_unicast_write_with_state
    MulticastWriteSetState,                 // fabric_multicast_noc_unicast_write_set_state + _with_state
    MulticastScatterWrite,                  // fabric_multicast_noc_scatter_write
    MulticastScatterWriteWithState,         // fabric_multicast_noc_scatter_write_with_state
    MulticastScatterWriteSetState,          // fabric_multicast_noc_scatter_write_set_state + _with_state
    MulticastFusedAtomicIncWrite,           // fabric_multicast_noc_fused_unicast_with_atomic_inc
    MulticastFusedAtomicIncWriteWithState,  // fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state
    MulticastFusedAtomicIncWriteSetState,  // fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state + _with_state
    ScatterWrite,                          // fabric_unicast_noc_scatter_write
    ScatterWriteWithState,                 // fabric_unicast_noc_scatter_write_with_state
    ScatterWriteSetState                   // fabric_unicast_noc_scatter_write_set_state + _with_state
};

// ---- Reusable defaults for addrgen tests ----
inline constexpr uint32_t kDefaultMeshId = 0;
inline constexpr ChipId kDefaultSrcChip = 0;
inline constexpr ChipId kDefaultDstChip = 1;
inline constexpr bool kDefaultUseDramDst = false;
inline constexpr uint32_t kDefaultTensorBytes = 1u << 20;  // 1 MiB
inline constexpr uint32_t kDefaultPageSize = 4096;         // 4 KiB
inline constexpr tt::tt_metal::CoreCoord kDefaultCore = {0, 0};
inline constexpr AddrgenApiVariant kDefaultApiVariant = AddrgenApiVariant::UnicastWrite;
inline constexpr uint32_t kDefaultMeshRows = 0;
inline constexpr uint32_t kDefaultMeshCols = 0;

// Test parameters for addrgen write correctness tests
struct AddrgenTestParams {
    uint32_t mesh_id = kDefaultMeshId;
    ChipId src_chip = kDefaultSrcChip;
    ChipId dst_chip = kDefaultDstChip;
    bool use_dram_dst = kDefaultUseDramDst;
    uint32_t tensor_bytes = kDefaultTensorBytes;
    uint32_t page_size = kDefaultPageSize;
    tt::tt_metal::CoreCoord sender_core = kDefaultCore;
    tt::tt_metal::CoreCoord receiver_core = kDefaultCore;
    AddrgenApiVariant api_variant = kDefaultApiVariant;  // Which API variant to test
    uint32_t mesh_rows = kDefaultMeshRows;  // For multicast: receiver mesh dimensions
    uint32_t mesh_cols = kDefaultMeshCols;
};

}  // namespace tt::tt_fabric::test
