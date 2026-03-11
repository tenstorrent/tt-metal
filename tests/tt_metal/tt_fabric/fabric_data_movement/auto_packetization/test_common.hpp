// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

// API variants for raw-size (non-addrgen) auto-packetization testing.
// These correspond to the raw-size write APIs that accept explicit `size` +
// `NocUnicastCommandHeader` parameters rather than addrgen-based address generation.
enum class RawApiVariant {
    UnicastWrite,              // fabric_unicast_noc_unicast_write (mesh namespace, raw-size)
    UnicastWriteConnMgr,       // fabric_unicast_noc_unicast_write (connection manager variant)
    MulticastWrite,            // fabric_multicast_noc_unicast_write (mesh namespace, raw-size)
    MulticastWriteConnMgr,     // fabric_multicast_noc_unicast_write (connection manager variant)
    LinearUnicastWrite,        // linear namespace raw-size unicast
    LinearMulticastWrite,      // linear namespace raw-size multicast
    LinearSparseMulticastWrite // linear namespace sparse multicast
};

// ---- Payload size constants for raw-size packetization tests ----
// NOTE: kChunkingPayloadBytes cannot be a constexpr because FABRIC_MAX_PACKET_SIZE
// is a runtime/device-side macro. Instead, tests compute the payload size at runtime as:
//   payload_bytes = kPayloadChunkCount * FABRIC_MAX_PACKET_SIZE + kPayloadRemainderBytes
inline constexpr uint32_t kPayloadChunkCount = 2;          // test sends 2 full chunks + remainder
inline constexpr uint32_t kPayloadRemainderBytes = 512;    // remainder after full chunks
inline constexpr uint32_t kSmallPayloadBytes = 2048;       // fits in single packet (< MAX)
inline constexpr uint32_t kChunkingMultiplier = 2;          // chunks = 2x MAX + remainder

// ---- Reusable defaults for raw-size packetization tests ----
inline constexpr uint32_t kDefaultMeshId = 0;
inline constexpr ChipId kDefaultSrcChip = 0;
inline constexpr ChipId kDefaultDstChip = 1;
inline constexpr ChipId kDefaultDst2Chip = 5;  // Second destination for connection manager variants
inline constexpr bool kDefaultUseDramDst = false;
inline constexpr uint32_t kDefaultTensorBytes = 1u << 20;  // 1 MiB
inline constexpr uint32_t kDefaultPageSize = 4096;          // 4 KiB
inline constexpr tt::tt_metal::CoreCoord kDefaultCore = {0, 0};
inline constexpr RawApiVariant kDefaultApiVariant = RawApiVariant::UnicastWrite;
inline constexpr uint32_t kDefaultMeshRows = 0;
inline constexpr uint32_t kDefaultMeshCols = 0;

// Test parameters for raw-size auto-packetization correctness tests.
// Mirrors the structure of AddrgenTestParams but uses RawApiVariant.
struct RawTestParams {
    uint32_t mesh_id = kDefaultMeshId;
    ChipId src_chip = kDefaultSrcChip;
    ChipId dst_chip = kDefaultDstChip;
    ChipId dst2_chip = kDefaultDst2Chip;  // Second destination for connection manager variants
    bool use_dram_dst = kDefaultUseDramDst;
    uint32_t tensor_bytes = kDefaultTensorBytes;
    uint32_t page_size = kDefaultPageSize;
    tt::tt_metal::CoreCoord sender_core = kDefaultCore;
    tt::tt_metal::CoreCoord receiver_core = kDefaultCore;
    RawApiVariant api_variant = kDefaultApiVariant;  // Which API variant to test
    uint32_t mesh_rows = kDefaultMeshRows;  // For multicast: receiver mesh dimensions
    uint32_t mesh_cols = kDefaultMeshCols;
};

}  // namespace tt::tt_fabric::test
