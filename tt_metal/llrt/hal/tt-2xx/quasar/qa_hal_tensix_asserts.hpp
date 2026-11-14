// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides static asserts. Consumers really do want it even if they don't use any symbol from it.
// IWYU pragma: always_keep

#include "dev_mem_map.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"
#include "hostdevcommon/fabric_common.h"

// Validate assumptions on mailbox layout on host compile
// Constexpr definitions allow for printing of breaking values at compile time
static_assert(MEM_MAILBOX_BASE + sizeof(mailboxes_t) <= MEM_MAILBOX_END);
static constexpr uint32_t TENSIX_LAUNCH_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t TENSIX_PROFILER_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert(TENSIX_LAUNCH_CHECK == 0);
static_assert(TENSIX_PROFILER_CHECK == 0);
static_assert(sizeof(launch_msg_t) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
static_assert(offsetof(subordinate_map_t, dm1) == 0);

static_assert(sizeof(tt::tt_fabric::routing_l1_info_t) == MEM_ROUTING_TABLE_SIZE, "Struct size mismatch!");
static_assert(
    sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t) == MEM_TENSIX_FABRIC_CONNECTIONS_SIZE,
    "Struct size mismatch!");
static_assert(
    offsetof(tt::tt_fabric::tensix_fabric_connections_l1_info_t, read_write) ==
        MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO,
    "Read-write connections offset must be 432 bytes!");
static_assert(
    sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t) % 16 == 0, "Struct size must be 16-byte aligned!");
static_assert(MEM_TENSIX_ROUTING_TABLE_BASE % 16 == 0, "Tensix routing table base must be 16-byte aligned");
static_assert(MEM_ROUTING_TABLE_SIZE % 16 == 0, "Tensix routing table size must be 16-byte aligned");
static_assert(MEM_TENSIX_FABRIC_CONNECTIONS_BASE % 16 == 0, "Tensix fabric connections base must be 16-byte aligned");
static_assert(MEM_TENSIX_FABRIC_CONNECTIONS_SIZE % 16 == 0, "Tensix fabric connections size must be 16-byte aligned");
static_assert(
    MEM_TENSIX_FABRIC_CONNECTIONS_BASE - MEM_TENSIX_ROUTING_TABLE_BASE == sizeof(tt::tt_fabric::routing_l1_info_t));

// ---------------------------------------------------------------------------------------------------
// Golden cross-check for the single-definition L1 layout (plan B2 / D end state). Deleting the old
// relational #define chain in dev_mem_map.h left tt_l1_compute_layout()/kL1Reservations as the sole
// source of the region bases. These static_asserts pin kIdentityL1Layout to the historical static
// chain, anchored on MEM_DM_LOCAL_BASE and the firmware/global bases -- which are NOT movable macros
// (never rewritten to the constexpr accessor), so this is an INDEPENDENT spec, not a tautology. Any
// drift in the layout routine or the reservations becomes a compile error here rather than a silent
// host/device memory-map mismatch. If the map is changed intentionally, update these golds.
namespace {
inline constexpr uint32_t kGoldNocCounter = MEM_DM_LOCAL_BASE + MEM_DM_LOCAL_SIZE * NUM_DM_CORES;
inline constexpr uint32_t kGoldFabricCounter = kGoldNocCounter + MEM_NOC_COUNTER_L1_SIZE;
inline constexpr uint32_t kGoldFabricConnLock = kGoldFabricCounter + MEM_FABRIC_COUNTER_L1_SIZE;
inline constexpr uint32_t kGoldRoutingTable = kGoldFabricConnLock + MEM_FABRIC_CONNECTION_LOCK_SIZE;
inline constexpr uint32_t kGoldRoutingPath = kGoldRoutingTable + MEM_OFFSET_OF_ROUTING_PATHS;
inline constexpr uint32_t kGoldExitNode = kGoldRoutingPath + MEM_TENSIX_ROUTING_PATH_SIZE;
inline constexpr uint32_t kGoldFabricConns = kGoldExitNode + MEM_EXIT_NODE_TABLE_SIZE + MEM_ROUTING_TABLE_PADDING;
inline constexpr uint32_t kGoldPacketPool = kGoldFabricConns + MEM_TENSIX_FABRIC_CONNECTIONS_SIZE;
inline constexpr uint32_t kGoldMapEnd = kGoldPacketPool + MEM_PACKET_HEADER_POOL_SIZE;
inline constexpr uint32_t kGoldBankToNoc = kGoldMapEnd + MEM_DM_LOCAL_SIZE + NUM_TRISC_CORES * MEM_TRISC_LOCAL_SIZE;
inline constexpr uint32_t kGoldLogicalToVirtual = kGoldBankToNoc + MEM_BANK_TO_NOC_SIZE;
inline constexpr uint32_t kGoldR(L1Region r) {
    return static_cast<uint32_t>(kIdentityL1Layout.region_base[static_cast<std::size_t>(r)]);
}
}  // namespace
// Non-movable regions: firmware/global/local bases (static relational chain, unchanged).
static_assert(kGoldR(L1Region::DM_GLOBAL) == MEM_DM_GLOBAL_BASE);
static_assert(kGoldR(L1Region::TRISC0_GLOBAL) == MEM_TRISC0_GLOBAL_BASE);
static_assert(kGoldR(L1Region::TRISC1_GLOBAL) == MEM_TRISC1_GLOBAL_BASE);
static_assert(kGoldR(L1Region::TRISC2_GLOBAL) == MEM_TRISC2_GLOBAL_BASE);
static_assert(kGoldR(L1Region::TRISC3_GLOBAL) == MEM_TRISC3_GLOBAL_BASE);
static_assert(kGoldR(L1Region::DM_LOCAL) == MEM_DM_LOCAL_BASE);
static_assert(kGoldR(L1Region::TRISC0_FW) == MEM_TRISC0_FIRMWARE_BASE);
static_assert(kGoldR(L1Region::TRISC1_FW) == MEM_TRISC1_FIRMWARE_BASE);
static_assert(kGoldR(L1Region::TRISC2_FW) == MEM_TRISC2_FIRMWARE_BASE);
static_assert(kGoldR(L1Region::TRISC3_FW) == MEM_TRISC3_FIRMWARE_BASE);
// Movable regions: the reclaimable chain (must reproduce today's static addresses at the identity seed).
static_assert(kGoldR(L1Region::NOC_COUNTER) == kGoldNocCounter);
static_assert(kGoldR(L1Region::FABRIC_COUNTER) == kGoldFabricCounter);
static_assert(kGoldR(L1Region::FABRIC_CONNECTION_LOCK) == kGoldFabricConnLock);
static_assert(kGoldR(L1Region::ROUTING_TABLE) == kGoldRoutingTable);
static_assert(kGoldR(L1Region::ROUTING_PATH) == kGoldRoutingPath);
static_assert(kGoldR(L1Region::EXIT_NODE_TABLE) == kGoldExitNode);
static_assert(kGoldR(L1Region::FABRIC_CONNECTIONS) == kGoldFabricConns);
static_assert(kGoldR(L1Region::PACKET_HEADER_POOL) == kGoldPacketPool);
static_assert(kGoldR(L1Region::MAP_END) == kGoldMapEnd);
static_assert(kGoldR(L1Region::INIT_LOCAL_SCRATCH) == kGoldMapEnd);
static_assert(kGoldR(L1Region::BANK_TO_NOC_SCRATCH) == kGoldBankToNoc);
static_assert(kGoldR(L1Region::LOGICAL_TO_VIRTUAL_SCRATCH) == kGoldLogicalToVirtual);
static_assert(static_cast<uint32_t>(kIdentityL1Layout.map_end) == kGoldMapEnd);
// Anchor MEM_MAP_END (== the historical static value) so the reclaimed worker-L1 boundary is pinned.
static_assert(MEM_MAP_END == kGoldMapEnd);
