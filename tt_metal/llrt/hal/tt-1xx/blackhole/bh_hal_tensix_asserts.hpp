// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
static_assert((MEM_MAILBOX_BASE + offsetof(mailboxes_t, go_message_index)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
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
