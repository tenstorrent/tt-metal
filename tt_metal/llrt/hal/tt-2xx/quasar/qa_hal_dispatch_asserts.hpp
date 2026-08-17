// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides static asserts. Consumers really do want it even if they don't use any symbol from it.
// IWYU pragma: always_keep

#include "dev_mem_map.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"

static_assert(DISPATCH_MEM_MAP_END <= MEM_L1_SIZE, "Dispatch-engine L1 layout exceeds MEM_L1_SIZE");
static_assert(
    MEM_DISPATCH_DM0_KERNEL_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0,
    "Dispatch DM0 kernel base must be NOC-write aligned");

static_assert(MEM_MAILBOX_BASE + sizeof(mailboxes_t) <= MEM_DISPATCH_MAILBOX_END);
static constexpr uint32_t DISPATCH_LAUNCH_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t DISPATCH_PROFILER_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert(DISPATCH_LAUNCH_CHECK == 0);
static_assert(DISPATCH_PROFILER_CHECK == 0);

// Each region begins where the previous one ends. Deliberately written against the generic MEM_* names, which
// dev_mem_map.h redirects to the dispatch layout under COMPILE_FOR_DISPATCH_ENGINE -- a redirect that is missing
// leaves a Tensix-derived address in the chain and breaks one of these links.
static_assert(MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE <= MEM_INTERRUPT_TABLE_BASE);
static_assert(MEM_INTERRUPT_TABLE_BASE + MEM_INTERRUPT_TABLE_SIZE == MEM_DM_FIRMWARE_BASE);
static_assert(MEM_DM_FIRMWARE_BASE + MEM_DM_FIRMWARE_SIZE == MEM_DM_GLOBAL_BASE);
static_assert(MEM_DM_GLOBAL_BASE + MEM_DM_GLOBAL_SIZE * (NUM_DM_CORES + 1) == MEM_DM_LOCAL_BASE);
static_assert(MEM_DM_LOCAL_BASE + MEM_DM_LOCAL_SIZE * NUM_DM_CORES == MEM_NOC_COUNTER_BASE);
static_assert(MEM_NOC_COUNTER_BASE + MEM_NOC_COUNTER_L1_SIZE == MEM_FABRIC_COUNTER_BASE);
static_assert(MEM_FABRIC_COUNTER_BASE + MEM_FABRIC_COUNTER_L1_SIZE == MEM_FABRIC_CONNECTION_LOCK_BASE);
static_assert(MEM_FABRIC_CONNECTION_LOCK_BASE + MEM_FABRIC_CONNECTION_LOCK_SIZE == MEM_TENSIX_ROUTING_TABLE_BASE);
static_assert(MEM_TENSIX_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS == MEM_TENSIX_ROUTING_PATH_BASE);
static_assert(MEM_TENSIX_ROUTING_PATH_BASE + MEM_TENSIX_ROUTING_PATH_SIZE == MEM_TENSIX_EXIT_NODE_TABLE_BASE);
static_assert(
    MEM_TENSIX_EXIT_NODE_TABLE_BASE + MEM_EXIT_NODE_TABLE_SIZE + MEM_ROUTING_TABLE_PADDING ==
    MEM_TENSIX_FABRIC_CONNECTIONS_BASE);
static_assert(MEM_TENSIX_FABRIC_CONNECTIONS_BASE + MEM_TENSIX_FABRIC_CONNECTIONS_SIZE == MEM_PACKET_HEADER_POOL_BASE);
static_assert(MEM_PACKET_HEADER_POOL_BASE + MEM_PACKET_HEADER_POOL_SIZE == MEM_MAP_END);
static_assert(MEM_PACKET_HEADER_POOL_BASE % 16 == 0, "Dispatch packet header pool base must be 16-byte aligned");

// Init-local staging and the kernel config ring share the scratch range above MEM_MAP_END; both must stay clear of
// the kernel text that follows.
static_assert(MEM_MAP_END == MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH);
static_assert(MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH + MEM_DM_LOCAL_SIZE == MEM_BANK_TO_NOC_SCRATCH);
static_assert(MEM_BANK_TO_NOC_SCRATCH + MEM_BANK_TO_NOC_SIZE == MEM_LOGICAL_TO_VIRTUAL_SCRATCH);
static_assert(MEM_LOGICAL_TO_VIRTUAL_SCRATCH + MEM_LOGICAL_TO_VIRTUAL_SIZE <= MEM_DISPATCH_DM0_KERNEL_BASE);
static_assert(MEM_MAP_END + MEM_KERNEL_CONFIG_SIZE <= MEM_DISPATCH_DM0_KERNEL_BASE);
