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

static_assert(
    MEM_DISPATCH_PACKET_HEADER_POOL_BASE % 16 == 0, "Dispatch packet header pool base must be 16-byte aligned");

// Init-local staging and the kernel config ring share the scratch range above MEM_DISPATCH_MAP_END; both must stay
// clear of the kernel text that follows.
static_assert(MEM_DISPATCH_LOGICAL_TO_VIRTUAL_SCRATCH + MEM_LOGICAL_TO_VIRTUAL_SIZE <= MEM_DISPATCH_DM0_KERNEL_BASE);
static_assert(MEM_DISPATCH_MAP_END + MEM_DISPATCH_KERNEL_CONFIG_SIZE <= MEM_DISPATCH_DM0_KERNEL_BASE);
