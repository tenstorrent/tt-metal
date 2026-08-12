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

// Both derivations of the layout must land on the same addresses, or firmware would run at addresses the host never
// programmed.
static_assert(MEM_INTERRUPT_TABLE_BASE == MEM_DISPATCH_INTERRUPT_TABLE_BASE);
static_assert(MEM_DM_FIRMWARE_BASE == MEM_DISPATCH_DM_FIRMWARE_BASE);
static_assert(MEM_DM_GLOBAL_BASE == MEM_DISPATCH_DM_GLOBAL_BASE);
static_assert(MEM_DM_LOCAL_BASE == MEM_DISPATCH_DM_LOCAL_BASE);
static_assert(MEM_MAP_END == MEM_DISPATCH_MAP_END);
static_assert(MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH == MEM_DISPATCH_DM0_INIT_LOCAL_L1_BASE_SCRATCH);
static_assert(MEM_BANK_TO_NOC_SCRATCH == MEM_DISPATCH_BANK_TO_NOC_SCRATCH);
static_assert(MEM_LOGICAL_TO_VIRTUAL_SCRATCH == MEM_DISPATCH_LOGICAL_TO_VIRTUAL_SCRATCH);
