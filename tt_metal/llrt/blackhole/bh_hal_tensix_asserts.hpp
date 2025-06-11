// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides static asserts. Consumers really do want it even if they don't use any symbol from it.
// IWYU pragma: always_keep

#include "dev_mem_map.h"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"

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
