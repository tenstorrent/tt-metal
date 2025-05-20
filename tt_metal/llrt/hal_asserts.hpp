// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"

// Validate assumptions on mailbox layout on host compile
// Constexpr definitions allow for printing of breaking values at compile time
#ifdef NCRISC_HAS_IRAM
// These are only used in ncrisc-halt.S
static_assert(MEM_MAILBOX_BASE + offsetof(mailboxes_t, subordinate_sync.dm1) == MEM_SUBORDINATE_RUN_MAILBOX_ADDRESS);
static_assert(
    MEM_MAILBOX_BASE + offsetof(mailboxes_t, ncrisc_halt.stack_save) == MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS);
#endif
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
#include "eth_l1_address_map.h"
static_assert(
    eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE + sizeof(mailboxes_t) <=
    eth_l1_mem::address_map::ERISC_MEM_MAILBOX_END);
static_assert(MEM_IERISC_MAILBOX_BASE + sizeof(mailboxes_t) <= MEM_IERISC_MAILBOX_END);
static constexpr uint32_t ETH_LAUNCH_CHECK =
    (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t ETH_PROFILER_CHECK =
    (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) %
    TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert(ETH_LAUNCH_CHECK == 0);
static_assert(ETH_PROFILER_CHECK == 0);
static_assert(MEM_IERISC_FIRMWARE_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
static_assert(MEM_IERISC_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_IERISC_MAILBOX_END);
#else
static_assert(MEM_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_MAILBOX_END);
static constexpr uint32_t TENSIX_LAUNCH_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t TENSIX_PROFILER_CHECK =
    (MEM_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert(TENSIX_LAUNCH_CHECK == 0);
static_assert(TENSIX_PROFILER_CHECK == 0);
static_assert(sizeof(launch_msg_t) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
#endif
