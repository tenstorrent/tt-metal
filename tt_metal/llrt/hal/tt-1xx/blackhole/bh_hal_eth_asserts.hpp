// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides static asserts. Consumers really do want it even if they don't use any symbol from it.
// IWYU pragma: always_keep

#include "dev_mem_map.h"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"
#include "eth_l1_address_map.h"
#include "tests/tt_metal/tt_metal/tunneling/lite_fabric_memory_defs.h"

// Validate assumptions on mailbox layout on host compile
// Constexpr definitions allow for printing of breaking values at compile time
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
static_assert(MEM_AERISC_FABRIC_LITE_BARRIER == MEM_IERISC_FABRIC_LITE_BARRIER);
static_assert(MEM_AERISC_FABRIC_LITE_BARRIER == MEM_ERISC_FABRIC_LITE_BARRIER);
static_assert(MEM_ERISC_LITE_FABRIC_RESERVED_BASE == MEM_LITE_FABRIC_MEMORY_BASE);
static_assert(MEM_ERISC_LITE_FABRIC_RESERVED_SIZE == MEM_LITE_FABRIC_MEMORY_SIZE);
static_assert(MEM_NOC_ATOMIC_RET_VAL_ADDR == MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR);
static_assert(MEMORY_LAYOUT_END < MEM_LITE_FABRIC_MEMORY_END);
