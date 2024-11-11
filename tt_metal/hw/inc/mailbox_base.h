// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef _MAILBOX_BASE_H_
#define _MAILBOX_BASE_H_

#include <cstdint>

#include "eth_l1_address_map.h" // ERISC_MEM_MAILBOX_BASE
#include "dev_mem_map.h" // MEM_MAILBOX_BASE, MEM_IERISC_MAILBOX_BASE

constexpr inline std::uint32_t get_mailbox_base() {

#ifdef COMPILE_FOR_ERISC
    return eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
#elif COMPILE_FOR_IDLE_ERISC
    return MEM_IERISC_MAILBOX_BASE;
#else
    return MEM_MAILBOX_BASE;
#endif
}

#endif
