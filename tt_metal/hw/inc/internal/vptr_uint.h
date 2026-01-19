// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef MODELT
#include "modelt_accessor.h"
using vptr_uint = struct modelt_accessor_proxy;
using vptr_pc_buf = vptr_uint;
using vptr_mailbox = vptr_uint;
#elif defined(TENSIX_FIRMWARE)
#include <cstdint>
using vptr_uint = volatile uint32_t*;
using vptr_pc_buf = vptr_uint;
using vptr_mailbox = vptr_uint;
#else
#include <cstdint>
#include "emule_hw_proxy_pointer.h"
using vptr_uint = volatile uint32_t*;
using vptr_pc_buf = EmuleHardwareProxyPointer;
using vptr_mailbox = EmuleHardwareProxyPointer;
#endif
