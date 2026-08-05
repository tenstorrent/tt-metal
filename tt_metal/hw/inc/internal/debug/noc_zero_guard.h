// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// WATCHER guard for the device-zero "zero -> barrier -> reuse" contract.
//
// On Quasar the L1 zero (Noc::async_write_zeros) borrows the overlay write command buffer and only
// restores it in Noc::write_zeros_l1_barrier(); issuing any NoC write in between silently writes
// zeros instead of the intended data. This per-RISC flag is set by the L1 zero and cleared by its
// barrier, and Noc::async_write asserts it is clear.

#include "api/debug/assert.h"

#if defined(WATCHER_ENABLED)
namespace noc_zero_detail {
[[maybe_unused]] inline bool g_zero_mode_active = false;
}  // namespace noc_zero_detail
#define NOC_ZERO_MODE_ENTER() (::noc_zero_detail::g_zero_mode_active = true)
#define NOC_ZERO_MODE_EXIT() (::noc_zero_detail::g_zero_mode_active = false)
#define NOC_ASSERT_NOT_ZERO_MODE() ASSERT(!::noc_zero_detail::g_zero_mode_active)
#else
#define NOC_ZERO_MODE_ENTER()
#define NOC_ZERO_MODE_EXIT()
#define NOC_ASSERT_NOT_ZERO_MODE()
#endif
