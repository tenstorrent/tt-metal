// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal port shim for tt_metal/hw/inc/api/debug/assert.h
//
// The in-tree header pulls in internal/debug/watcher_common.h,
// internal/hw_thread.h and internal/tt-2xx/quasar/error_handling.h -- all of
// which reach for Tensix hardware state (watcher mailboxes, hart identity,
// device error handling) that does not exist on a plain RISC-V core.
//
// ASSERT() traps with `ebreak`, mirroring the device build's non-watcher path.

#pragma once

#define ASSERT(condition, ...) (void(not(condition) ? ({ asm("ebreak"); }), 0 : 0))
#define ASSERT_ENABLED 1

#ifndef WAYPOINT
#define WAYPOINT(x)
#endif
