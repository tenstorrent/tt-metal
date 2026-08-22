// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal shim for api/debug/assert.h (no Tensix watcher/error_handling).
// ASSERT() → ebreak.

#pragma once

#define ASSERT(condition, ...) (void(not(condition) ? ({ asm("ebreak"); }), 0 : 0))
#define ASSERT_ENABLED 1

#ifndef WAYPOINT
#define WAYPOINT(x)
#endif
