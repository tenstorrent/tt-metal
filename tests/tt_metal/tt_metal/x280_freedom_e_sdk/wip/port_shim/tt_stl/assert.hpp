// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal shim for tt_stl/assert.hpp (host-only fmt/backtrace/exceptions).
// Same macros: print then trap.

#pragma once

#include <cstdio>
#include <cstdlib>

namespace tt::assert::baremetal {

[[noreturn]] inline void fail(const char* file, int line, const char* kind, const char* msg) {
    printf("\n%s at %s:%d: %s\n", kind, file, line, msg);
    fflush(stdout);
    // No exceptions and nothing to unwind to; halt so the failure is visible.
    abort();
}

}  // namespace tt::assert::baremetal

#ifdef DEBUG
#define TT_ASSERT(condition, ...)                                                       \
    do {                                                                                \
        if (!(condition)) {                                                             \
            ::tt::assert::baremetal::fail(__FILE__, __LINE__, "TT_ASSERT", #condition); \
        }                                                                               \
    } while (0)
#else
#define TT_ASSERT(condition, ...) ((void)0)
#endif

#define TT_FATAL(condition, ...)                                                       \
    do {                                                                               \
        if (!(condition)) {                                                            \
            ::tt::assert::baremetal::fail(__FILE__, __LINE__, "TT_FATAL", #condition); \
        }                                                                              \
    } while (0)

#define TT_THROW(...) ::tt::assert::baremetal::fail(__FILE__, __LINE__, "TT_THROW", "unconditional throw")
