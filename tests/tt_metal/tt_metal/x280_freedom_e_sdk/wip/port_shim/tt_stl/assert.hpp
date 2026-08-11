// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal port shim for tt_stl/tt_stl/assert.hpp
//
// fabric_edm_packet_header.hpp includes <tt_stl/assert.hpp> on its non-kernel
// (host) path. The real header is firmly host-only: <fmt/format.h>, <cxxabi.h>,
// <execinfo.h> (backtrace), <iostream>, <sstream>, <stdexcept> and tt-logger.
// None of that survives on a bare-metal target with no OS, no unwinder and
// -fno-exceptions.
//
// This shim provides the same three macros with bare-metal behaviour: report
// over whatever stdout the BSP wired up, then trap.

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

// TT_ASSERT is a debug-only check in the real header too (compiled out in
// release builds), so keep it cheap but honour the condition when DEBUG is set.
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
