// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/slow_dispatch.cpp -- the x280_slow_dispatch object library.
//
// Sets TT_METAL_SLOW_DISPATCH_MODE=1 during static initialisation.
// tt-metal launches kernels on core-local RISCs (idle and active
// ERISC, DRISC) only under slow dispatch, and reads the variable once, when a
// device open first builds its runtime options, so the value must be in place
// before any Metal object exists. An executable opts in by linking this
// library in CMakeLists.txt; an object library keeps the initialiser from
// being dropped by the linker. kSlowDispatchLinked is the symbol
// harness/slow_dispatch.hpp's guard references, so a guarded executable
// without this library fails to link.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace x280::harness {

extern const bool kSlowDispatchLinked = [] {
    const char* previous = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (previous != nullptr && std::strcmp(previous, "1") != 0) {
        std::fprintf(
            stderr, "x280_slow_dispatch: overriding TT_METAL_SLOW_DISPATCH_MODE=\"%s\" with \"1\"\n", previous);
    }
    if (setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", /*overwrite=*/1) != 0) {
        std::fprintf(
            stderr, "x280_slow_dispatch: setenv(TT_METAL_SLOW_DISPATCH_MODE) failed: %s\n", std::strerror(errno));
        std::abort();
    }
    return true;
}();

}  // namespace x280::harness
