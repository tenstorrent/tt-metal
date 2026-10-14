// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/slow_dispatch.hpp -- the guard for executables that link
// x280_slow_dispatch (harness/slow_dispatch.cpp).

#ifndef X280_HARNESS_SLOW_DISPATCH_HPP
#define X280_HARNESS_SLOW_DISPATCH_HPP

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace x280::harness {

/** Defined by x280_slow_dispatch; true once its initialiser has set the variable. */
extern const bool kSlowDispatchLinked;

/**
 * Checks that the executable links x280_slow_dispatch. A caller without the
 * library fails to link, because this references kSlowDispatchLinked. At run
 * time, success when TT_METAL_SLOW_DISPATCH_MODE is "1"; otherwise a failure
 * naming the value and the x280_slow_dispatch link that sets it. Assert it in
 * SetUp before any MeshDevice opens:
 * `ASSERT_TRUE(x280::harness::slow_dispatch_linked());`.
 */
inline ::testing::AssertionResult slow_dispatch_linked() {
    const char* mode = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (kSlowDispatchLinked && mode != nullptr && std::strcmp(mode, "1") == 0) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "TT_METAL_SLOW_DISPATCH_MODE is "
                                         << (mode == nullptr ? std::string("unset") : "\"" + std::string(mode) + "\"")
                                         << ", not 1: this executable launches on a core-local RISC and must link the "
                                            "x280_slow_dispatch object library (x280/CMakeLists.txt)";
}

}  // namespace x280::harness

#endif  // X280_HARNESS_SLOW_DISPATCH_HPP
