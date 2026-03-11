// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>

namespace ttml::core {
std::string demangle(const char* name);

inline bool is_watcher_enabled() {
    const char* watcher = std::getenv("TT_METAL_WATCHER");
    const char* asserts = std::getenv("TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS");
    return (watcher != nullptr && watcher[0] != '\0') || (asserts != nullptr && std::string(asserts) == "1");
}
}  // namespace ttml::core

#define SKIP_FOR_WATCHER()                                        \
    do {                                                          \
        if (ttml::core::is_watcher_enabled()) {                   \
            GTEST_SKIP() << "Skipping test with watcher enabled"; \
        }                                                         \
    } while (0)
