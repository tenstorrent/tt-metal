// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace ttml {
namespace test_utils {

inline bool is_watcher_enabled() {
    constexpr auto TT_METAL_WATCHER_ENV_VAR = "TT_METAL_WATCHER";
    constexpr auto TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS_ENV_VAR = "TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS";

    auto watcher_env_ptr = std::getenv(TT_METAL_WATCHER_ENV_VAR);
    auto lightweight_asserts_env_ptr = std::getenv(TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS_ENV_VAR);

    bool watcher_set = watcher_env_ptr != nullptr && std::string(watcher_env_ptr).length() > 0;
    bool lightweight_asserts_set =
        lightweight_asserts_env_ptr != nullptr && std::string(lightweight_asserts_env_ptr) == "1";

    return watcher_set || lightweight_asserts_set;
}

// Macro to skip tests when watcher is enabled
#define SKIP_FOR_WATCHER()                                              \
    do {                                                                \
        if (ttml::test_utils::is_watcher_enabled()) {                   \
            GTEST_SKIP() << "Test is not passing with watcher enabled"; \
        }                                                               \
    } while (0)

}  // namespace test_utils
}  // namespace ttml
