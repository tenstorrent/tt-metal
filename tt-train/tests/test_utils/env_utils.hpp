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

    auto watcher_env_ptr = std::getenv(TT_METAL_WATCHER_ENV_VAR);
    return watcher_env_ptr != nullptr && std::string(watcher_env_ptr) == "1";
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
