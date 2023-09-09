/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdio.h>
#include "rtoptions.hpp"

namespace tt {

namespace llrt {

// Note: global initialization order is non-deterministic
// This is ok so long as this gets initialized before decisions are based on
// env state
RunTimeOptions OptionsG;

RunTimeOptions::RunTimeOptions() : watcher_interval(0) {

    const char *watcher_enable_str = getenv("TT_METAL_WATCHER");
    if (watcher_enable_str != nullptr) {
        int sleep_secs = 0;
        sscanf(watcher_enable_str, "%d", &sleep_secs);
        constexpr int watcher_default_sleep_secs = 2 * 60;
        watcher_interval = (sleep_secs == 0) ? watcher_default_sleep_secs : sleep_secs;
    }
}

} // namespace llrt

} // namespace tt
