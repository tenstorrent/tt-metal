/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdio.h>
#include <cstring>

#include "rtoptions.hpp"

namespace tt {

namespace llrt {

// Note: global initialization order is non-deterministic
// This is ok so long as this gets initialized before decisions are based on
// env state
RunTimeOptions OptionsG;

RunTimeOptions::RunTimeOptions() : watcher_interval_ms(0), watcher_dump_all(false) {

    const char *watcher_enable_str = getenv("TT_METAL_WATCHER");
    if (watcher_enable_str != nullptr) {
        int sleep_val = 0;
        sscanf(watcher_enable_str, "%d", &sleep_val);
        if (strstr(watcher_enable_str, "ms") == nullptr) {
            sleep_val *= 1000;
        }
        constexpr int watcher_default_sleep_msecs = 2 * 60 * 1000;
        watcher_interval_ms = (sleep_val == 0) ? watcher_default_sleep_msecs : sleep_val;
    }

    const char *watcher_dump_all_str = getenv("TT_METAL_WATCHER_DUMP_ALL");
    if (watcher_dump_all_str != nullptr) {
        watcher_dump_all = true;
    }
}

} // namespace llrt

} // namespace tt
