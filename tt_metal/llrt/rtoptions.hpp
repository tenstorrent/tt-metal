/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

namespace tt {

namespace llrt {

class RunTimeOptions {
    int watcher_interval;

public:
    RunTimeOptions();

    inline bool get_watcher_enabled() { return watcher_interval != 0; }
    inline int get_watcher_interval() { return watcher_interval; }
};


extern RunTimeOptions OptionsG;

} // namespace llrt

} // namespace tt
