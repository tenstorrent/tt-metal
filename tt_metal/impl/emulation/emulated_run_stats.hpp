// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::emule {

struct EmulatedRunStats {
    uint32_t num_cores = 0;
    std::vector<std::string> kernel_paths;  // unique kernel source basenames
};

const EmulatedRunStats& get_last_emulated_run_stats();

}  // namespace tt::tt_metal::emule
