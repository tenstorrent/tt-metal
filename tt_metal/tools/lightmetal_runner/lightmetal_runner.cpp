// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include <tt-metalium/logger.hpp>
#include <tt-metalium/assert.hpp>
#include "lightmetal_types.hpp"

using namespace tt;

int main(int argc, char* argv[]) {
    // Process cmdline arguments
    std::string program_filename = argv[0];
    TT_FATAL(argc == 2, "Invalid number of supplied arguments. Usage: {} <binary_file>", program_filename.c_str());
    std::string binary_filename = argv[1];

    // Read the Light Metal Binary file into blob, transfer ownership and execute it.
    LightMetalBinary binary_blob = LightMetalBinary::LoadFromFile(binary_filename);
    tt::tt_metal::LightMetalReplay lm_replay(std::move(binary_blob));

    if (!lm_replay.ExecuteLightMetalBinary()) {
        log_fatal("Light Metal Binary {} failed to execute or encountered errors.", binary_filename);
        return 1;
    } else {
        log_info(tt::LogMetalTrace, "Light Metal Binary {} executed successfully", binary_filename);
        return 0;
    }
}
