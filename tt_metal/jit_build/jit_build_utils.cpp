// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <stdlib.h>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <chrono>
#include <fstream>
#include <tt-logger/tt-logger.hpp>

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, const bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    static std::mutex io_mutex;

    // Emit at ERROR log level so CI log always captures the full command, even when normal verbose output is off.
    log_error(tt::LogAlways, "JIT_BUILD_CMD: {}", cmd);

    // Optional: when TT_METAL_FAIL_AFTER_CMD_DUMP is set, abort immediately so CI keeps the log artifact.
    if (std::getenv("TT_METAL_FAIL_AFTER_CMD_DUMP") != nullptr) {
        log_error(tt::LogAlways, "Aborting after command dump because TT_METAL_FAIL_AFTER_CMD_DUMP is set");
        return false;  // caller treats this as command failure
    }

    int ret = 0;  // Initialize to success
    if (getenv("TT_METAL_BACKEND_DUMP_RUN_CMD") or verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:" << std::endl;
            std::cout << cmd << std::endl << std::endl;
        }
        ret = system(cmd.c_str());
        // {
        //     std::lock_guard<std::mutex> lk(io_mutex);
        //     cout << "===== DONE SYSTEM COMMAND: " << cmd << std::endl;
        // }

    } else {
        std::string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
        ret = system(redirected_cmd.c_str());
    }
    // Always append the raw command into a side-car file next to the normal log
    if (!log_file.empty()) {
        try {
            std::ofstream cmd_log_stream(log_file + ".cmd", std::ios::app);
            cmd_log_stream << cmd << std::endl;
        } catch (...) {
            // Best-effort; do not fail build on logging error
        }
    }
    return (ret == 0);
}

void create_file(const std::string& file_path_str) {
    namespace fs = std::filesystem;

    fs::path file_path(file_path_str);
    fs::create_directories(file_path.parent_path());

    std::ofstream ofs(file_path);
    ofs.close();
}

}  // namespace tt::jit_build::utils
