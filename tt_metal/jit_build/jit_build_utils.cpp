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

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, const bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;
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
