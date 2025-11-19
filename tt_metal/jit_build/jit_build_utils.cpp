// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <fstream>
#include <tt-logger/tt-logger.hpp>
#include "tt_stl/assert.hpp"

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

BuildLock::BuildLock(const std::string& out_dir) : fd_(-1) {
    std::string lock_path = out_dir + ".jit_build_lock";
    fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ == -1) {
        TT_THROW("Failed to create lock file for JIT build lock: {}", lock_path);
    }
    struct flock fl = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0,  // Lock the whole file
    };

    if (fcntl(fd_, F_OFD_SETLKW, &fl) == -1) {
        close(fd_);
        TT_THROW("Failed to acquire JIT build lock on file: {}", lock_path);
    }
}

BuildLock::~BuildLock() {
    // File lock is automatically released when the file descriptor is closed
    close(fd_);
}

}  // namespace tt::jit_build::utils
