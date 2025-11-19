// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <pthread.h>
#include <cstdlib>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <fstream>
#include <tt-logger/tt-logger.hpp>
#include "tt_stl/assert.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, const bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;
    // Use cached env var from rtoptions instead of calling getenv() on every invocation
    const bool dump_commands = tt::tt_metal::MetalContext::instance().rtoptions().get_dump_build_commands();
    if (dump_commands || verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(cmd.c_str());
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

struct SharedMutex {
    volatile bool initialized;
    pthread_mutex_t mutex;
};

BuildLock::BuildLock(const std::string& out_dir) : fd_(-1) {
    std::string lock_path = out_dir + ".jit_build_lock";
    fd_ = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ == -1) {
        perror("open");
        TT_THROW("Failed to create lock file for JIT build lock: {}", lock_path);
    }
    struct flock fl = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0,  // Lock the whole file
    };

    if (fcntl(fd_, F_OFD_SETLKW, &fl) == -1) {
        perror("fcntl");
        close(fd_);
        TT_THROW("Failed to acquire JIT build lock on file: {}", lock_path);
    }
}

BuildLock::~BuildLock() {
    // File lock is automatically released when the file descriptor is closed
    close(fd_);
}

}  // namespace tt::jit_build::utils
