// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

#include "impl/context/metal_context.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, const bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;

#ifndef NDEBUG
    // Clear ASan/LSan environment variables for subprocesses to prevent the sanitizer
    // runtime from being injected into the cross-compiler toolchain. The sanitizers are
    // meant for testing our code, not external tools like the RISC-V compiler.
    const std::string sanitized_cmd = "unset LD_PRELOAD ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS; " + cmd;
#else
    const std::string& sanitized_cmd = cmd;
#endif

    // Use cached env var from rtoptions instead of calling getenv() on every invocation
    const bool dump_commands = tt::tt_metal::MetalContext::instance().rtoptions().get_dump_build_commands();
    if (dump_commands || verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(sanitized_cmd.c_str());
    } else {
        std::string redirected_cmd = sanitized_cmd + " >> " + log_file + " 2>&1";
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
