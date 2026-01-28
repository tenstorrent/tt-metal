// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <system_error>

#include <tt-logger/tt-logger.hpp>
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

uint64_t FileRenamer::unique_id_ = []() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> distr;
    return distr(rd);
}();

FileRenamer::FileRenamer(const std::string& target_path) : target_path_(target_path) {
    std::filesystem::path path(target_path);
    if (path.has_extension()) {
        path.replace_extension(fmt::format("{}{}", unique_id_, path.extension().string()));
        temp_path_ = path.string();
    } else {
        temp_path_ = fmt::format("{}.{}", target_path, unique_id_);
    }
}

FileRenamer::~FileRenamer() {
    std::error_code ec;
    if (target_path_.empty()) {
        return;
    }
    std::filesystem::rename(temp_path_, target_path_, ec);
    if (ec) {
        log_error(
            tt::LogBuildKernels,
            "Failed to rename temporary file {} to target file {}: {}",
            temp_path_,
            target_path_,
            ec.message());
    }
}

}  // namespace tt::jit_build::utils
