// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <cerrno>
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
#include <tt_stl/fmt.hpp>
#include "common/filesystem_utils.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::filesystem::path& log_file, bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;

    if (verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(cmd.c_str());
    } else {
        std::string redirected_cmd = cmd + " >> " + log_file.string() + " 2>&1";
        ret = system(redirected_cmd.c_str());
    }

    return (ret == 0);
}

void create_file(const std::filesystem::path& file_path) {
    tt::filesystem::safe_create_directories(file_path.parent_path());

    // just making sure the file is there. Don't need to worry about the state
    [[maybe_unused]] std::error_code open_ec;
    [[maybe_unused]] auto _ = tt::filesystem::retry_on_estale_ec(
        [&](std::error_code& ec) {
            std::ofstream file(file_path);
            if (!file.is_open() || file.fail()) {
                ec.assign(errno, std::system_category());
                return false;
            }
            file.close();
            return true;
        },
        open_ec);
}

uint64_t FileRenamer::unique_id_ = []() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> distr;
    return distr(rd);
}();

std::filesystem::path FileRenamer::generate_temp_path(const std::filesystem::path& target_path) {
    // stem() gives you the filename without the last extension, and extension() is empty when
    // there isn't one, so this covers both cases:
    // foo.txt -> foo.42.txt
    // foo -> foo.42

    std::filesystem::path filename = target_path.stem();
    filename += ".";
    filename += std::to_string(unique_id_);
    filename += target_path.extension();
    return target_path.parent_path() / filename;
}

FileRenamer::FileRenamer(const std::filesystem::path& target_path) :
    temp_path_(generate_temp_path(target_path)), target_path_(target_path) {}

FileRenamer::~FileRenamer() {
    if (target_path_.empty()) {
        return;
    }
    if (!tt::filesystem::safe_rename(temp_path_, target_path_)) {
        log_error(
            tt::LogBuildKernels, "Failed to rename temporary file {} to target file {}", temp_path_, target_path_);
    }
}

}  // namespace tt::jit_build::utils
