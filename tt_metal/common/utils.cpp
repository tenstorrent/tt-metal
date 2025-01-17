// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utils.hpp>
#include <mutex>
#include "tracy/Tracy.hpp"
#include <rtoptions.hpp>

#include <filesystem>
namespace fs = std::filesystem;

namespace tt {
namespace utils {
bool run_command(const string& cmd, const string& log_file, const bool verbose) {
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
        string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
        ret = system(redirected_cmd.c_str());
    }

    return (ret == 0);
}

void create_file(const string& file_path_str) {
    fs::path file_path(file_path_str);
    fs::create_directories(file_path.parent_path());

    std::ofstream ofs(file_path);
    ofs.close();
}

const std::string& get_reports_dir() {
    static std::string outpath;
    if (outpath == "") {
        outpath = tt::llrt::RunTimeOptions::get_instance().get_root_dir() + "/generated/reports/";
    }
    return outpath;
}

size_t DefinesHash::operator()(const std::map<std::string, std::string>& c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it) {
        hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    }
    return hash_value;
}

}  // namespace utils

}  // namespace tt
