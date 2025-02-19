// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <regex>
#include <map>
#include <mutex>
#include <fstream>
#include <utils.hpp>

#include <assert.hpp>

using namespace std;

namespace tt::tt_metal {

// Maps full_kernel_name to defines used to compile it
map<string, string> kernel_defines_and_args_;  // Maps full_kernel_name to defines used to compile it
// Guards kernel_defines_and_args_ for multi-threaded access
std::mutex mutex_kernel_defines_and_args_;

// Replace all occurrences of 'from' in 'source' with 'to'
void str_replace_all(string& source, const string& from, const string& to) {
    for (size_t pos = 0; (pos = source.find(from, pos)) != string::npos; pos += to.length()) {
        source.replace(pos, from.length(), to);
    }
}

void log_kernel_defines_and_args(
    const string& out_dir, const string& full_kernel_name, const string& defines_and_args_str) {
    std::lock_guard<std::mutex> lock(mutex_kernel_defines_and_args_);
    string defines_as_csv(defines_and_args_str);

    str_replace_all(defines_as_csv, "KERNEL_COMPILE_TIME_", "");
    str_replace_all(defines_as_csv, "-D", ",");
    str_replace_all(defines_as_csv, " ", "");
    str_replace_all(defines_as_csv, ",", ", ");

    if (kernel_defines_and_args_.find(out_dir) == kernel_defines_and_args_.end()) {
        kernel_defines_and_args_[out_dir] = defines_as_csv;
    } else {
        if (kernel_defines_and_args_[out_dir] != defines_as_csv) {
            log_error(
                "Multiple distinct kernel arguments found for: {}. Existing:\n{}, New:     \n{}",
                out_dir,
                kernel_defines_and_args_[full_kernel_name],
                defines_as_csv);
        }
    }
}

void dump_kernel_defines_and_args(const string& out_kernel_root_path) {
    // Make sure the directory exists
    tt::utils::create_file(out_kernel_root_path);

    string kernel_args_csv = out_kernel_root_path + "kernel_args.csv";

    std::lock_guard<std::mutex> lock(mutex_kernel_defines_and_args_);
    ofstream file(kernel_args_csv, ios::trunc);
    if (file.is_open()) {
        for (auto const& [full_kernel_name, defines_and_args_str] : kernel_defines_and_args_) {
            file << full_kernel_name << defines_and_args_str << "\n";
        }
        file.close();
    } else {
        TT_THROW("Failed to open file: {}", kernel_args_csv);
    }
}

}  // namespace tt::tt_metal
