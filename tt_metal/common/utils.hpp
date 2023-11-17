// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "assert.hpp"

using std::string;
using std::cout;

#include <filesystem>
namespace fs = std::filesystem;

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose);
    void create_file(string file_path_str);
    std::string get_root_dir();
    const std::string &get_reports_dir();

    // Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
    template <typename SizeT>
    inline void hash_combine(SizeT& seed, const SizeT value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    inline std::vector<std::string> strsplit(std::string input, char delimiter) {
        std::vector<std::string> result = {};
        std::stringstream ss(input);

        while (ss.good()) {
            std::string substr;
            getline(ss, substr, delimiter);
            result.push_back(substr);
        }
        return result;
    }
}
}
