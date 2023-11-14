/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tt_metal/host_api.hpp"

// Check whether the given file contains a list of strings. Doesn't check for strings between
// lines in the file.
bool FileContainsAllStrings(string file_name, vector<string> &must_contain) {
    std::fstream log_file;
    log_file.open(file_name, std::fstream::in);

    // Construct a set of required strings, we'll remove each one when it's found.
    set<string> must_contain_set(must_contain.begin(), must_contain.end());

    if (log_file.is_open()) {
        string line;
        while (getline(log_file, line)) {
            // Check for all target strings in the current line
            vector<string> found_on_current_line;
            for (const string &s : must_contain_set) {
                if (line.find(s) != string::npos)
                    found_on_current_line.push_back(s);
            }

            // Remove all strings found on this line from the set to continue searching for
            for (const string &s : found_on_current_line)
                must_contain_set.erase(s);

            // If all strings have been found, return true
            if (must_contain_set.empty())
                return true;
        }
    }

    // If the log file doesn't exist, is empty, or doesn't contain all strings, return false.
    string missing_strings = "";
    for (const string &s : must_contain_set)
        missing_strings += s + ",";
    tt::log_info(
        tt::LogTest,
        "Test Error: Expected file {} to contain the following strings: {}",
        file_name,
        missing_strings);
    return false;
}
