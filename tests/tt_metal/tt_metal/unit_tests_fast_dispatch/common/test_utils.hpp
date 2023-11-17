// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tt_metal/host_api.hpp"

// Helper function to open a file as an fstream, and check that it was opened properly.
inline bool OpenFile(string &file_name, std::fstream &file_stream, std::ios_base::openmode mode) {
    file_stream.open(file_name, mode);
    if (file_stream.is_open()) {
        return true;
    } else {
        tt::log_info(
            tt::LogTest,
            "Test Error: Couldn't open file {}.",
            file_name
        );
        return false;
    }
}

// Check whether the given file contains a list of strings. Doesn't check for
// strings between lines in the file.
inline bool FileContainsAllStrings(string file_name, vector<string> &must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in))
        return false;

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

// Checkes whether two files are identical.
inline bool FilesAreIdentical(string file_name_a, string file_name_b) {
    // Open each file
    std::fstream file_a, file_b;
    if (!OpenFile(file_name_a, file_a, std::fstream::in))
        return false;
    if (!OpenFile(file_name_b, file_b, std::fstream::in))
        return false;

    // Go through line-by-line
    string line_a, line_b;
    int line_num = 1;
    while (getline(file_a, line_a) && getline(file_b, line_b)) {
        if (line_a != line_b) {
            tt::log_info(
                tt::LogTest,
                "Test Error: Line {} of {} and {} did not match:\n\t{}\n\t{}",
                line_num,
                file_name_a,
                file_name_b,
                line_a,
                line_b
            );
            return false;
        }
        line_num++;
    }

    // Make sure that there's no lines left over in either file
    if (getline(file_a, line_a)) {
        tt::log_info(
            tt::LogTest,
            "Test Error: file {} has more lines than file {}",
            file_name_a,
            file_name_b
        );
        return false;
    }
    if (getline(file_b, line_b)) {
        tt::log_info(
            tt::LogTest,
            "Test Error: file {} has more lines than file {}",
            file_name_b,
            file_name_a
        );
        return false;
    }

    // If no checks failed, then the files match
    return true;
}
