// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_api.hpp"
#include "tt_metal/llrt/llrt.hpp"

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

// Helper function to dump a file
inline void DumpFile(string file_name) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        tt::log_info(tt::LogTest, "File \'{}\' does not exist!", file_name);
        return;
    }

    tt::log_info(tt::LogTest, "File \'{}\' contains:", file_name);
    string line;
    while (getline(log_file, line))
        tt::log_info(tt::LogTest, "{}", line);
}

// Compare two strings with a (single-character) wildcard
inline bool StringCompareWithWildcard(const string& s1, const string& s2, char wildcard) {
    if (s1.size() != s2.size())
        return false;

    for (int idx = 0; idx < s1.size(); idx++) {
        if (s1[idx] != s2[idx] && s1[idx] != wildcard && s2[idx] != wildcard)
            return false;
    }

    return true;
}

// Check if s1 is in s2, with wildcard character support
inline bool StringContainsWithWildcard(const string& s1, const string& s2, char wildcard) {
    int substr_len = s1.size();
    int superstr_len = s2.size();
    if (substr_len > superstr_len)
        return false;

    for (int idx = 0; idx <= superstr_len - substr_len; idx++) {
        string substr = s2.substr(idx, substr_len);
        if (StringCompareWithWildcard(s1, substr, wildcard))
            return true;
    }

    return false;
}

// Check whether the given file contains a list of strings. Doesn't check for
// strings between lines in the file.
inline bool FileContainsAllStrings(string file_name, const std::vector<string> &must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in))
        return false;

    // Construct a set of required strings, we'll remove each one when it's found.
    std::set<string> must_contain_set(must_contain.begin(), must_contain.end());

    if (log_file.is_open()) {
        string line;
        while (getline(log_file, line)) {
            // Check for all target strings in the current line
            std::vector<string> found_on_current_line;
            for (const string &s : must_contain_set) {
                if (StringContainsWithWildcard(s, line, '*'))
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
    log_file.close();

    // If the log file doesn't exist, is empty, or doesn't contain all strings, return false.
    string missing_strings = "";
    for (const string &s : must_contain_set)
        missing_strings += s + ",";
    tt::log_info(
        tt::LogTest,
        "Test Error: Expected file {} to contain the following strings: {}",
        file_name,
        missing_strings);
    DumpFile(file_name);
    return false;
}

// Check whether the given file contains a list of strings (in order). Doesn't check for strings
// between lines in a file.
inline bool FileContainsAllStringsInOrder(string file_name, const std::vector<string> &must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in))
        return false;

    // Construct a queue of required strings, we'll remove each one when it's found.
    std::deque<string> must_contain_queue(must_contain.begin(), must_contain.end());

    if (log_file.is_open()) {
        string line;
        while (getline(log_file, line)) {
            // Check for all target strings in the current line
            while (
                !must_contain_queue.empty() &&
                StringContainsWithWildcard(must_contain_queue.front(), line, '*')
            ) {
                must_contain_queue.pop_front();
            }

            // If all strings have been found, return true
            if (must_contain_queue.empty())
                return true;
        }
    }
    log_file.close();

    // If the log file doesn't exist, is empty, or doesn't contain all strings, return false.
    string missing_strings = "";
    for (const string &s : must_contain_queue)
        missing_strings += s + ",";
    tt::log_info(
        tt::LogTest,
        "Test Error: Expected file {} to contain the following strings: {}",
        file_name,
        missing_strings);
    DumpFile(file_name);
    return false;
}

// Checkes whether a given file matches a golden string.
inline bool FilesMatchesString(string file_name, const string& expected) {
    // Open the input file.
    std::fstream file;
    if (!OpenFile(file_name, file, std::fstream::in)) {
        tt::log_info(
            tt::LogTest,
            "Test Error: file {} could not be opened.",
            file_name
        );
        return false;
    }

    // Read the expected string into a stream
    std::istringstream expect_stream(expected);

    // Go through line-by-line
    string line_a, line_b;
    int line_num = 0;
    while (getline(file, line_a) && getline(expect_stream, line_b)) {
        line_num++;
        if (!StringCompareWithWildcard(line_a, line_b, '*')) {
            tt::log_info(
                tt::LogTest,
                "Test Error: Line {} of {} did not match expected:\n\t{}\n\t{}",
                line_num,
                file_name,
                line_a,
                line_b
            );
            return false;
        }
    }

    // Make sure that there's no lines left over in either stream
    if (getline(file, line_a)) {
        tt::log_info(
            tt::LogTest,
            "Test Error: file {} has more lines than expected (>{}).",
            file_name,
            line_num
        );
        return false;
    }
    if (getline(expect_stream, line_b)) {
        tt::log_info(
            tt::LogTest,
            "Test Error: file {} has less lines than expected ({}).",
            file_name,
            line_num
        );
        return false;
    }

    // If no checks failed, then the file matches expected.
    return true;
}
