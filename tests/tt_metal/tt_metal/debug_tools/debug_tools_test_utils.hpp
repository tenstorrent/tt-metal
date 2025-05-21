// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"

#include <string_view>

// Helper function to open a file as an fstream, and check that it was opened properly.
inline bool OpenFile(string &file_name, std::fstream &file_stream, std::ios_base::openmode mode) {
    file_stream.open(file_name, mode);
    if (file_stream.is_open()) {
        return true;
    }
    tt::log_info(tt::LogTest, "Test Error: Couldn't open file {}.", file_name);
    return false;
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
    while (getline(log_file, line)) {
        tt::log_info(tt::LogTest, "{}", line);
    }
}

// Wildcard is '?', just like a glob.
inline bool StringCompareWithWildcard(const std::string_view str, const std::string_view pattern) {
    size_t pattern_size = pattern.size();
    if (pattern_size != str.size())
        return false;
    for (int idx = 0; idx < pattern_size; idx++) {
        if (str[idx] != pattern[idx] && pattern[idx] != '?') {
            return false;
        }
    }
    return true;
}

// Check if haystack contains needle, return true. needle may contain
// '?' to match any character (just like glob).
inline bool StringContainsWithWildcard(std::string_view haystack, std::string_view needle) {
    size_t needle_size = needle.size();
    if (needle_size == 0 || needle.front() == '?') {
        // The needle is empty, or begins with '?', fail in order to
        // force test to be fixed.
        return false;
    }
    if (needle_size > haystack.size()) {
        return false;
    }
    char first = needle.front();

    for (size_t idx = 0, limit = haystack.size() - needle_size;
         (idx = haystack.find(first, idx)) <= limit; idx++) {
        std::string_view substr(&haystack[idx], needle_size);
        if (StringCompareWithWildcard(substr, needle)) {
            return true;
        }
    }

    return false;
}

// Check whether the given file contains a list of strings in any order. Doesn't check for
// strings between lines in the file.
inline bool FileContainsAllStrings(string file_name, const std::vector<string> &must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in))
        return false;

    // Construct a set of required strings, we'll remove each one when it's found.
    std::set<std::string_view> must_contain_set;
    for (auto const &str : must_contain) {
        must_contain_set.insert(str);
    }

    for (;;) {
        if (must_contain_set.empty()) {
            // Found them all.
            return true;
        }

        string line;
        if (!getline(log_file, line)) {
            break;
        }

        // Check for all target strings in the current line
        std::vector<std::string_view> found_on_current_line;
        for (const auto &s : must_contain_set) {
            if (StringContainsWithWildcard(line, s)) {
                found_on_current_line.push_back(s);
            }
        }

        // Remove all strings found on this line from the set to continue searching for
        for (const auto &s : found_on_current_line)
            must_contain_set.erase(s);
    }

    // Reached EOF with strings yet to find.
    string missing_strings = "";
    for (const auto &s : must_contain_set) {
        missing_strings.append(&", \""[missing_strings.empty() ? 2 : 0]).append(s).push_back('"');
    }
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

    // Construct a deque of required strings, we'll remove each one when it's found.
    std::deque<std::string_view> must_contain_queue;
    for (auto const &str : must_contain) {
        must_contain_queue.push_back(str);
    }

    for (;;) {
        if (must_contain_queue.empty()) {
            // Found them all
            return true;
        }

        string line;
        if (!getline(log_file, line)) {
            break;
        }

        // Check for all target strings in the current line
        for (; !must_contain_queue.empty(); must_contain_queue.pop_front()) {
            if (!StringContainsWithWildcard(line, must_contain_queue.front())) {
                break;
            }
        }
    }

    // Reached EOF with strings yet to find.
    string missing_strings = "";
    for (const auto &s : must_contain_queue) {
        missing_strings.append(&", \""[missing_strings.empty() ? 2 : 0]).append(s).push_back('"');
    }
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
        if (!StringCompareWithWildcard(line_a, line_b)) {
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
