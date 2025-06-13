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
    log_info(tt::LogTest, "Test Error: Couldn't open file {}.", file_name);
    return false;
}

// Helper function to dump a file
inline void DumpFile(string file_name) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        log_info(tt::LogTest, "File \'{}\' does not exist!", file_name);
        return;
    }

    log_info(tt::LogTest, "File \'{}\' contains:", file_name);
    string line;
    while (getline(log_file, line)) {
        log_info(tt::LogTest, "{}", line);
    }
}

std::string_view::size_type FloatingGlobEndsAt(const std::string_view haystack,
                                               const std::string_view needle,
                                               unsigned globs);

// Check of pattern matches at the beginning of str.
inline std::string_view::size_type AnchoredGlobEndsAt(const std::string_view str,
                                                      const std::string_view pattern,
                                                      unsigned globs) {
    if (str.size() + globs < pattern.size()) {
        return str.npos;
    }

    for (std::string_view::size_type idx = 0; idx != pattern.size(); idx++) {
        if (pattern[idx] == '*') {
            auto result = FloatingGlobEndsAt(str.substr(idx), pattern.substr(idx + 1), globs - 1);
            if (result != str.npos) {
                // An empty suffix matches the whole string.
                result = result ? result + idx : str.size();
            }
            return result;
        } else if (idx >= str.size()) {
            return str.npos;
        } else if (pattern[idx] == '?') {
            continue;
        } else if (str[idx] != pattern[idx]) {
            return str.npos;
        }
    }
    return pattern.size();;
}

// Look for needle in haystack. We look backwards through haystack, so
// that glob use will find the longest match.
inline std::string_view::size_type FloatingGlobEndsAt(const std::string_view haystack,
                                                      const std::string_view needle,
                                                      unsigned globs) {
    if (needle.empty()) {
        // Empty needle matches at end.
        return haystack.size();
    }
    char first = needle.front();
    if (first == '*') {
        // '*' at front, handle as an anchored glob.
        return AnchoredGlobEndsAt(haystack, needle, globs);
    }
    if (haystack.size() + globs < needle.size()) {
        return haystack.npos;
    }

    for (std::string_view::size_type idx = haystack.size() + globs - needle.size();; idx--) {
        if (first != '?') {
            // no wildcard at front, scan for first char to begin search.
            idx = haystack.rfind(first, idx);
            if (idx == haystack.npos) {
                break;
            }
        }
        // Try an anchored match here.
        auto result = AnchoredGlobEndsAt(haystack.substr(idx), needle, globs);
        if (result != haystack.npos) {
            return result + idx;
        }
        if (!idx)
            break;
    }

    return haystack.npos;
}

// Count the number of '*' characters.
inline unsigned GlobCount(const std::string_view glob) {
    unsigned count = 0;
    for (std::string_view::size_type idx = 0; (idx = glob.find('*', idx)) != glob.npos; idx++)
        count++;
    return count;
}
// str matches pattern, allowing '?' and '*' globbing.
inline bool StringMatchesGlob(const std::string_view str, const std::string_view pattern) {
    return AnchoredGlobEndsAt(str, pattern, GlobCount(pattern)) == str.size();
}

// haystack contains needle, allowing '?' and '*' globbing.
inline bool StringContainsGlob(const std::string_view haystack, const std::string_view needle) {
    return FloatingGlobEndsAt(haystack, needle, GlobCount(needle)) != haystack.npos;
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
            if (StringContainsGlob(line, s)) {
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
    log_info(
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
            if (!StringContainsGlob(line, must_contain_queue.front())) {
                break;
            }
        }
    }

    // Reached EOF with strings yet to find.
    string missing_strings = "";
    for (const auto &s : must_contain_queue) {
        missing_strings.append(&", \""[missing_strings.empty() ? 2 : 0]).append(s).push_back('"');
    }
    log_info(
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
        log_info(
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
        if (!StringMatchesGlob(line_a, line_b)) {
            log_info(
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
        log_info(
            tt::LogTest,
            "Test Error: file {} has more lines than expected (>{}).",
            file_name,
            line_num
        );
        return false;
    }
    if (getline(expect_stream, line_b)) {
        log_info(
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
