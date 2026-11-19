// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <impl/debug/dprint_server.hpp>
#include "llrt.hpp"
#include <fmt/format.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Helper function to open a file as an fstream, and check that it was opened properly.
inline bool OpenFile(const std::string& file_name, std::fstream& file_stream, std::ios_base::openmode mode) {
    file_stream.open(file_name, mode);
    if (file_stream.is_open()) {
        return true;
    }
    log_info(tt::LogTest, "Test Error: Couldn't open file {}.", file_name);
    return false;
}

// Helper function to dump a file
inline void DumpFile(const std::string& file_name) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        log_info(tt::LogTest, "File \'{}\' does not exist!", file_name);
        return;
    }

    log_info(tt::LogTest, "File \'{}\' contains:", file_name);
    std::string line;
    while (getline(log_file, line)) {
        log_info(tt::LogTest, "{}", line);
    }
}

// getline for the checks below: skips the lines that firmware printed, which the DPRINT server tags for
// DevicePrintFixture, so that the checks see only what kernels printed. DumpFile shows every line.
inline bool GetLineSkippingFirmware(std::istream& stream, std::string& line) {
    while (std::getline(stream, line)) {
        if (!line.starts_with(tt::tt_metal::DPrintServer::kFirmwareLineTag)) {
            return true;
        }
    }
    return false;
}

std::string_view::size_type FloatingGlobEndsAt(std::string_view haystack, std::string_view needle, unsigned globs);

// Check of pattern matches at the beginning of str.
inline std::string_view::size_type AnchoredGlobEndsAt(
    const std::string_view str, const std::string_view pattern, unsigned globs) {
    if (str.size() + globs < pattern.size()) {
        return str.npos;
    }

    // NOLINTBEGIN(bugprone-branch-clone)
    for (std::string_view::size_type idx = 0; idx != pattern.size(); idx++) {
        if (pattern[idx] == '*') {
            auto result = FloatingGlobEndsAt(str.substr(idx), pattern.substr(idx + 1), globs - 1);
            if (result != str.npos) {
                // An empty suffix matches the whole string.
                result = result ? result + idx : str.size();
            }
            return result;
        }
        if (idx >= str.size()) {
            return str.npos;
        }
        if (pattern[idx] == '?') {
            continue;
        }
        if (str[idx] != pattern[idx]) {
            return str.npos;
        }
    }
    // NOLINTEND(bugprone-branch-clone)

    return pattern.size();
}

// Look for needle in haystack. We look backwards through haystack, so
// that glob use will find the longest match.
inline std::string_view::size_type FloatingGlobEndsAt(
    const std::string_view haystack, const std::string_view needle, unsigned globs) {
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
        if (!idx) {
            break;
        }
    }

    return haystack.npos;
}

// Count the number of '*' characters.
inline unsigned GlobCount(const std::string_view glob) {
    unsigned count = 0;
    for (std::string_view::size_type idx = 0; (idx = glob.find('*', idx)) != glob.npos; idx++) {
        count++;
    }
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
inline bool FileContainsAllStrings(const std::string& file_name, const std::vector<std::string>& must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        return false;
    }

    // Construct a set of required strings, we'll remove each one when it's found.
    std::set<std::string_view> must_contain_set;
    for (const auto& str : must_contain) {
        must_contain_set.insert(str);
    }

    for (;;) {
        if (must_contain_set.empty()) {
            // Found them all.
            return true;
        }

        std::string line;
        if (!GetLineSkippingFirmware(log_file, line)) {
            break;
        }

        // Check for all target strings in the current line
        std::vector<std::string_view> found_on_current_line;
        for (const auto& s : must_contain_set) {
            if (StringContainsGlob(line, s)) {
                found_on_current_line.push_back(s);
            }
        }

        // Remove all strings found on this line from the set to continue searching for
        for (const auto& s : found_on_current_line) {
            must_contain_set.erase(s);
        }
    }

    // Reached EOF with strings yet to find.
    std::string missing_strings;
    for (const auto& s : must_contain_set) {
        missing_strings.append(&", \""[missing_strings.empty() ? 2 : 0]).append(s).push_back('"');
    }
    log_info(
        tt::LogTest, "Test Error: Expected file {} to contain the following strings: {}", file_name, missing_strings);
    DumpFile(file_name);
    return false;
}

// Check that the given file contains NONE of the listed strings. Returns false as soon as any one
// is found. Doesn't check for strings spanning lines in the file.
inline bool FileContainsNoneOfStrings(const std::string& file_name, const std::vector<std::string>& must_not_contain) {
    if (must_not_contain.empty()) {
        return true;
    }

    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        return false;
    }

    std::string line;
    while (GetLineSkippingFirmware(log_file, line)) {
        for (const auto& forbidden : must_not_contain) {
            if (StringContainsGlob(line, forbidden)) {
                log_info(
                    tt::LogTest,
                    "Test Error: file {} unexpectedly contains forbidden string \"{}\"",
                    file_name,
                    forbidden);
                DumpFile(file_name);
                return false;
            }
        }
    }

    return true;
}

// Check whether the given file contains a list of strings (in order). Doesn't check for strings
// between lines in a file.
inline bool FileContainsAllStringsInOrder(const std::string& file_name, const std::vector<std::string>& must_contain) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        return false;
    }

    // Construct a deque of required strings, we'll remove each one when it's found.
    std::deque<std::string_view> must_contain_queue;
    for (const auto& str : must_contain) {
        must_contain_queue.push_back(str);
    }

    for (;;) {
        if (must_contain_queue.empty()) {
            // Found them all
            return true;
        }

        std::string line;
        if (!GetLineSkippingFirmware(log_file, line)) {
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
    std::string missing_strings;
    for (const auto& s : must_contain_queue) {
        missing_strings.append(&", \""[missing_strings.empty() ? 2 : 0]).append(s).push_back('"');
    }
    log_info(
        tt::LogTest, "Test Error: Expected file {} to contain the following strings: {}", file_name, missing_strings);
    DumpFile(file_name);
    return false;
}

// Delete all lines from file that start with a given prefix
inline bool DeleteLinesStartingWith(const std::string& file_name, const std::string& prefix) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        log_info(tt::LogTest, "File '{}' does not exist!", file_name);
        return false;
    }
    std::string content;
    std::string line;
    while (getline(log_file, line)) {
        if (line.starts_with(prefix)) {
            // Skip lines that begin with prefix
            continue;
        }
        content += line + "\n";
    }
    log_file.close();
    if (!OpenFile(file_name, log_file, std::fstream::out | std::fstream::trunc)) {
        std::cout << "Could not open file " << file_name << " for writing!" << std::endl;
        return false;
    }
    log_file << content;
    log_file.close();
    return true;
}
// Checks whether a given file matches a golden string, line by line. Golden lines may use the '*' and '?' globs.
inline bool FilesMatchesString(const std::string& file_name, const std::string& expected) {
    // Open the input file.
    std::fstream file;
    if (!OpenFile(file_name, file, std::fstream::in)) {
        log_info(tt::LogTest, "Test Error: file {} could not be opened.", file_name);
        return false;
    }

    // Read the expected string into a stream
    std::istringstream expect_stream(expected);

    // Go through line-by-line. Read both streams before checking either, so that a line left over in one of them
    // is never consumed unchecked.
    std::string line_a, line_b;
    int line_num = 0;
    for (;;) {
        const bool has_line_a = GetLineSkippingFirmware(file, line_a);
        const bool has_line_b = static_cast<bool>(getline(expect_stream, line_b));
        if (!has_line_a && !has_line_b) {
            // If no checks failed, then the file matches expected.
            return true;
        }
        line_num++;
        if (!has_line_b) {
            log_info(
                tt::LogTest,
                "Test Error: file {} has more than the {} expected line(s); line {} is:\n\t{}",
                file_name,
                line_num - 1,
                line_num,
                line_a);
            return false;
        }
        if (!has_line_a) {
            log_info(
                tt::LogTest,
                "Test Error: file {} has only {} line(s), fewer than expected; line {} should be:\n\t{}",
                file_name,
                line_num - 1,
                line_num,
                line_b);
            return false;
        }
        if (!StringMatchesGlob(line_a, line_b)) {
            log_info(
                tt::LogTest,
                "Test Error: Line {} of {} did not match expected:\n\t{}\n\t{}",
                line_num,
                file_name,
                line_a,
                line_b);
            return false;
        }
    }
}

// Checks that the file holds exactly the expected lines, in order. Expected lines may use the '*' and '?' globs for
// text that changes between runs, such as an address. On a mismatch, logs the whole file.
inline bool FileMatchesLines(const std::string& file_name, const std::vector<std::string>& expected_lines) {
    std::string expected;
    for (const auto& line : expected_lines) {
        expected.append(line).push_back('\n');
    }
    if (FilesMatchesString(file_name, expected)) {
        return true;
    }
    DumpFile(file_name);
    return false;
}

// Splits a DPRINT line written with the prefix, "<device>:<x>-<y>:<RISC>: <text>", into the prefix without its
// trailing ": " (for example "0:1-2:TR0") and the text. A line without the prefix comes back whole, with an empty
// prefix.
inline std::pair<std::string_view, std::string_view> SplitDevicePrintPrefix(std::string_view line) {
    const auto is_number = [](std::string_view s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    const auto separator = line.find(": ");
    if (separator == std::string_view::npos) {
        return {{}, line};
    }
    const std::string_view prefix = line.substr(0, separator);
    const auto device_end = prefix.find(':');
    const auto core_end = device_end == std::string_view::npos ? device_end : prefix.find(':', device_end + 1);
    if (core_end == std::string_view::npos) {
        return {{}, line};
    }
    const std::string_view device = prefix.substr(0, device_end);
    const std::string_view core = prefix.substr(device_end + 1, core_end - device_end - 1);
    const std::string_view risc = prefix.substr(core_end + 1);
    const auto core_separator = core.find('-');
    if (!is_number(device) || core_separator == std::string_view::npos || !is_number(core.substr(0, core_separator)) ||
        !is_number(core.substr(core_separator + 1)) || risc.empty() || risc.find_first_of(": ") != risc.npos) {
        return {{}, line};
    }
    return {prefix, line.substr(separator + 2)};
}

// The "<device>:<x>-<y>" part of a prefix returned by SplitDevicePrintPrefix, for example "0:1-2" for "0:1-2:TR0".
inline std::string_view DevicePrintPrefixCore(std::string_view prefix) { return prefix.substr(0, prefix.rfind(':')); }

// Groups the lines of a DPRINT log written with the prefix by that prefix, keeping each RISC's lines in order. Keys
// are prefixes as returned by SplitDevicePrintPrefix; lines without the prefix go under the empty key.
inline std::map<std::string, std::vector<std::string>> ReadLinesPerRisc(const std::string& file_name) {
    std::map<std::string, std::vector<std::string>> lines_per_risc;
    std::fstream file;
    if (!OpenFile(file_name, file, std::fstream::in)) {
        return lines_per_risc;
    }
    std::string line;
    while (GetLineSkippingFirmware(file, line)) {
        const auto [prefix, text] = SplitDevicePrintPrefix(line);
        lines_per_risc[std::string(prefix)].emplace_back(text);
    }
    return lines_per_risc;
}

// Checks that `actual` holds exactly the `expected` lines, in order. Expected lines may use the '*' and '?' globs. On
// a mismatch, logs the first difference, naming the lines `label` (for example "0:1-2:TR0 in <file>").
inline bool LinesMatchGlobs(
    std::string_view label, const std::vector<std::string>& actual, const std::vector<std::string>& expected) {
    for (std::size_t i = 0; i < std::max(actual.size(), expected.size()); i++) {
        if (i >= actual.size()) {
            log_info(
                tt::LogTest,
                "Test Error: {} has {} line(s), expected {}; line {} should be:\n\t{}",
                label,
                actual.size(),
                expected.size(),
                i + 1,
                expected[i]);
            return false;
        }
        if (i >= expected.size()) {
            log_info(
                tt::LogTest,
                "Test Error: {} has {} line(s), expected {}; line {} is:\n\t{}",
                label,
                actual.size(),
                expected.size(),
                i + 1,
                actual[i]);
            return false;
        }
        if (!StringMatchesGlob(actual[i], expected[i])) {
            log_info(
                tt::LogTest,
                "Test Error: line {} of {} did not match expected:\n\t{}\n\t{}",
                i + 1,
                label,
                actual[i],
                expected[i]);
            return false;
        }
    }
    return true;
}

// Checks a DPRINT log written with the prefix: each RISC printed exactly its expected lines, in order, and nothing
// else was printed. Keys are prefixes such as "0:1-2:TR0"; expected lines may use the '*' and '?' globs. On a
// mismatch, logs the whole file.
inline bool FileMatchesLinesPerRisc(
    const std::string& file_name, const std::map<std::string, std::vector<std::string>>& expected) {
    const auto actual = ReadLinesPerRisc(file_name);
    bool matches = true;
    for (const auto& [prefix, lines] : actual) {
        if (!expected.contains(prefix)) {
            log_info(
                tt::LogTest,
                "Test Error: {} printed {} unexpected line(s) in {}; the first is:\n\t{}",
                prefix.empty() ? std::string("a line without the prefix") : prefix,
                lines.size(),
                file_name,
                lines.front());
            matches = false;
        }
    }
    const std::vector<std::string> no_lines;
    for (const auto& [prefix, expected_lines] : expected) {
        const auto it = actual.find(prefix);
        if (!LinesMatchGlobs(
                fmt::format("{} in {}", prefix, file_name),
                it == actual.end() ? no_lines : it->second,
                expected_lines)) {
            matches = false;
        }
    }
    if (!matches) {
        DumpFile(file_name);
    }
    return matches;
}
