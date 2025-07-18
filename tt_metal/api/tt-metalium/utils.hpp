// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>

namespace tt {
namespace utils {
bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);
const std::string& get_reports_dir();

// Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename E, std::enable_if_t<std::is_enum<E>::value, bool> = true>
auto underlying_type(const E& e) {
    return static_cast<typename std::underlying_type<E>::type>(e);
}
}  // namespace utils
}  // namespace tt
