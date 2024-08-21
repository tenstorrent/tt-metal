// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>
#include <string>
#include <tuple>

namespace tt {

template <class T>
constexpr std::false_type false_type_t{};

template <typename T>
T parse_env(const char *env_name, const T &default_value) {
    char* env_value = std::getenv(env_name);
    if (env_value == nullptr)
        return default_value;

    if constexpr (std::is_same_v<T, bool>) {
        return static_cast<bool>(std::stoi(env_value, 0, 0));
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::string{env_value};
    } else if constexpr (std::is_same_v<T, int>) {
        return std::stoi(env_value, 0, 0);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return std::stoul(env_value, 0, 0);
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return std::stoull(env_value, 0, 0);
    } else {
        static_assert(false_type_t<T>, "No specialization for type");
    }
}

template <typename T>
T parse_trigger(const char *env_name, const T &default_value) {
    T retval = parse_env<T>(env_name, default_value);
    unsetenv(env_name);
    return retval;
}

}  // namespace tt

// Explicit specializations
template bool tt::parse_env<bool>(const char *, const bool &);
template std::string tt::parse_env<std::string>(const char *, const std::string &);
template int tt::parse_env<int>(const char *, const int &);
template uint32_t tt::parse_env<uint32_t>(const char *, const uint32_t &);
template uint64_t tt::parse_env<uint64_t>(const char *, const uint64_t &);
