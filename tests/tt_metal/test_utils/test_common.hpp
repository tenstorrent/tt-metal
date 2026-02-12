// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <algorithm>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>
#include <iostream>
#include <sstream>

using namespace tt::tt_metal;  // test only

namespace test_args {

template <class T>
constexpr std::false_type always_false{};

template <class T>
T parse(const std::string& s) {
    if constexpr (std::is_same_v<T, std::uint32_t>) {
        return std::stoul(s, nullptr, 0);
    } else if constexpr (std::is_same_v<T, int>) {
        return std::stoi(s, nullptr, 0);
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return std::stoull(s, nullptr, 0);
    } else if constexpr (std::is_same_v<T, bool>) {
        return static_cast<bool>(std::stoi(s, nullptr, 0));
    } else if constexpr (std::is_same_v<T, std::string>) {
        return s;
    } else {
        static_assert(test_args::always_false<T>, "No specialization for type");
    }
}

inline std::string strip(const std::string& s) {
    std::string whitespace = " \t\n";
    std::size_t start = s.find_first_not_of(whitespace);
    std::size_t end = s.find_last_not_of(whitespace);
    end += bool(end != std::string::npos);
    return s.substr(start, end);
}

inline std::string get_command_option(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::string>& default_value = std::nullopt) {
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(test_args), std::end(test_args), option);
    if (option_pointer != std::end(test_args) && std::next(option_pointer) != std::end(test_args)) {
        return *std::next(option_pointer);
    }
    if (not default_value.has_value()) {
        throw std::runtime_error("Option not found!");
    }
    return default_value.value();
}

inline std::uint32_t get_command_option_uint32(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::uint32_t>& default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stoul(param, nullptr, 0);
}

inline std::int32_t get_command_option_int32(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::uint32_t>& default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stoi(param, nullptr, 0);
}

inline double get_command_option_double(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<double>& default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stod(param, nullptr);
}

inline bool has_command_option(const std::vector<std::string>& test_args, const std::string& option) {
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(test_args), std::end(test_args), option);
    return option_pointer != std::end(test_args);
}

inline std::tuple<std::string, std::vector<std::string>> get_command_option_and_remaining_args(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::string>& default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(remaining_args), std::end(remaining_args), option);
    if (option_pointer != std::end(remaining_args) and (option_pointer + 1) != std::end(remaining_args)) {
        std::string value = *(option_pointer + 1);
        remaining_args.erase(option_pointer, option_pointer + 2);
        return {value, remaining_args};
    }
    if (not default_value.has_value()) {
        throw std::runtime_error("Option not found!");
    }
    return {default_value.value(), remaining_args};
}

inline std::tuple<std::uint32_t, std::vector<std::string>> get_command_option_uint32_and_remaining_args(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::uint32_t>& default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stoul(param, nullptr, 0), remaining_args};
}

inline std::tuple<std::uint64_t, std::vector<std::string>> get_command_option_uint64_and_remaining_args(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::uint64_t>& default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }

    return {std::stoull(param, nullptr, 0), remaining_args};
}

inline std::tuple<std::int32_t, std::vector<std::string>> get_command_option_int32_and_remaining_args(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<std::int32_t>& default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stoi(param, nullptr, 0), remaining_args};
}

inline std::tuple<double, std::vector<std::string>> get_command_option_double_and_remaining_args(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<double>& default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stod(param, nullptr), remaining_args};
}

inline std::tuple<bool, std::vector<std::string>> has_command_option_and_remaining_args(
    const std::vector<std::string>& test_args, const std::string& option) {
    std::vector<std::string> remaining_args = test_args;
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(remaining_args), std::end(remaining_args), option);
    bool value = option_pointer != std::end(remaining_args);
    if (value) {
        remaining_args.erase(option_pointer);
    }
    return {value, remaining_args};
}

template <class T>
inline void split_string_into_vector(
    std::vector<T>& output_vector, std::string input_command_modified, const char* delimiter) {
    if (!input_command_modified.empty()) {
        size_t current_pos = input_command_modified.find(delimiter);
        while (current_pos != std::string::npos) {
            output_vector.push_back(test_args::parse<T>(input_command_modified.substr(0, current_pos)));
            input_command_modified.erase(0, current_pos + 1);
            current_pos = input_command_modified.find(delimiter);
        }
        if (!input_command_modified.empty()) {
            output_vector.push_back(test_args::parse<T>(input_command_modified));
        }
    }
}

template <class T>
T get(
    const std::vector<std::string>& test_args,
    const std::string& option,
    const std::optional<T>& default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        if constexpr (std::is_same_v<T, std::string>) {
            param = get_command_option(test_args, option, default_value.value());
        } else {
            param = get_command_option(test_args, option, std::to_string(default_value.value()));
        }
    } else {
        param = get_command_option(test_args, option);
    }

    return test_args::parse<T>(param);
}

inline void validate_remaining_args(const std::vector<std::string>& remaining_args) {
    if (remaining_args.size() == 1) {
        // Only executable is left, so all good
        return;
    }
    std::cout << "Remaining test_args:" << std::endl;
    for (int i = 1; i < remaining_args.size(); i++) {
        std::cout << "\t" << remaining_args.at(i) << std::endl;
    }
    throw std::runtime_error("Not all test_args were parsed");
}

}  // namespace test_args
