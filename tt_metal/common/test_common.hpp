// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <functional>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "common/metal_soc_descriptor.h"

// Needed for TargetDevice enum
#include "common/base.hpp"

inline std::string get_soc_description_file(const tt::ARCH &arch, tt::TargetDevice target_device, string output_dir = "") {
    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    string tt_metal_home;
    if (getenv("TT_METAL_HOME")) {
        tt_metal_home = getenv("TT_METAL_HOME");
    } else {
        tt_metal_home = "./";
    }
    if (tt_metal_home.back() != '/') {
        tt_metal_home += "/";
    }
    if (target_device == tt::TargetDevice::Simulator){
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported");
            case tt::ARCH::GRAYSKULL: throw std::runtime_error("GRAYSKULL arch not supported");
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported");
            case tt::ARCH::WORMHOLE_B0: throw std::runtime_error("WORMHOLE_B0 arch not supported");
            case tt::ARCH::BLACKHOLE: return tt_metal_home + "tt_metal/soc_descriptors/blackhole_simulation_1x2_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    } else {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported"); // will be overwritten in tt_global_state constructor
            case tt::ARCH::GRAYSKULL: return tt_metal_home + "tt_metal/soc_descriptors/grayskull_120_arch.yaml";
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported");
            case tt::ARCH::WORMHOLE_B0: return tt_metal_home + "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml";
            case tt::ARCH::BLACKHOLE: return tt_metal_home + "tt_metal/soc_descriptors/blackhole_140_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    }
    return "";
}

namespace test_args {

template <class T>
constexpr std::false_type always_false{};

template <class T>
T parse(std::string const &s) {
    if constexpr (std::is_same_v<T, std::uint32_t>) {
        return std::stoul(s, 0, 0);
    } else if constexpr (std::is_same_v<T, int>) {
        return std::stoi(s, 0, 0);
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return std::stoull(s, 0, 0);
    } else if constexpr (std::is_same_v<T, bool>) {
        return static_cast<bool>(std::stoi(s, 0, 0));
    } else if constexpr (std::is_same_v<T, std::string>) {
        return s;
    } else {
        static_assert(test_args::always_false<T>, "No specialization for type");
    }
}

inline std::string strip(std::string const &s) {
    std::string whitespace = " \t\n";
    std::size_t start = s.find_first_not_of(whitespace);
    std::size_t end = s.find_last_not_of(whitespace);
    end += bool(end != std::string::npos);
    return s.substr(start, end);
}

inline std::string get_command_option(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::string> &default_value = std::nullopt) {
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(test_args), std::end(test_args), option);
    if (option_pointer != std::end(test_args) and option_pointer++ != std::end(test_args)) {
        return *option_pointer;
    }
    if (not default_value.has_value()) {
        throw std::runtime_error("Option not found!");
    }
    return default_value.value();
}

inline std::uint32_t get_command_option_uint32(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::uint32_t> &default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stoul(param, 0, 0);
}

inline std::int32_t get_command_option_int32(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::uint32_t> &default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stoi(param, 0, 0);
}

inline double get_command_option_double(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<double> &default_value = std::nullopt) {
    std::string param;
    if (default_value.has_value()) {
        param = get_command_option(test_args, option, std::to_string(default_value.value()));
    } else {
        param = get_command_option(test_args, option);
    }
    return std::stod(param, 0);
}

inline bool has_command_option(const std::vector<std::string> &test_args, const std::string &option) {
    std::vector<std::string>::const_iterator option_pointer =
        std::find(std::begin(test_args), std::end(test_args), option);
    return option_pointer != std::end(test_args);
}

inline std::tuple<std::string, std::vector<std::string>> get_command_option_and_remaining_args(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::string> &default_value = std::nullopt) {
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

inline  std::tuple<std::uint32_t, std::vector<std::string>> get_command_option_uint32_and_remaining_args(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::uint32_t> &default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stoul(param, 0, 0), remaining_args};
}

inline  std::tuple<std::uint64_t, std::vector<std::string>> get_command_option_uint64_and_remaining_args(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::uint64_t> &default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }

    return {std::stoull(param, 0, 0), remaining_args};
}

inline std::tuple<std::int32_t, std::vector<std::string>> get_command_option_int32_and_remaining_args(
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<std::int32_t> &default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stoi(param, 0, 0), remaining_args};
}

inline std::tuple<double, std::vector<std::string>> get_command_option_double_and_remaining_args(
    const std::vector<std::string> &test_args, const std::string &option, const std::optional<double> &default_value = std::nullopt) {
    std::vector<std::string> remaining_args = test_args;
    std::string param;
    if (default_value.has_value()) {
        std::tie(param, remaining_args) =
            get_command_option_and_remaining_args(test_args, option, std::to_string(default_value.value()));
    } else {
        std::tie(param, remaining_args) = get_command_option_and_remaining_args(test_args, option);
    }
    return {std::stod(param, 0), remaining_args};
}

inline std::tuple<bool, std::vector<std::string>> has_command_option_and_remaining_args(
    const std::vector<std::string> &test_args, const std::string &option) {
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
    std::vector<T> &output_vector, const std::string input_command, const char *delimiter) {
    std::string input_command_modified = input_command;
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
    const std::vector<std::string> &test_args,
    const std::string &option,
    const std::optional<T> &default_value = std::nullopt) {
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

inline void validate_remaining_args(const std::vector<std::string> &remaining_args) {
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
