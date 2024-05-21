// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <tuple>

namespace ttnn {

namespace core {

struct Config {
    std::string cache_path = "/home/.cache/ttnn";
    std::string model_cache_path = "/home/.cache/ttnn/models";
    std::string tmp_dir = "/tmp/ttnn";
    bool enable_model_cache = false;
    bool enable_fast_runtime_mode = false;
    bool throw_exception_on_fallback = false;
    bool enable_logging = false;
    bool enable_graph_report = false;
    bool enable_detailed_buffer_report = false;
    bool enable_detailed_tensor_report = false;
    bool enable_comparison_mode = false;
    float comparison_mode_pcc = 0.9999;
    std::string root_report_path = "generated/ttnn/reports";
    std::optional<std::string> report_name = std::nullopt;

    static constexpr auto attribute_names = std::make_tuple(
        "cache_path",
        "model_cache_path",
        "tmp_dir",
        "enable_model_cache",
        "enable_fast_runtime_mode",
        "throw_exception_on_fallback",
        "enable_logging",
        "enable_graph_report",
        "enable_detailed_buffer_report",
        "enable_detailed_tensor_report",
        "enable_comparison_mode",
        "comparison_mode_pcc",
        "root_report_path",
        "report_name");

    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->cache_path),
            std::cref(this->model_cache_path),
            std::cref(this->tmp_dir),
            std::cref(this->enable_model_cache),
            std::cref(this->enable_fast_runtime_mode),
            std::cref(this->throw_exception_on_fallback),
            std::cref(this->enable_logging),
            std::cref(this->enable_graph_report),
            std::cref(this->enable_detailed_buffer_report),
            std::cref(this->enable_detailed_tensor_report),
            std::cref(this->enable_comparison_mode),
            std::cref(this->comparison_mode_pcc),
            std::cref(this->root_report_path),
            std::cref(this->report_name));
    }
};

inline Config CONFIG{};

}  // namespace core

using core::CONFIG;
using core::Config;
}  // namespace ttnn
