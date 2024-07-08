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
    bool enable_fast_runtime_mode = true;
    bool throw_exception_on_fallback = false;
    bool enable_logging = false;
    bool enable_graph_report = false;
    bool enable_detailed_buffer_report = false;
    bool enable_detailed_tensor_report = false;
    bool enable_comparison_mode = false;
    float comparison_mode_pcc = 0.9999;
    std::string root_report_path = "generated/ttnn/reports";
    std::optional<std::string> report_name = std::nullopt;
};

inline Config CONFIG{};

}  // namespace core

using core::CONFIG;
using core::Config;
}  // namespace ttnn
