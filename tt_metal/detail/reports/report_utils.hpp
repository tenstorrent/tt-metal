// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::detail {

inline const std::string& get_reports_dir() {
    static std::string outpath;
    if (outpath.empty()) {
        outpath = tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir() + "/generated/reports/";
    }
    return outpath;
}

inline const std::string& metal_reports_dir() {
    static const std::string reports_path = get_reports_dir();
    return reports_path;
}

}  // namespace tt::tt_metal::detail
