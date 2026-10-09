// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal::detail {

// logs_dir is expected to end with a trailing '/' (see RunTimeOptions::get_logs_dir()).
inline std::string metal_reports_dir(const std::string& logs_dir) { return logs_dir + "generated/reports/"; }

}  // namespace tt::tt_metal::detail
