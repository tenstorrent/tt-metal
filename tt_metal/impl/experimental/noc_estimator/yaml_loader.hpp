// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include <vector>
#include <tt-metalium/experimental/noc_estimator/types.hpp>

namespace tt::tt_metal::experimental::noc_estimator {

struct LoadedData {
    std::vector<uint32_t> transaction_sizes;
    std::map<GroupKey, LatencyData> entries;
};

LoadedData load_latency_data_from_yaml(const std::string& yaml_path);

}  // namespace tt::tt_metal::experimental::noc_estimator
