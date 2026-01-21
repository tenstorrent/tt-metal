// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include <vector>
#include "../common/types.hpp"

namespace tt::noc_estimator::loader {

struct LoadedData {
    std::vector<uint32_t> transaction_sizes;
    std::map<common::GroupKey, common::LatencyData> entries;
};

LoadedData load_latency_data_from_yaml(const std::string& yaml_path);

}  // namespace tt::noc_estimator::loader
