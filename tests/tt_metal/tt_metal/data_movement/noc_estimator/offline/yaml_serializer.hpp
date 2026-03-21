// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include <tt-metalium/experimental/noc_estimator/types.hpp>

namespace tt::tt_metal::experimental::noc_estimator::offline {

bool save_latency_data_to_yaml(const std::map<GroupKey, LatencyData>& data, const std::string& yaml_path);

}  // namespace tt::tt_metal::experimental::noc_estimator::offline
