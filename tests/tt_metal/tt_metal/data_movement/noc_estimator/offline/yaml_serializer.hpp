// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include "../common/types.hpp"

namespace tt::noc_estimator::offline {

bool save_latency_data_to_yaml(
    const std::map<common::GroupKey, common::LatencyData>& data, const std::string& yaml_path);

}  // namespace tt::noc_estimator::offline
