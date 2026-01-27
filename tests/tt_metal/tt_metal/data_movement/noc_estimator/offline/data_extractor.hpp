// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "csv_reader.hpp"
#include "../common/types.hpp"

namespace tt::noc_estimator::offline {

// Extract latencies for standard transaction sizes from data points
common::LatencyData extract_latencies(const std::vector<DataPoint>& points);

}  // namespace tt::noc_estimator::offline
