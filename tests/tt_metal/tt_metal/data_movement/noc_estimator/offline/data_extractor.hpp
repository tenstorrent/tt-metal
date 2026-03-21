// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "csv_reader.hpp"
#include <tt-metalium/experimental/noc_estimator/types.hpp>

namespace tt::tt_metal::experimental::noc_estimator::offline {

// Extract latencies for standard transaction sizes from data points
LatencyData extract_latencies(const std::vector<DataPoint>& points);

}  // namespace tt::tt_metal::experimental::noc_estimator::offline
