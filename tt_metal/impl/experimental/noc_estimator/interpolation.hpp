// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/noc_estimator/types.hpp>
#include <map>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental::noc_estimator {

// Interpolate latency for a given transaction size from LatencyData
double interpolate_latency(const LatencyData& data, const std::vector<uint32_t>& sizes, uint32_t transaction_size);

// Interpolate across numeric fields (num_transactions, num_subordinates)
double interpolate_latency_nd(
    const GroupKey& key,
    uint32_t transaction_size,
    const std::vector<uint32_t>& sizes,
    const std::map<GroupKey, LatencyData>& entries);

// Check if any data exists for the non-numeric fields of the key
bool has_matching_data(const GroupKey& key, const std::map<GroupKey, LatencyData>& entries);

// Find data with parameter relaxation (returns latency, sets relaxed_param name if relaxation was needed)
double find_with_relaxation(
    const GroupKey& key,
    uint32_t transaction_size,
    const std::vector<uint32_t>& sizes,
    const std::map<GroupKey, LatencyData>& entries,
    std::string& relaxed_param);

}  // namespace tt::tt_metal::experimental::noc_estimator
