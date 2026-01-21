// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_extractor.hpp"
#include <map>

namespace tt::noc_estimator::offline {

common::LatencyData extract_latencies(const std::vector<DataPoint>& points) {
    common::LatencyData result;
    const auto& sizes = common::STANDARD_TRANSACTION_SIZES;
    result.latencies.resize(sizes.size(), 0.0);

    // Build map of transaction_size -> latency from data points
    std::map<uint32_t, double> size_to_latency;
    for (const auto& p : points) {
        size_to_latency[p.transaction_size_bytes] = p.latency_cycles;
    }
    if (size_to_latency.empty()) {
        return result;
    }

    // Extract latency for each standard size
    for (size_t i = 0; i < sizes.size(); i++) {
        uint32_t size = sizes[i];
        auto it = size_to_latency.find(size);

        if (it != size_to_latency.end()) {
            result.latencies[i] = it->second;
        } else {
            // Interpolate from neighbors
            auto lower = size_to_latency.lower_bound(size);
            if (lower == size_to_latency.begin()) {
                result.latencies[i] = lower->second;
            } else if (lower == size_to_latency.end()) {
                result.latencies[i] = size_to_latency.rbegin()->second;
            } else {
                auto upper = lower;
                --lower;
                double t = static_cast<double>(size - lower->first) / static_cast<double>(upper->first - lower->first);
                result.latencies[i] = lower->second + t * (upper->second - lower->second);
            }
        }
    }

    return result;
}

}  // namespace tt::noc_estimator::offline
