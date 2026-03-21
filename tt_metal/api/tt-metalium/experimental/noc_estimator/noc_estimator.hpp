// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/experimental/noc_estimator/types.hpp>

namespace tt::tt_metal::experimental::noc_estimator {

struct NocEstimatorParams {
    NocMechanism mechanism = NocMechanism::UNICAST;
    NocPattern pattern = NocPattern::ONE_TO_ONE;
    MemoryType memory = MemoryType::L1;
    Architecture arch = Architecture::WORMHOLE_B0;
    uint32_t num_transactions = 64;
    uint32_t num_transactions_per_barrier = 1;
    uint32_t transaction_size_bytes = 512;
    uint32_t num_subordinates = 1;
    bool same_axis = false;
    bool stateful = false;
    bool loopback = false;
    uint32_t noc_index = 0;
};

// Estimation result
struct NocEstimate {
    double bandwidth_bytes_per_cycle = 0.0;
    double latency_cycles = 0.0;
};

// Main estimator function
NocEstimate estimate_noc_performance(const NocEstimatorParams& params);

// Convenience functions
double estimate_noc_bandwidth(const NocEstimatorParams& params);
double estimate_noc_latency(const NocEstimatorParams& params);

}  // namespace tt::tt_metal::experimental::noc_estimator
