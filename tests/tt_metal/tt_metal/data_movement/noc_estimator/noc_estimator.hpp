// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/types.hpp"

namespace tt::noc_estimator {

// Use common enums to avoid duplication
using Architecture = common::Architecture;
using NocMechanism = common::NocMechanism;
using MemoryType = common::MemoryType;
using NocPattern = common::NocPattern;

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
    bool linked = false;
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

}  // namespace tt::noc_estimator
