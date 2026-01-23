// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_estimator.hpp"
#include "loader/yaml_loader.hpp"
#include "common/types.hpp"
#include "common/interpolation.hpp"
#include <map>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tt::noc_estimator {

static constexpr const char* DEFAULT_YAML_PATH =
    "tests/tt_metal/tt_metal/data_movement/noc_estimator/noc_latencies.yaml";

static std::map<common::GroupKey, common::LatencyData> g_entries;
static std::vector<uint32_t> g_transaction_sizes;
static std::once_flag g_init_once;

static bool initialize_from_yaml(const std::string& yaml_path) {
    auto loaded = loader::load_latency_data_from_yaml(yaml_path);
    g_entries = std::move(loaded.entries);
    g_transaction_sizes = std::move(loaded.transaction_sizes);
    return !g_entries.empty();
}

NocEstimate estimate_noc_performance(const NocEstimatorParams& params) {
    NocEstimate result{};

    // Auto-initialize on first call
    std::call_once(g_init_once, []() {
        std::cout << "Auto-initializing from default YAML: " << DEFAULT_YAML_PATH << "\n";
        if (!initialize_from_yaml(DEFAULT_YAML_PATH)) {
            throw std::runtime_error("Failed to auto-load latency data. Generate YAML first.");
        }
    });

    if (g_entries.empty()) {
        throw std::runtime_error("No latency data loaded.");
    }

    // Build group key with all parameters
    common::GroupKey key{
        .mechanism = static_cast<common::NocMechanism>(params.mechanism),
        .pattern = static_cast<common::NocPattern>(params.pattern),
        .memory = static_cast<common::MemoryType>(params.memory),
        .arch = static_cast<common::Architecture>(params.arch),
        .num_transactions = params.num_transactions_per_barrier,
        .num_subordinates = params.num_subordinates,
        .same_axis = params.same_axis,
        .linked = params.linked};

    double num_transactions_d = static_cast<double>(params.num_transactions);

    // Calculate the number of iterations needed to process all transactions
    uint32_t num_iterations = 1;

    if (params.num_transactions > params.num_transactions_per_barrier) {
        num_iterations =
            (params.num_transactions + params.num_transactions_per_barrier - 1) / params.num_transactions_per_barrier;
    }

    auto it = g_entries.find(key);
    if (it != g_entries.end()) {
        result.latency_cycles =
            common::interpolate_latency(it->second, g_transaction_sizes, params.transaction_size_bytes);
    } else if (common::has_matching_data(key, g_entries)) {
        result.latency_cycles =
            common::interpolate_latency_nd(key, params.transaction_size_bytes, g_transaction_sizes, g_entries);
    } else {
        // No exact match - try relaxation
        std::string relaxed_param;
        result.latency_cycles = common::find_with_relaxation(
            key, params.transaction_size_bytes, g_transaction_sizes, g_entries, relaxed_param);
        if (result.latency_cycles > 0) {
            std::cerr << "Warning: Used fallback (relaxed " << relaxed_param << ")\n";
        } else {
            throw std::runtime_error("No match found for the given parameters, even with relaxation.");
        }
    }

    result.latency_cycles *= num_iterations;

    // Calculate bandwidth from latency
    if (result.latency_cycles > 0) {
        result.bandwidth_bytes_per_cycle =
            static_cast<double>(params.transaction_size_bytes) * num_transactions_d / result.latency_cycles;
    }

    return result;
}

double estimate_noc_bandwidth(const NocEstimatorParams& params) {
    return estimate_noc_performance(params).bandwidth_bytes_per_cycle;
}

double estimate_noc_latency(const NocEstimatorParams& params) {
    return estimate_noc_performance(params).latency_cycles;
}

}  // namespace tt::noc_estimator
