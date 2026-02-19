// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_estimator.hpp"
#include <cmath>
#include <exception>
#include <iostream>

using namespace tt::tt_metal::noc_estimator;

static int g_failures = 0;

static void expect_near_pct(const char* label, double actual, double expected, double percent) {
    const double tolerance = std::abs(expected) * (percent / 100.0);
    const double delta = std::abs(actual - expected);
    if (delta > tolerance) {
        std::cout << "FAIL: " << label << " expected " << expected << " +/- " << percent << "%, got " << actual
                  << "\n\n";
        g_failures++;
    }
}

int main() {
    // Test 1: Blackhole multicast
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 1,
            .num_transactions_per_barrier = 1,
            .transaction_size_bytes = 64,
            .num_subordinates = 4,
            .loopback = true};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 1 - Blackhole multicast (64 bytes, 1 txn, 4 subordinates):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 1 latency", result.latency_cycles, 396.0, 2.0);
        // expect_gt_zero("Test 1 bandwidth", result.bandwidth_bytes_per_cycle);
    }

    // Test 2: Wormhole unicast to multiple subordinates
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::WORMHOLE_B0,
            .num_transactions = 100,
            .num_transactions_per_barrier = 100,
            .transaction_size_bytes = 2048,
            .num_subordinates = 25};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 2 - Wormhole unicast (2048 bytes, 100 txns, 25 subordinates):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 2 latency", result.latency_cycles, 6995.0, 2.0);
        // expect_gt_zero("Test 2 bandwidth", result.bandwidth_bytes_per_cycle);
    }

    // Test 3: Interpolated value
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 16,
            .num_transactions_per_barrier = 16,
            .transaction_size_bytes = 7500,  // Interpolated
            .num_subordinates = 4};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 3 - Interpolated (7500 bytes, 16 txns, 4 subordinates):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 3 latency", result.latency_cycles, 6900.0, 2.0);
    }

    // Test 4: Compare MULTICAST_LINKED vs MULTICAST
    {
        NocEstimatorParams params_unlinked{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .num_transactions_per_barrier = 4,
            .transaction_size_bytes = 2048,
            .num_subordinates = 4};

        NocEstimatorParams params_linked{
            .mechanism = NocMechanism::MULTICAST_LINKED,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .num_transactions_per_barrier = 4,
            .transaction_size_bytes = 2048,
            .num_subordinates = 4};

        NocEstimate result_unlinked = estimate_noc_performance(params_unlinked);
        NocEstimate result_linked = estimate_noc_performance(params_linked);

        std::cout << "Test 4 - MULTICAST_LINKED vs MULTICAST comparison (2048 bytes, 4 txns):\n";
        std::cout << "  MULTICAST latency: " << result_unlinked.latency_cycles << " cycles\n";
        std::cout << "  MULTICAST_LINKED latency: " << result_linked.latency_cycles << " cycles\n";
        std::cout << "  Difference: " << (result_unlinked.latency_cycles - result_linked.latency_cycles)
                  << " cycles\n\n";
        expect_near_pct("Test 4 unlinked latency", result_unlinked.latency_cycles, 1577.0, 2.0);
        expect_near_pct("Test 4 linked latency", result_linked.latency_cycles, 749.0, 2.0);
    }

    // Test 5: Numeric interpolation on num_transactions
    // Data has 1, 4, 16, 64, 100 - request 50 (between 16 and 64)
    {
        std::cout << "=== Interpolation tests ===\n\n";

        // First get bounds to compare
        NocEstimatorParams params_lower{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 16,
            .num_transactions_per_barrier = 16,
            .transaction_size_bytes = 2048,
            .num_subordinates = 4};

        NocEstimatorParams params_upper{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 64,
            .num_transactions_per_barrier = 64,
            .transaction_size_bytes = 2048,
            .num_subordinates = 4};

        NocEstimatorParams params_interp{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 50,  // Between 16 and 64
            .num_transactions_per_barrier = 50,
            .transaction_size_bytes = 2048,
            .num_subordinates = 4};

        NocEstimate result_lower = estimate_noc_performance(params_lower);
        NocEstimate result_upper = estimate_noc_performance(params_upper);
        NocEstimate result_interp = estimate_noc_performance(params_interp);

        std::cout << "Test 5 - Interpolation on num_transactions:\n";
        std::cout << "  num_txns=16:  " << result_lower.latency_cycles << " cycles\n";
        std::cout << "  num_txns=50:  " << result_interp.latency_cycles << " cycles (interpolated)\n";
        std::cout << "  num_txns=64:  " << result_upper.latency_cycles << " cycles\n";
        expect_near_pct("Test 5 lower bound latency", result_lower.latency_cycles, 5273.0, 2.0);
        expect_near_pct("Test 5 upper bound latency", result_upper.latency_cycles, 20100.0, 2.0);

        // Verify interpolated value is between bounds
        bool in_range = (result_interp.latency_cycles >= result_lower.latency_cycles &&
                         result_interp.latency_cycles <= result_upper.latency_cycles) ||
                        (result_interp.latency_cycles <= result_lower.latency_cycles &&
                         result_interp.latency_cycles >= result_upper.latency_cycles);
        std::cout << "  In range: " << (in_range ? "YES" : "NO") << "\n\n";
    }

    // Test 6: 2D interpolation (both num_transactions AND num_subordinates)
    {
        std::cout << "Test 6 - 2D Interpolation (num_transactions=50, num_subordinates=15):\n";

        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 50,  // Interpolated (between 16 and 64)
            .num_transactions_per_barrier = 50,
            .transaction_size_bytes = 2048,
            .num_subordinates = 15};  // Interpolated (between 9 and 25)

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "  Latency: " << result.latency_cycles << " cycles (bilinear interpolation)\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 6 latency", result.latency_cycles, 17270.0, 2.0);
    }

    // Test 7: Blackhole unicast with limited buffer size
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ONE,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 128,
            .num_transactions_per_barrier = 64,
            .transaction_size_bytes = 2048};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 7 - Blackhole unicast (2048 bytes, 128 txns (64 per barrier)):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 7 latency", result.latency_cycles, 5090.0, 2.0);
    }

    // Test 8: Blackhole unicast same_axis comparison (same_axis=false vs same_axis=true)
    {
        NocEstimatorParams params_diff_axis{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ONE,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .num_transactions_per_barrier = 4,
            .transaction_size_bytes = 2048,
            .same_axis = false};

        NocEstimatorParams params_same_axis{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ONE,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .num_transactions_per_barrier = 4,
            .transaction_size_bytes = 2048,
            .same_axis = true};

        NocEstimate result_diff_axis = estimate_noc_performance(params_diff_axis);
        NocEstimate result_same_axis = estimate_noc_performance(params_same_axis);

        std::cout << "Test 8 - Blackhole unicast same_axis comparison (2048 bytes, 4 txns):\n";
        std::cout << "  same_axis=false latency: " << result_diff_axis.latency_cycles << " cycles\n";
        std::cout << "  same_axis=true  latency: " << result_same_axis.latency_cycles << " cycles\n";
        std::cout << "  Difference: " << (result_diff_axis.latency_cycles - result_same_axis.latency_cycles)
                  << " cycles\n\n";
        expect_near_pct("Test 8 diff_axis latency", result_diff_axis.latency_cycles, 495.0, 2.0);
        expect_near_pct("Test 8 same_axis latency", result_same_axis.latency_cycles, 349.0, 2.0);
    }

    // Test 9: DRAM one core read from one DRAM buffer
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_FROM_ONE,
            .memory = MemoryType::DRAM,
            .arch = Architecture::WORMHOLE_B0,
            .num_transactions = 1,
            .transaction_size_bytes = 2048};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 9 - DRAM one core read from one DRAM buffer (2048 bytes, 1 txn):\n";

        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_bytes_per_cycle << " bytes/cycle\n\n";
        expect_near_pct("Test 9 latency", result.latency_cycles, 470.0, 2.0);
    }

    if (g_failures > 0) {
        std::cout << "Total failures: " << g_failures << "\n";
    }
    return g_failures == 0 ? 0 : 1;
}
