// Test: Noc Estimator with various inputs

#include "noc_estimator.hpp"
#include <iostream>

using namespace tt::noc_estimator;

int main() {
    // Test 1: Blackhole multicast
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 1,
            .transaction_size_bytes = 64,
            .num_peers = 4};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 1 - Blackhole multicast (64 bytes, 1 txn, 4 peers):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles (expected ~396)\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
    }

    // Test 2: Wormhole unicast
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::WORMHOLE_B0,
            .num_transactions = 100,
            .transaction_size_bytes = 2048,
            .num_peers = 25};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 2 - Wormhole unicast (2048 bytes, 100 txns, 25 peers):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles (expected ~6985)\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
    }

    // Test 3: Interpolated value
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 16,
            .transaction_size_bytes = 7500,  // Interpolated
            .num_peers = 4};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 3 - Interpolated (7500 bytes, 16 txns, 4 peers):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
    }

    // Test 4: Compare linked vs non-linked
    {
        NocEstimatorParams params_unlinked{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .transaction_size_bytes = 2048,
            .num_peers = 4,
            .linked = false};

        NocEstimatorParams params_linked{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .transaction_size_bytes = 2048,
            .num_peers = 4,
            .linked = true};

        NocEstimate result_unlinked = estimate_noc_performance(params_unlinked);
        NocEstimate result_linked = estimate_noc_performance(params_linked);

        std::cout << "Test 4 - Linked vs Unlinked comparison (2048 bytes, 4 txns):\n";
        std::cout << "  Unlinked latency: " << result_unlinked.latency_cycles << " cycles (expected ~644)\n";
        std::cout << "  Linked latency: " << result_linked.latency_cycles << " cycles (expected ~542)\n";
        std::cout << "  Difference: " << (result_unlinked.latency_cycles - result_linked.latency_cycles)
                  << " cycles\n\n";
    }

    // Test 5: Numeric interpolation on num_transactions
    // Data has 1, 4, 16, 64, 100 - request 50 (between 16 and 64)
    {
        std::cout << "=== INTERPOLATION TESTS ===\n\n";

        // First get bounds to compare
        NocEstimatorParams params_lower{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 16,
            .transaction_size_bytes = 2048,
            .num_peers = 4};

        NocEstimatorParams params_upper{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 64,
            .transaction_size_bytes = 2048,
            .num_peers = 4};

        NocEstimatorParams params_interp{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 50,  // Between 16 and 64
            .transaction_size_bytes = 2048,
            .num_peers = 4};

        NocEstimate result_lower = estimate_noc_performance(params_lower);
        NocEstimate result_upper = estimate_noc_performance(params_upper);
        NocEstimate result_interp = estimate_noc_performance(params_interp);

        std::cout << "Test 5 - Interpolation on num_transactions:\n";
        std::cout << "  num_txns=16:  " << result_lower.latency_cycles << " cycles (expected ~1497)\n";
        std::cout << "  num_txns=50:  " << result_interp.latency_cycles << " cycles (interpolated)\n";
        std::cout << "  num_txns=64:  " << result_upper.latency_cycles << " cycles (expected ~4871)\n";

        // Verify interpolated value is between bounds
        bool in_range = (result_interp.latency_cycles >= result_lower.latency_cycles &&
                         result_interp.latency_cycles <= result_upper.latency_cycles) ||
                        (result_interp.latency_cycles <= result_lower.latency_cycles &&
                         result_interp.latency_cycles >= result_upper.latency_cycles);
        std::cout << "  In range: " << (in_range ? "YES" : "NO") << "\n\n";
    }

    // Test 6: 2D interpolation (both num_transactions AND num_peers)
    {
        std::cout << "Test 6 - 2D Interpolation (num_transactions=50, num_peers=15):\n";

        NocEstimatorParams params{
            .mechanism = NocMechanism::MULTICAST,
            .pattern = NocPattern::ONE_TO_ALL,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 50,  // Interpolated (between 16 and 64)
            .transaction_size_bytes = 2048,
            .num_peers = 15};  // Interpolated (between 4 and 25)

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "  Latency: " << result.latency_cycles << " cycles (bilinear interpolation)\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
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
        std::cout << "  Latency: " << result.latency_cycles << " cycles (expected ~5102)\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
    }

    std::cout << "=== RELAXATION TESTS ===\n\n";

    // Test 8: Relaxation - Blackhole unicast same axis
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_TO_ONE,
            .arch = Architecture::BLACKHOLE,
            .num_transactions = 4,
            .transaction_size_bytes = 2048,
            .same_axis = true};

        NocEstimate result = estimate_noc_performance(params);
        std::cout << "Test 8 - Blackhole unicast (2048 bytes, 4 txn, same axis):\n";
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gbps << " bytes/cycle\n\n";
    }

    // Test 9: Relaxation - ONE_FROM_ONE with DRAM
    {
        NocEstimatorParams params{
            .mechanism = NocMechanism::UNICAST,
            .pattern = NocPattern::ONE_FROM_ONE,
            .memory = MemoryType::DRAM,
            .arch = Architecture::WORMHOLE_B0,
            .num_transactions = 1,
            .transaction_size_bytes = 2048};

        std::cout << "Test 9 - Relaxation: ONE_FROM_ONE + DRAM (no exact match):\n";
        NocEstimate result = estimate_noc_performance(params);
        std::cout << "  Latency: " << result.latency_cycles << " cycles\n";
        std::cout << "  (Should raise an error since we cannot relax memory access)\n\n";
    }

    return 0;
}
