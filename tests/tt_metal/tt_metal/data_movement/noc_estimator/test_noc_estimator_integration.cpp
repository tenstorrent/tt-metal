// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests that validate the specific NOC estimator parameter configurations
// used by the integration sites in ttnn (common.cpp, conv2d, OpPerformanceModel).

#include <tt-metalium/experimental/noc_estimator/noc_estimator.hpp>
#include <cmath>
#include <iostream>
#include <string>

using namespace tt::tt_metal::experimental::noc_estimator;

static int g_failures = 0;

static void expect_positive(const char* label, double value) {
    if (value <= 0.0) {
        std::cout << "FAIL: " << label << " expected positive, got " << value << "\n";
        g_failures++;
    } else {
        std::cout << "PASS: " << label << " = " << value << "\n";
    }
}

static void expect_in_range(const char* label, double value, double min_val, double max_val) {
    if (value < min_val || value > max_val) {
        std::cout << "FAIL: " << label << " expected [" << min_val << ", " << max_val << "], got " << value << "\n";
        g_failures++;
    } else {
        std::cout << "PASS: " << label << " = " << value << " (in [" << min_val << ", " << max_val << "])\n";
    }
}

// common.cpp: DRAM read (get_cycles_for_transaction_size with is_dram=true, is_read=true)
static void test_dram_read() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_FROM_ONE;
        params.memory = MemoryType::DRAM_INTERLEAVED;
        params.num_transactions = 64;

        for (uint32_t size : {64u, 512u, 2048u, 65536u}) {
            params.transaction_size_bytes = size;
            auto est = estimate_noc_performance(params);
            float freq = (arch == Architecture::BLACKHOLE) ? 1.2f : 1.0f;
            float bw_gbps = est.bandwidth_bytes_per_cycle * freq;
            std::string label =
                std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " DRAM read " + std::to_string(size) + "B";
            expect_positive(label.c_str(), est.latency_cycles);
            expect_in_range((label + " BW(GB/s)").c_str(), bw_gbps, 0.1, 100.0);
        }
    }
}

// common.cpp: DRAM write (is_dram=true, is_read=false)
static void test_dram_write() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_TO_ONE;
        params.memory = MemoryType::DRAM_INTERLEAVED;
        params.num_transactions = 64;
        params.transaction_size_bytes = 2048;

        auto est = estimate_noc_performance(params);
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " DRAM write";
        expect_positive(label.c_str(), est.latency_cycles);
        expect_positive((label + " BW").c_str(), est.bandwidth_bytes_per_cycle);
    }
}

// common.cpp: L1 local (is_local=true, loopback=true)
static void test_l1_local() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_TO_ONE;
        params.memory = MemoryType::L1;
        params.loopback = true;
        params.num_transactions = 64;
        params.transaction_size_bytes = 1024;

        auto est = estimate_noc_performance(params);
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " L1 local";
        expect_positive(label.c_str(), est.latency_cycles);
        float freq = (arch == Architecture::BLACKHOLE) ? 1.2f : 1.0f;
        float bw_gbps = est.bandwidth_bytes_per_cycle * freq;
        expect_in_range((label + " BW(GB/s)").c_str(), bw_gbps, 0.5, 100.0);
    }
}

// common.cpp: L1 remote read (is_local=false, is_read=true)
static void test_l1_remote_read() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_FROM_ONE;
        params.memory = MemoryType::L1;
        params.num_transactions = 64;
        params.transaction_size_bytes = 512;

        auto est = estimate_noc_performance(params);
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " L1 remote read";
        expect_positive(label.c_str(), est.latency_cycles);
    }
}

// common.cpp: L1 remote write (is_local=false, is_read=false)
static void test_l1_remote_write() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_TO_ONE;
        params.memory = MemoryType::L1;
        params.num_transactions = 64;
        params.transaction_size_bytes = 512;

        auto est = estimate_noc_performance(params);
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " L1 remote write";
        expect_positive(label.c_str(), est.latency_cycles);
    }
}

// conv2d: get_all_dram_noc_transfer_rate (ONE_FROM_ALL, DRAM_INTERLEAVED)
static void test_conv2d_dram_all() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_FROM_ALL;
        params.memory = MemoryType::DRAM_INTERLEAVED;
        params.num_transactions = 64;
        params.transaction_size_bytes = 2048;

        auto est = estimate_noc_performance(params);
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " conv2d DRAM all";
        expect_positive(label.c_str(), est.latency_cycles);
    }
}

// conv2d: get_mcast_many_l1_linked_noc_transfer_rate (MULTICAST_LINKED, ONE_TO_ALL, L1)
static void test_conv2d_mcast_linked() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::MULTICAST_LINKED;
        params.pattern = NocPattern::ONE_TO_ALL;
        params.memory = MemoryType::L1;
        params.num_transactions = 64;
        params.transaction_size_bytes = 4096;
        params.num_subordinates = 8;

        auto est = estimate_noc_performance(params);
        float freq = (arch == Architecture::BLACKHOLE) ? 1.2f : 1.0f;
        float bw_gbps = est.bandwidth_bytes_per_cycle * freq;
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " conv2d mcast linked";
        expect_positive(label.c_str(), est.latency_cycles);
        expect_in_range((label + " BW(GB/s)").c_str(), bw_gbps, 0.1, 80.0);
    }
}

// OpPerformanceModelGeneral: peak DRAM BW at 65536B
static void test_peak_dram_bw() {
    for (auto arch : {Architecture::WORMHOLE_B0, Architecture::BLACKHOLE}) {
        NocEstimatorParams params;
        params.arch = arch;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_FROM_ONE;
        params.memory = MemoryType::DRAM_INTERLEAVED;
        params.transaction_size_bytes = 65536;
        params.num_transactions = 64;

        auto est = estimate_noc_performance(params);
        float freq = (arch == Architecture::BLACKHOLE) ? 1.2f : 1.0f;
        float peak_gbps = est.bandwidth_bytes_per_cycle * freq;
        float max_theoretical = (arch == Architecture::BLACKHOLE) ? 512.0f : 258.0f;
        std::string label = std::string(arch == Architecture::BLACKHOLE ? "BH" : "WH") + " peak DRAM BW";
        expect_in_range(label.c_str(), peak_gbps, 5.0, max_theoretical);
    }
}

// BW should generally increase with transaction size (monotonicity)
static void test_bw_monotonicity() {
    float prev_bw = 0.0;
    uint32_t sizes[] = {64, 256, 1024, 4096, 16384, 65536};
    bool monotonic = true;

    for (auto size : sizes) {
        NocEstimatorParams params;
        params.arch = Architecture::WORMHOLE_B0;
        params.mechanism = NocMechanism::UNICAST;
        params.pattern = NocPattern::ONE_TO_ONE;
        params.memory = MemoryType::DRAM_INTERLEAVED;
        params.num_transactions = 64;
        params.transaction_size_bytes = size;

        auto est = estimate_noc_performance(params);
        if (est.bandwidth_bytes_per_cycle < prev_bw * 0.95) {
            std::cout << "FAIL: BW not monotonic at size " << size << ": " << est.bandwidth_bytes_per_cycle << " < "
                      << prev_bw << "\n";
            monotonic = false;
            g_failures++;
        }
        prev_bw = est.bandwidth_bytes_per_cycle;
    }
    if (monotonic) {
        std::cout << "PASS: BW monotonically increases with transaction size\n";
    }
}

int main() {
    std::cout << "=== NOC Estimator Integration Tests ===\n\n";

    std::cout << "--- common.cpp transfer types ---\n";
    test_dram_read();
    test_dram_write();
    test_l1_local();
    test_l1_remote_read();
    test_l1_remote_write();

    std::cout << "\n--- conv2d transfer types ---\n";
    test_conv2d_dram_all();
    test_conv2d_mcast_linked();

    std::cout << "\n--- OpPerformanceModelGeneral ---\n";
    test_peak_dram_bw();

    std::cout << "\n--- Properties ---\n";
    test_bw_monotonicity();

    std::cout << "\n"
              << (g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << " (failures: " << g_failures << ")\n";
    return g_failures > 0 ? 1 : 0;
}
