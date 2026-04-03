// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host driver for NOC overhead microbenchmark.
// Deploys noc_overhead_ubench kernel on a tensix or eth core and reads back per-operation cycle costs.
//
// Usage: test_noc_overhead [num_iterations] [use_eth] [eth_core_idx]
//   num_iterations: number of iterations per test (default 1000)
//   use_eth:        0 = tensix core (0,0), 1 = first active eth core (default 0)
//   eth_core_idx:   which active eth core to use when use_eth=1 (default 0)

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

using namespace tt;

static const char* TEST_NAMES[] = {
    "fence (invalidate_l1_cache)",
    "wall clock read (TS overhead)",
    "NOC_STATUS_READ: NIU_MST_RD_RESP_RECEIVED",
    "NOC_STATUS_READ: NIU_MST_NONPOSTED_WR_REQ_SENT",
    "NOC_STATUS_READ: NIU_MST_WR_ACK_RECEIVED",
    "NOC_STATUS_READ: NIU_MST_ATOMIC_RESP_RECEIVED",
    "NOC_STATUS_READ: NIU_MST_POSTED_WR_REQ_SENT",
    "L1 read: noc_reads_num_issued",
    "ncrisc_noc_reads_flushed (MMIO+L1+cmp)",
    "ncrisc_noc_nonposted_atomics_flushed",
    "noc_async_full_barrier (nothing outstanding)",
    "noc_semaphore_inc (issue only, to self)",
    "noc_semaphore_inc + flush (full round-trip to self)",
    "volatile L1 read (sem poll, single read)",
};
constexpr uint32_t NUM_TESTS = 14;

void print_results(const std::vector<uint32_t>& results, uint32_t num_iterations, const std::string& core_label) {
    log_info(tt::LogTest, "\n=== Results for {} ===", core_label);
    log_info(tt::LogTest, "{:>4} {:>8} {:>8} {:>8} {:>10}  {}", "ID", "Min", "Max", "Mean", "Sum", "Test");
    log_info(tt::LogTest, "{}", std::string(80, '-'));

    for (uint32_t t = 0; t < NUM_TESTS; t++) {
        uint32_t test_id = results[t * 5 + 0];
        uint32_t min_val = results[t * 5 + 1];
        uint32_t max_val = results[t * 5 + 2];
        uint32_t sum_lo = results[t * 5 + 3];
        uint32_t sum_hi = results[t * 5 + 4];
        uint64_t sum = (static_cast<uint64_t>(sum_hi) << 32) | sum_lo;
        double mean = static_cast<double>(sum) / num_iterations;

        log_info(
            tt::LogTest, "{:>4} {:>8} {:>8} {:>8.1f} {:>10}  {}", test_id, min_val, max_val, mean, sum, TEST_NAMES[t]);
    }

    uint64_t ts_sum = (static_cast<uint64_t>(results[1 * 5 + 4]) << 32) | results[1 * 5 + 3];
    double ts_overhead = static_cast<double>(ts_sum) / num_iterations;

    log_info(tt::LogTest, "\n--- Adjusted costs (subtracting TS overhead of {:.1f} cycles) ---", ts_overhead);
    log_info(tt::LogTest, "{:>4} {:>8}  {}", "ID", "Adj.Mean", "Test");
    log_info(tt::LogTest, "{}", std::string(60, '-'));
    for (uint32_t t = 0; t < NUM_TESTS; t++) {
        uint32_t sum_lo = results[t * 5 + 3];
        uint32_t sum_hi = results[t * 5 + 4];
        uint64_t sum = (static_cast<uint64_t>(sum_hi) << 32) | sum_lo;
        double mean = static_cast<double>(sum) / num_iterations;
        double adjusted = mean - ts_overhead;

        log_info(tt::LogTest, "{:>4} {:>8.1f}  {}", t, adjusted, TEST_NAMES[t]);
    }
}

int main(int argc, char** argv) {
    uint32_t num_iterations = (argc > 1) ? std::stoi(argv[1]) : 1000;
    uint32_t use_eth = (argc > 2) ? std::stoi(argv[2]) : 0;
    uint32_t eth_core_idx = (argc > 3) ? std::stoi(argv[3]) : 0;

    // For eth mode, open all devices so active eth cores can complete FW handshake
    size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    std::vector<tt_metal::IDevice*> all_devices;
    if (use_eth && num_devices > 1) {
        for (size_t i = 0; i < num_devices; i++) {
            all_devices.push_back(tt_metal::CreateDevice(i));
        }
    } else {
        all_devices.push_back(tt_metal::CreateDevice(0));
    }
    tt_metal::IDevice* device = all_devices[0];

    constexpr uint32_t result_words = NUM_TESTS * 5;

    if (!use_eth) {
        // Tensix mode
        CoreCoord logical_core(0, 0);
        auto virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
        log_info(
            tt::LogTest,
            "NOC Overhead Benchmark: {} iterations on Tensix (0,0) virtual=({},{})",
            num_iterations,
            virtual_core.x,
            virtual_core.y);

        constexpr uint32_t l1_base = 150 * 1024;
        uint32_t sem_addr = (l1_base + 15) & ~15u;
        uint32_t ts_buf_addr = (sem_addr + 16 + 15) & ~15u;

        tt_metal::Program program = tt_metal::CreateProgram();
        auto kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_overhead_ubench.cpp",
            logical_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

        tt_metal::SetRuntimeArgs(
            program, kernel, logical_core, {ts_buf_addr, num_iterations, virtual_core.x, virtual_core.y, sem_addr});

        tt_metal::detail::CompileProgram(device, program);
        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> results(result_words, 0);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            results.data(), result_words * sizeof(uint32_t), tt_cxy_pair(device->id(), virtual_core), ts_buf_addr);

        print_results(
            results, num_iterations, fmt::format("Tensix (0,0) virt=({},{})", virtual_core.x, virtual_core.y));

    } else {
        // Eth mode
        bool slow_dispatch = (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr);
        auto active_eth_cores = tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
            device->id(), !slow_dispatch);
        TT_FATAL(!active_eth_cores.empty(), "No active ethernet cores found");

        std::vector<CoreCoord> eth_cores(active_eth_cores.begin(), active_eth_cores.end());
        TT_FATAL(
            eth_core_idx < eth_cores.size(), "eth_core_idx {} >= {} active eth cores", eth_core_idx, eth_cores.size());

        CoreCoord eth_logical = eth_cores[eth_core_idx];
        auto eth_virtual = device->virtual_core_from_logical_core(eth_logical, CoreType::ETH);
        log_info(
            tt::LogTest,
            "NOC Overhead Benchmark: {} iterations on ETH({},{}) virtual=({},{})",
            num_iterations,
            eth_logical.x,
            eth_logical.y,
            eth_virtual.x,
            eth_virtual.y);

        uint32_t eth_unreserved = tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED);
        uint32_t sem_addr = (eth_unreserved + 15) & ~15u;
        uint32_t ts_buf_addr = (sem_addr + 16 + 15) & ~15u;

        tt_metal::Program program = tt_metal::CreateProgram();
        auto kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_overhead_ubench.cpp",
            eth_logical,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = {}});

        tt_metal::SetRuntimeArgs(
            program, kernel, eth_logical, {ts_buf_addr, num_iterations, eth_virtual.x, eth_virtual.y, sem_addr});

        tt_metal::detail::CompileProgram(device, program);
        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> results(result_words, 0);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            results.data(), result_words * sizeof(uint32_t), tt_cxy_pair(device->id(), eth_virtual), ts_buf_addr);

        print_results(
            results,
            num_iterations,
            fmt::format("ETH({},{}) virt=({},{})", eth_logical.x, eth_logical.y, eth_virtual.x, eth_virtual.y));
    }

    for (auto* d : all_devices) {
        tt_metal::CloseDevice(d);
    }
    return 0;
}
