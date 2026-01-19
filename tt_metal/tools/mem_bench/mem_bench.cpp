// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include "context.hpp"
#include "device.hpp"
#include "device_utils.hpp"
#include "host_utils.hpp"
#include "tt-metalium/program.hpp"
#include "tt_metal/impl/dispatch/util/size_literals.hpp"
#include "vector_aligned.hpp"
#include "work_thread.hpp"
#include <llrt/tt_cluster.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::tools::mem_bench;

// Global variable to store user-specified device ID
std::optional<int> g_user_device_id;

// Read L1 counters (cycles, bytes rd, bytes wr) and increment test_results
void read_inc_data_from_cores(const Context& ctx, IDevice* device, const CoreRange& cores, TestResult& test_results) {
    auto dev_cycles = read_cores(device, cores, ctx.device_address.cycles);
    auto dev_bytes_read = read_cores(device, cores, ctx.device_address.rd_bytes);
    auto dev_bytes_written = read_cores(device, cores, ctx.device_address.wr_bytes);
    auto dev_clk = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id()) * 1e6;  // Hz

    double total_cycles = std::reduce(dev_cycles.begin(), dev_cycles.end(), 0ULL);

    test_results.total_cores_cycles += total_cycles;
    test_results.total_cores_time += total_cycles / dev_clk;
    // Reduce with 64 bits to prevent overflow as values read from device is 32 bits
    test_results.total_cores_bytes_rd += std::reduce(dev_bytes_read.begin(), dev_bytes_read.end(), 0ULL);
    test_results.total_cores_bytes_wr += std::reduce(dev_bytes_written.begin(), dev_bytes_written.end(), 0ULL);

    test_results.kernel_0_cycles += dev_cycles[0];
    test_results.kernel_0_time += dev_cycles[0] / dev_clk;
    test_results.kernel_0_bytes_rd += dev_bytes_read[0];
    test_results.kernel_0_bytes_wr += dev_bytes_written[0];
}

// Report device bandwidth to the benchmark state
// Average bw will be reported as "dev_bw" as well as the bw for the
// first core will also be reported by itself as "kernel_0_bw".
void report_device_bw(benchmark::State& state, const TestResult& test_results) {
    state.counters["dev_bw"] =
        (test_results.total_cores_bytes_rd + test_results.total_cores_bytes_wr) / test_results.total_cores_time;
    state.counters["dev_rd_bytes"] = test_results.total_cores_bytes_rd;
    state.counters["dev_wr_bytes"] = test_results.total_cores_bytes_wr;
    state.counters["dev_rd_bw"] = test_results.total_cores_bytes_rd / test_results.total_cores_time;
    state.counters["dev_wr_bw"] = test_results.total_cores_bytes_wr / test_results.total_cores_time;
    state.counters["dev_cycles"] = test_results.total_cores_cycles;

    state.counters["kernel_0_bw"] =
        (test_results.kernel_0_bytes_rd + test_results.kernel_0_bytes_wr) / test_results.kernel_0_time;
    state.counters["kernel_0_rd_bw"] = test_results.kernel_0_bytes_rd / test_results.kernel_0_time;
    state.counters["kernel_0_wr_bw"] = test_results.kernel_0_bytes_wr / test_results.kernel_0_time;
    state.counters["kernel_0_cycles"] = test_results.kernel_0_cycles;
}

// Benchmark various memcpy_to_device transfer sizes.
// Reports host bw.
TestResult mem_bench_page_sizing(benchmark::State& state) {
    auto device_ids = get_device_ids_for_single_device(g_user_device_id);
    const uint32_t device_id = device_ids.empty() ? 0 : device_ids[0];
    TestResult results;
    Context ctx{
        {},
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        0,                                      // Readers
        0,                                      // Writers
        true,                                   // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };

    auto src_data = generate_random_src_data(ctx.total_size);
    auto* hugepage = get_hugepage(device_id, 0);
    auto hugepage_size = get_hugepage_size(device_id);
    bool cached = state.range(2);

    for ([[maybe_unused]] auto _ : state) {
        const double iteration_time =
            cached ? copy_to_hugepage(hugepage, hugepage_size, src_data, ctx.total_size, ctx.page_size, true)
                   : copy_to_hugepage(hugepage, hugepage_size, src_data, ctx.total_size, ctx.page_size, false);
        results.host_bytes_processed += ctx.total_size;
        results.host_time_elapsed += iteration_time;

        state.SetIterationTime(iteration_time);
    }
    state.SetBytesProcessed(ctx.total_size * state.iterations());
    return results;
}

// Benchmark memcpy_to_device on multiple threads to try saturating host bandwidth.
// Reports host bw.
TestResult mem_bench_copy_multithread(benchmark::State& state) {
    static_assert((MEMCPY_ALIGNMENT & ((MEMCPY_ALIGNMENT)-1)) == 0);
    auto device_ids = get_device_ids_for_single_device(g_user_device_id);
    const uint32_t device_id = device_ids.empty() ? 0 : device_ids[0];
    TestResult results;
    Context ctx{
        {},
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        static_cast<int>(state.range(2)),       // Threads
        0,                                      // Readers
        0,                                      // Writers
        true,                                   // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };
    auto src_data = generate_random_src_data(ctx.total_size);
    auto* hugepage = get_hugepage(device_id, 0);
    const auto hugepage_size = get_hugepage_size(device_id);
    const auto bytes_per_thread = ((ctx.total_size / ctx.threads) + (MEMCPY_ALIGNMENT)-1) & -(MEMCPY_ALIGNMENT);
    const auto last_thread_bytes = ctx.total_size - (bytes_per_thread * (ctx.threads - 1));

    for ([[maybe_unused]] auto _ : state) {
        auto iteration_time = execute_work_synced_start(
            ctx.threads,
            [&](int thread_idx) {
                uint64_t thread_dst = (uint64_t)hugepage + (thread_idx * bytes_per_thread);
                uint64_t thread_bytes = (thread_idx == ctx.threads - 1) ? last_thread_bytes : bytes_per_thread;
                std::span<uint32_t> thread_src{src_data};
                thread_src = thread_src.subspan(
                    (thread_idx * bytes_per_thread) / sizeof(uint32_t), thread_bytes / sizeof(uint32_t));
                copy_to_hugepage<false>(
                    (void*)thread_dst, hugepage_size, thread_src, thread_bytes, ctx.page_size, false);
            },
            []() {});

        results.host_bytes_processed += ctx.total_size;
        results.host_time_elapsed += iteration_time;

        state.SetIterationTime(iteration_time);
    }

    state.SetBytesProcessed(ctx.total_size * state.iterations());
    return results;
}

// Benchmark memcpy_to_device while the device is reading or writing the hugepage.
// Can be read or write not both. For both, use mem_bench_copy_with_read_and_write_kernel.
// Reports host bw and device bw.
TestResult mem_bench_copy_with_active_kernel(benchmark::State& state) {
    TestResult results;
    auto device_ids = get_device_ids_for_single_device(g_user_device_id);
    auto devices = tt::tt_metal::detail::CreateDevices(device_ids);
    const uint32_t device_id = device_ids.empty() ? 0 : device_ids[0];
    IDevice* device = devices[device_id];
    Context ctx{
        devices,
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        static_cast<int>(state.range(2)),       // Readers
        static_cast<int>(state.range(3)),       // Writers
        static_cast<bool>(state.range(4)),      // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };
    if (ctx.number_reader_kernels && ctx.number_writer_kernels) {
        TT_THROW(
            "Cannot have both reader and writes in this test method. Use mem_bench_copy_with_read_and_write_kernel "
            "instead.");
    }

    auto src_data = generate_random_src_data(ctx.total_size);
    auto* hugepage = get_hugepage(device->id(), 0);
    auto hugepage_size = get_hugepage_size(device->id());

    for ([[maybe_unused]] auto _ : state) {
        auto pgm = CreateProgram();
        std::optional<CoreRange> configured_cores;
        if (ctx.number_reader_kernels) {
            configured_cores = configure_kernels(device, pgm, ctx, 0, ctx.number_reader_kernels, false, hugepage_size);
        } else {
            configured_cores = configure_kernels(device, pgm, ctx, 0, ctx.number_writer_kernels, true, hugepage_size);
        }
        double host_copy_time = 1;  // Set to 1 so it doesn't divide by 0 if host copy is disabled

        double wait_for_kernel_time = execute_work_synced_start(
            1,
            [device, &pgm](int /*thread_idx*/) {
                // Program
                tt::tt_metal::detail::LaunchProgram(device, pgm, true);
            },
            [&]() {
                if (ctx.enable_host_copy_with_kernels) {
                    // Host copy while waiting for program
                    host_copy_time =
                        copy_to_hugepage(hugepage, hugepage_size, src_data, ctx.total_size, ctx.page_size, false);
                    results.host_bytes_processed += ctx.total_size;
                    results.host_time_elapsed += host_copy_time;
                }
            });

        results.host_wait_for_kernel_time_elapsed += wait_for_kernel_time;

        read_inc_data_from_cores(ctx, device, configured_cores.value(), results);

        state.SetIterationTime(host_copy_time);
    }
    if (ctx.enable_host_copy_with_kernels) {
        state.SetBytesProcessed(ctx.total_size * state.iterations());
    } else {
        state.SetBytesProcessed(0);
    }

    report_device_bw(state, results);
    tt::tt_metal::detail::CloseDevices(devices);
    return results;
}

// Host writing to a hugepage while the device pulls from another hugepage.
// Reports host bw and device bw.
TestResult mem_bench_copy_active_kernel_different_page(benchmark::State& state) {
    TestResult results;
    auto device_ids = get_device_ids_for_single_device(g_user_device_id);
    auto devices = tt::tt_metal::detail::CreateDevices(device_ids);
    const uint32_t device_id = device_ids.empty() ? 0 : device_ids[0];
    IDevice* device = devices[device_id];
    Context ctx{
        devices,
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        static_cast<int>(state.range(2)),       // Readers
        0,                                      // Writers
        true,                                   // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };

    auto src_data = generate_random_src_data(ctx.total_size);
    auto device_hugepage_size = get_hugepage_size(device->id());

    // 2nd open device is not required
    auto* host_hugepage = get_hugepage(device->id() + 1, 0);
    auto host_hugepage_size = get_hugepage_size(device->id() + 1);

    for ([[maybe_unused]] auto _ : state) {
        auto pgm = CreateProgram();
        auto configured_cores =
            configure_kernels(device, pgm, ctx, 0, ctx.number_reader_kernels, false, device_hugepage_size).value();
        double host_copy_time = 0;

        double wait_for_kernel_time = execute_work_synced_start(
            1,
            [device, &pgm](int /*thread_idx*/) {
                // Program
                tt::tt_metal::detail::LaunchProgram(device, pgm, true);
            },
            [&]() {
                // Host copy while waiting for program
                host_copy_time =
                    copy_to_hugepage(host_hugepage, host_hugepage_size, src_data, ctx.total_size, ctx.page_size, false);
                results.host_bytes_processed += ctx.total_size;
                results.host_time_elapsed += host_copy_time;
            });

        results.host_wait_for_kernel_time_elapsed += wait_for_kernel_time;

        read_inc_data_from_cores(ctx, device, configured_cores, results);

        state.SetIterationTime(host_copy_time);
    }

    state.SetBytesProcessed(ctx.total_size * state.iterations());

    report_device_bw(state, results);
    tt::tt_metal::detail::CloseDevices(devices);
    return results;
}

// Common Multi MMIO device test.
TestResult mem_bench_multi_mmio_devices(
    benchmark::State& state, std::map<ChipId, IDevice*>& devices, const Context& ctx) {
    TestResult results;

    // One thread to wait for program on each device

    for ([[maybe_unused]] auto _ : state) {
        std::map<int, Program> programs;                  // device : programs
        std::map<int, CoreRange> configured_core_ranges;  // device : cores
        for (auto [device_id, device] : devices) {
            programs[device_id] = CreateProgram();
            Program& pgm = programs[device_id];
            auto device_hugepage_size = get_hugepage_size(device_id);
            configured_core_ranges.insert(
                {device_id,
                 configure_kernels(device, pgm, ctx, 0, ctx.number_reader_kernels, false, device_hugepage_size)
                     .value()});
        }

        execute_work_synced_start(
            1,
            [devices, &programs](int /*thread_idx*/) {
                // Program
                for (auto& [device_id, pgm] : programs) {
                    tt::tt_metal::detail::LaunchProgram(devices.at(device_id), pgm, false);
                }
            },
            []() {});

        // Wait all programs to complete
        for (auto& [device_id, pgm] : programs) {
            tt::tt_metal::detail::WaitProgramDone(devices.at(device_id), pgm);
        }

        // Read counters from each core
        for (auto& [device_id, core_range] : configured_core_ranges) {
            read_inc_data_from_cores(ctx, devices.at(device_id), core_range, results);
        }

        // This test does not report host bw
        state.SetIterationTime(1);
    }

    state.SetBytesProcessed(0);
    report_device_bw(state, results);
    state.counters["num_mmio_devices"] = devices.size();

    return results;
}

// Multi MMIO devices reading on the same NUMA node.
TestResult mem_bench_multi_mmio_devices_reading_same_node(benchmark::State& state) {
    // Node 0
    auto device_ids = get_device_ids_for_multi_device_same_node(g_user_device_id);
    auto devices = tt::tt_metal::detail::CreateDevices(device_ids);

    Context ctx{
        devices,
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        static_cast<int>(state.range(2)),       // Readers on each device
        0,                                      // Writers
        false,                                  // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };

    TestResult results = mem_bench_multi_mmio_devices(state, devices, ctx);
    tt::tt_metal::detail::CloseDevices(devices);

    return results;
}

// Multi MMIO devices reading on different NUMA nodes.
TestResult mem_bench_multi_mmio_devices_reading_different_node(benchmark::State& state) {
    auto device_ids = get_device_ids_for_multi_device_different_nodes(g_user_device_id);
    auto devices = tt::tt_metal::detail::CreateDevices(device_ids);

    Context ctx{
        devices,
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        static_cast<int>(state.range(2)),       // Readers on each device
        0,                                      // Writers
        false,                                  // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };

    TestResult results = mem_bench_multi_mmio_devices(state, devices, ctx);
    tt::tt_metal::detail::CloseDevices(devices);

    return results;
}

// Benchmark memcpy_to_device while device is reading (prefetching) and writing (dispatching data back to host)
// First half of hugepage will be written to by host
// Second half will be written to by device
TestResult mem_bench_copy_with_read_and_write_kernel(benchmark::State& state) {
    auto device_ids = get_device_ids_for_single_device(g_user_device_id);
    auto devices = tt::tt_metal::detail::CreateDevices(device_ids);
    const uint32_t device_id = device_ids.empty() ? 0 : device_ids[0];
    IDevice* device = devices[device_id];
    Context ctx{
        devices,
        static_cast<uint32_t>(state.range(0)),  // Total size
        static_cast<uint32_t>(state.range(1)),  // Page size
        0,                                      // Threads
        static_cast<int>(state.range(2)),       // Readers
        static_cast<int>(state.range(3)),       // Writers
        static_cast<bool>(state.range(4)),      // Enable host copy
        0,                                      // Iterations is managed by the benchmark framework
    };

    auto src_data = generate_random_src_data(ctx.total_size);
    auto* hugepage = get_hugepage(device->id(), 0);
    auto hugepage_size = get_hugepage_size(device->id());

    // Don't need to separate device results
    // Readers will have 0 bytes written
    // Writers will have 0 bytes read. Will not mix.
    TestResult results;

    for ([[maybe_unused]] auto _ : state) {
        auto pgm = CreateProgram();
        auto configured_read_cores =
            configure_kernels(device, pgm, ctx, 0, ctx.number_reader_kernels, false, hugepage_size / 2).value();
        // Offset write cores to second half of PCIe
        // Use second row
        auto configured_write_cores =
            configure_kernels(
                device, pgm, ctx, 1, ctx.number_writer_kernels, true, hugepage_size / 2, hugepage_size / 2)
                .value();
        double host_copy_time = 0;

        double wait_for_kernel_time = execute_work_synced_start(
            1,
            [device, &pgm](int /*thread_idx*/) {
                // Program
                tt::tt_metal::detail::LaunchProgram(device, pgm, true);
            },
            [&]() {
                // Host copy while waiting for program
                host_copy_time =
                    copy_to_hugepage(hugepage, hugepage_size / 2, src_data, ctx.total_size, ctx.page_size, false);
                results.host_bytes_processed += ctx.total_size;
                results.host_time_elapsed += host_copy_time;
            });

        results.host_wait_for_kernel_time_elapsed += wait_for_kernel_time;

        read_inc_data_from_cores(ctx, device, configured_read_cores, results);
        read_inc_data_from_cores(ctx, device, configured_write_cores, results);

        state.SetIterationTime(host_copy_time);
    }

    state.SetBytesProcessed(ctx.total_size * state.iterations());
    report_device_bw(state, results);
    tt::tt_metal::detail::CloseDevices(devices);
    return results;
}

void global_bench_args(benchmark::internal::Benchmark* b) { b->UseManualTime()->Iterations(5); }

void register_basic_benchmark_suite() {
    // Host copying to hugepage (no device activity)
    ::benchmark::RegisterBenchmark("Host Copy (Cached)", mem_bench_page_sizing)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {16, 8_KB, 16_KB, 32_KB},
            {true},
        });
    // N cores reading the hugepage on the host
    ::benchmark::RegisterBenchmark("Device Reading Host", mem_bench_copy_with_active_kernel)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2, 3, 4},
            {0},
            {false},
        });
    // N cores writing the hugepage on the host
    ::benchmark::RegisterBenchmark("Device Writing Host", mem_bench_copy_with_active_kernel)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {0},
            {1, 2, 3, 4},
            {false},
        });
}

void register_full_benchmark_suite() {
    ::benchmark::RegisterBenchmark("Host Copy Page Sizing", mem_bench_page_sizing)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {16, 8_KB, 16_KB, 32_KB},
            {false},
        });
    ::benchmark::RegisterBenchmark("Host Copy Saturation", mem_bench_copy_multithread)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2, 3, 4, 5, 6, 7, 8},
        });
    ::benchmark::RegisterBenchmark(
        "Host Copy with Active Kernel on Different Hugepages", mem_bench_copy_active_kernel_different_page)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2, 3, 4},
        });
    ::benchmark::RegisterBenchmark(
        "Host Copy with Active Kernel Reading and Writing", mem_bench_copy_with_read_and_write_kernel)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2},
            {1, 2},
            {true},
        });
    ::benchmark::RegisterBenchmark(
        "Multiple MMIO Devices Reading (Same NUMA node)", mem_bench_multi_mmio_devices_reading_same_node)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2},
        });
    ::benchmark::RegisterBenchmark(
        "Multiple MMIO Devices Reading (Different NUMA node)", mem_bench_multi_mmio_devices_reading_different_node)
        ->Apply(global_bench_args)
        ->ArgsProduct({
            {1_GB},
            {32_KB},
            {1, 2},
        });
}

void print_help() {
    ::benchmark::PrintDefaultHelp();
    std::cout << "          [--help] Shows this help message\n";
    std::cout << "          [--full] Run all tests\n";
    std::cout << "          [--device-id=<id>] Use specific device ID instead of auto-selection\n";
    std::cout << "\nCounters\n";
    std::cout << "          bytes_per_second: Aggregate Host copy to hugepage bandwidth. 0 if not measured.\n";
    std::cout << "          dev_bw: Average single device core PCIe r/w bandwidth. 0 if not measured.\n";
    std::cout << "          kernel_0: Single core PCIe r/w bandwidth. 0 if not measured.\n";
}

bool has_flag(const std::vector<std::string>& input_args, const std::string& flag) {
    for (const auto& arg : input_args) {
        if (arg == flag) {
            return true;
        }
    }

    return false;
}

std::optional<int> get_device_id_from_args(const std::vector<std::string>& input_args) {
    for (const auto& arg : input_args) {
        if (arg.starts_with("--device-id=")) {
            std::string device_id_str = arg.substr(12);  // Skip "--device-id="
            try {
                return std::stoi(device_id_str);
            } catch (const std::exception&) {
                std::cerr << "Invalid device ID: " << device_id_str << std::endl;
                return std::nullopt;
            }
        }
    }
    return std::nullopt;
}

int main(int argc, char* argv[]) {
    std::vector<std::string> input_args(argv, argv + argc);
    if (has_flag(input_args, "--help")) {
        print_help();
        return 0;
    }

    // Parse device ID if specified
    g_user_device_id = get_device_id_from_args(input_args);
    if (g_user_device_id.has_value()) {
        std::cout << "Using device ID: " << g_user_device_id.value() << std::endl;
    }

    // Force TT_METAL options
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "true", true);
    setenv("TT_METAL_CLEAR_L1", "1", true);
    // May be overridden by the user
    setenv("TT_LOGGER_LEVEL", "FATAL", false);

    char arg0_default[] = "benchmark";
    char* args_default = arg0_default;
    if (!argv) {
        argc = 1;
        argv = &args_default;
    }

    // Run basic benchmarks
    register_basic_benchmark_suite();

    // Run all benchmarks
    if (has_flag(input_args, "--full")) {
        register_full_benchmark_suite();
    }

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
