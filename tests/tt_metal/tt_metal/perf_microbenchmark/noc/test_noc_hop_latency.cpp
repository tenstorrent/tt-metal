// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOC hop latency microbenchmark.
// Measures round-trip semaphore ping-pong latency to characterize per-hop NOC latency.
//
// Modes:
//   0 = tensix-tensix: fix sender at (0,0), sweep responder across all tensix cores
//   1 = eth-sender:    fix sender on eth core, sweep responder across all tensix cores
//   2 = eth-receiver:  sweep sender across all tensix cores, fix responder on eth core
//
// Usage: test_noc_hop_latency [num_iterations] [num_warmup] [noc_index] [mode]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <llrt/tt_cluster.hpp>

using namespace tt;

struct PingResult {
    double mean_cycles;  // total round-trip
    double min_cycles;
    double max_cycles;
    double stddev_cycles;
    double mean_issue;  // noc_semaphore_inc issue time
    double mean_poll;   // poll wait time
    double stddev_issue;
    double stddev_poll;
    uint32_t manhattan_distance;
    std::vector<uint32_t> raw_issue;  // per-iteration issue cycles
    std::vector<uint32_t> raw_poll;   // per-iteration poll cycles
};

struct CoreSpec {
    CoreCoord logical;
    CoreType type;  // WORKER or ETH
};

PingResult run_ping_pong(
    tt_metal::IDevice* device,
    CoreSpec sender,
    CoreSpec responder,
    uint32_t num_iterations,
    uint32_t num_warmup,
    tt_metal::NOC sender_noc,
    uint32_t noc_index,
    CoreCoord noc_grid_size) {
    tt_metal::Program program = tt_metal::CreateProgram();

    auto sender_virtual = device->virtual_core_from_logical_core(sender.logical, sender.type);
    auto responder_virtual = device->virtual_core_from_logical_core(responder.logical, responder.type);

    // L1 addresses: eth uses HAL unreserved base, tensix uses high L1 (safe region above kernel data)
    uint32_t eth_unreserved = tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED);
    constexpr uint32_t tensix_sem_base = 150 * 1024;  // 150KB — well above kernel config/stack

    // 16-byte aligned addresses for NOC semaphore operations
    auto align16 = [](uint32_t addr) { return (addr + 15) & ~15u; };

    // Sender addresses
    uint32_t sender_base = (sender.type == CoreType::ETH) ? eth_unreserved : tensix_sem_base;
    uint32_t sender_sem_addr = align16(sender_base);
    uint32_t timestamp_buf_addr = align16(sender_sem_addr + 16);  // semaphore is 4 bytes, pad to 16
    // Ready semaphore: used when responder is eth so sender waits for eth to be ready
    // Place after timestamp buffer (2 * num_iterations * 4 bytes)
    uint32_t ready_sem_addr = (responder.type == CoreType::ETH)
                                  ? align16(timestamp_buf_addr + 2 * num_iterations * sizeof(uint32_t))
                                  : 0;  // 0 = no handshake

    // Responder addresses
    uint32_t responder_base = (responder.type == CoreType::ETH) ? eth_unreserved : tensix_sem_base;
    uint32_t responder_sem_addr = align16(responder_base);

    // Sender NOC: on BH, ERISC0 is locked to NOC0
    tt_metal::NOC actual_sender_noc = sender_noc;
    if (sender.type == CoreType::ETH) {
        actual_sender_noc = tt_metal::NOC::NOC_0;
    }

    // Responder uses opposite NOC so both directions traverse the same physical path
    // Exception: ERISC0 on BH is locked to NOC0
    tt_metal::NOC responder_noc;
    if (responder.type == CoreType::ETH) {
        responder_noc = tt_metal::NOC::NOC_0;
    } else {
        responder_noc = (actual_sender_noc == tt_metal::NOC::NOC_0) ? tt_metal::NOC::NOC_1 : tt_metal::NOC::NOC_0;
    }

    // Sender kernel
    tt_metal::KernelHandle sender_kernel;
    if (sender.type == CoreType::ETH) {
        sender_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_latency_ping_sender.cpp",
            sender.logical,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = {}});
    } else {
        sender_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_latency_ping_sender.cpp",
            sender.logical,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = actual_sender_noc});
    }
    tt_metal::SetRuntimeArgs(
        program,
        sender_kernel,
        sender.logical,
        {responder_virtual.x,
         responder_virtual.y,
         responder_sem_addr,
         sender_sem_addr,
         timestamp_buf_addr,
         num_iterations,
         num_warmup,
         ready_sem_addr});

    // Responder kernel
    uint32_t total_pings = num_warmup + num_iterations;
    if (responder.type == CoreType::ETH) {
        auto responder_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_latency_ping_responder.cpp",
            responder.logical,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = {}});
        tt_metal::SetRuntimeArgs(
            program,
            responder_kernel,
            responder.logical,
            {sender_virtual.x, sender_virtual.y, sender_sem_addr, responder_sem_addr, total_pings, ready_sem_addr});
    } else {
        auto responder_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/noc/noc_latency_ping_responder.cpp",
            responder.logical,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = responder_noc});
        tt_metal::SetRuntimeArgs(
            program,
            responder_kernel,
            responder.logical,
            {sender_virtual.x, sender_virtual.y, sender_sem_addr, responder_sem_addr, total_pings, 0u});
    }

    tt_metal::detail::CompileProgram(device, program);
    tt_metal::detail::LaunchProgram(device, program);

    // Read timestamps from sender L1 — pairs of [issue_time, poll_time]
    std::vector<uint32_t> timestamps(2 * num_iterations, 0);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        timestamps.data(),
        2 * num_iterations * sizeof(uint32_t),
        tt_cxy_pair(device->id(), sender_virtual),
        timestamp_buf_addr);

    // Compute stats for total, issue, and poll phases
    uint64_t sum_total = 0, sum_issue = 0, sum_poll = 0;
    uint32_t min_val = UINT32_MAX, max_val = 0;
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t issue = timestamps[2 * i];
        uint32_t poll = timestamps[2 * i + 1];
        uint32_t total = issue + poll;
        sum_total += total;
        sum_issue += issue;
        sum_poll += poll;
        min_val = std::min(min_val, total);
        max_val = std::max(max_val, total);
    }
    double mean_total = static_cast<double>(sum_total) / num_iterations;
    double mean_issue = static_cast<double>(sum_issue) / num_iterations;
    double mean_poll = static_cast<double>(sum_poll) / num_iterations;

    double var_total = 0, var_issue = 0, var_poll = 0;
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t issue = timestamps[2 * i];
        uint32_t poll = timestamps[2 * i + 1];
        uint32_t total = issue + poll;
        var_total += (total - mean_total) * (total - mean_total);
        var_issue += (issue - mean_issue) * (issue - mean_issue);
        var_poll += (poll - mean_poll) * (poll - mean_poll);
    }
    double stddev_total = std::sqrt(var_total / num_iterations);
    double stddev_issue = std::sqrt(var_issue / num_iterations);
    double stddev_poll = std::sqrt(var_poll / num_iterations);

    // Manhattan distance using physical NOC0 coordinates (not virtual/translated)
    // Virtual coords for eth are in translated space (x=20+, y=25) which doesn't reflect actual hop count.
    // Physical NOC0 coords are within the 17x12 grid and give correct Manhattan distance.
    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    CoreCoord sender_phys = soc_desc.get_physical_core_from_logical_core(sender.logical, sender.type);
    CoreCoord responder_phys = soc_desc.get_physical_core_from_logical_core(responder.logical, responder.type);

    uint32_t abs_dx =
        (sender_phys.x > responder_phys.x) ? (sender_phys.x - responder_phys.x) : (responder_phys.x - sender_phys.x);
    uint32_t abs_dy =
        (sender_phys.y > responder_phys.y) ? (sender_phys.y - responder_phys.y) : (responder_phys.y - sender_phys.y);
    uint32_t dx, dy;
    if (noc_index == 0) {
        dx = abs_dx;
        dy = abs_dy;
    } else {
        // NOC1 routes the opposite way around the torus grid
        dx = noc_grid_size.x - abs_dx;
        dy = noc_grid_size.y - abs_dy;
    }

    // Store raw per-iteration samples
    std::vector<uint32_t> raw_issue(num_iterations), raw_poll(num_iterations);
    for (uint32_t i = 0; i < num_iterations; i++) {
        raw_issue[i] = timestamps[2 * i];
        raw_poll[i] = timestamps[2 * i + 1];
    }

    return PingResult{
        .mean_cycles = mean_total,
        .min_cycles = static_cast<double>(min_val),
        .max_cycles = static_cast<double>(max_val),
        .stddev_cycles = stddev_total,
        .mean_issue = mean_issue,
        .mean_poll = mean_poll,
        .stddev_issue = stddev_issue,
        .stddev_poll = stddev_poll,
        .manhattan_distance = dx + dy,
        .raw_issue = std::move(raw_issue),
        .raw_poll = std::move(raw_poll)};
}

void print_results(
    const std::map<std::pair<uint32_t, uint32_t>, PingResult>& results,
    CoreCoord grid_size,
    const std::string& title,
    uint32_t noc_index) {
    double cycles_to_ns = 1000.0 / 1350.0;  // BH: 1350 MHz
    int col_w = 9;

    log_info(tt::LogTest, "\n=== {} (NOC{}) ===", title, noc_index);
    std::ostringstream header;
    std::string yx_label = "y\\x";
    header << std::setw(col_w) << yx_label;
    for (uint32_t x = 0; x < grid_size.x; x++) {
        header << std::setw(col_w) << x;
    }
    log_info(tt::LogTest, "{}", header.str());

    for (uint32_t y = 0; y < grid_size.y; y++) {
        std::ostringstream row;
        row << std::setw(col_w) << y;
        for (uint32_t x = 0; x < grid_size.x; x++) {
            auto it = results.find({x, y});
            if (it != results.end()) {
                row << std::setw(col_w) << std::fixed << std::setprecision(0) << it->second.mean_cycles;
            } else {
                row << std::setw(col_w) << "---";
            }
        }
        log_info(tt::LogTest, "{}", row.str());
    }

    // Distance summary with issue/poll breakdown
    struct DistStats {
        std::vector<double> totals;
        std::vector<double> issues;
        std::vector<double> polls;
    };
    std::map<uint32_t, DistStats> dist_to_stats;
    for (const auto& [coord, result] : results) {
        auto& ds = dist_to_stats[result.manhattan_distance];
        ds.totals.push_back(result.mean_cycles);
        ds.issues.push_back(result.mean_issue);
        ds.polls.push_back(result.mean_poll);
    }
    log_info(tt::LogTest, "\n--- Distance summary ---");
    log_info(
        tt::LogTest,
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "dist",
        "total",
        "issue",
        "poll",
        "min",
        "max",
        "count");
    for (const auto& [dist, ds] : dist_to_stats) {
        double sum_t = 0, sum_i = 0, sum_p = 0, min_v = 1e9, max_v = 0;
        for (size_t j = 0; j < ds.totals.size(); j++) {
            sum_t += ds.totals[j];
            sum_i += ds.issues[j];
            sum_p += ds.polls[j];
            min_v = std::min(min_v, ds.totals[j]);
            max_v = std::max(max_v, ds.totals[j]);
        }
        double n = ds.totals.size();
        log_info(
            tt::LogTest,
            "{:>6} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>8}",
            dist,
            sum_t / n,
            sum_i / n,
            sum_p / n,
            min_v,
            max_v,
            (int)n);
    }

    // Linear regression
    double sum_d = 0, sum_l = 0, sum_dl = 0, sum_dd = 0;
    uint32_t n = 0;
    for (auto& [coord, result] : results) {
        double d = result.manhattan_distance;
        double l = result.mean_cycles;
        sum_d += d;
        sum_l += l;
        sum_dl += d * l;
        sum_dd += d * d;
        n++;
    }
    if (n > 1) {
        double mean_d = sum_d / n;
        double mean_l = sum_l / n;
        double b = (sum_dl - n * mean_d * mean_l) / (sum_dd - n * mean_d * mean_d);
        double a = mean_l - b * mean_d;
        log_info(tt::LogTest, "\n--- Linear fit: round_trip = {:.1f} + {:.2f} * distance ---", a, b);
        log_info(tt::LogTest, "  Fixed overhead (one-way): {:.1f} cycles ({:.1f} ns)", a / 2, a / 2 * cycles_to_ns);
        log_info(tt::LogTest, "  Per-hop cost (one-way):   {:.2f} cycles ({:.2f} ns)", b / 2, b / 2 * cycles_to_ns);
    }
}

void dump_raw_csv(
    const std::map<std::pair<uint32_t, uint32_t>, PingResult>& results,
    const std::string& sender_label,
    uint32_t num_iterations,
    uint32_t noc_index,
    uint32_t mode,
    const std::string& csv_path) {
    std::ofstream csv(csv_path);
    // Header
    csv << "src,dst_x,dst_y,distance,iteration,issue_cycles,poll_cycles,total_cycles\n";

    for (const auto& [coord, result] : results) {
        auto [dx, dy] = coord;
        for (uint32_t i = 0; i < num_iterations; i++) {
            uint32_t issue = result.raw_issue[i];
            uint32_t poll = result.raw_poll[i];
            csv << sender_label << "," << dx << "," << dy << "," << result.manhattan_distance << "," << i << ","
                << issue << "," << poll << "," << (issue + poll) << "\n";
        }
    }
    log_info(tt::LogTest, "Raw CSV written to: {}", csv_path);
}

int main(int argc, char** argv) {
    uint32_t num_iterations = (argc > 1) ? std::stoi(argv[1]) : 100;
    uint32_t num_warmup = (argc > 2) ? std::stoi(argv[2]) : 10;
    uint32_t noc_index = (argc > 3) ? std::stoi(argv[3]) : 0;
    uint32_t mode = (argc > 4) ? std::stoi(argv[4]) : 0;

    auto noc_select = noc_index == 0 ? tt_metal::NOC::NOC_0 : tt_metal::NOC::NOC_1;

    const char* mode_names[] = {"tensix-tensix", "eth-sender", "eth-receiver"};
    log_info(
        tt::LogTest,
        "NOC Hop Latency Benchmark: {} iterations, {} warmup, NOC{}, mode={} ({})",
        num_iterations,
        num_warmup,
        noc_index,
        mode,
        mode_names[mode]);

    // For eth modes, open all devices so the active eth cores can complete FW handshake
    size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    std::vector<tt_metal::IDevice*> all_devices;
    if (mode > 0 && num_devices > 1) {
        for (size_t i = 0; i < num_devices; i++) {
            all_devices.push_back(tt_metal::CreateDevice(i));
        }
    } else {
        all_devices.push_back(tt_metal::CreateDevice(0));
    }
    tt_metal::IDevice* device = all_devices[0];

    auto grid_size = device->compute_with_storage_grid_size();
    auto noc_grid_size = device->grid_size();  // Full NOC torus grid dimensions
    log_info(
        tt::LogTest,
        "Worker grid: {}x{}, NOC grid: {}x{}, {} devices opened",
        grid_size.x,
        grid_size.y,
        noc_grid_size.x,
        noc_grid_size.y,
        all_devices.size());

    // Get active eth cores for modes 1 and 2
    std::vector<CoreCoord> eth_cores;
    if (mode > 0) {
        bool slow_dispatch = (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr);
        auto active_eth_cores = tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
            device->id(), !slow_dispatch);
        TT_FATAL(!active_eth_cores.empty(), "No active ethernet cores found on device 0");
        for (const auto& ec : active_eth_cores) {
            eth_cores.push_back(ec);
        }
        log_info(tt::LogTest, "Found {} active ethernet cores", eth_cores.size());
        for (const auto& ec : eth_cores) {
            auto ev = device->virtual_core_from_logical_core(ec, CoreType::ETH);
            log_info(tt::LogTest, "  eth logical=({},{}), virtual=({},{})", ec.x, ec.y, ev.x, ev.y);
        }
    }

    if (mode == 0) {
        // Tensix-tensix: fix sender at (0,0), sweep responder
        std::map<std::pair<uint32_t, uint32_t>, PingResult> results;
        CoreSpec sender{CoreCoord(0, 0), CoreType::WORKER};
        for (uint32_t ry = 0; ry < grid_size.y; ry++) {
            for (uint32_t rx = 0; rx < grid_size.x; rx++) {
                CoreCoord responder_logical(rx, ry);
                if (responder_logical == sender.logical) {
                    continue;
                }

                CoreSpec responder{responder_logical, CoreType::WORKER};
                auto result = run_ping_pong(
                    device, sender, responder, num_iterations, num_warmup, noc_select, noc_index, noc_grid_size);
                results[{rx, ry}] = result;

                log_info(
                    tt::LogTest,
                    "  T({},{}) -> T({},{}): dist={}, total={:.1f} cyc, issue={:.1f} (sd={:.1f}), poll={:.1f} "
                    "(sd={:.1f})",
                    0,
                    0,
                    rx,
                    ry,
                    result.manhattan_distance,
                    result.mean_cycles,
                    result.mean_issue,
                    result.stddev_issue,
                    result.mean_poll,
                    result.stddev_poll);
            }
        }
        print_results(results, grid_size, "Tensix-Tensix round-trip (cycles), sender=T(0,0)", noc_index);
        dump_raw_csv(
            results,
            "T(0;0)",
            num_iterations,
            noc_index,
            mode,
            fmt::format("noc_hop_latency_mode{}_noc{}.csv", mode, noc_index));

    } else if (mode == 1) {
        // Eth-sender: sweep ALL active eth cores as sender, each sweeping all tensix responders
        for (const auto& eth_logical : eth_cores) {
            std::map<std::pair<uint32_t, uint32_t>, PingResult> results;
            CoreSpec sender{eth_logical, CoreType::ETH};
            auto eth_virtual = device->virtual_core_from_logical_core(eth_logical, CoreType::ETH);
            log_info(
                tt::LogTest,
                "\n>>> Eth sender: logical=({},{}), virtual=({},{})",
                eth_logical.x,
                eth_logical.y,
                eth_virtual.x,
                eth_virtual.y);

            for (uint32_t ry = 0; ry < grid_size.y; ry++) {
                for (uint32_t rx = 0; rx < grid_size.x; rx++) {
                    CoreSpec responder{CoreCoord(rx, ry), CoreType::WORKER};
                    auto result = run_ping_pong(
                        device, sender, responder, num_iterations, num_warmup, noc_select, noc_index, noc_grid_size);
                    results[{rx, ry}] = result;

                    log_info(
                        tt::LogTest,
                        "  E({},{}) -> T({},{}): dist={}, total={:.1f} cyc, issue={:.1f} (sd={:.1f}), poll={:.1f} "
                        "(sd={:.1f})",
                        eth_logical.x,
                        eth_logical.y,
                        rx,
                        ry,
                        result.manhattan_distance,
                        result.mean_cycles,
                        result.mean_issue,
                        result.stddev_issue,
                        result.mean_poll,
                        result.stddev_poll);
                }
            }
            print_results(
                results,
                grid_size,
                fmt::format("Eth-Sender round-trip (cycles), sender=E({},{})", eth_logical.x, eth_logical.y),
                noc_index);
            dump_raw_csv(
                results,
                fmt::format("E({};{})", eth_logical.x, eth_logical.y),
                num_iterations,
                noc_index,
                mode,
                fmt::format(
                    "noc_hop_latency_mode{}_noc{}_eth{}_{}.csv", mode, noc_index, eth_logical.x, eth_logical.y));
        }

    } else if (mode == 2) {
        // Eth-receiver: sweep ALL active eth cores as responder, each with all tensix senders
        for (const auto& eth_logical : eth_cores) {
            std::map<std::pair<uint32_t, uint32_t>, PingResult> results;
            CoreSpec responder{eth_logical, CoreType::ETH};
            auto eth_virtual = device->virtual_core_from_logical_core(eth_logical, CoreType::ETH);
            log_info(
                tt::LogTest,
                "\n>>> Eth responder: logical=({},{}), virtual=({},{})",
                eth_logical.x,
                eth_logical.y,
                eth_virtual.x,
                eth_virtual.y);

            for (uint32_t ry = 0; ry < grid_size.y; ry++) {
                for (uint32_t rx = 0; rx < grid_size.x; rx++) {
                    CoreSpec sender{CoreCoord(rx, ry), CoreType::WORKER};
                    auto result = run_ping_pong(
                        device, sender, responder, num_iterations, num_warmup, noc_select, noc_index, noc_grid_size);
                    results[{rx, ry}] = result;

                    log_info(
                        tt::LogTest,
                        "  T({},{}) -> E({},{}): dist={}, total={:.1f} cyc, issue={:.1f} (sd={:.1f}), poll={:.1f} "
                        "(sd={:.1f})",
                        rx,
                        ry,
                        eth_logical.x,
                        eth_logical.y,
                        result.manhattan_distance,
                        result.mean_cycles,
                        result.mean_issue,
                        result.stddev_issue,
                        result.mean_poll,
                        result.stddev_poll);
                }
            }
            print_results(
                results,
                grid_size,
                fmt::format("Eth-Receiver round-trip (cycles), responder=E({},{})", eth_logical.x, eth_logical.y),
                noc_index);
            dump_raw_csv(
                results,
                fmt::format("E({};{})", eth_logical.x, eth_logical.y),
                num_iterations,
                noc_index,
                mode,
                fmt::format(
                    "noc_hop_latency_mode{}_noc{}_eth{}_{}.csv", mode, noc_index, eth_logical.x, eth_logical.y));
        }
    }

    for (auto* d : all_devices) {
        tt_metal::CloseDevice(d);
    }
    return 0;
}
