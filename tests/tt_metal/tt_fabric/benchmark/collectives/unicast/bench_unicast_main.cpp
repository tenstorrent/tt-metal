// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <cstdint>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

// Single MeshDevice-based fixture with Fabric set to 2D.
struct Fixture : public ::tt::tt_metal::MeshDeviceFixtureBase {
    Fixture() :
        ::tt::tt_metal::MeshDeviceFixtureBase(Config{
            .num_cqs = 1,
            .trace_region_size = 1u << 20,  // enable mesh trace capture (e.g., 1 MiB)
            .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

// Shorthands
using tt::tt_fabric::bench::PerfParams;
using tt::tt_fabric::bench::PerfStats;
using tt::tt_fabric::bench::run_repeated;
using tt::tt_fabric::bench::warmup_once;

namespace {

struct RunOptions {
    int iters = 100;
    int warmup = 1;
    int trace_iters = 1;
    bool no_trace = false;     // reserved
    bool enable_sync = false;  // reserved
    std::string csv_path;      // if non-empty, append one row
};

void usage(const char* argv0) {
    std::string bin = std::filesystem::path(argv0).filename().string();
    if (bin.empty()) {
        bin = "build/test/tt_metal/tt_fabric/bench_unicast";
    }

    const std::string text = fmt::format(
        R"(Usage:
  {bin} --src-dev <mesh:chip|chip> --dst-dev <mesh:chip|chip> --size <bytes>
         [--page <bytes>] [--src-type <l1|dram>] [--dst-type <l1|dram>]
         [--send-core x,y] [--recv-core x,y]
         [--iters N] [--warmup N] [--trace-iters N]
         [--no-trace] [--enable-sync]
         [--csv <path>]

Notes:
- This binary runs ONE configuration per invocation. For sweeps, invoke it repeatedly (see examples below).
- --src-type/--dst-type are accepted for future use; current code uses DRAM src and L1/DRAM dst.
- Trace semantics: each measured iteration does
    1) warm-up (outside trace),
    2) BeginTraceCapture → enqueue the workload N=--trace-iters times → end_mesh_trace,
    3) replay_mesh_trace once and time it, then divides by N to get per-iter time.
  Thus:
    * --trace-iters controls how many enqueues are captured per trace.
    * --iters controls how many times the capture+replay cycle is repeated to collect stats (p50/mean, etc.).

Examples:

# Single run, 1 MiB tensor, 4 KiB pages, capture 64 enqueues per trace, repeat 10 times for stats:
  {bin} --src-dev 0:0 --dst-dev 0:1 --size 1048576 --page 4096 \
        --send-core 0,0 --recv-core 0,0 \
        --iters 10 --warmup 1 --trace-iters 64 \
        --csv artifacts/unicast.csv

# Sweep multiple sizes via helper script:
  python tests/tt_metal/tt_fabric/benchmark/collectives/unicast/run_unicast_sweep.py \
        --src 0:0 --dst 0:1 --sizes 4096,32768,1048576 \
        --recv-core 0,0 --iters 10 --warmup 1 --trace-iters 64 \
        --out-dir artifacts

Legend:
  <mesh:chip|chip>   e.g., "0:1" (mesh 0, chip 1) or just "1" if single mesh.
  cores              x,y in logical worker-core coordinates (e.g., 0,0).
)",
        fmt::arg("bin", bin));

    log_error(tt::LogTest, "{}", text);
}

bool parse_mesh_chip(const std::string& s, uint32_t& mesh, int& chip) {
    size_t colon = s.find(':');
    if (colon == std::string::npos) {
        mesh = 0;
        chip = std::stoi(s);
        return true;
    }
    mesh = static_cast<uint32_t>(std::stoul(s.substr(0, colon)));
    chip = std::stoi(s.substr(colon + 1));
    return true;
}

bool parse_xy(const std::string& s, int& x, int& y) {
    auto comma = s.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    x = std::stoi(s.substr(0, comma));
    y = std::stoi(s.substr(comma + 1));
    return true;
}

// Fills RunOptions + PerfParams + raw src/dst strings. Returns false on bad args.
bool parse_cli_or_usage(
    int argc, char** argv, RunOptions& run, PerfParams& p, std::string& src_dev_str, std::string& dst_dev_str) {
    std::string src_type = "dram";  // reserved
    std::string dst_type = "l1";    // reserved

    auto need = [&](int i, int more) {
        if (i + more >= argc) {
            usage(argv[0]);
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--src-dev" && need(i, 1)) {
            src_dev_str = argv[++i];
        } else if (a == "--dst-dev" && need(i, 1)) {
            dst_dev_str = argv[++i];
        } else if (a == "--size" && need(i, 1)) {
            p.tensor_bytes = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--page" && need(i, 1)) {
            p.page_size = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--src-type" && need(i, 1)) {
            src_type = argv[++i];
        } else if (a == "--dst-type" && need(i, 1)) {
            dst_type = argv[++i];
            p.use_dram_dst = (dst_type == "dram");
        } else if (a == "--send-core" && need(i, 1)) {
            int x, y;
            if (!parse_xy(argv[++i], x, y)) {
                usage(argv[0]);
                return false;
            }
            p.sender_core = {x, y};
        } else if (a == "--recv-core" && need(i, 1)) {
            int x, y;
            if (!parse_xy(argv[++i], x, y)) {
                usage(argv[0]);
                return false;
            }
            p.receiver_core = {x, y};
        } else if (a == "--iters" && need(i, 1)) {
            run.iters = std::stoi(argv[++i]);
        } else if (a == "--warmup" && need(i, 1)) {
            run.warmup = std::stoi(argv[++i]);
        } else if (a == "--trace-iters" && need(i, 1)) {
            run.trace_iters = std::stoi(argv[++i]);
        } else if (a == "--no-trace") {
            run.no_trace = true;
        } else if (a == "--enable-sync") {
            run.enable_sync = true;
        } else if (a == "--csv" && need(i, 1)) {
            run.csv_path = argv[++i];
        } else {
            usage(argv[0]);
            return false;
        }
    }

    if (src_dev_str.empty() || dst_dev_str.empty() || p.tensor_bytes == 0) {
        usage(argv[0]);
        return false;
    }

    uint32_t src_mesh = 0, dst_mesh = 0;
    int src_chip = 0, dst_chip = 0;
    if (!parse_mesh_chip(src_dev_str, src_mesh, src_chip)) {
        usage(argv[0]);
        return false;
    }
    if (!parse_mesh_chip(dst_dev_str, dst_mesh, dst_chip)) {
        usage(argv[0]);
        return false;
    }

    p.mesh_id = src_mesh;  // assume same mesh
    p.src_chip = static_cast<tt::ChipId>(src_chip);
    p.dst_chip = static_cast<tt::ChipId>(dst_chip);

    return true;
}

void append_csv_if_requested(const RunOptions& run, const PerfParams& p, const PerfStats& stats) {
    if (run.csv_path.empty()) {
        return;
    }

    std::filesystem::create_directories(std::filesystem::path(run.csv_path).parent_path());
    bool newfile = !std::ifstream(run.csv_path).good();

    std::ofstream ofs(run.csv_path, std::ios::app);
    if (!ofs) {
        log_error(tt::LogTest, "Failed to open CSV path for append: {}", run.csv_path);
        return;
    }
    if (newfile) {
        ofs << "mesh,src_chip,dst_chip,send_x,send_y,recv_x,recv_y,sizeB,pageB,iters,warmup,"
               "p50_ms,p95_ms,mean_GB_s,p50_GB_s,p10_GB_s,cv_GB_s_pct\n";
    }
    ofs << p.mesh_id << "," << p.src_chip << "," << p.dst_chip << "," << p.sender_core.x << "," << p.sender_core.y
        << "," << p.receiver_core.x << "," << p.receiver_core.y << "," << p.tensor_bytes << "," << p.page_size << ","
        << run.iters << "," << run.warmup << "," << stats.p50_ms << "," << stats.p95_ms << "," << stats.mean_GB_s << ","
        << stats.p50_GB_s << "," << stats.p10_GB_s << "," << stats.cv_GB_s_pct << "\n";

    log_info(tt::LogTest, "Appended CSV row to {}", run.csv_path);
}

// Validate fabric ethernet link health before measuring unicast bandwidth.
//
// This benchmark reports GB/s = bytes / wall-time for a fixed payload. A degraded ethernet link
// (one that has retrained, or is accumulating CRC / uncorrected-codeword errors) adds backpressure
// on the fabric and inflates the measured time for the same bytes -- producing a stable "slow mode"
// that trips the run_unicast_sweep.py bandwidth check with low run-to-run variance, looking like a
// bandwidth regression when it is really a hardware link fault. On T3K this state is otherwise
// undetected: Cluster::disable_ethernet_cores_with_retrain() only inspects retrain counts on UBB
// (galaxy) boards. This gate mirrors validate_fabric_link_health_for_performance_tests() in the
// sibling "Fabric BW & Latency" ubench (tt_fabric_test_common.hpp) and surfaces the degraded link
// with an actionable diagnostic instead of a misleading bandwidth failure.
//
// Only counts that indicate an actual fault are checked (retrain, CRC errors, uncorrected
// codewords). Corrected codewords are expected to be non-zero under normal FEC operation and are
// NOT treated as a fault. Links that are down/unconnected (e.g. unused QSFP expansion ports) are
// skipped.
bool validate_fabric_link_health() {
    auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = metal_context.get_cluster();
    const auto& hal = metal_context.hal();
    const bool is_wormhole_b0 = cluster.arch() == tt::ARCH::WORMHOLE_B0;

    auto retrain_count_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    // CRC / uncorrected-codeword registers are only reliably readable on Wormhole today
    // (mirrors the WORMHOLE_B0 guard in the ReportSystemHealth report).
    auto crc_addr =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    auto uncorr_addr =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);

    struct DegradedLink {
        tt::ChipId chip_id;
        CoreCoord eth_core;
        uint32_t retrain_count;
        uint32_t crc_error_count;
        uint64_t uncorrected_codeword_count;
    };
    std::vector<DegradedLink> degraded_links;
    uint32_t links_checked = 0;

    std::vector<uint32_t> read_vec;
    for (const auto& chip_id : cluster.all_chip_ids()) {
        const auto& soc_desc = cluster.get_soc_desc(chip_id);
        for (const auto& [eth_core, chan] : soc_desc.logical_eth_core_to_chan_map) {
            // Only check links that are actually up (carrying fabric traffic); skip
            // down/unconnected ports such as unused QSFP expansion channels.
            if (!cluster.is_ethernet_link_up(chip_id, eth_core)) {
                continue;
            }
            links_checked++;

            tt_cxy_pair virtual_eth_core(
                chip_id,
                cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, tt::CoreType::ETH));

            uint32_t retrain_count_val = 0;
            cluster.read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
            retrain_count_val = read_vec[0];

            uint32_t crc_error_val = 0, uncorr_val_lo = 0, uncorr_val_hi = 0;
            if (is_wormhole_b0) {
                cluster.read_core(&crc_error_val, sizeof(uint32_t), virtual_eth_core, crc_addr);
                cluster.read_core(&uncorr_val_hi, sizeof(uint32_t), virtual_eth_core, uncorr_addr);
                cluster.read_core(&uncorr_val_lo, sizeof(uint32_t), virtual_eth_core, uncorr_addr + sizeof(uint32_t));
            }
            uint64_t uncorrected_codeword_count =
                (static_cast<uint64_t>(uncorr_val_hi) << 32) | static_cast<uint64_t>(uncorr_val_lo);

            if (retrain_count_val != 0 || crc_error_val != 0 || uncorrected_codeword_count != 0) {
                degraded_links.push_back(
                    {chip_id, eth_core, retrain_count_val, crc_error_val, uncorrected_codeword_count});
            }
        }
    }

    if (!degraded_links.empty()) {
        log_error(tt::LogTest, "=== FABRIC ETHERNET LINK HEALTH CHECK FAILED ===");
        log_error(
            tt::LogTest,
            "Cannot run unicast bandwidth benchmark - {} of {} active ethernet link(s) are degraded.",
            degraded_links.size(),
            links_checked);
        log_error(
            tt::LogTest,
            "A retraining or error-accumulating link inflates measured time and causes a stable low-bandwidth "
            "mode. This is a hardware/link-health issue, not a test-config issue.");
        for (const auto& link : degraded_links) {
            log_error(
                tt::LogTest,
                "  chip {} eth core {}: retrain_count={} crc_errors={} uncorrected_codewords={}",
                link.chip_id,
                link.eth_core.str(),
                link.retrain_count,
                link.crc_error_count,
                link.uncorrected_codeword_count);
        }
        return false;
    }

    log_info(
        tt::LogTest,
        "Fabric ethernet link health check passed: {} active links, 0 retrains / CRC errors / uncorrected codewords",
        links_checked);
    return true;
}

}  // anonymous namespace

int main(int argc, char** argv) {
    RunOptions run;
    PerfParams p;
    std::string src_dev_str, dst_dev_str;

    if (!parse_cli_or_usage(argc, argv, run, p, src_dev_str, dst_dev_str)) {
        return 2;
    }

    log_info(
        tt::LogTest,
        "Starting unicast bench: src={} dst={} sizeB={} pageB={} send_core=[{},{}] recv_core=[{},{}] iters={} "
        "warmup={}",
        src_dev_str,
        dst_dev_str,
        p.tensor_bytes,
        p.page_size,
        p.sender_core.x,
        p.sender_core.y,
        p.receiver_core.x,
        p.receiver_core.y,
        run.iters,
        run.warmup);

    // Bring up the fixture env and run
    Fixture fixture;
    fixture.setup();

    // Gate on ethernet link health: a degraded/retraining link inflates measured time and produces a
    // stable low-bandwidth mode that fails the sweep's bandwidth check without being a real regression.
    if (!validate_fabric_link_health()) {
        fixture.teardown();
        return 1;  // Hard exit - cannot run bandwidth benchmark on degraded ethernet links
    }

    warmup_once(&fixture, p, run.warmup);
    auto stats = run_repeated(&fixture, p, /*warmup_iters=*/0, /*iters=*/run.iters);

    fixture.teardown();

    // Human-readable result
    log_info(
        tt::LogTest,
        "Result: p50_ms={} p95_ms={} mean_GB_s={} p50_GB_s={} p10_GB_s={} cv_GB_s_pct={}",
        stats.p50_ms,
        stats.p95_ms,
        stats.mean_GB_s,
        stats.p50_GB_s,
        stats.p10_GB_s,
        stats.cv_GB_s_pct);

    // Optional CSV artifact
    append_csv_if_requested(run, p, stats);

    log_info(tt::LogTest, "Unicast bench completed successfully.");
    return 0;
}
