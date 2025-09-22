// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"

// We reuse the existing test fixture infra to bring devices up.
struct Fixture : public ::tt::tt_fabric::fabric_router_tests::Fabric2DFixture {
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }

    static void suite_setup() {
        ::tt::tt_fabric::fabric_router_tests::BaseFabricFixture::DoSetUpTestSuite(
            tt::tt_fabric::FabricConfig::FABRIC_2D);
    }
    static void suite_teardown() { ::tt::tt_fabric::fabric_router_tests::BaseFabricFixture::DoTearDownTestSuite(); }
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
    bool no_trace = false;     // reserved
    bool enable_sync = false;  // reserved
    std::string csv_path;      // if non-empty, append one row
};

static void usage(const char* argv0) {
    log_error(
        tt::LogTest,
        "{}",
        fmt::format(
            R"(Usage:
  {} --src-dev <mesh:chip|chip> --dst-dev <mesh:chip|chip> --size <bytes>
         [--page <bytes>] [--src-type <l1|dram|single_bank>] [--dst-type <l1|dram|single_bank>]
         [--send-core x,y] [--recv-core x,y]
         [--iters N] [--warmup N]
         [--no-trace] [--enable-sync]
         [--csv <path>]

Notes:
- This binary runs ONE configuration. For sweeps, call it in a loop from a script.
- --src-type/--dst-type are accepted for future use; current code uses DRAM src and L1/DRAM dst as in your kernels.
)",
            argv0));
}

static bool parse_mesh_chip(const std::string& s, uint32_t& mesh, int& chip) {
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

static bool parse_xy(const std::string& s, int& x, int& y) {
    auto comma = s.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    x = std::stoi(s.substr(0, comma));
    y = std::stoi(s.substr(comma + 1));
    return true;
}

// Fills RunOptions + PerfParams + raw src/dst strings. Returns false on bad args.
static bool parse_cli_or_usage(
    int argc, char** argv, RunOptions& run, PerfParams& p, std::string& src_dev_str, std::string& dst_dev_str) {
    // Workload defaults
    p.mesh_id = 0;
    p.src_chip = 0;
    p.dst_chip = 1;
    p.page_size = 2048;
    p.tensor_bytes = 128 * p.page_size;
    p.use_dram_dst = false;
    p.sender_core = {0, 0};
    p.receiver_core = {0, 0};

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
    p.src_chip = static_cast<chip_id_t>(src_chip);
    p.dst_chip = static_cast<chip_id_t>(dst_chip);

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
    Fixture::suite_setup();
    Fixture fixture;
    fixture.setup();

    warmup_once(&fixture, p, run.warmup);
    auto stats = run_repeated(&fixture, p, /*warmup_iters=*/0, /*iters=*/run.iters);

    fixture.teardown();
    Fixture::suite_teardown();

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
