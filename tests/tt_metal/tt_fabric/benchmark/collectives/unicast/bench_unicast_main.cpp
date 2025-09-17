// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"

// We reuse the existing test fixture infra to bring devices up.
struct Fixture : public ::tt::tt_fabric::fabric_router_tests::Fabric2DFixture {
    void TestBody() override {}
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }

    // Public wrappers so main() can prepare/cleanup the suite environment
    static void suite_setup() {
        ::tt::tt_fabric::fabric_router_tests::BaseFabricFixture::DoSetUpTestSuite(
            tt::tt_fabric::FabricConfig::FABRIC_2D);
    }
    static void suite_teardown() { ::tt::tt_fabric::fabric_router_tests::BaseFabricFixture::DoTearDownTestSuite(); }
};

// Shorthands
using tt::tt_fabric::bench::find_device_by_id;
using tt::tt_fabric::bench::PerfParams;
using tt::tt_fabric::bench::PerfStats;
using tt::tt_fabric::bench::run_repeated;
using tt::tt_fabric::bench::warmup_once;

// Simple parsers (no deps)
static bool parse_mesh_chip(const std::string& s, uint32_t& mesh, int& chip) {
    // Accept "m:c" or just "c"
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

static void usage(const char* argv0) {
    std::cerr <<
        R"(Usage:
  )" << argv0 << R"( --src-dev <mesh:chip|chip> --dst-dev <mesh:chip|chip> --size <bytes>
         [--page <bytes>] [--src-type <l1|dram|single_bank>] [--dst-type <l1|dram|single_bank>]
         [--send-core x,y] [--recv-core x,y]
         [--iters N] [--warmup N]
         [--no-trace] [--enable-sync]
         [--csv <path>] [--format <human|csv|json>]

Notes:
- This binary runs ONE configuration. For sweeps, call it in a loop from a script.
- --src-type/--dst-type are accepted for future use; current code uses DRAM src and L1/DRAM dst as in your kernels.
)";
}

int main(int argc, char** argv) {
    // Defaults
    PerfParams p;
    p.mesh_id = 0;
    p.src_chip = 0;
    p.dst_chip = 1;
    p.page_size = 2048;
    p.tensor_bytes = 128 * p.page_size;
    p.use_dram_dst = false;
    p.sender_core = {0, 0};
    p.receiver_core = {0, 0};

    int iters = 5, warmup = 1;
    bool no_trace = false;     // TODO: wire to profiler off
    bool enable_sync = false;  // TODO: add a start barrier
    std::string src_dev_str, dst_dev_str;
    std::string src_type = "dram", dst_type = "l1";  // placeholders for future
    std::string format = "human";                    // "human" | "csv" | "json"
    std::string csv_path;                            // if non-empty, append one row

    // Parse argv
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) {
                usage(argv[0]);
                return false;
            }
            return true;
        };

        if (a == "--src-dev" && need(1)) {
            src_dev_str = argv[++i];
        } else if (a == "--dst-dev" && need(1)) {
            dst_dev_str = argv[++i];
        } else if (a == "--size" && need(1)) {
            p.tensor_bytes = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--page" && need(1)) {
            p.page_size = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--src-type" && need(1)) {
            src_type = argv[++i];
        } else if (a == "--dst-type" && need(1)) {
            dst_type = argv[++i];
            p.use_dram_dst = (dst_type == "dram");
        } else if (a == "--send-core" && need(1)) {
            int x, y;
            if (!parse_xy(argv[++i], x, y)) {
                usage(argv[0]);
                return 2;
            }
            p.sender_core = {x, y};
        } else if (a == "--recv-core" && need(1)) {
            int x, y;
            if (!parse_xy(argv[++i], x, y)) {
                usage(argv[0]);
                return 2;
            }
            p.receiver_core = {x, y};
        } else if (a == "--iters" && need(1)) {
            iters = std::stoi(argv[++i]);
        } else if (a == "--warmup" && need(1)) {
            warmup = std::stoi(argv[++i]);
        } else if (a == "--no-trace") {
            no_trace = true;
        } else if (a == "--enable-sync") {
            enable_sync = true;
        } else if (a == "--csv" && need(1)) {
            csv_path = argv[++i];
        } else if (a == "--format" && need(1)) {
            format = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (src_dev_str.empty() || dst_dev_str.empty() || p.tensor_bytes == 0) {
        usage(argv[0]);
        return 2;
    }
    uint32_t src_mesh = 0, dst_mesh = 0;
    int src_chip = 0, dst_chip = 0;
    if (!parse_mesh_chip(src_dev_str, src_mesh, src_chip)) {
        usage(argv[0]);
        return 2;
    }
    if (!parse_mesh_chip(dst_dev_str, dst_mesh, dst_chip)) {
        usage(argv[0]);
        return 2;
    }
    p.mesh_id = src_mesh;  // assume same mesh
    p.src_chip = static_cast<chip_id_t>(src_chip);
    p.dst_chip = static_cast<chip_id_t>(dst_chip);

    // Bring up the fixture env and run
    Fixture::suite_setup();
    Fixture fixture;
    fixture.setup();
    warmup_once(&fixture, p, warmup);
    auto stats = run_repeated(&fixture, p, /*warmup_iters=*/0, /*iters=*/iters);
    fixture.teardown();
    Fixture::suite_teardown();

    // Output
    if (format == "human") {
        std::cout << "[bench] src=" << src_dev_str << " dst=" << dst_dev_str << " size=" << p.tensor_bytes
                  << " page=" << p.page_size << " iters=" << iters << " warmup=" << warmup << " p50_ms=" << stats.p50_ms
                  << " p95_ms=" << stats.p95_ms << " mean_gbps=" << stats.mean_gbps << " p50_gbps=" << stats.p50_gbps
                  << " p10_gbps=" << stats.p10_gbps << " cv_gbps_pct=" << stats.cv_gbps_pct << "\n";
    } else if (format == "csv") {
        // print a single CSV row to stdout
        std::cout << "mesh,src_chip,dst_chip,send_x,send_y,recv_x,recv_y,sizeB,pageB,iters,warmup,"
                  << "p50_ms,p95_ms,mean_gbps,p50_gbps,p10_gbps,cv_gbps_pct\n";
        std::cout << p.mesh_id << "," << p.src_chip << "," << p.dst_chip << "," << p.sender_core.x << ","
                  << p.sender_core.y << "," << p.receiver_core.x << "," << p.receiver_core.y << "," << p.tensor_bytes
                  << "," << p.page_size << "," << iters << "," << warmup << "," << stats.p50_ms << "," << stats.p95_ms
                  << "," << stats.mean_gbps << "," << stats.p50_gbps << "," << stats.p10_gbps << ","
                  << stats.cv_gbps_pct << "\n";
    } else if (format == "json") {
        std::cout << "{"
                  << "\"mesh\":" << p.mesh_id << ",\"src_chip\":" << p.src_chip << ",\"dst_chip\":" << p.dst_chip
                  << ",\"send_core\":[" << p.sender_core.x << "," << p.sender_core.y << "]"
                  << ",\"recv_core\":[" << p.receiver_core.x << "," << p.receiver_core.y << "]"
                  << ",\"sizeB\":" << p.tensor_bytes << ",\"pageB\":" << p.page_size << ",\"iters\":" << iters
                  << ",\"warmup\":" << warmup << ",\"p50_ms\":" << stats.p50_ms << ",\"p95_ms\":" << stats.p95_ms
                  << ",\"mean_gbps\":" << stats.mean_gbps << "}\n";
    }

    if (!csv_path.empty()) {
        // append one row; create header if new
        bool newfile = false;
        {
            std::ifstream chk(csv_path);
            newfile = !chk.good();
        }
        std::ofstream ofs(csv_path, std::ios::app);
        if (newfile) {
            ofs << "mesh,src_chip,dst_chip,send_x,send_y,recv_x,recv_y,sizeB,pageB,iters,warmup,"
                   "p50_ms,p95_ms,mean_gbps,p50_gbps,p10_gbps,cv_gbps_pct\n";
        }
        ofs << p.mesh_id << "," << p.src_chip << "," << p.dst_chip << "," << p.sender_core.x << "," << p.sender_core.y
            << "," << p.receiver_core.x << "," << p.receiver_core.y << "," << p.tensor_bytes << "," << p.page_size
            << "," << iters << "," << warmup << "," << stats.p50_ms << "," << stats.p95_ms << "," << stats.mean_gbps
            << "," << stats.p50_gbps << "," << stats.p10_gbps << "," << stats.cv_gbps_pct << "\n";
    }

    return 0;
}
