// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// N-chip all-gather step over PCIe peer-to-peer. Every chip runs one persistent worker that
// pushes its payload into every peer's L1 and flags it; the step completes when all peers'
// payloads have landed. Timed on-device, launch excluded. Companion of test_pcie_p2p.cpp.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "pcie_p2p_common.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::pcie_p2p;

namespace {

struct Options {
    std::vector<int> chips{0, 1, 2, 3};
    std::vector<uint32_t> sizes{64, 256, 1024, 4096, 16384, 65536};
    uint32_t iters = 1000;
    uint32_t warmup = 20;
    uint32_t region_base = 13;  // regions region_base .. region_base + n - 2 on each chip
    uint64_t noc_base = 32ull << 30;
    std::string ordering = "relaxed";
    uint32_t flag_bytes = 64;
    CoreCoord core{0, 0};
    double timeout_s = 3.0;
    bool force = false;
    bool dump = false;
};

std::vector<uint32_t> parse_list_u32(const std::string& s) {
    std::vector<uint32_t> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t c = s.find(',', p);
        if (c == std::string::npos) {
            c = s.size();
        }
        v.push_back(std::stoul(s.substr(p, c - p), nullptr, 0));
        p = c + 1;
    }
    return v;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            TT_FATAL(i + 1 < argc, "missing value for {}", a);
            return argv[++i];
        };
        if (a == "--chips") {
            o.chips.clear();
            for (auto c : parse_list_u32(next())) {
                o.chips.push_back(int(c));
            }
        } else if (a == "--sizes") {
            o.sizes = parse_list_u32(next());
        } else if (a == "--iters") {
            o.iters = std::stoul(next());
        } else if (a == "--warmup") {
            o.warmup = std::stoul(next());
        } else if (a == "--region-base") {
            o.region_base = std::stoul(next());
        } else if (a == "--noc-base") {
            o.noc_base = std::stoull(next(), nullptr, 0);
        } else if (a == "--ordering") {
            o.ordering = next();
        } else if (a == "--flag-bytes") {
            o.flag_bytes = std::stoul(next());
        } else if (a == "--timeout") {
            o.timeout_s = std::stod(next());
        } else if (a == "--core") {
            auto s = next();
            auto c = s.find(',');
            o.core = {std::stoul(s.substr(0, c)), std::stoul(s.substr(c + 1))};
        } else if (a == "--force") {
            o.force = true;
        } else if (a == "--dump") {
            o.dump = true;
        } else if (a == "-h" || a == "--help") {
            fmt::print(
                "usage: test_pcie_p2p_allgather [--chips 0,1,2,3] [--sizes 64,256,...] [--iters N] [--warmup N]\n"
                "        [--ordering relaxed|strict|posted] [--region-base R] [--noc-base HEX] [--core x,y]\n"
                "        [--flag-bytes 16|64] [--timeout S] [--force] [--dump]\n");
            exit(0);
        } else {
            TT_THROW("unknown arg {}", a);
        }
    }
    TT_FATAL(o.chips.size() >= 2 && o.chips.size() <= 8, "need 2..8 chips");
    TT_FATAL(o.region_base + o.chips.size() - 1 <= kIatuRegions, "not enough iATU regions from --region-base");
    TT_FATAL(o.iters + 8 <= 16384, "iters too large for the L1 results buffer");
    for (auto s : o.sizes) {
        TT_FATAL(s >= 16 && s % 16 == 0, "sizes must be multiples of 16 B");
    }
    return o;
}

struct Stats {
    double min, p50, p99, max, mean;
};
Stats stats_ns(std::vector<uint32_t> cyc, double mhz) {
    std::sort(cyc.begin(), cyc.end());
    auto ns = [&](uint32_t c) { return c * 1000.0 / mhz; };
    double sum = 0;
    for (auto c : cyc) {
        sum += c;
    }
    return {
        ns(cyc.front()),
        ns(cyc[cyc.size() / 2]),
        ns(cyc[cyc.size() * 99 / 100]),
        ns(cyc.back()),
        ns(uint32_t(sum / cyc.size()))};
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    const uint32_t n = opt.chips.size();

    auto meshes = distributed::MeshDevice::create_unit_meshes(opt.chips);
    auto& cluster = MetalContext::instance().get_cluster();
    TT_FATAL(cluster.arch() == tt::ARCH::BLACKHOLE, "Blackhole only");

    std::vector<Chip> chips(n);
    for (uint32_t i = 0; i < n; ++i) {
        init_chip(chips[i], opt.chips[i], meshes.at(opt.chips[i]));
    }
    if (opt.dump) {
        for (auto& c : chips) {
            dump_chip(c);
        }
    }

    // ---- L1 layout (same on every chip) ----
    uint32_t U = 0;
    for (auto& c : chips) {
        U = std::max(U, c.l1_unreserved);
    }
    const uint32_t max_size = *std::max_element(opt.sizes.begin(), opt.sizes.end());
    const uint32_t FLAGS = U;             // n x 64 B
    const uint32_t FLAG_SRC = U + 0x200;  // 64 B
    const uint32_t RESULTS = U + 0x1000;  // 64 KiB
    const uint32_t SRC = U + 0x11000;     // payload
    const uint32_t RECV = (SRC + max_size + 0xFFF) & ~0xFFFu;
    const uint32_t stride = (max_size + 0xFFF) & ~0xFFFu;
    TT_FATAL(
        RECV + 2 * n * stride <= kL1Size,
        "layout does not fit in L1: reduce sizes or chips (need {} B)",
        RECV + 2 * n * stride);
    fmt::print(
        "L1 layout: FLAGS 0x{:x} RESULTS 0x{:x} SRC 0x{:x} RECV 0x{:x} stride 0x{:x} (max payload {} B)\n",
        FLAGS,
        RESULTS,
        SRC,
        RECV,
        stride,
        max_size);

    // ---- windows and iATU regions: peer p is reachable from every chip at noc_base + p * 2 MiB ----
    fmt::print("\n=== setup: {} chips, ordering {} ===\n", n, opt.ordering);
    const uint64_t ord = tlb_ordering(opt.ordering);
    std::vector<uint64_t> win_off(n);
    for (uint32_t i = 0; i < n; ++i) {
        const CoreCoord v = chips[i].dev->virtual_core_from_logical_core(opt.core, CoreType::WORKER);
        win_off[i] = make_inbound_window(chips[i], v.x, v.y, 0, ord, "worker L1");
    }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t r = opt.region_base;
        for (uint32_t p = 0; p < n; ++p) {
            if (p == i) {
                continue;
            }
            make_outbound(chips[i], r++, opt.noc_base + uint64_t(p) * kTlb2M, chips[p], win_off[p], opt.force);
        }
    }
    for (uint32_t i = 0; i < n; ++i) {  // host self-check of every inbound window
        auto* win = reinterpret_cast<volatile uint32_t*>(chips[i].tlbs[0]->get_base());
        win[RECV / 4] = 0xC0DE0000u | i;
        auto v = read_l1(chips[i].dev, opt.core, RECV, 64);
        TT_FATAL(v[0] == (0xC0DE0000u | i), "chip {} inbound window does not reach its worker L1", chips[i].id);
    }

    const double mhz = chips[0].dev->get_clock_rate_mhz();
    const uint64_t timeout_cycles = uint64_t(opt.timeout_s * mhz * 1e6);
    const std::string kpath = "tests/tt_metal/tt_metal/perf_microbenchmark/pcie_p2p/kernels/p2p_allgather.cpp";
    const bool slow = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    fmt::print(
        "\n=== all-gather step latency, {} chips, {} iters (+{} warmup), clock {} MHz ===\n",
        n,
        opt.iters,
        opt.warmup,
        mhz);
    fmt::print(
        "{:>8} | {:>9} {:>9} {:>9} {:>9} {:>9} | per-chip step ns{:>{}}| viol\n",
        "bytes",
        "step ns",
        "mean ns",
        "p50 ns",
        "p99 ns",
        "max ns",
        "",
        std::max<int>(1, n * 7 - 15));
    fmt::print(
        "           (step = steady-state period of back-to-back all-gathers, max over chips; mean/p50/p99/max = "
        "per-iteration completion time, all chips)\n");
    for (uint32_t size : opt.sizes) {
        // reset flags, receive slots, results; fill payload
        for (uint32_t i = 0; i < n; ++i) {
            write_l1(chips[i].dev, opt.core, FLAGS, std::vector<uint32_t>(0x1000 / 4, 0));
            write_l1(chips[i].dev, opt.core, RECV, std::vector<uint32_t>(2 * n * stride / 4, 0));
            std::vector<uint32_t> pat(size / 4);
            for (uint32_t w = 0; w < pat.size(); ++w) {
                pat[w] = (0x50000000u | (i << 24)) + w;
            }
            write_l1(chips[i].dev, opt.core, SRC, pat);
        }
        std::vector<distributed::MeshWorkload> wls(n);
        for (uint32_t i = 0; i < n; ++i) {
            Program p;
            auto k = CreateKernel(
                p,
                kpath,
                opt.core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            SetRuntimeArgs(
                p,
                k,
                opt.core,
                std::vector<uint32_t>{
                    i,
                    n,
                    uint32_t(chips[i].pcie_noc_base),
                    uint32_t(chips[i].pcie_noc_base >> 32),
                    uint32_t(opt.noc_base),
                    uint32_t(opt.noc_base >> 32),
                    FLAGS,
                    FLAG_SRC,
                    RESULTS,
                    SRC,
                    RECV,
                    stride,
                    size,
                    opt.iters + opt.warmup,
                    uint32_t(timeout_cycles),
                    uint32_t(timeout_cycles >> 32),
                    opt.flag_bytes,
                    opt.warmup});
            auto zero = distributed::MeshCoordinate::zero_coordinate(chips[i].mesh->shape().dims());
            wls[i].add_program(distributed::MeshCoordinateRange(zero, zero), std::move(p));
        }
        if (slow) {
            std::vector<std::thread> th;
            for (uint32_t i = 0; i < n; ++i) {
                th.emplace_back(
                    [&, i] { distributed::EnqueueMeshWorkload(chips[i].mesh->mesh_command_queue(), wls[i], true); });
            }
            for (auto& t : th) {
                t.join();
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                distributed::EnqueueMeshWorkload(chips[i].mesh->mesh_command_queue(), wls[i], false);
            }
            for (uint32_t i = 0; i < n; ++i) {
                distributed::Finish(chips[i].mesh->mesh_command_queue());
            }
        }
        // collect
        std::vector<uint32_t> all;
        std::string per_chip;
        double step_max = 0;
        uint32_t viol = 0;
        bool ok = true;
        for (uint32_t i = 0; i < n; ++i) {
            auto r = read_l1(chips[i].dev, opt.core, RESULTS, 0x40 + (opt.iters + opt.warmup) * 4);
            if (r[0] != 1) {
                ok = false;
                fmt::print(
                    "  chip {}: status {} after {} iterations (waiting on peer {}), violations {}\n",
                    chips[i].id,
                    r[0],
                    r[1],
                    r[3],
                    r[2]);
                continue;
            }
            viol += r[2];
            std::vector<uint32_t> cyc(r.begin() + 8 + opt.warmup, r.begin() + 8 + opt.iters + opt.warmup);
            all.insert(all.end(), cyc.begin(), cyc.end());
            const double period_ns = ((uint64_t(r[5]) << 32) | r[4]) * 1000.0 / mhz / opt.iters;
            step_max = std::max(step_max, period_ns);
            per_chip += fmt::format("{:>7.0f}", period_ns);
        }
        if (!ok) {
            fmt::print("{:>8} | FAILED\n", size);
            break;
        }
        auto s = stats_ns(all, mhz);
        fmt::print(
            "{:>8} | {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} | {} | {}\n",
            size,
            step_max,
            s.mean,
            s.p50,
            s.p99,
            s.max,
            per_chip,
            viol);
    }

    for (auto& c : chips) {
        for (auto r : c.programmed_regions) {
            c.iatu.disable(r);
        }
        c.programmed_regions.clear();
        c.tlbs.clear();
    }
    for (auto& m : meshes) {
        m.second->close();
    }
    return 0;
}
