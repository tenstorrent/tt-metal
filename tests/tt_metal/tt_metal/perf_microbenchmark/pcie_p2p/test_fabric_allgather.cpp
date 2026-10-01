// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// N-chip all-gather step over TT-Fabric (Ethernet), the baseline for test_pcie_p2p_allgather.
// Identical protocol and timing method: one persistent worker per chip, timed on-device.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kL1Size = 1536u * 1024u;

struct Options {
    std::vector<int> chips{0, 1, 2, 3};
    std::vector<uint32_t> sizes{64, 256, 1024, 4096, 16384, 65536};
    uint32_t iters = 1000;
    uint32_t warmup = 20;
    CoreCoord core{0, 0};
    double timeout_s = 3.0;
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
        } else if (a == "--timeout") {
            o.timeout_s = std::stod(next());
        } else if (a == "--core") {
            auto s = next();
            auto c = s.find(',');
            o.core = {std::stoul(s.substr(0, c)), std::stoul(s.substr(c + 1))};
        } else if (a == "-h" || a == "--help") {
            fmt::print(
                "usage: test_fabric_allgather [--chips 0,1,2,3] [--sizes 64,...] [--iters N] [--warmup N] [--core x,y] "
                "[--timeout S]\n");
            exit(0);
        } else {
            TT_THROW("unknown arg {}", a);
        }
    }
    TT_FATAL(o.chips.size() >= 2 && o.chips.size() <= 8, "need 2..8 chips");
    TT_FATAL(o.iters + 8 <= 16384, "iters too large for the L1 results buffer");
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

    // Fabric routers are brought up for every device in the system, so open them all.
    const auto num_devices = GetNumAvailableDevices();
    std::vector<int> all_ids(num_devices);
    for (unsigned i = 0; i < num_devices; ++i) {
        all_ids[i] = int(i);
    }
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    const auto dispatch_core_config = MetalContext::instance().resolve_dispatch_core_config();
    auto meshes = distributed::MeshDevice::create_unit_meshes(
        all_ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
    auto& control_plane = MetalContext::instance().get_control_plane();

    std::vector<IDevice*> devs(n);
    std::vector<tt::tt_fabric::FabricNodeId> nodes;
    for (uint32_t i = 0; i < n; ++i) {
        devs[i] = meshes.at(opt.chips[i])->get_devices()[0];
        nodes.push_back(control_plane.get_fabric_node_id_from_physical_chip_id(opt.chips[i]));
        fmt::print(
            "chip {} -> fabric node mesh {} device {}\n", opt.chips[i], *nodes.back().mesh_id, nodes.back().chip_id);
    }
    const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    fmt::print("fabric max payload per packet: {} B\n", max_payload);

    uint32_t U = 0;
    for (auto* d : devs) {
        U = std::max<uint32_t>(U, d->allocator()->get_base_allocator_addr(HalMemType::L1));
    }
    const uint32_t max_size = *std::max_element(opt.sizes.begin(), opt.sizes.end());
    const uint32_t FLAGS = U;
    const uint32_t RESULTS = U + 0x1000;
    const uint32_t SRC = U + 0x11000;
    const uint32_t RECV = (SRC + max_size + 0xFFF) & ~0xFFFu;
    const uint32_t stride = (max_size + 0xFFF) & ~0xFFFu;
    TT_FATAL(RECV + 2 * n * stride <= kL1Size, "layout does not fit in L1 (need {} B)", RECV + 2 * n * stride);

    const double mhz = devs[0]->get_clock_rate_mhz();
    const uint64_t timeout_cycles = uint64_t(opt.timeout_s * mhz * 1e6);
    const std::string kpath = "tests/tt_metal/tt_metal/perf_microbenchmark/pcie_p2p/kernels/fabric_allgather.cpp";
    const bool slow = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    fmt::print(
        "\n=== fabric (2D, Ethernet) all-gather step latency, {} chips, {} iters (+{} warmup), clock {} MHz ===\n",
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
    for (uint32_t size : opt.sizes) {
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<uint32_t> z(0x1000 / 4, 0), zr(2 * n * stride / 4, 0), pat(size / 4);
            for (uint32_t w = 0; w < pat.size(); ++w) {
                pat[w] = (0x50000000u | (i << 24)) + w;
            }
            detail::WriteToDeviceL1(devs[i], opt.core, FLAGS, z);
            detail::WriteToDeviceL1(devs[i], opt.core, RECV, zr);
            detail::WriteToDeviceL1(devs[i], opt.core, SRC, pat);
        }
        std::vector<distributed::MeshWorkload> wls(n);
        for (uint32_t i = 0; i < n; ++i) {
            Program prog;
            std::map<std::string, std::string> defines = {{"FABRIC_2D", ""}};
            auto k = CreateKernel(
                prog,
                kpath,
                opt.core,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
            std::vector<uint32_t> args{
                i,
                n,
                FLAGS,
                RESULTS,
                SRC,
                RECV,
                stride,
                size,
                opt.iters + opt.warmup,
                opt.warmup,
                uint32_t(timeout_cycles),
                uint32_t(timeout_cycles >> 32),
                max_payload};
            // one connection per distinct outgoing router
            std::vector<std::pair<uint32_t, uint32_t>> conns;  // (peer index used to open it, link idx)
            std::vector<CoreCoord> conn_eth;
            std::vector<uint32_t> peer_args;
            for (uint32_t p = 0; p < n; ++p) {
                if (p == i) {
                    continue;
                }
                auto links = tt::tt_fabric::get_forwarding_link_indices(nodes[i], nodes[p]);
                TT_FATAL(!links.empty(), "no fabric link from chip {} to chip {}", opt.chips[i], opt.chips[p]);
                const uint32_t link = links[0];
                const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(nodes[i], nodes[p], link);
                uint32_t ci = 0;
                for (; ci < conn_eth.size(); ++ci) {
                    if (conn_eth[ci] == eth) {
                        break;
                    }
                }
                if (ci == conn_eth.size()) {
                    conn_eth.push_back(eth);
                    conns.push_back({p, link});
                }
                const CoreCoord dst_v = devs[p]->worker_core_from_logical_core(opt.core);
                peer_args.insert(
                    peer_args.end(),
                    {p, ci, nodes[p].chip_id, *nodes[p].mesh_id, uint32_t(dst_v.x), uint32_t(dst_v.y)});
                if (size == opt.sizes.front()) {
                    fmt::print(
                        "  chip {} -> chip {}: link {} via eth core ({},{}) conn {}\n",
                        opt.chips[i],
                        opt.chips[p],
                        link,
                        eth.x,
                        eth.y,
                        ci);
                }
            }
            TT_FATAL(conns.size() <= 4, "more than 4 distinct routers");
            args.push_back(conns.size());
            args.insert(args.end(), peer_args.begin(), peer_args.end());
            for (auto [peer, link] : conns) {
                tt::tt_fabric::append_fabric_connection_rt_args(nodes[i], nodes[peer], link, prog, opt.core, args);
            }
            SetRuntimeArgs(prog, k, opt.core, args);
            auto zero = distributed::MeshCoordinate::zero_coordinate(meshes.at(opt.chips[i])->shape().dims());
            wls[i].add_program(distributed::MeshCoordinateRange(zero, zero), std::move(prog));
        }
        if (slow) {
            std::vector<std::thread> th;
            for (uint32_t i = 0; i < n; ++i) {
                th.emplace_back([&, i] {
                    distributed::EnqueueMeshWorkload(meshes.at(opt.chips[i])->mesh_command_queue(), wls[i], true);
                });
            }
            for (auto& t : th) {
                t.join();
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                distributed::EnqueueMeshWorkload(meshes.at(opt.chips[i])->mesh_command_queue(), wls[i], false);
            }
            for (uint32_t i = 0; i < n; ++i) {
                distributed::Finish(meshes.at(opt.chips[i])->mesh_command_queue());
            }
        }
        std::vector<uint32_t> all;
        std::string per_chip;
        double step_max = 0;
        uint32_t viol = 0;
        bool ok = true;
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<uint32_t> r;
            detail::ReadFromDeviceL1(devs[i], opt.core, RESULTS, 0x40 + (opt.iters + opt.warmup) * 4, r);
            if (r[0] != 1) {
                ok = false;
                fmt::print(
                    "  chip {}: status {} after {} iterations (waiting on peer {}), violations {}\n",
                    opt.chips[i],
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
    for (auto& m : meshes) {
        m.second->close();
    }
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    return 0;
}
