// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Blackhole PCIe peer-to-peer microbenchmark: chip A's NoC writes directly into
// chip B's L1 (or DRAM) over PCIe, no host CPU and no host memory in the path.
//
// Setup, all from user space:
//   * inbound on B : a 2 MiB TLB window (BAR0) pointed at the target core
//   * outbound on A: a DesignWare iATU region, programmed through BAR2, whose
//                    target is B's BAR0 physical address + the window's offset
//   * the mirror of both for B -> A (ping-pong echo, bandwidth completion ack)
//   * kernels write to  PCIe-core NoC XY | (1 << 60) | iATU base | local offset
//
// The kernel driver is not involved beyond what tt-metal already uses. IOMMU
// must be off (or in passthrough) so that BAR physical == DMA address.

#include <fmt/format.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include "impl/allocator/allocator.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/tlb.hpp>
#include "pcie_p2p_common.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::pcie_p2p;

namespace {

struct Options {
    int chip_a = 0;
    int chip_b = 1;
    uint32_t iters = 2000;
    uint64_t bw_bytes = 256ull << 20;
    uint32_t chunk = 16384;
    uint32_t region_a = 15;           // iATU region on A for B's L1 window
    uint32_t region_a_dram = 14;      // iATU region on A for B's DRAM window
    uint32_t region_b = 15;           // iATU region on B for A's L1 window
    uint64_t noc_base = 32ull << 30;  // outbound NoC address where our regions start (32 GiB)
    std::string ordering = "strict";
    std::string target = "l1";  // l1 | dram
    uint32_t flag_bytes = 64;
    bool dump_only = false;
    bool no_bw = false;
    bool no_pingpong = false;
    bool force = false;
    CoreCoord core_a{0, 0};
    CoreCoord core_b{0, 0};
    uint32_t data_span = 512u << 10;  // bytes of B window cycled through in BW mode
    double timeout_s = 3.0;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            TT_FATAL(i + 1 < argc, "missing value for {}", a);
            return argv[++i];
        };
        if (a == "--a") {
            o.chip_a = std::stoi(next());
        } else if (a == "--b") {
            o.chip_b = std::stoi(next());
        } else if (a == "--iters") {
            o.iters = std::stoul(next());
        } else if (a == "--bw-bytes") {
            o.bw_bytes = std::stoull(next(), nullptr, 0);
        } else if (a == "--chunk") {
            o.chunk = std::stoul(next(), nullptr, 0);
        } else if (a == "--region-a") {
            o.region_a = std::stoul(next());
        } else if (a == "--region-a-dram") {
            o.region_a_dram = std::stoul(next());
        } else if (a == "--region-b") {
            o.region_b = std::stoul(next());
        } else if (a == "--noc-base") {
            o.noc_base = std::stoull(next(), nullptr, 0);
        } else if (a == "--ordering") {
            o.ordering = next();
        } else if (a == "--target") {
            o.target = next();
        } else if (a == "--flag-bytes") {
            o.flag_bytes = std::stoul(next());
        } else if (a == "--data-span") {
            o.data_span = std::stoul(next(), nullptr, 0);
        } else if (a == "--timeout") {
            o.timeout_s = std::stod(next());
        } else if (a == "--core-a") {
            auto s = next();
            auto c = s.find(',');
            o.core_a = {std::stoul(s.substr(0, c)), std::stoul(s.substr(c + 1))};
        } else if (a == "--core-b") {
            auto s = next();
            auto c = s.find(',');
            o.core_b = {std::stoul(s.substr(0, c)), std::stoul(s.substr(c + 1))};
        } else if (a == "--dump-only") {
            o.dump_only = true;
        } else if (a == "--no-bw") {
            o.no_bw = true;
        } else if (a == "--no-pingpong") {
            o.no_pingpong = true;
        } else if (a == "--force") {
            o.force = true;
        } else if (a == "-h" || a == "--help") {
            fmt::print(
                "usage: test_pcie_p2p [--a N --b M] [--iters N] [--bw-bytes N] [--chunk N] [--target l1|dram]\n"
                "                     [--ordering relaxed|strict|posted] [--region-a R --region-b R --region-a-dram "
                "R]\n"
                "                     [--noc-base HEX] [--core-a x,y --core-b x,y] [--data-span N] [--flag-bytes "
                "16|64]\n"
                "                     [--timeout S] [--dump-only] [--no-bw] [--no-pingpong] [--force]\n");
            exit(0);
        } else {
            TT_THROW("unknown arg {}", a);
        }
    }
    TT_FATAL(o.chunk % 64 == 0 && o.chunk <= 65536, "chunk must be a multiple of 64 B and <= 64 KiB");
    TT_FATAL(
        o.noc_base % kTlb2M == 0 && o.noc_base + 3 * kTlb2M <= kNocLocalLimit,
        "noc-base must be 2 MiB aligned and below 64 GiB");
    TT_FATAL(o.iters <= 16000, "iters <= 16000 (results live in L1)");
    return o;
}

void run_pair(
    distributed::MeshDevice& mesh_b,
    Program&& prog_b,
    distributed::MeshDevice& mesh_a,
    Program&& prog_a,
    bool responder) {
    auto zero_a = distributed::MeshCoordinate::zero_coordinate(mesh_a.shape().dims());
    auto zero_b = distributed::MeshCoordinate::zero_coordinate(mesh_b.shape().dims());
    distributed::MeshWorkload wl_a, wl_b;
    wl_a.add_program(distributed::MeshCoordinateRange(zero_a, zero_a), std::move(prog_a));
    if (responder) {
        wl_b.add_program(distributed::MeshCoordinateRange(zero_b, zero_b), std::move(prog_b));
    }
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread tb;
        if (responder) {
            tb = std::thread([&] { distributed::EnqueueMeshWorkload(mesh_b.mesh_command_queue(), wl_b, true); });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        distributed::EnqueueMeshWorkload(mesh_a.mesh_command_queue(), wl_a, true);
        if (tb.joinable()) {
            tb.join();
        }
    } else {
        if (responder) {
            distributed::EnqueueMeshWorkload(mesh_b.mesh_command_queue(), wl_b, false);
        }
        distributed::EnqueueMeshWorkload(mesh_a.mesh_command_queue(), wl_a, false);
        distributed::Finish(mesh_a.mesh_command_queue());
        if (responder) {
            distributed::Finish(mesh_b.mesh_command_queue());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    TT_FATAL(opt.chip_a != opt.chip_b, "need two different chips");

    auto meshes = distributed::MeshDevice::create_unit_meshes({opt.chip_a, opt.chip_b});
    auto& cluster = MetalContext::instance().get_cluster();
    TT_FATAL(cluster.arch() == tt::ARCH::BLACKHOLE, "Blackhole only");

    Chip A, B;
    init_chip(A, opt.chip_a, meshes.at(opt.chip_a));
    init_chip(B, opt.chip_b, meshes.at(opt.chip_b));
    fmt::print("\n=== before setup ===\n");
    dump_chip(A);
    dump_chip(B);
    if (opt.dump_only) {
        for (auto& m : meshes) {
            m.second->close();
        }
        return 0;
    }

    // ---- L1 layout (identical on both chips, relative to the unreserved base) ----
    const uint32_t U = std::max(A.l1_unreserved, B.l1_unreserved);
    const uint32_t FLAG = U;             // 64 B, seq written by the peer
    const uint32_t FLAG_SRC = U + 0x40;  // 64 B, local staging for the seq write
    const uint32_t RESULTS = U + 0x100;  // 64 KiB
    const uint32_t SRC = U + 0x20000;    // 64 KiB source buffer
    const uint32_t DATA = U + 0x40000;   // BW destination region on B (L1 target)
    TT_FATAL(
        opt.target != "l1" || DATA + opt.data_span <= 1536u * 1024u, "data span does not fit in L1; lower --data-span");

    const uint64_t ord = tlb_ordering(opt.ordering);
    const CoreCoord va = A.dev->virtual_core_from_logical_core(opt.core_a, CoreType::WORKER);
    const CoreCoord vb = B.dev->virtual_core_from_logical_core(opt.core_b, CoreType::WORKER);

    fmt::print("\n=== setup ===\n");
    // B inbound L1 window (A -> B flag + L1 data), A outbound region
    const uint64_t b_l1_off = make_inbound_window(B, vb.x, vb.y, 0, ord, "L1 of core B");
    make_outbound(A, opt.region_a, opt.noc_base, B, b_l1_off, opt.force);
    // A inbound L1 window (B -> A echo/ack), B outbound region
    const uint64_t a_l1_off = make_inbound_window(A, va.x, va.y, 0, ord, "L1 of core A");
    make_outbound(B, opt.region_b, opt.noc_base, A, a_l1_off, opt.force);
    // Optional DRAM data window on B
    uint64_t data_dst = A.pcie_noc_base | (opt.noc_base + DATA);
    uint32_t data_span = opt.data_span;
    std::shared_ptr<Buffer> dram_buf;  // keeps the DRAM target allocated for the whole run
    if (opt.target == "dram") {
        // Own the target memory: one 6 MiB page lands entirely in DRAM bank 0; carve a 2 MiB aligned window out of it.
        dram_buf = CreateBuffer(BufferConfig{
            .device = B.dev, .size = 6ull << 20, .page_size = 6ull << 20, .buffer_type = BufferType::DRAM});
        const auto* alloc = B.dev->allocator_impl().get();
        const uint64_t bank0_addr =
            uint64_t(dram_buf->address()) + uint64_t(int64_t(alloc->get_bank_offset(BufferType::DRAM, 0)));
        const uint64_t win_addr = (bank0_addr + kTlb2M - 1) & ~(kTlb2M - 1);
        TT_FATAL(win_addr + kTlb2M <= bank0_addr + (6ull << 20), "2 MiB window does not fit in the DRAM buffer");
        const uint32_t ch = alloc->get_dram_channel_from_bank_id(0);
        const CoreCoord dram_logical = B.dev->logical_core_from_dram_channel(ch);
        const CoreCoord dram_v = B.dev->virtual_core_from_logical_core(dram_logical, CoreType::DRAM);
        fmt::print(
            "  DRAM buffer on chip {} at 0x{:x} (bank 0 channel {} core ({},{})), window at bank addr 0x{:x}\n",
            B.id,
            dram_buf->address(),
            ch,
            dram_v.x,
            dram_v.y,
            win_addr);
        const uint64_t b_dram_off = make_inbound_window(B, dram_v.x, dram_v.y, win_addr, ord, "DRAM bank 0 of B");
        make_outbound(A, opt.region_a_dram, opt.noc_base + kTlb2M, B, b_dram_off, opt.force);
        data_dst = A.pcie_noc_base | (opt.noc_base + kTlb2M);
        data_span = std::min<uint32_t>(opt.data_span, kTlb2M);
    }
    const uint64_t flag_dst_on_b = A.pcie_noc_base | (opt.noc_base + FLAG);
    const uint64_t ack_dst_on_a = B.pcie_noc_base | (opt.noc_base + FLAG);
    fmt::print(
        "  A writes data to NoC 0x{:016x}, flag to 0x{:016x}; B acks to 0x{:016x}\n",
        data_dst,
        flag_dst_on_b,
        ack_dst_on_a);

    fmt::print("\n=== after setup ===\n");
    dump_chip(A);
    dump_chip(B);

    // Host-side sanity: write through B's new window from the host, read back via the normal path.
    {
        auto* win = reinterpret_cast<volatile uint32_t*>(B.tlbs[0]->get_base());
        win[(DATA / 4)] = 0xB0B0CAFE;
        auto v = read_l1(B.dev, opt.core_b, DATA, 64);
        fmt::print(
            "  host->B window self-check: wrote 0xB0B0CAFE via TLB {}, read back 0x{:08X} -> {}\n",
            B.tlbs[0]->get_tlb_id(),
            v[0],
            v[0] == 0xB0B0CAFE ? "OK" : "MISMATCH");
        TT_FATAL(v[0] == 0xB0B0CAFE, "B inbound window does not point at core B L1");
    }

    const double mhz_a = A.dev->get_clock_rate_mhz();
    const uint64_t timeout_cycles = uint64_t(opt.timeout_s * mhz_a * 1e6);
    const auto lo = [](uint64_t v) { return uint32_t(v); };
    const auto hi = [](uint64_t v) { return uint32_t(v >> 32); };
    const std::string kdir = "tests/tt_metal/tt_metal/perf_microbenchmark/pcie_p2p/kernels/";

    auto zero_flags = [&] {
        std::vector<uint32_t> z(0x100 / 4, 0), zr(0x10000 / 4, 0);
        for (Chip* c : {&A, &B}) {
            write_l1(c->dev, c == &A ? opt.core_a : opt.core_b, FLAG, z);
            write_l1(c->dev, c == &A ? opt.core_a : opt.core_b, RESULTS, zr);
        }
    };
    auto initiator = [&](uint32_t mode, uint32_t iters, uint64_t total, Program& p) {
        auto k = CreateKernel(
            p,
            kdir + "p2p_initiator.cpp",
            opt.core_a,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(
            p,
            k,
            opt.core_a,
            std::vector<uint32_t>{
                mode,
                lo(data_dst),
                hi(data_dst),
                lo(flag_dst_on_b),
                hi(flag_dst_on_b),
                FLAG,
                SRC,
                FLAG_SRC,
                RESULTS,
                iters,
                opt.chunk,
                uint32_t(total),
                data_span,
                lo(timeout_cycles),
                hi(timeout_cycles),
                opt.flag_bytes});
    };
    auto responder = [&](uint32_t iters, Program& p) {
        auto k = CreateKernel(
            p,
            kdir + "p2p_responder.cpp",
            opt.core_b,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(
            p,
            k,
            opt.core_b,
            std::vector<uint32_t>{
                FLAG,
                lo(ack_dst_on_a),
                hi(ack_dst_on_a),
                FLAG_SRC,
                RESULTS,
                iters,
                lo(timeout_cycles),
                hi(timeout_cycles),
                opt.flag_bytes});
    };

    // ---- 1. smoke: one 4 KiB write A -> B, verify on B via the ordinary host path ----
    fmt::print("\n=== smoke: A NoC write -> PCIe -> B ({}) ===\n", opt.target);
    {
        zero_flags();
        std::vector<uint32_t> pat(4096 / 4);
        for (uint32_t i = 0; i < pat.size(); ++i) {
            pat[i] = 0xA5000000u | i;
        }
        write_l1(A.dev, opt.core_a, SRC, pat);
        std::vector<uint32_t> zero(4096 / 4, 0);
        if (opt.target == "l1") {
            write_l1(B.dev, opt.core_b, DATA, zero);
        }
        Program pa, pb;
        initiator(0, 0, 0, pa);
        run_pair(*B.mesh, std::move(pb), *A.mesh, std::move(pa), false);
        auto ra = read_l1(A.dev, opt.core_a, RESULTS, 64);
        std::vector<uint32_t> got;
        if (opt.target == "l1") {
            got = read_l1(B.dev, opt.core_b, DATA, 4096);
        } else {
            auto* win = reinterpret_cast<volatile uint32_t*>(B.tlbs.back()->get_base());
            got.resize(4096 / 4);
            for (uint32_t i = 0; i < got.size(); ++i) {
                got[i] = win[i];
            }
        }
        uint32_t bad = 0;
        for (uint32_t i = 0; i < pat.size(); ++i) {
            bad += (got[i] != pat[i]);
        }
        fmt::print(
            "  initiator status {} ; B received 4096 B with {} mismatching words -> {}\n",
            ra[0],
            bad,
            bad ? "FAIL" : "PASS");
        if (bad) {
            fmt::print("  first words on B: 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x}\n", got[0], got[1], got[2], got[3]);
            for (Chip* c : {&A, &B}) {
                for (auto r : c->programmed_regions) {
                    c->iatu.disable(r);
                }
            }
            for (auto& m : meshes) {
                m.second->close();
            }
            return 1;
        }
    }

    // ---- 2. ping-pong latency ----
    if (!opt.no_pingpong) {
        fmt::print(
            "\n=== ping-pong: {} iters, {} B flag writes, ordering {} ===\n", opt.iters, opt.flag_bytes, opt.ordering);
        zero_flags();
        Program pa, pb;
        responder(opt.iters, pb);
        initiator(1, opt.iters, 0, pa);
        run_pair(*B.mesh, std::move(pb), *A.mesh, std::move(pa), true);
        auto ra = read_l1(A.dev, opt.core_a, RESULTS, 0x40 + opt.iters * 4);
        auto rb = read_l1(B.dev, opt.core_b, RESULTS, 64);
        fmt::print("  initiator status {} done {} | responder status {} done {}\n", ra[0], ra[1], rb[0], rb[1]);
        std::vector<uint32_t> rtt(ra.begin() + 8, ra.begin() + 8 + ra[1]);
        if (!rtt.empty()) {
            std::sort(rtt.begin(), rtt.end());
            auto ns = [&](uint32_t cyc) { return cyc * 1000.0 / mhz_a; };
            const uint32_t skip = std::min<uint32_t>(rtt.size() / 20, 50);  // drop warm-up samples from the mean
            double sum = 0;
            for (size_t i = skip; i < rtt.size(); ++i) {
                sum += rtt[i];
            }
            const double mean = sum / (rtt.size() - skip);
            fmt::print("  clock {} MHz\n", mhz_a);
            fmt::print(
                "  RTT   min {:.0f} ns  p50 {:.0f} ns  mean {:.0f} ns  p99 {:.0f} ns  max {:.0f} ns\n",
                ns(rtt.front()),
                ns(rtt[rtt.size() / 2]),
                ns(mean),
                ns(rtt[rtt.size() * 99 / 100]),
                ns(rtt.back()));
            fmt::print(
                "  one-way (RTT/2)   min {:.0f} ns  p50 {:.0f} ns  p99 {:.0f} ns\n",
                ns(rtt.front()) / 2,
                ns(rtt[rtt.size() / 2]) / 2,
                ns(rtt[rtt.size() * 99 / 100]) / 2);
        }
    }

    // ---- 3. bandwidth ----
    if (!opt.no_bw) {
        fmt::print(
            "\n=== bandwidth: {} MiB in {} B chunks into B {} (window span {} KiB) ===\n",
            opt.bw_bytes >> 20,
            opt.chunk,
            opt.target,
            data_span >> 10);
        zero_flags();
        std::vector<uint32_t> pat(opt.chunk / 4);
        for (uint32_t i = 0; i < pat.size(); ++i) {
            pat[i] = 0x5A000000u | i;
        }
        write_l1(A.dev, opt.core_a, SRC, pat);
        Program pa, pb;
        responder(1, pb);
        initiator(2, 1, opt.bw_bytes, pa);
        const auto t0 = std::chrono::steady_clock::now();
        run_pair(*B.mesh, std::move(pb), *A.mesh, std::move(pa), true);
        const double host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        auto ra = read_l1(A.dev, opt.core_a, RESULTS, 64);
        auto rb = read_l1(B.dev, opt.core_b, RESULTS, 64);
        const uint64_t issue = (uint64_t(ra[3]) << 32) | ra[2];
        const uint64_t e2e = (uint64_t(ra[5]) << 32) | ra[4];
        fmt::print("  initiator status {} chunks {} | responder status {}\n", ra[0], ra[1], rb[0]);
        if (ra[0] == 1) {
            const double issue_s = issue / (mhz_a * 1e6), e2e_s = e2e / (mhz_a * 1e6);
            fmt::print("  issue+NoC barrier : {:.3f} ms -> {:.2f} GB/s\n", issue_s * 1e3, opt.bw_bytes / issue_s / 1e9);
            fmt::print("  end-to-end (ack)  : {:.3f} ms -> {:.2f} GB/s\n", e2e_s * 1e3, opt.bw_bytes / e2e_s / 1e9);
            fmt::print("  (host wall incl. dispatch, not a measurement: {:.3f} ms)\n", host_s * 1e3);
            // verify the last chunk landed
            std::vector<uint32_t> got;
            if (opt.target == "l1") {
                got = read_l1(B.dev, opt.core_b, DATA, opt.chunk);
            } else {
                auto* win = reinterpret_cast<volatile uint32_t*>(B.tlbs.back()->get_base());
                got.resize(opt.chunk / 4);
                for (uint32_t i = 0; i < got.size(); ++i) {
                    got[i] = win[i];
                }
            }
            uint32_t bad = 0, first_bad = 0;
            for (uint32_t i = 0; i < pat.size(); ++i) {
                if (got[i] != pat[i]) {
                    if (!bad) {
                        first_bad = i;
                    }
                    ++bad;
                }
            }
            fmt::print("  data check on B (first chunk slot): {} mismatching words\n", bad);
            if (bad) {
                fmt::print(
                    "    first mismatch word {}: expected 0x{:08x} got 0x{:08x}\n",
                    first_bad,
                    pat[first_bad],
                    got[first_bad]);
            }
        }
    }

    // ---- cleanup ----
    for (Chip* c : {&A, &B}) {
        for (auto r : c->programmed_regions) {
            c->iatu.disable(r);
        }
    }
    A.tlbs.clear();
    B.tlbs.clear();
    for (auto& m : meshes) {
        m.second->close();
    }
    fmt::print("\ndone\n");
    return 0;
}
