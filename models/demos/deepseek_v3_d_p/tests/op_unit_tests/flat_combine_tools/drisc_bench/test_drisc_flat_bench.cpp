// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC flat-expert weight delivery benchmark (Blackhole, programmable DRAM cores). Can DRAM cores do the job of
// flat_routed_expert's 16 Tensix gate/up weight readers? Reproduces only the weight movement of one call on the flat
// plan's own cores (DRISC_FLAT_PLAN: a text dump of the plan: "H I E tile_bytes kblk np R", "gu x y ...",
// "down x y ..."):
//   gu     2 DRISC senders per bank (DRISC_FLAT_EP=1: one) stream the gate/up weights from their bank and write each
//          block (kblk x 2 np tiles) into its gate/up core's landing slots (drisc_flat_gu_sender.cpp) - the Tensix
//          readers' read + forward, with no Tensix core involved
//   dn     the down cores read their down slices rotating over the banks (flat_dn_rot_reader.cpp), as flat does
//   gu+dn  both at once (the DRAM load of a real call; x / h / y traffic not included)
// Bytes as one call of E experts; nothing is consumed (no credits). Timing from device wall clocks written by every
// kernel after a device-side start barrier (programs with DRAM-core kernels need slow dispatch, which launches cores
// ~1 ms apart; every kernel waits until all have arrived, then starts its clock).
// Compare with the Python Tensix-reader bench (tests/.../flat_combine_tools/dram_read_bench, worklog 8c / 8d):
// GLM 5.3 E8 readers + forward alone 222.6 us, readers + forward + rotating down 371.1 us.
//   TT_METAL_SLOW_DISPATCH_MODE=1 DRISC_FLAT_PLAN=plan.txt ./unit_tests_api --gtest_filter='DriscFlatBench*'
//   (DRISC_FLAT_NOC 0/1, DRISC_FLAT_SLOTS, DRISC_FLAT_EP, DRISC_FLAT_ITERS, DRISC_FLAT_DN_NOC 0/1 (default 0, as flat))
// DRAM-core and Tensix wall clocks are not synchronized: with both groups, read the per-group spans (each group's
// kernels start together after the shared barrier), not the overall span.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

uint32_t env_u32(const char* name, uint32_t dflt) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : dflt;
}

struct FlatPlan {
    uint32_t H = 0, I = 0, E = 0, tile = 0, kblk = 0, np = 0, R = 0;
    std::vector<CoreCoord> gu, down;
};

bool load_plan(const char* path, FlatPlan& p) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string line;
    std::getline(f, line);
    std::istringstream(line) >> p.H >> p.I >> p.E >> p.tile >> p.kblk >> p.np >> p.R;
    while (std::getline(f, line)) {
        std::istringstream s(line);
        std::string tag;
        s >> tag;
        auto& v = tag == "gu" ? p.gu : p.down;
        uint32_t x = 0, y = 0;
        while (s >> x >> y) {
            v.emplace_back(x, y);
        }
    }
    return !p.gu.empty() && !p.down.empty();
}

// Logical DRAM endpoint whose reads go through `noc` for bank `view` (as test_dram_kernels.cpp).
CoreCoord endpoint_for_noc(const metal_SocDescriptor& soc, uint32_t view, NOC noc) {
    const CoreCoord pref = soc.get_preferred_worker_core_for_dram_view(view, static_cast<uint8_t>(noc));
    const auto& eps = soc.dram_bank_endpoint_coords.at(view);
    for (uint32_t i = 0; i < eps.size(); i++) {
        if (eps[i] == pref) {
            return CoreCoord{view, i};
        }
    }
    TT_FATAL(false, "preferred DRAM endpoint for bank {} not found", view);
    return {};
}

}  // namespace

class DriscFlatBenchFixture : public UnitMeshAnyDispatchFixture {
protected:
    void SetUp() override {
        UnitMeshAnyDispatchFixture::SetUp();
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Blackhole only";
        }
        const auto& hal = MetalContext::instance().hal();
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
        if (!std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            GTEST_SKIP() << "programs with DRAM-core kernels need TT_METAL_SLOW_DISPATCH_MODE=1";
        }
        const char* path = std::getenv("DRISC_FLAT_PLAN");
        if (!path || !load_plan(path, plan_)) {
            GTEST_SKIP() << "set DRISC_FLAT_PLAN to a flat plan dump";
        }
        dev_ = devices_[0].get();
        range_ = distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0));
        drisc_l1_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        drisc_l1_noc_ = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        drisc_l1_size_ = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        tensix_l1_ = dev_->allocator()->get_base_allocator_addr(HalMemType::L1);
    }

    void run(Program& program) {
        distributed::MeshWorkload wl;
        wl.add_program(range_, std::move(program));
        distributed::EnqueueMeshWorkload(dev_->mesh_command_queue(), wl, false);
        distributed::Finish(dev_->mesh_command_queue());
    }

    std::pair<uint64_t, uint64_t> read_t(CoreCoord vcore, uint64_t noc_addr) {
        std::vector<uint32_t> t(4);
        MetalContext::instance().get_cluster().read_core(
            t.data(), 4 * sizeof(uint32_t), tt_cxy_pair(dev_->build_id(), vcore), noc_addr);
        return {(uint64_t(t[1]) << 32) | t[0], (uint64_t(t[3]) << 32) | t[2]};
    }

    FlatPlan plan_;
    distributed::MeshDevice* dev_ = nullptr;
    distributed::MeshCoordinateRange range_{distributed::MeshCoordinate(0, 0)};
    uint32_t drisc_l1_ = 0, drisc_l1_size_ = 0, tensix_l1_ = 0;
    uint64_t drisc_l1_noc_ = 0;
};

TEST_F(DriscFlatBenchFixture, GateUpDelivery) {
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(dev_->build_id());
    const uint32_t banks = soc.get_dram_compute_grid_size().x;
    const uint32_t clk_mhz = MetalContext::instance().get_cluster().get_device_aiclk(dev_->get_device_ids()[0]);
    const FlatPlan& p = plan_;
    const uint32_t Ht = p.H / 32, It = p.I / 32, nk = Ht / p.kblk;
    // one gate/up core's block of a chunk; DRISC_FLAT_WMULT=k sends k blocks per write (same bytes, k x larger
    // writes and k x fewer of them: what a layout with each receiver's blocks contiguous would allow)
    const uint32_t wmult = env_u32("DRISC_FLAT_WMULT", 1);
    const uint32_t blk = p.kblk * 2 * p.np * p.tile * wmult;
    const uint32_t n_readers = static_cast<uint32_t>(p.gu.size()) / p.R;  // the Tensix readers this replaces
    ASSERT_EQ(n_readers, 2 * banks) << "plan with 2 readers per bank expected";
    const uint32_t ep = env_u32("DRISC_FLAT_EP", 2), slots = env_u32("DRISC_FLAT_SLOTS", 4);
    const uint32_t iters = env_u32("DRISC_FLAT_ITERS", 5), land_slots = 4;
    const NOC drisc_noc = env_u32("DRISC_FLAT_NOC", 0) ? NOC::NOC_1 : NOC::NOC_0;
    ASSERT_TRUE(ep == 1 || ep == 2);
    ASSERT_LE(slots * blk + 2048, drisc_l1_size_) << "DRISC slots + timing / barrier scratch do not fit its L1";
    ASSERT_EQ((p.E * nk) % wmult, 0u);
    const uint32_t per_reader = p.E * nk * p.R / wmult;  // writes (blocks) per Tensix reader
    const uint32_t sender_mode = env_u32("DRISC_FLAT_SENDER_MODE", 0);  // 0 DMA + write, 1 writes only, 2 DMA only
    const uint64_t gu_bytes = uint64_t(per_reader) * n_readers * blk;

    // DRISC-usable endpoints (subchannel != the syseng-owned NOC0 worker endpoint)
    const uint32_t noc0_sub = endpoint_for_noc(soc, 0, NOC::NOC_0).y;
    std::vector<uint32_t> subs;
    for (uint32_t s = 0; s < soc.get_dram_compute_grid_size().y; ++s) {
        if (s != noc0_sub) {
            subs.push_back(s);
        }
    }
    ASSERT_GE(subs.size(), ep);

    // gate/up weights: one page per bank (both readers' regions), down: one page per bank (rotated slices)
    const uint32_t gu_per_bank = 2 * per_reader * blk;
    auto gu_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = uint64_t(gu_per_bank) * banks},
        {.page_size = gu_per_bank, .buffer_type = BufferType::DRAM},
        dev_);
    const uint32_t nd = static_cast<uint32_t>(p.down.size());
    std::vector<uint32_t> pcd(nd), dblk(nd);
    uint32_t sub_max = 0;
    const uint32_t dn_blocks = p.E * It / p.kblk;
    uint64_t dn_bytes = 0;
    for (uint32_t d = 0; d < nd; ++d) {
        pcd[d] = Ht / nd + (d < Ht % nd ? 1 : 0);
        dblk[d] = p.kblk * pcd[d] * p.tile;
        sub_max = std::max(sub_max, (dn_blocks + banks - 1) / banks * dblk[d]);
        dn_bytes += uint64_t(dn_blocks) * dblk[d];
    }
    const uint32_t dn_per_bank = nd * sub_max;
    auto dn_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = uint64_t(dn_per_bank) * banks},
        {.page_size = dn_per_bank, .buffer_type = BufferType::DRAM},
        dev_);
    const uint32_t dmax = *std::max_element(dblk.begin(), dblk.end());
    const uint32_t ring_bytes = 4 * dmax, dn_timing = tensix_l1_ + (ring_bytes + 31) / 32 * 32;  // 32 B aligned
    const uint32_t drisc_timing_off = (slots * blk + 31) / 32 * 32;
    // start barrier: a 32 B arrival line per kernel in the first down core's L1 (no kernel of this test uses it there)
    const CoreCoord bar_v = dev_->virtual_core_from_logical_core(p.down[0], CoreType::WORKER);
    const uint32_t bar_addr = tensix_l1_ + 512 * 1024;
    ASSERT_GT(bar_addr, dn_timing + 2048);

    log_info(
        LogTest,
        "DRISC flat bench: H {} I {} E {}, {} gate/up cores ({} per reader), {} down cores; gate/up {:.1f} MB in {} B "
        "writes, {} DRISC senders per bank on {}, {} slots, sender mode {}; down {:.1f} MB; aiclk {} MHz",
        p.H,
        p.I,
        p.E,
        p.gu.size(),
        p.R,
        nd,
        gu_bytes / 1e6,
        blk,
        ep,
        drisc_noc == NOC::NOC_0 ? "NOC0" : "NOC1",
        slots,
        sender_mode,
        dn_bytes / 1e6,
        clk_mhz);

    struct Mode {
        const char* name;
        bool gu, dn;
    };
    for (const Mode& mode : {Mode{"gu", true, false}, Mode{"dn", false, true}, Mode{"gu+dn", true, true}}) {
        std::vector<CoreCoord> drisc_v, dn_v;
        const uint32_t n_gu_k = mode.gu ? banks * ep : 0, n_all = n_gu_k + (mode.dn ? nd : 0);
        auto build = [&]() {
            Program program = CreateProgram();
            drisc_v.clear();
            dn_v.clear();
            if (mode.gu) {
                for (uint32_t b = 0; b < banks; ++b) {
                    for (uint32_t e = 0; e < ep; ++e) {
                        const CoreCoord lc{b, subs[e]};
                        // readers this sender stands in for: reader b + banks * e (2 senders per bank) or both
                        std::vector<uint32_t> rds = ep == 2 ? std::vector<uint32_t>{b + banks * e}
                                                            : std::vector<uint32_t>{b, b + banks};
                        std::vector<uint32_t> rt{
                            gu_buf->address() + (ep == 2 ? e * per_reader * blk : 0u),
                            drisc_l1_,
                            blk,
                            per_reader * static_cast<uint32_t>(rds.size()),
                            slots,
                            p.R * static_cast<uint32_t>(rds.size()),
                            tensix_l1_,
                            land_slots,
                            drisc_l1_ + drisc_timing_off,
                            bar_v.x,
                            bar_v.y,
                            bar_addr,
                            b * ep + e,
                            n_all,
                            sender_mode};
                        // receivers in block order: block i -> receiver i % R'; with one sender per bank its blocks
                        // alternate between the two readers' cores (R' = 2R)
                        for (uint32_t r : rds) {
                            for (uint32_t j = 0; j < p.R; ++j) {
                                const CoreCoord v =
                                    dev_->virtual_core_from_logical_core(p.gu[r * p.R + j], CoreType::WORKER);
                                rt.push_back(v.x);
                                rt.push_back(v.y);
                            }
                        }
                        auto k = CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_flat_gu_sender.cpp",
                            lc,
                            DramConfig{.noc = drisc_noc});
                        SetRuntimeArgs(program, k, lc, rt);
                        drisc_v.push_back(dev_->virtual_core_from_logical_core(lc, CoreType::DRAM));
                    }
                }
            }
            if (mode.dn) {
                std::vector<CoreRange> rs;
                for (const auto& c : p.down) {
                    rs.emplace_back(c, c);
                }
                auto k = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/flat_dn_rot_reader.cpp",
                    CoreRangeSet(rs),
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = env_u32("DRISC_FLAT_DN_NOC", 0) ? NOC::NOC_1 : NOC::NOC_0});
                for (uint32_t d = 0; d < nd; ++d) {
                    SetRuntimeArgs(
                        program,
                        k,
                        p.down[d],
                        {dn_buf->address() + d * sub_max,
                         d % banks,
                         dblk[d],
                         dn_blocks,
                         2u,
                         4u,
                         tensix_l1_,
                         dn_timing,
                         banks,
                         bar_v.x,
                         bar_v.y,
                         bar_addr,
                         n_gu_k + d,
                         n_all});
                    dn_v.push_back(dev_->virtual_core_from_logical_core(p.down[d], CoreType::WORKER));
                }
            }
            return program;
        };
        std::vector<double> span_all, span_gu, span_dn;
        for (uint32_t it = 0; it < iters + 1; ++it) {
            Program program = build();
            const std::vector<uint32_t> zeros(8 * 64, 0);
            MetalContext::instance().get_cluster().write_core(
                zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(dev_->build_id(), bar_v), bar_addr);
            MetalContext::instance().get_cluster().l1_barrier(dev_->build_id());
            run(program);
            if (it == 0) {
                continue;  // warm-up (kernel compile, first launch)
            }
            uint64_t s_all = UINT64_MAX, e_all = 0, s_g = UINT64_MAX, e_g = 0, s_d = UINT64_MAX, e_d = 0;
            for (const auto& v : drisc_v) {
                auto [a, b] = read_t(v, drisc_l1_noc_ + drisc_timing_off);
                s_g = std::min(s_g, a);
                e_g = std::max(e_g, b);
            }
            for (const auto& v : dn_v) {
                auto [a, b] = read_t(v, dn_timing);
                s_d = std::min(s_d, a);
                e_d = std::max(e_d, b);
            }
            s_all = std::min(s_g, s_d);
            e_all = std::max(e_g, e_d);
            const double us = 1.0 / clk_mhz;
            span_all.push_back((e_all - s_all) * us);
            if (mode.gu) {
                span_gu.push_back((e_g - s_g) * us);
            }
            if (mode.dn) {
                span_dn.push_back((e_d - s_d) * us);
            }
        }
        auto med = [](std::vector<double> v) {
            if (v.empty()) {
                return 0.0;
            }
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const double t = med(span_all);
        const double bytes = (mode.gu ? double(gu_bytes) : 0.0) + (mode.dn ? double(dn_bytes) : 0.0);
        log_info(
            LogTest,
            "DRISC flat bench {:6s}: {:8.1f} us (gate/up senders {:8.1f}, down cores {:8.1f}) -> {:6.1f} GB/s ({:5.1f}% "
            "of 512)",
            mode.name,
            t,
            med(span_gu),
            med(span_dn),
            bytes / t / 1e3,
            bytes / t / 1e3 / 512.0 * 100.0);
    }
}

// One DRAM tile (bank 0's first DRISC-usable subchannel) reading only its own bank and writing to ONE Tensix core:
// DMA only, NoC writes only (the same L1 slots re-sent), and both pipelined, for a near and a far receiver and two
// block sizes. Separates the tile's DMA read rate from its NoC write injection rate.
TEST_F(DriscFlatBenchFixture, SingleDriscOneTensix) {
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(dev_->build_id());
    const uint32_t clk_mhz = MetalContext::instance().get_cluster().get_device_aiclk(dev_->get_device_ids()[0]);
    const uint32_t noc0_sub = endpoint_for_noc(soc, 0, NOC::NOC_0).y;
    uint32_t sub = 0;
    for (uint32_t s = 0; s < soc.get_dram_compute_grid_size().y; ++s) {
        if (s != noc0_sub) {
            sub = s;
            break;
        }
    }
    const CoreCoord lc{0, sub};
    const CoreCoord dv = dev_->virtual_core_from_logical_core(lc, CoreType::DRAM);
    const auto grid = dev_->compute_with_storage_grid_size();
    const uint64_t total = 32ull << 20;  // bytes per run
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total * 8},
        {.page_size = static_cast<uint32_t>(total), .buffer_type = BufferType::DRAM},
        dev_);
    const uint32_t bar_addr = tensix_l1_ + 512 * 1024;
    log_info(LogTest, "single DRISC: bank 0 subchannel {} (virtual {}), {} MB per run", sub, dv.str(), total >> 20);
    for (const CoreCoord rx_l : {CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}}) {
        const CoreCoord rx = dev_->virtual_core_from_logical_core(rx_l, CoreType::WORKER);
        for (const uint32_t blk : {9216u, 16384u}) {
            const uint32_t slots = 4, n = static_cast<uint32_t>(total / blk);
            ASSERT_LE(slots * blk + 2048, drisc_l1_size_);
            const uint32_t timing_off = (slots * blk + 31) / 32 * 32;
            for (const uint32_t mode : {2u, 1u, 0u}) {
                std::vector<double> us;
                for (uint32_t it = 0; it < 4; ++it) {
                    Program program = CreateProgram();
                    auto k = CreateKernel(
                        program,
                        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_flat_gu_sender.cpp",
                        lc,
                        DramConfig{.noc = NOC::NOC_0});
                    SetRuntimeArgs(
                        program,
                        k,
                        lc,
                        {buf->address(),
                         drisc_l1_,
                         blk,
                         n,
                         slots,
                         1u,
                         tensix_l1_,
                         4u,
                         drisc_l1_ + timing_off,
                         rx.x,
                         rx.y,
                         bar_addr,
                         0u,
                         1u,
                         mode,
                         rx.x,
                         rx.y});
                    const std::vector<uint32_t> zeros(64, 0);
                    MetalContext::instance().get_cluster().write_core(
                        zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(dev_->build_id(), rx), bar_addr);
                    MetalContext::instance().get_cluster().l1_barrier(dev_->build_id());
                    run(program);
                    if (it == 0) {
                        continue;
                    }
                    auto [a, b] = read_t(dv, drisc_l1_noc_ + timing_off);
                    us.push_back(double(b - a) / clk_mhz);
                }
                std::sort(us.begin(), us.end());
                const double t = us[us.size() / 2];
                log_info(
                    LogTest,
                    "single DRISC -> Tensix {} ({}): {:22s} blk {:5d} B: {:8.1f} us -> {:6.1f} GB/s",
                    rx_l.str(),
                    rx.str(),
                    mode == 2 ? "DMA only" : (mode == 1 ? "NoC writes only" : "DMA + NoC writes"),
                    blk,
                    t,
                    double(total) / t / 1e3);
            }
        }
    }
}

// Both DRISC-usable tiles of bank 0, each reading its half of the bank region and writing to its own Tensix core:
// does the pair reach the bank's 64 GB/s?
TEST_F(DriscFlatBenchFixture, TwoDriscOneBank) {
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(dev_->build_id());
    const uint32_t clk_mhz = MetalContext::instance().get_cluster().get_device_aiclk(dev_->get_device_ids()[0]);
    const uint32_t noc0_sub = endpoint_for_noc(soc, 0, NOC::NOC_0).y;
    std::vector<uint32_t> subs;
    for (uint32_t s = 0; s < soc.get_dram_compute_grid_size().y; ++s) {
        if (s != noc0_sub) {
            subs.push_back(s);
        }
    }
    ASSERT_GE(subs.size(), 2u);
    const uint64_t per_tile = 16ull << 20;
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 2 * per_tile * 8},
        {.page_size = static_cast<uint32_t>(2 * per_tile), .buffer_type = BufferType::DRAM},
        dev_);
    const std::array<CoreCoord, 2> rx_l = {CoreCoord{0, 0}, CoreCoord{1, 0}};
    const CoreCoord bar = dev_->virtual_core_from_logical_core(rx_l[0], CoreType::WORKER);
    const uint32_t bar_addr = tensix_l1_ + 512 * 1024;
    for (const uint32_t blk : {9216u, 16384u}) {
        const uint32_t slots = 4, n = static_cast<uint32_t>(per_tile / blk);
        const uint32_t timing_off = (slots * blk + 31) / 32 * 32;
        for (const uint32_t mode : {2u, 1u, 0u}) {
            std::vector<double> us;
            for (uint32_t it = 0; it < 4; ++it) {
                Program program = CreateProgram();
                std::vector<CoreCoord> dvs;
                for (uint32_t e = 0; e < 2; ++e) {
                    const CoreCoord lc{0, subs[e]};
                    const CoreCoord rx = dev_->virtual_core_from_logical_core(rx_l[e], CoreType::WORKER);
                    auto k = CreateKernel(
                        program,
                        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_flat_gu_sender.cpp",
                        lc,
                        DramConfig{.noc = NOC::NOC_0});
                    SetRuntimeArgs(
                        program,
                        k,
                        lc,
                        {buf->address() + static_cast<uint32_t>(e * per_tile),
                         drisc_l1_,
                         blk,
                         n,
                         slots,
                         1u,
                         tensix_l1_,
                         4u,
                         drisc_l1_ + timing_off,
                         bar.x,
                         bar.y,
                         bar_addr,
                         e,
                         2u,
                         mode,
                         rx.x,
                         rx.y});
                    dvs.push_back(dev_->virtual_core_from_logical_core(lc, CoreType::DRAM));
                }
                const std::vector<uint32_t> zeros(64, 0);
                MetalContext::instance().get_cluster().write_core(
                    zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(dev_->build_id(), bar), bar_addr);
                MetalContext::instance().get_cluster().l1_barrier(dev_->build_id());
                run(program);
                if (it == 0) {
                    continue;
                }
                uint64_t s0 = UINT64_MAX, e1 = 0;
                for (const auto& dv : dvs) {
                    auto [a, b] = read_t(dv, drisc_l1_noc_ + timing_off);
                    s0 = std::min(s0, a);
                    e1 = std::max(e1, b);
                }
                us.push_back(double(e1 - s0) / clk_mhz);
            }
            std::sort(us.begin(), us.end());
            const double t = us[us.size() / 2];
            log_info(
                LogTest,
                "two DRISCs on bank 0 -> 2 Tensix: {:22s} blk {:5d} B: {:8.1f} us -> {:6.1f} GB/s (both tiles)",
                mode == 2 ? "DMA only" : (mode == 1 ? "NoC writes only" : "DMA + NoC writes"),
                blk,
                t,
                double(2 * per_tile) / t / 1e3);
        }
    }
}
