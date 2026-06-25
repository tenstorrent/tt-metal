// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Probe how the X280 NoC TLB reaches a Tensix tile on NOC1. Writes a known value
// to core (0,0) L1, then the FW DMA-reads it back three ways (sel0+translated,
// sel1+translated, sel1+NOC1-coord) and reports which returns the right value.
//
// Build:  make -C tools/x280_bm   ;   cmake --build build_Release --target test_x280_noc1_probe

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>

#include <impl/context/metal_context.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

using tt::Cluster;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;

static constexpr uint64_t LIM_BASE = 0x08000000ULL;
static constexpr uint64_t RESET_UNIT_BASE = 0x80030000ULL;
static constexpr uint64_t L2CPU_RESET_REG = RESET_UNIT_BASE + 0x14;
static constexpr uint64_t L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;
static constexpr uint64_t PLL4_BASE = 0x80020500ULL;
static constexpr uint64_t PLL_CNTL_1 = PLL4_BASE + 0x4;
static constexpr uint64_t PLL_CNTL_5 = PLL4_BASE + 0x14;
static constexpr uint64_t MBOX_PARAMS = 0x08011000ULL;
static constexpr uint64_t MBOX_RESULTS = 0x08011040ULL;
static constexpr uint64_t R_DONE = MBOX_RESULTS + 0x30;
static constexpr uint64_t DONE_MAGIC = 0x10C1C0FFEEULL;
static constexpr uint64_t SRC_L1 = 0x80000ULL;
static constexpr uint64_t KNOWN = 0xABCD1234EE55AA00ULL;

static const CoreCoord ARC_TILE{8, 0};
struct PllSolution {
    int fbdiv;
    int postdiv[4];
};
static bool pll_solution(int mhz, PllSolution& out) {
    switch (mhz) {
        case 1000: out = {80, {1, 1, 1, 1}}; return true;
        case 200: out = {128, {15, 15, 15, 15}}; return true;
        case 800: out = {64, {1, 1, 1, 1}}; return true;
        case 1750: out = {140, {1, 1, 1, 1}}; return true;
        default: return false;
    }
}

class X280 {
public:
    X280(Cluster& c, int chip, int l2) : cluster_(c), chip_(chip), l2cpu_(l2) {
        l2_ = virt({8, 3});
        arc_ = virt(ARC_TILE);
    }
    uint32_t reg_rd(const tt_cxy_pair& t, uint64_t a) const {
        uint32_t v = 0;
        cluster_.read_reg(&v, t, a);
        return v;
    }
    void reg_wr(const tt_cxy_pair& t, uint64_t a, uint32_t v) const { cluster_.write_reg(&v, t, a); }
    uint64_t lim_rd_u64(uint64_t a) const {
        uint64_t v = 0;
        cluster_.read_core(&v, sizeof(v), l2_, a);
        return v;
    }
    void load_lim(const std::vector<uint8_t>& bin) const {
        cluster_.write_core(bin.data(), (uint32_t)bin.size(), l2_, LIM_BASE);
    }
    void write_block(const std::vector<uint8_t>& b, uint64_t a) const {
        cluster_.write_core(b.data(), (uint32_t)b.size(), l2_, a);
    }
    void assert_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v & ~(1u << (l2cpu_ + 4)));
    }
    void release_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v | (1u << (l2cpu_ + 4)));
        (void)reg_rd(arc_, L2CPU_RESET_REG);
    }
    void set_reset_vectors(uint64_t e) const {
        uint32_t lo = (uint32_t)(e & 0xFFFFFFFF), hi = (uint32_t)(e >> 32);
        for (int c = 0; c < 4; c++) {
            reg_wr(l2_, L2CPU_REG_BASE + c * 8, lo);
            reg_wr(l2_, L2CPU_REG_BASE + c * 8 + 4, hi);
        }
    }
    void set_pll(int mhz) const {
        PllSolution sol;
        if (!pll_solution(mhz, sol)) {
            throw std::runtime_error("pll");
        }
        uint32_t c5 = reg_rd(arc_, PLL_CNTL_5);
        uint8_t pd[4] = {
            uint8_t(c5 & 0xFF), uint8_t((c5 >> 8) & 0xFF), uint8_t((c5 >> 16) & 0xFF), uint8_t((c5 >> 24) & 0xFF)};
        uint32_t c1 = reg_rd(arc_, PLL_CNTL_1), c1_low = c1 & 0x0000FFFF;
        uint16_t fb = uint16_t((c1 >> 16) & 0xFFFF);
        auto wc5 = [&]() {
            uint32_t v = pd[0] | (uint32_t(pd[1]) << 8) | (uint32_t(pd[2]) << 16) | (uint32_t(pd[3]) << 24);
            reg_wr(arc_, PLL_CNTL_5, v);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        auto wc1 = [&]() {
            reg_wr(arc_, PLL_CNTL_1, c1_low | (uint32_t(fb) << 16));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        for (int i = 0; i < 4; i++) {
            while (pd[i] < sol.postdiv[i]) {
                pd[i]++;
                wc5();
            }
        }
        while (fb != sol.fbdiv) {
            fb += (sol.fbdiv > fb) ? 1 : -1;
            wc1();
        }
        for (int i = 0; i < 4; i++) {
            while (pd[i] > sol.postdiv[i]) {
                pd[i]--;
                wc5();
            }
        }
    }
    tt_cxy_pair l2() const { return l2_; }

private:
    tt_cxy_pair virt(CoreCoord phys) const {
        return tt_cxy_pair(chip_, cluster_.get_virtual_coordinate_from_physical_coordinates(chip_, phys));
    }
    Cluster& cluster_;
    int chip_, l2cpu_;
    tt_cxy_pair l2_, arc_;
};

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open " + p);
    }
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (d.size() % 4) {
        d.push_back(0);
    }
    return d;
}
template <typename T>
static void pack(std::vector<uint8_t>& b, size_t o, T v) {
    std::memcpy(b.data() + o, &v, sizeof(T));
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    std::string bin_path = "tools/x280_bm/build/noc1_probe.bin";
    int device_id = 0, l2cpu = 0, pll = 1000;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bin") {
            bin_path = argv[++i];
        } else if (a == "--device") {
            device_id = std::stoi(argv[++i]);
        }
    }
    auto bin = read_file(bin_path);
    printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    if (std::system(("tt-smi -r " + std::to_string(device_id)).c_str()) != 0) {
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));

    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();

    // Coords for logical core (0,0): translated (proven NOC0 path) + NOC0 + NOC1.
    CoreCoord trans =
        cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{0, 0}, CoreType::WORKER);
    const auto& sd = cluster.get_soc_desc(device_id);
    tt::umd::CoreCoord log0(0, 0, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL);
    tt::umd::CoreCoord noc0 = sd.translate_coord_to(log0, tt::CoordSystem::NOC0);
    tt::umd::CoreCoord noc1 = sd.translate_coord_to(log0, tt::CoordSystem::NOC1);
    printf(
        "[coords] core(0,0): translated (%u,%u)  NOC0 (%u,%u)  NOC1 (%u,%u)\n",
        (unsigned)trans.x,
        (unsigned)trans.y,
        (unsigned)noc0.x,
        (unsigned)noc0.y,
        (unsigned)noc1.x,
        (unsigned)noc1.y);

    // Write a known value to core(0,0) L1 0x80000 (via the translated coord).
    cluster.write_core(&KNOWN, 8, tt_cxy_pair(device_id, trans), SRC_L1);

    X280 x280(cluster, device_id, l2cpu);
    x280.assert_reset();
    x280.load_lim(bin);
    {
        std::vector<uint8_t> z(0x40, 0);
        x280.write_block(z, MBOX_RESULTS);
    }  // clear stale done
    std::vector<uint8_t> params(64, 0);
    pack<uint32_t>(params, 0x00, (uint32_t)trans.x);
    pack<uint32_t>(params, 0x04, (uint32_t)trans.y);
    pack<uint32_t>(params, 0x08, (uint32_t)noc1.x);
    pack<uint32_t>(params, 0x0C, (uint32_t)noc1.y);
    pack<uint64_t>(params, 0x10, SRC_L1);
    x280.write_block(params, MBOX_PARAMS);
    x280.set_reset_vectors(LIM_BASE);
    x280.set_pll(pll);
    x280.release_reset();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    bool done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (x280.lim_rd_u64(R_DONE) == DONE_MAGIC) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!done) {
        fprintf(stderr, "timeout\n");
        return 1;
    }

    uint32_t rcA = (uint32_t)x280.lim_rd_u64(MBOX_RESULTS + 0x00);
    uint64_t vA = x280.lim_rd_u64(MBOX_RESULTS + 0x08);
    uint32_t rcB = (uint32_t)x280.lim_rd_u64(MBOX_RESULTS + 0x10);
    uint64_t vB = x280.lim_rd_u64(MBOX_RESULTS + 0x18);
    uint32_t rcC = (uint32_t)x280.lim_rd_u64(MBOX_RESULTS + 0x20);
    uint64_t vC = x280.lim_rd_u64(MBOX_RESULTS + 0x28);

    auto rcs = [](uint32_t r) { return r == 0 ? "ok" : r == 1 ? "err" : "timeout"; };
    printf("\n=== X280 NOC1 probe (known value 0x%016llx) ===\n", (unsigned long long)KNOWN);
    printf(
        "  A sel0 + translated : rc=%s val=0x%016llx %s\n",
        rcs(rcA),
        (unsigned long long)vA,
        vA == KNOWN ? "(ground truth ✓)" : "");
    printf(
        "  B sel1 + translated : rc=%s val=0x%016llx %s\n",
        rcs(rcB),
        (unsigned long long)vB,
        (rcB == 0 && vB == KNOWN) ? "<-- NOC1 works w/ translated ✓" : "");
    printf(
        "  C sel1 + NOC1 coord : rc=%s val=0x%016llx %s\n",
        rcs(rcC),
        (unsigned long long)vC,
        (rcC == 0 && vC == KNOWN) ? "<-- NOC1 works w/ NOC1 coord ✓" : "");
    std::fflush(stdout);
    std::_Exit(0);
}
