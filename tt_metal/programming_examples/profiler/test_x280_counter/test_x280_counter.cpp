// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 bare-metal proof-of-life (step 1), host side, as a tt-metal
// programming example.
//
// Boots the single-hart heartbeat firmware (tools/x280_bm, built into
// counter.bin) into the Blackhole X280 (L2CPU) L3 LIM and polls the counter
// the firmware publishes, over the NOC, printing it on the host. A steadily
// increasing value proves the X280 firmware is alive.
//
// Everything goes through tt-metal's UMD (the low-level Cluster) -- a single
// access path. We do NOT use pyluwen here: pyluwen + UMD in one process
// corrupts ARC firmware. A board reset (tt-smi -r) happens before the device
// is opened so the L2CPU is in a resettable state (re-asserting reset on a
// running L2CPU is a no-op on this hardware).
//
// Boot sequence mirrors tt-llm-engine x280/host/loader.py:
//   assert L2CPU reset -> NOC-write counter.bin to LIM 0x08000000 ->
//   set the four hart reset vectors -> step the L2CPU PLL -> release reset.
// The ARC reset-unit / PLL registers are reached as NOC reg writes to the
// ARC tile (8,0); LIM + reset vectors as NOC ops to the L2CPU tile (8,3).
//
// Build:  make -C tools/x280_bm                            (builds counter.bin)
//         cmake --build build_Release --target test_x280_counter
// Run (from the repo root, libs on the path):
//   export TT_METAL_HOME=$PWD
//   export LD_LIBRARY_PATH=$(find build_Release -name '*.so*' -type f \
//       -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
//   ./build_Release/programming_examples/profiler/test_x280_counter --count 20
//
// Flags (all optional): --bin <path> --device N --l2cpu N --pll MHZ
//                       --interval-ms N --count N --no-reset --no-boot

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

using tt::Cluster;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;

// ---- Hardware constants (from tt-llm-engine x280/host/loader.py) ----
static constexpr uint64_t LIM_BASE = 0x08000000ULL;         // L3 LIM SRAM, at-reset
static constexpr uint64_t HB_COUNTER_ADDR = 0x08010000ULL;  // matches src/counter.c
static constexpr uint64_t TRAP_DIAG_ADDR = 0x0800FFE0ULL;   // matches ld/x280-lim.ld
static constexpr uint64_t RESET_UNIT_BASE = 0x80030000ULL;
static constexpr uint64_t L2CPU_RESET_REG = RESET_UNIT_BASE + 0x14;  // bit (4+N) per L2CPU
static constexpr uint64_t L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;    // reset-vector reg file
// L2CPU index -> physical NOC0 coordinate of its tile.
static CoreCoord l2cpu_tile(int idx) {
    switch (idx) {
        case 0: return {8, 3};
        case 1: return {8, 9};
        case 2: return {8, 5};
        case 3: return {8, 7};
        default: return {8, 3};
    }
}
static const CoreCoord ARC_TILE{8, 0};  // ARC reset-unit / PLL registers (NOC-mapped)

// ---- PLL (PLL4 = L2CPU), from tt-llm-engine x280/host/clock.py ----
static constexpr uint64_t PLL4_BASE = 0x80020500ULL;
static constexpr uint64_t PLL_CNTL_1 = PLL4_BASE + 0x4;   // refdiv | postdiv | fbdiv(u16)
static constexpr uint64_t PLL_CNTL_5 = PLL4_BASE + 0x14;  // 4x u8 postdiv

struct PllSolution {
    int fbdiv;
    int postdiv[4];
};
static bool pll_solution(int mhz, PllSolution& out) {
    switch (mhz) {
        case 200: out = {128, {15, 15, 15, 15}}; return true;
        case 800: out = {64, {1, 1, 1, 1}}; return true;
        case 1000: out = {80, {1, 1, 1, 1}}; return true;
        case 1750: out = {140, {1, 1, 1, 1}}; return true;
        default: return false;
    }
}

class X280 {
public:
    X280(Cluster& cluster, int chip, int l2cpu) : cluster_(cluster), chip_(chip), l2cpu_(l2cpu) {
        l2_ = virt(l2cpu_tile(l2cpu_));
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

    void assert_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v & ~(1u << (l2cpu_ + 4)));
    }
    void release_reset() const {
        uint32_t v = reg_rd(arc_, L2CPU_RESET_REG);
        reg_wr(arc_, L2CPU_RESET_REG, v | (1u << (l2cpu_ + 4)));
        (void)reg_rd(arc_, L2CPU_RESET_REG);  // read-back flush
    }

    void load_lim(const std::vector<uint8_t>& bin) const {
        cluster_.write_core(bin.data(), static_cast<uint32_t>(bin.size()), l2_, LIM_BASE);
    }

    void set_reset_vectors(uint64_t entry) const {
        uint32_t lo = static_cast<uint32_t>(entry & 0xFFFFFFFF);
        uint32_t hi = static_cast<uint32_t>(entry >> 32);
        for (int core = 0; core < 4; core++) {  // all four harts share the vector
            reg_wr(l2_, L2CPU_REG_BASE + core * 8, lo);
            reg_wr(l2_, L2CPU_REG_BASE + core * 8 + 4, hi);
        }
    }

    // Step the PLL post-dividers / feedback divider one notch at a time toward
    // the target (faithful port of clock.set_l2cpu_pll), to avoid glitching.
    void set_pll(int mhz) const {
        PllSolution sol;
        if (!pll_solution(mhz, sol)) {
            throw std::runtime_error("no PLL solution for " + std::to_string(mhz) + " MHz");
        }
        uint32_t c5 = reg_rd(arc_, PLL_CNTL_5);
        uint8_t pd[4] = {
            uint8_t(c5 & 0xFF), uint8_t((c5 >> 8) & 0xFF), uint8_t((c5 >> 16) & 0xFF), uint8_t((c5 >> 24) & 0xFF)};
        uint32_t c1 = reg_rd(arc_, PLL_CNTL_1);
        uint32_t c1_low = c1 & 0x0000FFFF;  // preserve refdiv + postdiv bytes
        uint16_t fb = uint16_t((c1 >> 16) & 0xFFFF);

        auto write_c5 = [&]() {
            uint32_t v = pd[0] | (uint32_t(pd[1]) << 8) | (uint32_t(pd[2]) << 16) | (uint32_t(pd[3]) << 24);
            reg_wr(arc_, PLL_CNTL_5, v);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        auto write_c1 = [&]() {
            reg_wr(arc_, PLL_CNTL_1, c1_low | (uint32_t(fb) << 16));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };

        // Increase post-dividers first.
        for (int i = 0; i < 4; i++) {
            while (pd[i] < sol.postdiv[i]) {
                pd[i]++;
                write_c5();
            }
        }
        // Step the feedback divider.
        while (fb != sol.fbdiv) {
            fb += (sol.fbdiv > fb) ? 1 : -1;
            write_c1();
        }
        // Decrease post-dividers last.
        for (int i = 0; i < 4; i++) {
            while (pd[i] > sol.postdiv[i]) {
                pd[i]--;
                write_c5();
            }
        }
    }

    tt_cxy_pair l2() const { return l2_; }
    tt_cxy_pair arc() const { return arc_; }

private:
    tt_cxy_pair virt(CoreCoord phys) const {
        CoreCoord v = cluster_.get_virtual_coordinate_from_physical_coordinates(chip_, phys);
        return tt_cxy_pair(chip_, v);
    }

    Cluster& cluster_;
    int chip_;
    int l2cpu_;
    tt_cxy_pair l2_;
    tt_cxy_pair arc_;
};

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open firmware: " + path);
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (data.size() % 4) {
        data.push_back(0);  // pad to 4 bytes
    }
    return data;
}

int main(int argc, char** argv) {
    std::string bin_path = "tools/x280_bm/build/counter.bin";
    int device_id = 0, l2cpu = 0, pll = 1000, interval_ms = 100, count = 30;
    bool do_reset = true, do_boot = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--bin") {
            bin_path = next();
        } else if (a == "--device") {
            device_id = std::stoi(next());
        } else if (a == "--l2cpu") {
            l2cpu = std::stoi(next());
        } else if (a == "--pll") {
            pll = std::stoi(next());
        } else if (a == "--interval-ms") {
            interval_ms = std::stoi(next());
        } else if (a == "--count") {
            count = std::stoi(next());
        } else if (a == "--no-reset") {
            do_reset = false;
        } else if (a == "--no-boot") {
            do_boot = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    std::vector<uint8_t> bin;
    if (do_boot) {
        bin = read_file(bin_path);
        printf("[fw] %s (%zu bytes)\n", bin_path.c_str(), bin.size());
    }

    // Board reset BEFORE opening the device, so the L2CPU is resettable and
    // ARC re-initializes NOC routing. (tt-smi -r tears down any running X280.)
    if (do_reset) {
        std::string cmd = "tt-smi -r " + std::to_string(device_id);
        printf("[boot] %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            fprintf(stderr, "tt-smi reset failed\n");
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    // Open the chip via tt-metal; this initializes the Cluster we drive below.
    auto mesh = MeshDevice::create_unit_mesh(device_id);
    Cluster& cluster = MetalContext::instance().get_cluster();

    X280 x280(cluster, device_id, l2cpu);
    printf(
        "[boot] L2CPU %d = NOC tile (%u,%u); ARC virt (%u,%u)\n",
        l2cpu,
        (unsigned)x280.l2().x,
        (unsigned)x280.l2().y,
        (unsigned)x280.arc().x,
        (unsigned)x280.arc().y);

    if (do_boot) {
        x280.assert_reset();
        x280.load_lim(bin);

        // Verify the load landed.
        std::vector<uint8_t> rb(8);
        cluster.read_core(rb.data(), 8, x280.l2(), LIM_BASE);
        if (std::memcmp(rb.data(), bin.data(), 8) != 0) {
            fprintf(stderr, "[boot] LIM readback mismatch\n");
            return 1;
        }
        printf("[boot] loaded %zu bytes to LIM 0x%08llx, readback OK\n", bin.size(), (unsigned long long)LIM_BASE);

        x280.set_reset_vectors(LIM_BASE);
        printf("[boot] PLL -> %d MHz\n", pll);
        x280.set_pll(pll);
        x280.release_reset();
        printf("[boot] released L2CPU %d from reset\n", l2cpu);
    }

    printf("[poll] reading 0x%08llx every %d ms\n\n", (unsigned long long)HB_COUNTER_ADDR, interval_ms);
    uint64_t prev = 0;
    bool have_prev = false;
    for (int i = 0; i < count; i++) {
        uint64_t val = x280.lim_rd_u64(HB_COUNTER_ADDR);
        if (have_prev) {
            printf("[poll %5d] counter = %llu  (+%lld)\n", i, (unsigned long long)val, (long long)(val - prev));
        } else {
            printf("[poll %5d] counter = %llu\n", i, (unsigned long long)val);
        }
        prev = val;
        have_prev = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    // The trap-diag slot is only written by the FW's trap handler; if the
    // counter advanced, no trap occurred and the slot is just uninitialized
    // LIM. Report it only as a hint when the counter looks stuck.
    printf("\n[done] last counter = %llu\n", (unsigned long long)prev);
    if (!have_prev || prev == 0) {
        uint64_t mcause = x280.lim_rd_u64(TRAP_DIAG_ADDR);
        printf("[done] counter did not advance; trap mcause slot = 0x%016llx\n", (unsigned long long)mcause);
    }
    return 0;
}
