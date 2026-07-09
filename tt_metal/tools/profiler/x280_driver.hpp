// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// X280Driver — boot/control of a Blackhole SiFive X280 (L2CPU) tile from the host via UMD.
//
// The X280 is the (sole) consumer of the per-RISC SPSC kernel-profiler zone rings: each Tensix
// RISC streams zone markers into an L1 ring and BLOCKS when it fills, so a drainer must advance
// the consumer head or producers deadlock. This driver loads a drainer firmware (.bin) into the
// X280's LIM, points its reset vectors at it, sets the core PLL, and releases reset. It is the
// reusable boot path shared by the standalone test harness and RealtimeProfilerManager.
//
// Blackhole-only. All addresses are X280/L2CPU NOC register + LIM addresses.

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <umd/device/types/xy_pair.hpp>

#include "core_coord.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::profiler {

// --- X280 / L2CPU register + LIM map (Blackhole) ---
inline constexpr uint64_t X280_LIM_BASE = 0x08000000ULL;  // local SRAM; firmware loads here
inline constexpr uint64_t X280_RESET_UNIT_BASE = 0x80030000ULL;
inline constexpr uint64_t X280_L2CPU_RESET_REG = X280_RESET_UNIT_BASE + 0x14;
inline constexpr uint64_t X280_L2CPU_REG_BASE = 0xFFFFF7FEFFF10000ULL;  // per-hart reset vector regs
inline constexpr uint64_t X280_PLL4_BASE = 0x80020500ULL;
inline constexpr uint64_t X280_PLL_CNTL_1 = X280_PLL4_BASE + 0x4;
inline constexpr uint64_t X280_PLL_CNTL_5 = X280_PLL4_BASE + 0x14;

inline constexpr int X280_NRISC = 5;                  // Tensix RISCs per core feeding the SPSC rings
inline constexpr uint32_t X280_PROF_CTRL_WORDS = 32;  // profiler control_vector length (words)

// Physical NOC coords of the four L2CPU tiles + the ARC tile on Blackhole.
inline CoreCoord x280_l2cpu_tile(int idx) {
    switch (idx) {
        case 1: return {8, 9};
        case 2: return {8, 5};
        case 3: return {8, 7};
        default: return {8, 3};
    }
}
inline const CoreCoord X280_ARC_TILE{8, 0};

struct X280PllSolution {
    int fbdiv;
    int postdiv[4];
};
inline bool x280_pll_solution(int mhz, X280PllSolution& out) {
    switch (mhz) {
        case 200: out = {128, {15, 15, 15, 15}}; return true;
        case 800: out = {64, {1, 1, 1, 1}}; return true;
        case 1000: out = {80, {1, 1, 1, 1}}; return true;
        case 1750: out = {140, {1, 1, 1, 1}}; return true;
        default: return false;
    }
}

class X280Driver {
public:
    X280Driver(::tt::Cluster& cluster, int chip, int l2cpu) : cluster_(cluster), chip_(chip), l2cpu_(l2cpu) {
        l2_ = virt(x280_l2cpu_tile(l2cpu_));
        arc_ = virt(X280_ARC_TILE);
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
    void lim_wr_u64(uint64_t a, uint64_t v) const { cluster_.write_core(&v, sizeof(v), l2_, a); }

    void write_block(const void* data, uint32_t size, uint64_t addr) const {
        cluster_.write_core(data, size, l2_, addr);
    }
    void read_block(void* data, uint32_t size, uint64_t addr) const { cluster_.read_core(data, size, l2_, addr); }

    void assert_reset() const {
        uint32_t v = reg_rd(arc_, X280_L2CPU_RESET_REG);
        reg_wr(arc_, X280_L2CPU_RESET_REG, v & ~(1u << (l2cpu_ + 4)));
    }
    void release_reset() const {
        uint32_t v = reg_rd(arc_, X280_L2CPU_RESET_REG);
        reg_wr(arc_, X280_L2CPU_RESET_REG, v | (1u << (l2cpu_ + 4)));
        (void)reg_rd(arc_, X280_L2CPU_RESET_REG);
    }
    void load_lim(const std::vector<uint8_t>& bin) const {
        cluster_.write_core(bin.data(), static_cast<uint32_t>(bin.size()), l2_, X280_LIM_BASE);
    }
    void set_reset_vectors(uint64_t entry) const {
        uint32_t lo = static_cast<uint32_t>(entry & 0xFFFFFFFF);
        uint32_t hi = static_cast<uint32_t>(entry >> 32);
        for (int core = 0; core < 4; core++) {
            reg_wr(l2_, X280_L2CPU_REG_BASE + core * 8, lo);
            reg_wr(l2_, X280_L2CPU_REG_BASE + core * 8 + 4, hi);
        }
    }
    // Ramp the X280 core PLL to `mhz` by stepping postdiv/fbdiv one notch at a time (the HW
    // requires monotonic single-step changes).
    void set_pll(int mhz) const {
        X280PllSolution sol;
        if (!x280_pll_solution(mhz, sol)) {
            throw std::runtime_error("no X280 PLL solution for " + std::to_string(mhz) + " MHz");
        }
        uint32_t c5 = reg_rd(arc_, X280_PLL_CNTL_5);
        uint8_t pd[4] = {
            uint8_t(c5 & 0xFF), uint8_t((c5 >> 8) & 0xFF), uint8_t((c5 >> 16) & 0xFF), uint8_t((c5 >> 24) & 0xFF)};
        uint32_t c1 = reg_rd(arc_, X280_PLL_CNTL_1);
        uint32_t c1_low = c1 & 0x0000FFFF;
        uint16_t fb = uint16_t((c1 >> 16) & 0xFFFF);
        auto write_c5 = [&]() {
            uint32_t v = pd[0] | (uint32_t(pd[1]) << 8) | (uint32_t(pd[2]) << 16) | (uint32_t(pd[3]) << 24);
            reg_wr(arc_, X280_PLL_CNTL_5, v);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        auto write_c1 = [&]() {
            reg_wr(arc_, X280_PLL_CNTL_1, c1_low | (uint32_t(fb) << 16));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        };
        for (int i = 0; i < 4; i++) {
            while (pd[i] < sol.postdiv[i]) {
                pd[i]++;
                write_c5();
            }
        }
        while (fb != sol.fbdiv) {
            fb += (sol.fbdiv > fb) ? 1 : -1;
            write_c1();
        }
        for (int i = 0; i < 4; i++) {
            while (pd[i] > sol.postdiv[i]) {
                pd[i]--;
                write_c5();
            }
        }
    }

    // One-time-per-cold-power-cycle L3 LIM ECC prime. Routes LIM's SRAM through the L3 cache controller
    // (WayEnable=0xF, every master pinned to Way 15) and touches each 64 B line of the read-as-zero L3
    // Zero Device, so the controller writes valid data+ECC into the physical SRAM backing LIM. WayEnable
    // is increase-only, so LIM is a cache (unusable) until the caller warm-resets the chip AFTER this —
    // the ECC then persists across that reset and LIM_BASE becomes valid SRAM. Mirrors
    // test_x280_profcons --primeecc. `prime_bytes` must cover the firmware + mailboxes + stacks.
    void prime_lim_ecc(uint64_t prime_bytes = 0x60000) const {
        constexpr uint64_t kL3WayEnable = 0x02010008ULL;
        constexpr uint64_t kL3WayMaskBase = 0x02010800ULL;
        constexpr uint32_t kL3NumMasters = 38;
        constexpr uint64_t kZeroDeviceBase = 0x0A000000ULL;
        reg_wr(l2_, kL3WayEnable, 0xF);
        for (uint32_t m = 0; m < kL3NumMasters; m++) {
            reg_wr(l2_, kL3WayMaskBase + static_cast<uint64_t>(m) * 8, 0x8000);  // force alloc into Way 15
        }
        for (uint64_t off = 0; off < prime_bytes; off += 64) {
            reg_wr(l2_, kZeroDeviceBase + off, 0);  // touch each line -> fetch+merge+writeback valid ECC
        }
    }

    tt_cxy_pair l2() const { return l2_; }
    tt_cxy_pair arc() const { return arc_; }

private:
    tt_cxy_pair virt(CoreCoord phys) const {
        CoreCoord v = cluster_.get_virtual_coordinate_from_physical_coordinates(chip_, phys);
        return tt_cxy_pair(chip_, v);
    }
    ::tt::Cluster& cluster_;
    int chip_;
    int l2cpu_;
    tt_cxy_pair l2_;
    tt_cxy_pair arc_;
};

}  // namespace tt::tt_metal::profiler
