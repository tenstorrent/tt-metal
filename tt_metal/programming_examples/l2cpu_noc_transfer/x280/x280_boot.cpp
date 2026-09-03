// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Boot tool for the Blackhole L2CPU x280 harts, raw UMD only (no tt-metal).
// Follows the reference sequence in tenstorrent/tt-bh-linux boot.py:
//   1. load firmware into the tile's LIM over the NOC (works with harts in reset)
//   2. set every hart's reset vector (external-peripherals regs on the tile)
//   3. step PLL4 (the L2CPU clock) to the 200 MHz solution
//   4. read-modify-write ARC reset-unit L2CPU_RESET (offset 0x30014), bit 4+idx
//
// HARDWARE BUG: harts can be released exactly once per chip reset. This tool
// refuses to run "boot" if the tile's reset bit is already set. A crashed or
// wedged firmware means tt-smi reset.
//
// Usage:  x280_boot status
//         x280_boot boot <fw.bin>        (tile 0 = NOC0 (8,3))

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "umd/device/cluster.hpp"

using namespace tt::umd;

namespace {

// Reference-faithful (tt-bh-linux) DRAM boot path. Firmware executes from cached
// GDDR; loaded and mailbox-read via the uncached GDDR alias so no cache flush is
// needed for host<->firmware visibility.
constexpr uint64_t kLoadNoc = 0x30000000ULL;          // uncached GDDR alias: host load + readback
constexpr uint64_t kResetVector = 0x400030000000ULL;  // cached GDDR alias: hart's initial PC
constexpr uint64_t kMailbox = 0x30100000ULL;          // uncached GDDR alias, +1 MiB
constexpr uint64_t kL3WayEnable = 0x02010000ULL + 8;  // L3_REG_BASE + 8 = CCACHE0_WAYENABLE
constexpr uint64_t kL2PrefetchBase = 0x02030000ULL;
constexpr uint64_t kResetVecBase = 0xfffff7fefff10000ULL;  // + hart*8: lo32, hi32
constexpr uint64_t kArcPll4Cntl1 = 0x20504;                // ARC offsets (base 0x8000_0000)
constexpr uint64_t kArcPll4Cntl5 = 0x20514;
constexpr uint64_t kArcL2cpuReset = 0x30014;
constexpr int kTileIndex = 0;  // L2CPU0 = NOC0 (8,3), reset bit 4

// ARC peripheral registers (PLL4, reset unit) live in ARC APB space; the offsets
// below are APB offsets (e.g. reset unit = ARC_RESET_UNIT_OFFSET 0x30000, so
// kArcL2cpuReset 0x30014 = reset unit + 0x14). UMD renamed read_from_arc ->
// read_from_arc_apb (same {ptr, offset, size} signature).
uint32_t arc_rd32(TTDevice* dev, uint64_t off) {
    uint32_t v = 0;
    dev->read_from_arc_apb(&v, off, sizeof(v));
    return v;
}

void arc_wr32(TTDevice* dev, uint64_t off, uint32_t v) { dev->write_to_arc_apb(&v, off, sizeof(v)); }

void print_status(Cluster& cluster, TTDevice* dev, const CoreCoord& l2cpu) {
    uint32_t pll1 = arc_rd32(dev, kArcPll4Cntl1);
    uint32_t pll5 = arc_rd32(dev, kArcPll4Cntl5);
    uint32_t reset = arc_rd32(dev, kArcL2cpuReset);
    printf(
        "PLL4_CNTL_1 = 0x%08x  (refdiv=%u postdiv=%u fbdiv=%u)\n", pll1, pll1 & 0xff, (pll1 >> 8) & 0xff, pll1 >> 16);
    printf(
        "PLL4_CNTL_5 = 0x%08x  (postdivs %u %u %u %u)\n",
        pll5,
        pll5 & 0xff,
        (pll5 >> 8) & 0xff,
        (pll5 >> 16) & 0xff,
        (pll5 >> 24) & 0xff);
    printf("L2CPU_RESET = 0x%08x  (tile0 (8,3) released: %s)\n", reset, (reset >> 4) & 1 ? "YES" : "no");

    uint64_t mbox[8] = {};
    cluster.read_from_device(mbox, 0, l2cpu, kMailbox, sizeof(mbox));
    printf(
        "mailbox: heartbeat=%lu fw_state=0x%lx hartid=%lu traps=%lu mcause=0x%lx cmo_ok=%lu boot_marker=0x%lx\n",
        mbox[0],
        mbox[1],
        mbox[2],
        mbox[3],
        mbox[4],
        mbox[5],
        mbox[6]);
}

// Step one byte-field of a PLL register toward a target, one unit per write,
// mirroring tt-bh-linux clock.py.
void step_field(TTDevice* dev, uint64_t reg, int byte_lo, int width_bits, uint32_t target) {
    for (;;) {
        uint32_t v = arc_rd32(dev, reg);
        uint32_t mask = (width_bits == 16) ? 0xffffu : 0xffu;
        uint32_t cur = (v >> byte_lo) & mask;
        if (cur == target) {
            return;
        }
        uint32_t next = cur + (target > cur ? 1 : -1);
        v = (v & ~(mask << byte_lo)) | (next << byte_lo);
        arc_wr32(dev, reg, v);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// 200 MHz solution from tt-bh-linux clock.py: fbdiv=128, postdivs={15,15,15,15}.
// Order: raise postdivs first, then move fbdiv, then lower postdivs.
void set_pll4_200mhz(TTDevice* dev) {
    for (int i = 0; i < 4; i++) {
        uint32_t cur = (arc_rd32(dev, kArcPll4Cntl5) >> (8 * i)) & 0xff;
        if (cur < 15) {
            step_field(dev, kArcPll4Cntl5, 8 * i, 8, 15);
        }
    }
    step_field(dev, kArcPll4Cntl1, 16, 16, 128);
    for (int i = 0; i < 4; i++) {
        step_field(dev, kArcPll4Cntl5, 8 * i, 8, 15);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s status | boot <fw.bin>\n", argv[0]);
        return 1;
    }

    Cluster cluster;  // default options: UMD's own soc descriptor, which knows the L2CPU cores
    TTDevice* dev = cluster.get_tt_device(0);
    CoreCoord l2cpu(8, 3, tt::CoreType::L2CPU, tt::CoordSystem::NOC0);

    if (std::string(argv[1]) == "status") {
        print_status(cluster, dev, l2cpu);
        return 0;
    }

    if (std::string(argv[1]) != "boot" || argc < 3) {
        fprintf(stderr, "usage: %s status | boot <fw.bin>\n", argv[0]);
        return 1;
    }

    uint32_t reset = arc_rd32(dev, kArcL2cpuReset);
    if ((reset >> (4 + kTileIndex)) & 1) {
        fprintf(stderr, "tile already released from reset (one-shot!) — refusing. tt-smi reset to retry.\n");
        return 1;
    }

    // 0. Enable the whole L3 as cache (tt-bh-linux does this before loading, when
    //    running from DRAM). This removes the LIM scratchpad, so the mailbox lives
    //    in uncached GDDR instead.
    uint32_t wayenable = 0xf;
    cluster.write_to_device(&wayenable, 4, 0, l2cpu, kL3WayEnable);
    uint32_t way_rb = 0;
    cluster.read_from_device(&way_rb, 0, l2cpu, kL3WayEnable, 4);
    printf("L3 cache ways enabled (CCACHE0_WAYENABLE=0x%x)\n", way_rb);

    // 1. Load firmware into GDDR (via the uncached alias) and verify. A readback
    //    mismatch here means GDDR isn't accessible — we abort BEFORE consuming the
    //    one-shot reset, so a DRAM problem never costs a reset cycle.
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<char> fw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (fw.empty()) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    fw.resize((fw.size() + 3) & ~3ULL);
    cluster.write_to_device(fw.data(), fw.size(), 0, l2cpu, kLoadNoc);
    std::vector<char> check(fw.size());
    cluster.read_from_device(check.data(), 0, l2cpu, kLoadNoc, check.size());
    if (memcmp(fw.data(), check.data(), fw.size()) != 0) {
        fprintf(stderr, "firmware read-back mismatch at GDDR — is DRAM trained? aborting before reset.\n");
        return 1;
    }
    printf(
        "firmware loaded + verified: %zu bytes at GDDR (exec alias 0x%llx)\n",
        fw.size(),
        (unsigned long long)kResetVector);

    // 2. Clear the mailbox (uncached GDDR alias).
    std::vector<char> zeros(0x100, 0);
    cluster.write_to_device(zeros.data(), zeros.size(), 0, l2cpu, kMailbox);

    // 3. Reset vectors for all four harts -> cached GDDR exec alias.
    for (int hart = 0; hart < 4; hart++) {
        uint32_t lo = (uint32_t)(kResetVector & 0xffffffff);
        uint32_t hi = (uint32_t)(kResetVector >> 32);
        cluster.write_to_device(&lo, 4, 0, l2cpu, kResetVecBase + hart * 8);
        cluster.write_to_device(&hi, 4, 0, l2cpu, kResetVecBase + hart * 8 + 4);
        uint32_t rb_lo = 0, rb_hi = 0;
        cluster.read_from_device(&rb_lo, 0, l2cpu, kResetVecBase + hart * 8, 4);
        cluster.read_from_device(&rb_hi, 0, l2cpu, kResetVecBase + hart * 8 + 4, 4);
        if (rb_lo != lo || rb_hi != hi) {
            fprintf(stderr, "reset vector readback mismatch on hart %d — aborting\n", hart);
            return 1;
        }
    }
    printf("reset vectors set (4 harts -> 0x%llx)\n", (unsigned long long)kResetVector);

    // 4. L2CPU PLL to the 200 MHz solution (release must happen at low speed).
    uint32_t pll1 = arc_rd32(dev, kArcPll4Cntl1);
    if ((pll1 >> 16) == 0) {
        fprintf(stderr, "PLL4 fbdiv is 0 — PLL looks unconfigured; not safe to release. aborting.\n");
        return 1;
    }
    set_pll4_200mhz(dev);
    printf("PLL4 stepped to 200 MHz solution\n");

    // 5. Release the tile's harts (one-shot!), then confirm the bit actually set.
    reset = arc_rd32(dev, kArcL2cpuReset);
    arc_wr32(dev, kArcL2cpuReset, reset | (1u << (4 + kTileIndex)));
    uint32_t reset_after = arc_rd32(dev, kArcL2cpuReset);
    printf(
        "released tile %d from reset (L2CPU_RESET 0x%x -> 0x%x, bit%d=%d)\n",
        kTileIndex,
        reset,
        reset_after,
        4 + kTileIndex,
        (reset_after >> (4 + kTileIndex)) & 1);

    // 6. Configure L2 prefetchers (tt-bh-linux does this after release).
    for (uint32_t off : {0x0000u, 0x2000u, 0x4000u, 0x6000u}) {
        uint32_t basic = 0x15811, user = 0x38c84e;
        cluster.write_to_device(&basic, 4, 0, l2cpu, kL2PrefetchBase + off);
        cluster.write_to_device(&user, 4, 0, l2cpu, kL2PrefetchBase + 4 + off);
    }

    // 7. Watch the heartbeat. boot_marker distinguishes "hart executed at all"
    //    (marker set, from the first instructions in start.S) from "never ran".
    for (int i = 0; i < 5; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        uint64_t mbox[8] = {};
        cluster.read_from_device(mbox, 0, l2cpu, kMailbox, sizeof(mbox));
        printf(
            "t+%dms boot_marker=0x%lx heartbeat=%lu fw_state=0x%lx hartid=%lu traps=%lu cmo_ok=%lu\n",
            200 * (i + 1),
            mbox[6],
            mbox[0],
            mbox[1],
            mbox[2],
            mbox[3],
            mbox[5]);
    }

    // 8. If the firmware is the bandwidth build, it sets a done flag at
    //    mailbox+0xF0 and writes result records at +0x100. Poll for it and, if
    //    present, print MB/s (mcycle counts at the 200 MHz PLL solution).
    constexpr double kFreqHz = 200e6;
    const char* tag_name[6] = {
        "posted sd  4KiB",
        "posted sd 16KiB",
        "posted sd 64KiB",
        "posted sd 256KiB",
        "default sd 64KiB",
        "posted sw 64KiB"};
    uint32_t done = 0;
    for (int i = 0; i < 30 && done != 0x00B7D09Eu; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cluster.read_from_device(&done, 0, l2cpu, kMailbox + 0xF0, 4);
    }
    if (done == 0x00B7D09Eu) {
        printf("\nTLB-window store bandwidth (x280 -> own GDDR via NOC loopback, %.0f MHz):\n", kFreqHz / 1e6);
        uint32_t rec[6 * 4] = {};
        cluster.read_from_device(rec, 0, l2cpu, kMailbox + 0x100, sizeof(rec));
        for (int i = 0; i < 6; i++) {
            uint32_t tag = rec[i * 4 + 0], size = rec[i * 4 + 1];
            uint64_t cyc = rec[i * 4 + 2] | ((uint64_t)rec[i * 4 + 3] << 32);
            double secs = cyc / kFreqHz;
            double mbps = secs > 0 ? (size / secs) / 1e6 : 0.0;
            printf(
                "  %-16s %7u B  %10lu cyc  %6.2f cyc/8B  %8.1f MB/s\n",
                tag < 6 ? tag_name[tag] : "?",
                size,
                cyc,
                cyc * 8.0 / size,
                mbps);
        }
    } else {
        printf("\n(no bandwidth results — firmware is the echo build, or sweep didn't finish)\n");
    }
    return 0;
}
