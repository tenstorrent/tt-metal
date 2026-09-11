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
// Usage:  x280_boot [--chip N] status
//         x280_boot [--chip N] niu                  (L2CPU tile NIU config + slave counters)
//         x280_boot [--chip N] rd <addr> [nwords]   (dump x280-physical memory, e.g. the uncached GDDR alias)
//         x280_boot [--chip N] boot <fw.bin>        (tile 0 = NOC0 (8,3))
// --chip selects the UMD chip id (default 0); the tool otherwise touches one chip.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
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

void print_status(Cluster& cluster, TTDevice* dev, const CoreCoord& l2cpu, tt::ChipId chip) {
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
    cluster.read_from_device(mbox, chip, l2cpu, kMailbox, sizeof(mbox));
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

// The L2CPU tile's own NIU registers. UMD places the NOC control-register block of
// the NOC2AXI tiles (PCIe/ARC/L2CPU) at a 64-bit base (blackhole_implementation.hpp
// NOC0_CONTROL_REG_ADDR_BASE_MAP), not at the 0xFFB2_0000 base Tensix/ETH/DRAM use.
// Register offsets follow tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:
// NOC_NODE_ID +0x44, NOC_CFG(n) +0x100+4n, NOC_STATUS(n) +0x200+4n.
constexpr uint64_t kL2cpuNiuBase = 0xFFFFFFFFFF000000ULL;

void print_niu(Cluster& cluster, const CoreCoord& l2cpu, tt::ChipId chip) {
    auto rd = [&](uint64_t off) {
        uint32_t v = 0xFFFFFFFFu;
        cluster.read_from_device_reg(&v, chip, l2cpu, kL2cpuNiuBase + off, 4);
        return v;
    };
    try {
        const uint32_t node_id = rd(0x44);
        const uint32_t cfg0 = rd(0x100);
        printf(
            "L2CPU NIU regs @0x%llx: NOC_NODE_ID=0x%08x (x=%u y=%u)  NIU_CFG_0=0x%08x (NOC_ID_TRANSLATE_EN[14]=%u)\n",
            (unsigned long long)kL2cpuNiuBase,
            node_id,
            node_id & 0x3f,
            (node_id >> 6) & 0x3f,
            cfg0,
            (cfg0 >> 14) & 1);
        struct {
            const char* name;
            uint32_t idx;
        } regs[] = {
            {"NIU_SLV_REQ_ACCEPTED", 0x34},
            {"NIU_SLV_RD_REQ_RECEIVED", 0x35},
            {"NIU_SLV_RD_RESP_SENT", 0x32},
            {"NIU_SLV_NONPOSTED_WR_REQ_RECEIVED", 0x3A},
            {"NIU_SLV_POSTED_WR_REQ_RECEIVED", 0x3B},
            {"NIU_SLV_WR_ACK_SENT", 0x31},
            {"NIU_SLV_NONPOSTED_ATOMIC_RECEIVED", 0x36},
            {"NIU_SLV_POSTED_ATOMIC_RECEIVED", 0x37},
            {"NIU_SLV_ATOMIC_RESP_SENT", 0x30},
        };
        for (const auto& r : regs) {
            printf("  %-36s = %u\n", r.name, rd(0x200 + r.idx * 4));
        }
    } catch (const std::exception& e) {
        printf("L2CPU NIU register read failed: %s\n", e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Optional leading "--chip N" selects the UMD chip id; everything else is unchanged.
    tt::ChipId chip = 0;
    int argi = 1;
    if (argc >= 3 && std::string(argv[1]) == "--chip") {
        chip = static_cast<tt::ChipId>(std::strtoul(argv[2], nullptr, 0));
        argi = 3;
    }
    const int nargs = argc - argi;
    if (nargs < 1) {
        fprintf(stderr, "usage: %s [--chip N] status | niu | rd <addr> [n] | boot <fw.bin>\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[argi];

    Cluster cluster;  // default options: UMD's own soc descriptor, which knows the L2CPU cores
    TTDevice* dev = cluster.get_tt_device(chip);
    CoreCoord l2cpu(8, 3, tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
    printf("chip %u\n", static_cast<unsigned>(chip));

    if (cmd == "status") {
        print_status(cluster, dev, l2cpu, chip);
        return 0;
    }
    if (cmd == "niu") {
        print_niu(cluster, l2cpu, chip);
        return 0;
    }
    if (cmd == "rd" && nargs >= 2) {
        const uint64_t addr = std::strtoull(argv[argi + 1], nullptr, 0);
        const uint32_t n = nargs >= 3 ? static_cast<uint32_t>(std::strtoul(argv[argi + 2], nullptr, 0)) : 8u;
        std::vector<uint32_t> words(n);
        cluster.read_from_device(words.data(), chip, l2cpu, addr, n * 4);
        for (uint32_t i = 0; i < n; i++) {
            if (i % 8 == 0) {
                printf("%s0x%08llx:", i ? "\n" : "", (unsigned long long)(addr + i * 4));
            }
            printf(" %08x", words[i]);
        }
        printf("\n");
        return 0;
    }

    if (cmd != "boot" || nargs < 2) {
        fprintf(stderr, "usage: %s [--chip N] status | niu | rd <addr> [n] | boot <fw.bin>\n", argv[0]);
        return 1;
    }
    const char* fw_path = argv[argi + 1];

    uint32_t reset = arc_rd32(dev, kArcL2cpuReset);
    if ((reset >> (4 + kTileIndex)) & 1) {
        fprintf(stderr, "tile already released from reset (one-shot!) — refusing. tt-smi reset to retry.\n");
        return 1;
    }

    // 0. Enable the whole L3 as cache (tt-bh-linux does this before loading, when
    //    running from DRAM). This removes the LIM scratchpad, so the mailbox lives
    //    in uncached GDDR instead.
    uint32_t wayenable = 0xf;
    cluster.write_to_device(&wayenable, 4, chip, l2cpu, kL3WayEnable);
    uint32_t way_rb = 0;
    cluster.read_from_device(&way_rb, chip, l2cpu, kL3WayEnable, 4);
    printf("L3 cache ways enabled (CCACHE0_WAYENABLE=0x%x)\n", way_rb);

    // 1. Load firmware into GDDR (via the uncached alias) and verify. A readback
    //    mismatch here means GDDR isn't accessible — we abort BEFORE consuming the
    //    one-shot reset, so a DRAM problem never costs a reset cycle.
    std::ifstream f(fw_path, std::ios::binary);
    std::vector<char> fw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (fw.empty()) {
        fprintf(stderr, "cannot read %s\n", fw_path);
        return 1;
    }
    fw.resize((fw.size() + 3) & ~3ULL);
    cluster.write_to_device(fw.data(), fw.size(), chip, l2cpu, kLoadNoc);
    std::vector<char> check(fw.size());
    cluster.read_from_device(check.data(), chip, l2cpu, kLoadNoc, check.size());
    if (memcmp(fw.data(), check.data(), fw.size()) != 0) {
        fprintf(stderr, "firmware read-back mismatch at GDDR — is DRAM trained? aborting before reset.\n");
        return 1;
    }
    printf(
        "firmware loaded + verified: %zu bytes at GDDR (exec alias 0x%llx)\n",
        fw.size(),
        (unsigned long long)kResetVector);

    // 2. Clear the mailbox region (uncached GDDR alias): status, request, connection,
    //    response and inbox blocks all live in the first 64 KiB past kMailbox.
    std::vector<char> zeros(0x10000, 0);
    cluster.write_to_device(zeros.data(), zeros.size(), chip, l2cpu, kMailbox);

    // 3. Reset vectors for all four harts -> cached GDDR exec alias.
    for (int hart = 0; hart < 4; hart++) {
        uint32_t lo = (uint32_t)(kResetVector & 0xffffffff);
        uint32_t hi = (uint32_t)(kResetVector >> 32);
        cluster.write_to_device(&lo, 4, chip, l2cpu, kResetVecBase + hart * 8);
        cluster.write_to_device(&hi, 4, chip, l2cpu, kResetVecBase + hart * 8 + 4);
        uint32_t rb_lo = 0, rb_hi = 0;
        cluster.read_from_device(&rb_lo, chip, l2cpu, kResetVecBase + hart * 8, 4);
        cluster.read_from_device(&rb_hi, chip, l2cpu, kResetVecBase + hart * 8 + 4, 4);
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
        cluster.write_to_device(&basic, 4, chip, l2cpu, kL2PrefetchBase + off);
        cluster.write_to_device(&user, 4, chip, l2cpu, kL2PrefetchBase + 4 + off);
    }

    // 7. Watch the heartbeat. boot_marker distinguishes "hart executed at all"
    //    (marker set, from the first instructions in start.S) from "never ran".
    for (int i = 0; i < 5; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        uint64_t mbox[8] = {};
        cluster.read_from_device(mbox, chip, l2cpu, kMailbox, sizeof(mbox));
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
        cluster.read_from_device(&done, chip, l2cpu, kMailbox + 0xF0, 4);
    }
    if (done == 0x00B7D09Eu) {
        printf("\nTLB-window store bandwidth (x280 -> own GDDR via NOC loopback, %.0f MHz):\n", kFreqHz / 1e6);
        uint32_t rec[6 * 4] = {};
        cluster.read_from_device(rec, chip, l2cpu, kMailbox + 0x100, sizeof(rec));
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
