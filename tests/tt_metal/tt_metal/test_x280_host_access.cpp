// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// X280 host-access spike: drives L2CPU discovery, state inspection, and a bounded
// LIM write through the UMD cluster Metal already owns, never a second owner.
//
//   X280HostAccessReadOnlyGate    -- no device writes at all
//   X280HostAccessLimScratchWrite -- one restoring 64-byte LIM write; requires
//                                    primed LIM ECC
//   X280HostAccessArcMessage      -- one ARC NOP; changes no state
//   X280HostAccessAllOnesRejection-- unclocks one tile to prove the all-ones
//                                    guard fires, then restores it
//   X280HostAccessPllSlew         -- 800 -> 200 -> 800 MHz monotonic slew with
//                                    per-step readback, verified via telemetry;
//                                    refuses to run if any L2CPU is out of reset
//
// Register access uses the _reg path, which applies alignment validation and
// issues 32-bit chunked transfers with strict ordering through the UC window.
// LIM is memory rather than registers, so it uses the block path.
//
// Run with:
//   TT_METAL_HOME=$PWD TT_METAL_SLOW_DISPATCH_MODE=1 TT_VISIBLE_DEVICES=N \
//     ./build_Release/test/tt_metal/unit_tests_legacy \
//     --gtest_filter="*X280HostAccess*"

#include "common/device_fixture.hpp"
#include "impl/context/metal_context.hpp"

#include <llrt/tt_cluster.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/blackhole_arc.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/noc_id.hpp>
#include <umd/device/types/telemetry.hpp>
#include <umd/device/arc/arc_telemetry_reader.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace tt::tt_metal;

namespace {

// L2CPU_RESET bit (4+N) selects hardware L2CPU N. The NoC0 coordinate for each N
// is fixed by tt-isa-documentation BlackholeA0/L2CPUTile/README.md:40-43, and is
// NOT the order UMD stores them in (UMD sorts by Y). Convert explicitly.
int hw_l2cpu_from_noc0(size_t x, size_t y) {
    if (x != 8) {
        return -1;
    }
    switch (y) {
        case 3: return 0;
        case 9: return 1;
        case 5: return 2;
        case 7: return 3;
        default: return -1;
    }
}

constexpr uint64_t CCACHE0_CONFIG = 0x02010000;
constexpr uint64_t CCACHE0_WAYENABLE = 0x02010008;
constexpr uint64_t L2CPU_RESET = 0x80030014;

constexpr uint64_t LIM_BASE = 0x08000000;
// Scratch target. Constraints, in order of how easy they are to get wrong:
//   - inside the 2 MiB static TLB window anchored at LIM base, else the access
//     silently drops to the dynamic WC/Relaxed path;
//   - below LIM_BASE + 0x1E0000, since backed capacity at WayEnable=0 is only
//     15 x 128 KiB even though the address window is a full 2 MiB;
//   - clear of the resident-idle image at LIM_BASE (~512 B plus stack to +0xEE0),
//     the ECC-prime marker at +0x800, and the LIM sentinel at +0x100000.
constexpr uint64_t LIM_SCRATCH = LIM_BASE + 0x40000;
constexpr uint64_t ECC_PRIME_MARKER_ADDR = LIM_BASE + 0x800;
constexpr uint64_t ECC_PRIME_MARKER_MAGIC = 0xECC17E57C0DED00DULL;
// One full 64-byte cache line: avoids any read-modify-write of a line whose ECC
// might be uninitialised, independent of the prime state.
constexpr size_t LIM_SCRATCH_WORDS = 16;

// An unclocked L2CPU answers NOC reads with all ones. That must be rejected as an
// invalid observation rather than masked into an apparently valid field value.
constexpr uint32_t ALL_ONES = 0xFFFFFFFFu;

uint32_t read_reg(tt::umd::Cluster& umd, int chip, const tt::umd::CoreCoord& core, uint64_t addr) {
    uint32_t val = 0;
    umd.read_from_device_reg(&val, chip, core, addr, sizeof(val));
    return val;
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, X280HostAccessReadOnlyGate) {
    // Borrow the cluster Metal already owns. get_driver() hands out a reference to
    // its unique_ptr; nothing here constructs a Cluster, so there is no second owner.
    const auto& cluster = MetalContext::instance().get_cluster();
    auto& umd = const_cast<tt::umd::Cluster&>(*cluster.get_driver());

    // 1. MMIO-local? Decides whether the TLB path and register semantics exist at
    //    all: RemoteChip::get_tlb_manager() throws and its _reg calls degrade to
    //    plain reads.
    auto all_chips = umd.get_target_device_ids();
    auto mmio_chips = umd.get_target_mmio_device_ids();
    printf("chips=%zu mmio=%zu\n", all_chips.size(), mmio_chips.size());
    EXPECT_EQ(all_chips.size(), mmio_chips.size()) << "non-MMIO chips present; TLB path unavailable";

    const int chip = *all_chips.begin();

    // 2. ARC route. x==11 means ARC is reachable over AXI (BAR); x==2 means the
    //    NOC-to-ARC-tile route only. get_pcie_x_coordinate() is private and
    //    is_arc_available_over_axi() protected, so derive it from the unharvested
    //    PCIe tile instead.
    const auto& sd = umd.get_soc_descriptor(chip);
    auto pcie = sd.get_cores(tt::CoreType::PCIE, tt::CoordSystem::NOC0);
    ASSERT_FALSE(pcie.empty()) << "no unharvested PCIe tile";
    printf("pcie noc0 = (%zu,%zu) -> arc_over_axi=%s\n", pcie[0].x, pcie[0].y, pcie[0].x == 11 ? "yes" : "no");

    // 3. NOC selection is thread-local mutable state and there is no public
    //    getter (get_selected_noc_id lives in the internal device/noc_access.hpp),
    //    so pin it for the duration instead of asserting it. All coordinates below
    //    are NoC0, matching the loader's hardcoded pairs. RAII restores on exit.
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    // 4. L2CPU inventory, with the UMD-index -> hardware-number conversion.
    auto l2cpu = sd.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
    auto l2cpu_tr = sd.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::TRANSLATED);
    printf("L2CPU: %zu noc0, %zu translated\n", l2cpu.size(), l2cpu_tr.size());
    ASSERT_FALSE(l2cpu.empty()) << "no unharvested L2CPU tiles";
    for (size_t i = 0; i < l2cpu.size(); i++) {
        int hw = hw_l2cpu_from_noc0(l2cpu[i].x, l2cpu[i].y);
        printf("  umd_idx %zu -> noc0 (%zu,%zu) -> hw %d [reset bit %d]\n", i, l2cpu[i].x, l2cpu[i].y, hw, hw + 4);
        EXPECT_GE(hw, 0) << "L2CPU at an undocumented NoC0 coordinate";
    }

    // 5. Telemetry. ENABLED_L2CPU is indexed by hardware number, not UMD order.
    if (auto* tel = umd.get_tt_device(chip)->get_arc_telemetry_reader()) {
        if (tel->is_entry_available(tt::umd::ENABLED_L2CPU)) {
            printf("ENABLED_L2CPU = 0x%x\n", tel->read_entry(tt::umd::ENABLED_L2CPU));
        }
        for (uint8_t tag = tt::umd::L2CPUCLK0; tag <= tt::umd::L2CPUCLK3; tag++) {
            if (tel->is_entry_available(tag)) {
                printf("L2CPUCLK%d = %u MHz\n", tag - tt::umd::L2CPUCLK0, tel->read_entry(tag));
            }
        }
    }

    // 6/7. Non-destructive state reads. Reject all-ones before interpreting.
    const auto& tile = l2cpu[0];
    uint32_t cfg = read_reg(umd, chip, tile, CCACHE0_CONFIG);
    uint32_t wayen = read_reg(umd, chip, tile, CCACHE0_WAYENABLE);
    printf("CCACHE0_CONFIG=0x%08x CCACHE0_WAYENABLE=0x%08x\n", cfg, wayen);
    EXPECT_NE(cfg, ALL_ONES) << "all-ones L3 CSR read: tile unclocked or transport broken";
    EXPECT_NE(wayen, ALL_ONES) << "all-ones WayEnable read; must not be masked into a valid value";
    if (wayen != ALL_ONES) {
        // LIM capacity is (15 - WayEnable) * 128 KiB, starting at 0x08000000.
        printf("LIM available = %u KiB\n", (15u - (wayen & 0xF)) * 128u);
    }

    auto arc = sd.get_cores(tt::CoreType::ARC, tt::CoordSystem::NOC0);
    ASSERT_FALSE(arc.empty()) << "no ARC core";
    uint32_t reset = read_reg(umd, chip, arc[0], L2CPU_RESET);
    printf("ARC noc0 = (%zu,%zu)  L2CPU_RESET = 0x%08x\n", arc[0].x, arc[0].y, reset);
    EXPECT_NE(reset, ALL_ONES) << "all-ones L2CPU_RESET read";
    for (size_t i = 0; i < l2cpu.size(); i++) {
        int hw = hw_l2cpu_from_noc0(l2cpu[i].x, l2cpu[i].y);
        if (hw >= 0) {
            printf("  hw L2CPU %d: %s\n", hw, (reset & (1u << (hw + 4))) ? "out of reset" : "held in reset");
        }
    }
}

// First write of the spike: a bounded, restoring LIM scratch write through
// Metal's borrowed cluster. LIM is memory, not registers, so the block path is
// correct here -- the Strict/UC _reg rule applies to CSRs and the PLL.
//
// Requires primed LIM ECC. On an unprimed tile, partial writes silently corrupt;
// loader.py::ensure_ecc_primed (via prime_lim.py) primes and asks for the ASIC
// reset that completes it.
TEST_F(MeshDeviceSingleCardFixture, X280HostAccessLimScratchWrite) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto& umd = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    const int chip = *umd.get_target_device_ids().begin();
    const auto& sd = umd.get_soc_descriptor(chip);
    auto l2cpu = sd.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
    ASSERT_FALSE(l2cpu.empty());
    const auto& tile = l2cpu[0];

    // Refuse to write unprimed LIM: the failure mode is silent corruption.
    std::vector<uint32_t> marker(2, 0);
    umd.read_from_device(marker.data(), chip, tile, ECC_PRIME_MARKER_ADDR, 2 * sizeof(uint32_t));
    uint64_t seen = (static_cast<uint64_t>(marker[1]) << 32) | marker[0];
    printf("ECC marker @ LIM+0x800 = 0x%016lx (%s)\n", seen, seen == ECC_PRIME_MARKER_MAGIC ? "PRIMED" : "UNPRIMED");
    ASSERT_EQ(seen, ECC_PRIME_MARKER_MAGIC) << "LIM ECC is unprimed; run prime_lim.py + board reset first";

    uint32_t wayen = read_reg(umd, chip, tile, CCACHE0_WAYENABLE) & 0xF;
    printf("WayEnable=%u -> LIM backed capacity = %u KiB\n", wayen, (15u - wayen) * 128u);
    ASSERT_LT(LIM_SCRATCH - LIM_BASE, (15u - wayen) * 128u * 1024u) << "scratch beyond backed LIM";

    // Save, write a pattern, verify, restore, verify restored.
    std::vector<uint32_t> original(LIM_SCRATCH_WORDS, 0);
    umd.read_from_device(original.data(), chip, tile, LIM_SCRATCH, LIM_SCRATCH_WORDS * sizeof(uint32_t));

    std::vector<uint32_t> pattern(LIM_SCRATCH_WORDS);
    for (size_t i = 0; i < LIM_SCRATCH_WORDS; i++) {
        pattern[i] = 0x5A5A0000u | static_cast<uint32_t>(i);
    }
    umd.write_to_device(pattern.data(), LIM_SCRATCH_WORDS * sizeof(uint32_t), chip, tile, LIM_SCRATCH);

    std::vector<uint32_t> readback(LIM_SCRATCH_WORDS, 0);
    umd.read_from_device(readback.data(), chip, tile, LIM_SCRATCH, LIM_SCRATCH_WORDS * sizeof(uint32_t));
    printf(
        "wrote 0x%08x..0x%08x, read 0x%08x..0x%08x\n",
        pattern.front(),
        pattern.back(),
        readback.front(),
        readback.back());
    EXPECT_EQ(readback, pattern) << "LIM scratch readback mismatch";

    umd.write_to_device(original.data(), LIM_SCRATCH_WORDS * sizeof(uint32_t), chip, tile, LIM_SCRATCH);
    std::vector<uint32_t> restored(LIM_SCRATCH_WORDS, 0);
    umd.read_from_device(restored.data(), chip, tile, LIM_SCRATCH, LIM_SCRATCH_WORDS * sizeof(uint32_t));
    EXPECT_EQ(restored, original) << "failed to restore original LIM contents";
    printf("scratch restored\n");
}

// ---------------------------------------------------------------------------
// PLL slew, ported from x280/host/clock.py::set_l2cpu_pll.
//
// PLL4 is per-chip and serves all four L2CPUs. Direct MMIO manipulation through
// the ARC tile is the documented bring-up method (tt-isa-documentation
// BlackholeA0/L2CPUTile/README.md:33), not a workaround.
//
// Two properties are load-bearing and both are preserved here:
//   - monotonic single-step slew: each divider moves by +/-1 per write, so no
//     intermediate value puts an out-of-range clock on the tile;
//   - phase order: raise the postdivs that must increase, then move fbdiv, then
//     lower the postdivs that must decrease, bounding the output frequency while
//     the VCO multiplier changes.
//
// Every write is followed by a readback. That is the mechanism that makes the
// slew safe rather than an optimisation: the read forces the write to complete
// before the next issues, so same-address coalescing cannot collapse the
// staircase regardless of window attributes.
//
// Route is write_to_device_reg to the ARC core (the NOC path), not
// write_to_arc_apb, which resolves to bar_write32 on Galaxy where ARC is
// reachable over AXI -- the access class implicated in the documented chassis
// hang.
namespace {

constexpr uint64_t PLL4_CNTL_1 = 0x80020504;  // refdiv:8 postdiv:8 fbdiv:16
constexpr uint64_t PLL4_CNTL_5 = 0x80020514;  // postdiv0..3, one byte each

struct PllSolution {
    uint32_t mhz;
    uint16_t fbdiv;
    uint8_t postdiv[4];
};
// Subset of clock.py's `solutions` table.
constexpr PllSolution PLL_800 = {800, 64, {1, 1, 1, 1}};
constexpr PllSolution PLL_200 = {200, 128, {15, 15, 15, 15}};

void write_reg(tt::umd::Cluster& umd, int chip, const tt::umd::CoreCoord& core, uint64_t addr, uint32_t val) {
    // Note the argument order differs between the write and read variants:
    // write takes size second, read takes it last.
    umd.write_to_device_reg(&val, sizeof(val), chip, core, addr);
    uint32_t seen = 0;
    umd.read_from_device_reg(&seen, chip, core, addr, sizeof(seen));
    ASSERT_EQ(seen, val) << "register readback mismatch at addr " << addr;
}

// Step one postdiv field to its target, +/-1 per write, whole word each time.
void slew_postdiv(
    tt::umd::Cluster& umd, int chip, const tt::umd::CoreCoord& arc, uint8_t (&cur)[4], int field, uint8_t target) {
    while (cur[field] != target) {
        cur[field] = static_cast<uint8_t>(cur[field] + (target > cur[field] ? 1 : -1));
        uint32_t word = cur[0] | (cur[1] << 8) | (cur[2] << 16) | (cur[3] << 24);
        write_reg(umd, chip, arc, PLL4_CNTL_5, word);
    }
}

void slew_fbdiv(
    tt::umd::Cluster& umd, int chip, const tt::umd::CoreCoord& arc, uint16_t& cur, uint16_t target, uint32_t low16) {
    while (cur != target) {
        cur = static_cast<uint16_t>(cur + (target > cur ? 1 : -1));
        write_reg(umd, chip, arc, PLL4_CNTL_1, (static_cast<uint32_t>(cur) << 16) | low16);
    }
}

uint32_t poll_l2cpuclk(tt::umd::TTDevice* dev, uint32_t expect_mhz, int tries = 40) {
    auto* tel = dev->get_arc_telemetry_reader();
    uint32_t last = 0;
    for (int i = 0; i < tries; i++) {
        last = tel->read_entry(tt::umd::L2CPUCLK0);
        if (last == expect_mhz) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return last;
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, X280HostAccessPllSlew) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto& umd = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    const int chip = *umd.get_target_device_ids().begin();
    const auto& sd = umd.get_soc_descriptor(chip);
    auto arc_cores = sd.get_cores(tt::CoreType::ARC, tt::CoordSystem::NOC0);
    ASSERT_FALSE(arc_cores.empty());
    const auto& arc = arc_cores[0];
    auto* dev = umd.get_tt_device(chip);

    // All four L2CPUs must be in reset: PLL4 is per-chip, so a running hart would
    // see the clock move under it.
    uint32_t reset = read_reg(umd, chip, arc, L2CPU_RESET);
    ASSERT_NE(reset, ALL_ONES) << "all-ones L2CPU_RESET read";
    ASSERT_EQ(reset & 0xF0u, 0u) << "an L2CPU is out of reset; refusing to move its clock";

    const uint32_t orig_c1 = read_reg(umd, chip, arc, PLL4_CNTL_1);
    const uint32_t orig_c5 = read_reg(umd, chip, arc, PLL4_CNTL_5);
    ASSERT_NE(orig_c1, ALL_ONES);
    ASSERT_NE(orig_c5, ALL_ONES);
    printf(
        "PLL_CNTL_1=0x%08x (refdiv=%u postdiv=%u fbdiv=%u)  PLL_CNTL_5=0x%08x\n",
        orig_c1,
        orig_c1 & 0xFF,
        (orig_c1 >> 8) & 0xFF,
        (orig_c1 >> 16) & 0xFFFF,
        orig_c5);
    printf("L2CPUCLK0 before = %u MHz\n", dev->get_arc_telemetry_reader()->read_entry(tt::umd::L2CPUCLK0));

    const uint32_t c1_low16 = orig_c1 & 0xFFFFu;  // preserve refdiv/postdiv
    uint16_t fbdiv = static_cast<uint16_t>((orig_c1 >> 16) & 0xFFFF);
    uint8_t pd[4] = {
        static_cast<uint8_t>(orig_c5 & 0xFF),
        static_cast<uint8_t>((orig_c5 >> 8) & 0xFF),
        static_cast<uint8_t>((orig_c5 >> 16) & 0xFF),
        static_cast<uint8_t>((orig_c5 >> 24) & 0xFF)};

    auto apply = [&](const PllSolution& sol) {
        for (int f = 0; f < 4; f++) {  // raise first
            if (sol.postdiv[f] > pd[f]) {
                slew_postdiv(umd, chip, arc, pd, f, sol.postdiv[f]);
            }
        }
        slew_fbdiv(umd, chip, arc, fbdiv, sol.fbdiv, c1_low16);
        for (int f = 0; f < 4; f++) {  // then lower
            if (sol.postdiv[f] < pd[f]) {
                slew_postdiv(umd, chip, arc, pd, f, sol.postdiv[f]);
            }
        }
    };

    apply(PLL_200);
    uint32_t at_200 = poll_l2cpuclk(dev, 200);
    printf("after slew to 200: L2CPUCLK0 = %u MHz\n", at_200);
    EXPECT_EQ(at_200, 200u) << "telemetry did not confirm 200 MHz";

    apply(PLL_800);
    uint32_t at_800 = poll_l2cpuclk(dev, 800);
    printf("after slew to 800: L2CPUCLK0 = %u MHz\n", at_800);
    EXPECT_EQ(at_800, 800u) << "telemetry did not confirm return to 800 MHz";

    EXPECT_EQ(read_reg(umd, chip, arc, PLL4_CNTL_1), orig_c1) << "PLL_CNTL_1 not restored";
    EXPECT_EQ(read_reg(umd, chip, arc, PLL4_CNTL_5), orig_c5) << "PLL_CNTL_5 not restored";
    printf("PLL registers restored\n");
}

// Demonstrates the ARC message path through the borrowed cluster, independently
// of LIM and of telemetry.
//
// NOP is deliberate: it is the one message that changes no state, so this proves
// the path exists without side effects. Note AICLK_GO_BUSY (0x52) is the message
// flagged as an open Galaxy chassis-safety question -- do not substitute it.
//
// Nothing else in this spike needs an ARC message: telemetry comes from
// ArcTelemetryReader and ARC-tile registers from ordinary NOC reads. This test
// exists to establish the capability, not because a caller requires it.
TEST_F(MeshDeviceSingleCardFixture, X280HostAccessArcMessage) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto& umd = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    const int chip = *umd.get_target_device_ids().begin();

    // Positive demonstration: get_clocks() round-trips through ARC and returns
    // real data, so it proves the message path works through the borrowed cluster.
    auto clocks = umd.get_clocks();
    ASSERT_FALSE(clocks.empty()) << "no clocks returned; ARC message path failed";
    for (const auto& [id, mhz] : clocks) {
        printf("ARC get_clocks: chip %d = %d MHz\n", id, mhz);
    }

    // NOP is enumerated in UMD's ArcMessageType but is NOT implemented by the CMFW
    // on this firmware bundle: the queue protocol answers with status 0xFF and UMD
    // turns that into "Message code: 17 not recognized by ARC firmware". That is a
    // well-formed rejection, which still exercises the full round trip -- record
    // which codes this firmware honours rather than trusting the enum.
    try {
        uint32_t ret3 = 0;
        uint32_t ret4 = 0;
        const int rc = umd.arc_msg(
            chip,
            static_cast<uint32_t>(tt::umd::blackhole::ArcMessageType::NOP),
            /*wait_for_done=*/true,
            /*args=*/{},
            tt::umd::timeout::ARC_MESSAGE_TIMEOUT,
            &ret3,
            &ret4);
        printf("arc_msg(NOP) accepted: rc=%d ret3=0x%x ret4=0x%x\n", rc, ret3, ret4);
    } catch (const std::exception& e) {
        printf("arc_msg(NOP) rejected by CMFW (expected on this bundle): %s\n", e.what());
    }
}

// Exercises the all-ones rejection by deliberately unclocking one L2CPU tile.
//
// Why this is safe to do and safe to undo:
//   - PLL4 lives in the ARC tile, not the L2CPU tile, so unclocking the L2CPU
//     cannot prevent us from writing the divider back. Restoration is always
//     reachable.
//   - postdiv = 0 is the encoding CMFW itself uses to disable an L2CPU clock
//     (clock_control_tt_bh_enable writes the enable flag straight into the
//     postdiv field), so this is not an invented state.
//   - Every hart is in reset, so no software sees its clock disappear.
//
// Only postdiv0 is touched, which is hardware L2CPU 0 at NoC0 (8,3). After the
// mutation only EXPECT_* is used, never ASSERT_*, so a failed check still falls
// through to the restore; a scope guard covers the exception path.
TEST_F(MeshDeviceSingleCardFixture, X280HostAccessAllOnesRejection) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto& umd = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    const int chip = *umd.get_target_device_ids().begin();
    const auto& sd = umd.get_soc_descriptor(chip);
    auto arc_cores = sd.get_cores(tt::CoreType::ARC, tt::CoordSystem::NOC0);
    auto l2cpu = sd.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
    ASSERT_FALSE(arc_cores.empty());
    ASSERT_FALSE(l2cpu.empty());
    const auto& arc = arc_cores[0];
    auto* dev = umd.get_tt_device(chip);

    // Tile for hardware L2CPU 0, which postdiv0 clocks.
    const tt::umd::CoreCoord* tile0 = nullptr;
    for (const auto& c : l2cpu) {
        if (hw_l2cpu_from_noc0(c.x, c.y) == 0) {
            tile0 = &c;
        }
    }
    ASSERT_NE(tile0, nullptr) << "no L2CPU tile maps to hardware index 0";

    uint32_t reset = read_reg(umd, chip, arc, L2CPU_RESET);
    ASSERT_NE(reset, ALL_ONES);
    ASSERT_EQ(reset & 0xF0u, 0u) << "an L2CPU is out of reset; refusing to stop its clock";

    const uint32_t orig_c5 = read_reg(umd, chip, arc, PLL4_CNTL_5);
    ASSERT_NE(orig_c5, ALL_ONES);
    ASSERT_EQ(read_reg(umd, chip, *tile0, CCACHE0_CONFIG) != ALL_ONES, true)
        << "tile already reads all ones before we touched anything";

    // Restore on every exit path, including an exception.
    struct Restore {
        tt::umd::Cluster& umd;
        int chip;
        const tt::umd::CoreCoord& arc;
        uint32_t value;
        ~Restore() { umd.write_to_device_reg(&value, sizeof(value), chip, arc, PLL4_CNTL_5); }
    } restore{umd, chip, arc, orig_c5};

    // Stop hardware L2CPU 0's clock: postdiv0 -> 0, other fields untouched.
    const uint32_t stopped = orig_c5 & 0xFFFFFF00u;
    write_reg(umd, chip, arc, PLL4_CNTL_5, stopped);
    printf("PLL_CNTL_5 0x%08x -> 0x%08x (postdiv0 = 0, hw L2CPU 0 unclocked)\n", orig_c5, stopped);
    printf("L2CPUCLK0 now reads %u MHz\n", dev->get_arc_telemetry_reader()->read_entry(tt::umd::L2CPUCLK0));

    // An unresponsive tile must never hand back a value a caller could mistake for
    // real state. Two outcomes are safe, and which one occurs depends on the UMD
    // revision, so both are accepted: a current UMD brackets each MMIO access with
    // a per-op budget and raises when one overruns it, while an older UMD returns
    // whatever the hardware produced. Measured on Blackhole that value is all ones
    // (the access completes in ~55 ms and yields 0xffffffff), so neither path
    // produces anything plausible. The test fails only on a plausible value, which
    // is the genuinely dangerous outcome.
    bool rejected = false;
    bool all_ones_seen = false;
    std::string how;
    try {
        uint32_t cfg = read_reg(umd, chip, *tile0, CCACHE0_CONFIG);
        all_ones_seen = (cfg == ALL_ONES);
        rejected = all_ones_seen;
        how = all_ones_seen ? "returned all ones" : "returned a plausible value";
        printf("unclocked read: CCACHE0_CONFIG=0x%08x\n", cfg);
    } catch (const std::exception& e) {
        rejected = true;
        how = std::string("rejected by UMD: ") + std::string(e.what()).substr(0, 90);
    }
    printf("unclocked read outcome: %s\n", how.c_str());
    EXPECT_TRUE(rejected) << "an unclocked tile returned a plausible value";

    // Why rejecting matters: masking an all-ones read yields WayEnable=15 -- "all
    // ways committed to cache, LIM gone" -- indistinguishable from a genuinely
    // committed state, which would send a caller to a needless ASIC reset.
    printf("masking all ones would give WayEnable=%u (LIM would appear to be 0 KiB)\n", ALL_ONES & 0xFu);

    // Restore and confirm recovery.
    write_reg(umd, chip, arc, PLL4_CNTL_5, orig_c5);
    uint32_t back = poll_l2cpuclk(dev, 800);
    printf("after restore: L2CPUCLK0 = %u MHz\n", back);
    EXPECT_EQ(back, 800u) << "clock did not return to 800 MHz";
    EXPECT_NE(read_reg(umd, chip, *tile0, CCACHE0_CONFIG), ALL_ONES) << "tile still unreadable after restore";
    EXPECT_EQ(read_reg(umd, chip, arc, PLL4_CNTL_5), orig_c5) << "PLL_CNTL_5 not restored";
    printf("tile reclocked, PLL_CNTL_5 restored\n");
}
