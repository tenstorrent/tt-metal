// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The one case that moves PLL4 on real silicon.
//
//   X280PllSlew -- monotonic slew away from whatever frequency it finds and
//                  back, per-step readback, verified via telemetry
//
// Separate from x280_device_tests, and opted into by environment, because a
// PLL4 move can wedge the chip's CMFW: every open() of the chardev asks CMFW
// for a power state (tt-kmd chardev.c, tenstorrent_set_aggregated_power_state),
// and x280_hw_addrs.hpp records that no such message may be issued while a slew
// is in flight. A wedged chip answers PCIe with a continuous NonFatal error
// stream, which on a host running other device consumers burns enough CPU in
// AER recovery to take the machine down. The slew restores what it moves, but
// restoration does not undo a CMFW that stopped accepting power messages
// mid-slew -- that needs `tt-smi -r <device>`.
//
// No case here unclocks hardware L2CPU 0 (postdiv0 = 0): that cannot be undone.
// PLL4 lives in the ARC tile, but on a Blackhole Galaxy the host NOC read of
// the unclocked tile wedges the path to the ARC as well, so every subsequent
// register access -- including a restore -- times out around 220 ms and the
// chip stays unclocked. Recovering it needs a chassis reset, and the PCIe error
// stream in the meantime is enough to take the host down.
//
// Run with:
//   X280_ALLOW_PLL_SLEW=1 TT_METAL_RUNTIME_ROOT=<tt-metal> TT_VISIBLE_DEVICES=N
//   build/test/tt_metal/x280/x280_pll_test
// Omit TT_VISIBLE_DEVICES to sweep the whole mesh. Prefer a host where nothing
// else holds /dev/tenstorrent/*; a telemetry daemon counts.

#include "x280_device_test_support.hpp"

#include <gtest/gtest.h>

#include <tt-umd/arc/firmware_telemetry_reader.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/noc_id.hpp>
#include <tt-umd/types/telemetry.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace x280::test;

namespace {

bool opted_in() {
    const char* opt = std::getenv("X280_ALLOW_PLL_SLEW");
    return opt != nullptr && std::string(opt) == "1";
}

// This moves PLL4. Refusing by default is what keeps a routine device-test run
// from wedging a chip.
bool skip_unless_allowed(std::string& why) {
    if (!opted_in()) {
        why = "moves PLL4 on real silicon. Set X280_ALLOW_PLL_SLEW=1 to run.";
        return true;
    }
    if (watcher_enabled()) {
        why = "watcher rejects host access to ARC and L2CPU cores";
        return true;
    }
    return false;
}

struct PllSolution {
    uint32_t mhz;
    uint16_t fbdiv;
    std::array<uint8_t, 4> postdiv;
};
// Subset of the runtime's registered PLL4 solutions.
constexpr PllSolution PLL_800 = {800, 64, {1, 1, 1, 1}};
constexpr PllSolution PLL_200 = {200, 128, {15, 15, 15, 15}};
constexpr PllSolution PLL_1000 = {1000, 80, {1, 1, 1, 1}};
constexpr PllSolution PLL_1750 = {1750, 140, {1, 1, 1, 1}};

struct PllDividers {
    uint16_t fbdiv = 0;
    std::array<uint8_t, 4> postdiv{};
};

// Step one postdiv field to its target, +/-1 per write, whole word each time.
[[nodiscard]] bool slew_postdiv(
    const tt::Cluster& c, tt_cxy_pair arc, std::array<uint8_t, 4>& cur, int field, uint8_t target) {
    while (cur.at(field) != target) {
        cur.at(field) = static_cast<uint8_t>(cur.at(field) + (target > cur.at(field) ? 1 : -1));
        uint32_t word = cur[0] | (static_cast<uint32_t>(cur[1]) << 8) | (static_cast<uint32_t>(cur[2]) << 16) |
                        (static_cast<uint32_t>(cur[3]) << 24);
        // A step that did not land makes every later step a jump from an
        // unknown value, which is what the staircase exists to avoid.
        if (!write_reg(c, arc, x280::kPll4Cntl5, word)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool slew_fbdiv(const tt::Cluster& c, tt_cxy_pair arc, uint16_t& cur, uint16_t target, uint32_t low16) {
    while (cur != target) {
        cur = static_cast<uint16_t>(cur + (target > cur ? 1 : -1));
        if (!write_reg(c, arc, x280::kPll4Cntl1, (static_cast<uint32_t>(cur) << 16) | low16)) {
            return false;
        }
    }
    return true;
}

PllDividers unpack_dividers(uint32_t cntl1, uint32_t cntl5) {
    return {
        static_cast<uint16_t>((cntl1 >> 16) & 0xFFFF),
        {static_cast<uint8_t>(cntl5 & 0xFF),
         static_cast<uint8_t>((cntl5 >> 8) & 0xFF),
         static_cast<uint8_t>((cntl5 >> 16) & 0xFF),
         static_cast<uint8_t>((cntl5 >> 24) & 0xFF)}};
}

[[nodiscard]] bool slew_dividers(
    const tt::Cluster& c, tt_cxy_pair arc, PllDividers& current, const PllDividers& target, uint32_t cntl1_low16) {
    for (int field = 0; field < 4; field++) {
        if (target.postdiv.at(field) > current.postdiv.at(field) &&
            !slew_postdiv(c, arc, current.postdiv, field, target.postdiv.at(field))) {
            return false;
        }
    }
    if (!slew_fbdiv(c, arc, current.fbdiv, target.fbdiv, cntl1_low16)) {
        return false;
    }
    for (int field = 0; field < 4; field++) {
        if (target.postdiv.at(field) < current.postdiv.at(field) &&
            !slew_postdiv(c, arc, current.postdiv, field, target.postdiv.at(field))) {
            return false;
        }
    }
    return true;
}

const PllSolution* classify_solution(const PllDividers& dividers) {
    constexpr std::array<const PllSolution*, 4> known = {&PLL_200, &PLL_800, &PLL_1000, &PLL_1750};
    for (const PllSolution* solution : known) {
        if (solution->fbdiv == dividers.fbdiv && solution->postdiv == dividers.postdiv) {
            return solution;
        }
    }
    return nullptr;
}

// Exception cleanup re-reads the current fields and follows the same safe
// staircase as the normal path. Direct whole-register restoration can turn an
// interrupted slew into an unbounded divider jump.
class ScopedPllRestore {
public:
    ScopedPllRestore(const tt::Cluster& cluster, tt_cxy_pair arc, uint32_t original_cntl1, uint32_t original_cntl5) :
        cluster_(cluster),
        arc_(arc),
        original_cntl1_(original_cntl1),
        original_cntl5_(original_cntl5),
        original_(unpack_dividers(original_cntl1, original_cntl5)) {}

    ~ScopedPllRestore() {
        if (restored_) {
            return;
        }
        try {
            if (!restore()) {
                ADD_FAILURE() << "failed to restore PLL dividers through the safe staircase";
            }
        } catch (const std::exception& error) {
            ADD_FAILURE() << "failed to restore PLL dividers: " << error.what();
        }
    }

    [[nodiscard]] bool restore() {
        const uint32_t current_cntl1 = read_reg(cluster_, arc_, x280::kPll4Cntl1);
        const uint32_t current_cntl5 = read_reg(cluster_, arc_, x280::kPll4Cntl5);
        if (x280::is_all_ones(current_cntl1) || x280::is_all_ones(current_cntl5)) {
            ADD_FAILURE() << "cannot restore PLL from an all-ones divider read";
            return false;
        }

        PllDividers current = unpack_dividers(current_cntl1, current_cntl5);
        if (!slew_dividers(cluster_, arc_, current, original_, original_cntl1_ & 0xFFFFu)) {
            return false;
        }
        restored_ = read_reg(cluster_, arc_, x280::kPll4Cntl1) == original_cntl1_ &&
                    read_reg(cluster_, arc_, x280::kPll4Cntl5) == original_cntl5_;
        if (!restored_) {
            ADD_FAILURE() << "PLL divider readback did not match the saved starting state";
        }
        return restored_;
    }

private:
    const tt::Cluster& cluster_;
    tt_cxy_pair arc_;
    uint32_t original_cntl1_;
    uint32_t original_cntl5_;
    PllDividers original_;
    bool restored_ = false;
};

uint32_t poll_l2cpuclk(tt::umd::TTDevice* dev, uint32_t expect_mhz, int tries = 40) {
    auto* tel = dev->get_firmware_telemetry_reader();
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

// Independent direct-wrapper PLL slew.
//
// PLL4 is per-chip and serves all four L2CPUs. The test deliberately implements
// the staircase over tt::Cluster instead of calling the production transport.
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
// staircase regardless of window attributes. A step that does not land stops the
// slew for that chip and falls through to the restore.
//
// Route is tt::Cluster::write_reg to the ARC core (the NOC path), not
// write_to_arc_apb, which resolves to bar_write32 on Galaxy where ARC is
// reachable over AXI -- an access class that can hang the chassis.
TEST(X280Pll, X280PllSlew) {
    if (std::string why; skip_unless_allowed(why)) {
        GTEST_SKIP() << why;
    }
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    Coverage cov{"PllSlew", "chips"};
    for (const tt::ChipId chip : cluster().all_chip_ids()) {
        SCOPED_TRACE(chip_label(chip));
        cov.total++;
        const std::string label = chip_label(chip);
        const auto& sd = cluster().get_soc_desc(chip);

        tt_cxy_pair arc{};
        if (!arc_of(sd, chip, arc)) {
            cov.skip(label + ": no ARC core");
            continue;
        }
        auto* dev = driver().get_tt_device(chip);

        if (const std::string why = quiescent_or_skip(cluster(), arc); !why.empty()) {
            printf("%s: %s\n", label.c_str(), why.c_str());
            cov.skip(label + ": " + why);
            continue;
        }

        const uint32_t orig_c1 = read_reg(cluster(), arc, x280::kPll4Cntl1);
        const uint32_t orig_c5 = read_reg(cluster(), arc, x280::kPll4Cntl5);
        if (x280::is_all_ones(orig_c1) || x280::is_all_ones(orig_c5)) {
            printf("%s: all-ones PLL register read\n", label.c_str());
            cov.skip(label + ": all-ones PLL register read");
            continue;
        }
        printf(
            "\n--- %s ---\nPLL_CNTL_1=0x%08x (refdiv=%u postdiv=%u fbdiv=%u)  PLL_CNTL_5=0x%08x\n",
            label.c_str(),
            orig_c1,
            orig_c1 & 0xFF,
            (orig_c1 >> 8) & 0xFF,
            (orig_c1 >> 16) & 0xFFFF,
            orig_c5);
        printf("L2CPUCLK0 before = %u MHz\n", dev->get_firmware_telemetry_reader()->read_entry(tt::umd::L2CPUCLK0));

        ScopedPllRestore restore(cluster(), arc, orig_c1, orig_c5);

        const uint32_t c1_low16 = orig_c1 & 0xFFFFu;  // preserve refdiv/postdiv
        PllDividers current = unpack_dividers(orig_c1, orig_c5);

        auto apply = [&](const PllSolution& sol) {
            const PllDividers target{sol.fbdiv, sol.postdiv};
            return slew_dividers(cluster(), arc, current, target, c1_low16);
        };

        const PllSolution* start = classify_solution(current);
        if (start == nullptr) {
            cov.skip(label + ": starting PLL dividers are not a registered solution");
            continue;
        }
        const uint32_t start_mhz = dev->get_firmware_telemetry_reader()->read_entry(tt::umd::L2CPUCLK0);
        if (start_mhz != start->mhz) {
            ADD_FAILURE() << "divider classification says " << start->mhz << " MHz but telemetry reports " << start_mhz
                          << " MHz";
            cov.skip(label + ": divider and telemetry disagree");
            continue;
        }
        const PllSolution& away = start->mhz == PLL_200.mhz ? PLL_800 : PLL_200;

        if (apply(away)) {
            uint32_t at_away = poll_l2cpuclk(dev, away.mhz);
            printf("after slew to %u: L2CPUCLK0 = %u MHz\n", away.mhz, at_away);
            EXPECT_EQ(at_away, away.mhz) << "telemetry did not confirm " << away.mhz << " MHz";
        }

        // Attempted whether or not the outbound slew completed. restore() reads
        // the actual intermediate fields before stepping back.
        EXPECT_TRUE(restore.restore()) << "failed to slew back to " << start->mhz << " MHz";
        uint32_t at_start = poll_l2cpuclk(dev, start->mhz);
        printf("after slew back to %u: L2CPUCLK0 = %u MHz\n", start->mhz, at_start);
        EXPECT_EQ(at_start, start->mhz) << "telemetry did not confirm return to " << start->mhz << " MHz";

        EXPECT_EQ(read_reg(cluster(), arc, x280::kPll4Cntl1), orig_c1) << "PLL_CNTL_1 not restored";
        EXPECT_EQ(read_reg(cluster(), arc, x280::kPll4Cntl5), orig_c5) << "PLL_CNTL_5 not restored";
        cov.covered++;
    }
    cov.finish();
}
