// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared scaffolding for the explicit ARC and PLL transport-characterization
// binaries: driver borrowing, coordinate translation, checked register
// primitives, and sweep bookkeeping.
//
// Register access goes through Metal's own tt::Cluster wrapper. read_reg /
// write_reg reach UMD's alignment-validated, 32-bit, strictly ordered UC path.
// ARC messaging and telemetry stay on the driver because the wrapper does not
// expose them.

#pragma once

#include "impl/context/metal_context.hpp"

#include "x280/device/x280_state.hpp"
#include "x280_hw_addrs.hpp"

#include <gtest/gtest.h>

#include <llrt/tt_cluster.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace x280::test {

/** The process's tt::Cluster, owned by MetalContext and valid while it lives. */
inline const tt::Cluster& cluster() { return tt::tt_metal::MetalContext::instance().get_cluster(); }

/** Whether Metal's watcher is enabled; its host NoC sanitizer admits no L2CPU coordinate. */
inline bool watcher_enabled() { return tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled(); }

// Only ARC messaging and telemetry need this. operator* on a const unique_ptr
// yields a non-const referent, so borrowing the driver needs no cast; nothing here
// constructs a Cluster, so there is no second owner.
inline tt::umd::Cluster& driver() { return *cluster().get_driver(); }

inline std::string chip_label(tt::ChipId chip) { return "chip " + std::to_string(chip); }

// Hardware L2CPUs currently out of reset. All four bits are read, not just the
// unharvested tiles: PLL4 is per-chip and physically clocks every L2CPU, so any
// released core makes a clock move unsafe regardless of harvesting.
inline std::vector<int> out_of_reset(uint32_t reset) {
    std::vector<int> live;
    for (int hw = 0; hw < x280::kNumL2cpuTiles; hw++) {
        if (reset & (1u << x280::l2cpu_reset_bit(hw))) {
            live.push_back(hw);
        }
    }
    return live;
}

inline std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

// Sweep bookkeeping. A sweep that covered nothing must not report a silent pass,
// but the right non-pass depends on why nothing was covered, so skips carry that
// distinction: a booted fleet is a normal state the case does not apply to, while
// an unprimed one needs an operator action and says which.
struct Coverage {
    const char* what;
    const char* unit;
    size_t total = 0;
    size_t covered = 0;
    std::vector<std::string> skipped;
    // The tile is in a state the case simply does not apply to.
    void skip(const std::string& why) { skipped.push_back(why); }

    // GTEST_SKIP/ADD_FAILURE record against the running test even from a helper;
    // only the `return` they expand to is local, and this is the last call anyway.
    void finish() const {
        printf("\n%s covered %zu/%zu %s\n", what, covered, total, unit);
        for (const auto& s : skipped) {
            printf("  skipped: %s\n", s.c_str());
        }
        if (covered != 0) {
            return;
        }
        GTEST_SKIP() << "no eligible " << unit << ": " << skipped.size() << '/' << total
                     << " skipped, none of them fixable from here";
    }
};

inline uint32_t read_reg(const tt::Cluster& c, tt_cxy_pair core, uint64_t addr) {
    uint32_t val = 0;
    c.read_reg(&val, core, addr);
    return val;
}

// Returns false on a readback mismatch so the caller can stop and restore.
//
// A fatal assertion cannot do that job here. ASSERT_EQ returns from this helper
// alone, so a stepping loop would carry on past a write that did not land -- and
// HasFatalFailure(), which detects that, is sticky for the whole TEST, so under a
// sweep the first bad chip would silently disable the slew for every chip after
// it. A returned bool is scoped to the one step that failed.
[[nodiscard]] inline bool write_reg(const tt::Cluster& c, tt_cxy_pair core, uint64_t addr, uint32_t val) {
    c.write_reg(&val, core, addr);
    uint32_t seen = 0;
    c.read_reg(&seen, core, addr);
    if (seen != val) {
        ADD_FAILURE() << "register readback mismatch at 0x" << std::hex << addr << ": wrote 0x" << val << ", read 0x"
                      << seen;
        return false;
    }
    return true;
}

// The precondition the clock-moving cases share: every hart on the chip must
// already be in reset, since PLL4 is per-chip and a running hart would see its
// frequency move. Returns the skip reason, empty when the chip is quiescent.
//
// A precondition and never an action. A tile found running cannot be quieted --
// release is one-way, so a tile put back into reset needs a whole-ASIC reset to
// run again -- and a chip left in resident idle on purpose is a normal state
// rather than a failure. The chip is skipped, never forced.
inline std::string quiescent_or_skip(const tt::Cluster& c, tt_cxy_pair arc) {
    const uint32_t reset = read_reg(c, arc, x280::kL2cpuResetReg);
    if (x280::is_all_ones(reset)) {
        return "all-ones L2CPU_RESET read";
    }
    const std::vector<int> live = out_of_reset(reset);
    if (!live.empty()) {
        return "hw L2CPU " + join(live) + " out of reset";
    }
    return {};
}

// The ARC tile in TRANSLATED coordinates, which is what tt::Cluster resolves
// against. Returns false when the chip has no unharvested ARC core.
inline bool arc_of(const metal_SocDescriptor& sd, tt::ChipId chip, tt_cxy_pair& out) {
    auto arc = sd.get_cores(tt::CoreType::ARC, tt::CoordSystem::TRANSLATED);
    if (arc.empty()) {
        return false;
    }
    out = tt_cxy_pair(chip, arc[0].x, arc[0].y);
    return true;
}

}  // namespace x280::test
