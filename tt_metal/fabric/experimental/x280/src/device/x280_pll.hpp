// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// PLL4 programming sequence.
//
// PLL4 serves all four L2CPU tiles from one VCO, so a frequency change is
// chip-global even though the post-dividers are per-output. Direct MMIO
// programming through the ARC tile is the bring-up method.

#pragma once

#include "x280_metal_transport.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace x280 {

// Prefix-free diagnostic for a frequency with no PLL solution.
std::string unsupported_pll_frequency_diagnostic(uint32_t mhz);
bool is_supported_pll_frequency(uint32_t mhz);

struct PllClockObservation {
    int hw_index = -1;
    std::optional<uint32_t> mhz;
};

struct PllEstablishResult {
    bool confirmed = false;
    bool programmed = false;
    std::vector<PllClockObservation> observed;
    std::string diagnostic;
};

// Slews PLL4 to `mhz`.
//
// Two properties are load-bearing:
//
//   - every divider moves by +/-1 per register write, so no intermediate value
//     puts an out-of-range clock on a tile;
//   - the phase order is raise-the-postdivs-that-must-increase, slew fbdiv,
//     then lower-the-postdivs-that-must-decrease, which bounds the output
//     frequency while the VCO multiplier moves.
//
// Each write is followed by a readback. That is what serialises the staircase:
// without it, same-address stores can coalesce and the slew is no longer
// monotonic, which is the one failure the stepping exists to prevent.
//
// Deliberately carries no "all tiles must be in reset" guard. Programming the
// PLL under a running hart is the normal operation — it is how a caller sets
// the frequency it wants to run at, on the resident-idle fast path as much as
// after a cold boot — and the monotonic slew is what makes it safe. A caller
// that needs the stricter condition checks reset itself.
//
// Validates all arguments before I/O, programs only when the current telemetry
// does not already agree, then confirms all tiles in fair polling sweeps.
PllEstablishResult establish_l2cpu_pll(
    X280MetalTransport& transport,
    int chip_id,
    const tt::umd::CoreCoord& arc,
    const std::vector<int>& hw_indices,
    uint32_t mhz,
    std::chrono::milliseconds confirm_timeout);

}  // namespace x280
