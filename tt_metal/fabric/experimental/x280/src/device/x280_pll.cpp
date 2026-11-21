// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_pll.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"
#include "x280/device/x280_deadline.hpp"
#include "x280/device/x280_state.hpp"

#include <tt-umd/types/telemetry.hpp>

#include <sstream>
#include <unordered_set>

namespace x280 {

namespace {

constexpr std::chrono::milliseconds kClockPollInterval{100};

struct PllSolution {
    uint32_t mhz;
    uint16_t fbdiv;
    uint8_t postdiv[4];
};

// Registered PLL4 solutions. 12.5 MHz reference: mhz = 12.5 * fbdiv / postdiv.
constexpr PllSolution kPllSolutions[] = {
    {15, 120, {99, 99, 99, 99}},
    {200, 128, {15, 15, 15, 15}},
    {800, 64, {1, 1, 1, 1}},
    {1000, 80, {1, 1, 1, 1}},
    {1750, 140, {1, 1, 1, 1}},
};

}  // namespace

std::string unsupported_pll_frequency_diagnostic(uint32_t mhz) {
    std::ostringstream diagnostic;
    diagnostic << "no PLL solution for " << mhz << " MHz. Supported:";
    for (const auto& solution : kPllSolutions) {
        diagnostic << ' ' << solution.mhz;
    }
    return diagnostic.str();
}

namespace {

const PllSolution* find_solution(uint32_t mhz) {
    for (const auto& solution : kPllSolutions) {
        if (solution.mhz == mhz) {
            return &solution;
        }
    }
    return nullptr;
}

struct PllRegisters {
    uint32_t cntl1_low16;  // refdiv and the CNTL_1 postdiv, preserved verbatim
    uint16_t fbdiv;
    uint8_t postdiv[4];
};

PllRegisters read_pll(const X280MetalTransport& transport, int chip_id, const tt::umd::CoreCoord& arc) {
    const uint32_t cntl1 = transport.read_reg32_checked(chip_id, arc, kPll4Cntl1, "PLL4 CNTL_1");
    const uint32_t cntl5 = transport.read_reg32_checked(chip_id, arc, kPll4Cntl5, "PLL4 CNTL_5");

    PllRegisters regs;
    regs.cntl1_low16 = cntl1 & 0xFFFFu;
    regs.fbdiv = static_cast<uint16_t>((cntl1 >> 16) & 0xFFFFu);
    for (int field = 0; field < 4; field++) {
        regs.postdiv[field] = static_cast<uint8_t>((cntl5 >> (8 * field)) & 0xFFu);
    }
    return regs;
}

uint32_t pack_cntl5(const uint8_t (&postdiv)[4]) {
    return static_cast<uint32_t>(postdiv[0]) | (static_cast<uint32_t>(postdiv[1]) << 8) |
           (static_cast<uint32_t>(postdiv[2]) << 16) | (static_cast<uint32_t>(postdiv[3]) << 24);
}

// One divider field to its target, one step per write. The whole 32-bit word
// goes out each time, so the other three fields are preserved by carrying the
// current values along.
void step_postdiv(
    X280MetalTransport& transport,
    int chip_id,
    const tt::umd::CoreCoord& arc,
    PllRegisters& regs,
    int field,
    uint8_t target) {
    while (regs.postdiv[field] != target) {
        regs.postdiv[field] = static_cast<uint8_t>(regs.postdiv[field] + (target > regs.postdiv[field] ? 1 : -1));
        transport.write_reg32_verified(chip_id, arc, kPll4Cntl5, pack_cntl5(regs.postdiv));
    }
}

void step_fbdiv(
    X280MetalTransport& transport, int chip_id, const tt::umd::CoreCoord& arc, PllRegisters& regs, uint16_t target) {
    while (regs.fbdiv != target) {
        regs.fbdiv = static_cast<uint16_t>(regs.fbdiv + (target > regs.fbdiv ? 1 : -1));
        transport.write_reg32_verified(
            chip_id, arc, kPll4Cntl1, (static_cast<uint32_t>(regs.fbdiv) << 16) | regs.cntl1_low16);
    }
}

// Throws if `mhz` has no solution or if any readback disagrees.
void program_l2cpu_pll(X280MetalTransport& transport, int chip_id, const tt::umd::CoreCoord& arc, uint32_t mhz) {
    const PllSolution* solution = find_solution(mhz);
    if (solution == nullptr) {
        fail(unsupported_pll_frequency_diagnostic(mhz));
    }

    PllRegisters regs = read_pll(transport, chip_id, arc);

    // Raise first: a larger post-divider lowers the output, bounding it while
    // the VCO multiplier moves.
    for (int field = 0; field < 4; field++) {
        if (solution->postdiv[field] > regs.postdiv[field]) {
            step_postdiv(transport, chip_id, arc, regs, field, solution->postdiv[field]);
        }
    }

    step_fbdiv(transport, chip_id, arc, regs, solution->fbdiv);

    // Lower last, so the output only reaches its target once the VCO is there.
    for (int field = 0; field < 4; field++) {
        if (solution->postdiv[field] < regs.postdiv[field]) {
            step_postdiv(transport, chip_id, arc, regs, field, solution->postdiv[field]);
        }
    }
}

}  // namespace

bool is_supported_pll_frequency(uint32_t mhz) { return find_solution(mhz) != nullptr; }

PllEstablishResult establish_l2cpu_pll(
    X280MetalTransport& transport,
    int chip_id,
    const tt::umd::CoreCoord& arc,
    const std::vector<int>& hw_indices,
    uint32_t mhz,
    std::chrono::milliseconds confirm_timeout) {
    PllEstablishResult result;

    if (!is_supported_pll_frequency(mhz)) {
        result.diagnostic = unsupported_pll_frequency_diagnostic(mhz);
        return result;
    }
    if (confirm_timeout.count() <= 0) {
        result.diagnostic =
            "clock_confirm_timeout is " + std::to_string(confirm_timeout.count()) + " ms; it must be positive";
        return result;
    }
    if (hw_indices.empty()) {
        result.diagnostic = "cannot establish PLL4 without an enabled L2CPU telemetry endpoint";
        return result;
    }

    std::unordered_set<int> unique_indices;
    for (int hw_index : hw_indices) {
        if (hw_index < 0 || hw_index >= kNumL2cpuTiles) {
            result.diagnostic = "L2CPU telemetry index " + std::to_string(hw_index) +
                                " is outside the hardware range 0.." + std::to_string(kNumL2cpuTiles - 1);
            return result;
        }
        if (!unique_indices.insert(hw_index).second) {
            result.diagnostic = "duplicate L2CPU telemetry index " + std::to_string(hw_index);
            return result;
        }
    }

    auto sample_all = [&] {
        bool all_agree = true;
        result.observed.clear();
        result.observed.reserve(hw_indices.size());
        for (int hw_index : hw_indices) {
            PllClockObservation observation;
            observation.hw_index = hw_index;
            observation.mhz = transport.telemetry(chip_id, static_cast<uint8_t>(tt::umd::L2CPUCLK0 + hw_index));
            all_agree = all_agree && observation.mhz.has_value() && *observation.mhz == mhz;
            result.observed.push_back(observation);
        }
        return all_agree;
    };

    if (sample_all()) {
        result.confirmed = true;
        result.diagnostic = "PLL4 already reports " + std::to_string(mhz) + " MHz on every enabled L2CPU";
        return result;
    }

    program_l2cpu_pll(transport, chip_id, arc, mhz);
    result.programmed = true;

    const Deadline deadline{confirm_timeout};
    const auto poll = poll_until(deadline, kClockPollInterval, sample_all);
    result.confirmed = poll.satisfied;
    if (result.confirmed) {
        result.diagnostic = "PLL4 at " + std::to_string(mhz) + " MHz, confirmed on every enabled L2CPU after " +
                            std::to_string(poll.polls) + " telemetry sweeps";
        return result;
    }

    std::ostringstream diagnostic;
    diagnostic << "PLL4 did not report " << mhz << " MHz on every enabled L2CPU within " << confirm_timeout.count()
               << " ms (" << poll.polls << " fair telemetry sweeps; the bound may include one final telemetry sweep):";
    for (const auto& observation : result.observed) {
        diagnostic << " L2CPU " << observation.hw_index << '=';
        if (observation.mhz.has_value()) {
            diagnostic << *observation.mhz << " MHz;";
        } else {
            diagnostic << "unavailable;";
        }
    }
    result.diagnostic = diagnostic.str();
    return result;
}

}  // namespace x280
