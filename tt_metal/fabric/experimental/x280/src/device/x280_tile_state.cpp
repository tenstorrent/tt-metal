// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_tile_state.hpp"
#include "x280/device/x280_active.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

#include <algorithm>
#include <sstream>

namespace x280 {

std::string describe(const HartState& hart) {
    if (!hart.observation_valid) {
        return "INVALID(" + hart.invalid_reason + ")";
    }

    std::ostringstream out;
    out << "version=" << hex(hart.state.protocol_version) << " control_run=" << hex(hart.control.run_value)
        << " entry=" << hex(hart.control.entry) << " command=" << hex(hart.control.command)
        << " run=" << hex(hart.state.run_value) << " phase=" << hex(hart.state.phase)
        << " result=" << hex(hart.state.result) << " heartbeat=" << hex(hart.state.heartbeat) << " status="
        << hex(hart.state.terminal_status)
        // How far the harness carried this hart. Decimal, because it is a
        // small enumerated postcode rather than a wire word.
        << " stage=" << hart.state.stage;
    const auto commit = commit_state(hart);
    if (commit.phase_payload_pending) {
        out << " phase_commit=uncommitted";
    }
    if (commit.terminal_payload_pending) {
        out << " terminal_commit=uncommitted";
    }
    if (commit.trap || commit.trap_payload_pending) {
        out << " trap(run=" << hex(hart.trap.run_value) << " mcause=" << hex(hart.trap.mcause)
            << " mepc=" << hex(hart.trap.mepc) << " mtval=" << hex(hart.trap.mtval)
            << " probe_state=" << hart.trap.probe_state << " valid=" << hex(hart.trap.trap_valid)
            << (commit.trap ? " committed" : " uncommitted") << ")";
    }
    return out.str();
}

std::string describe(const TileState& state) {
    std::ostringstream out;
    out << label_chip_l2cpu(state.id.chip_id, state.id.hw_index) << " at "
        << label_noc0_xy(state.id.noc_x, state.id.noc_y) << ", " << state.id.num_harts << " harts, reset bit "
        << l2cpu_reset_bit(state.id.hw_index) << "\n";
    if (!state.observation_valid) {
        out << "  observation : INVALID -- " << state.invalid_reason
            << " (unclocked tile or broken transport; not classified as a state)\n";
        return out.str();
    }
    out << "  reset       : " << (state.reset_released ? "released" : "held") << "\n";
    out << "  wayenable   : " << hex(state.raw_wayenable) << " -> LIM backed " << (state.lim_backed_bytes / 1024)
        << " KiB\n";
    out << "  lim ecc     : " << to_string(state.ecc);
    if (state.ecc == LimEccState::LimUnavailable) {
        // Printing zeroes for words that were never read would look like an
        // observation of a silent tile.
        out << " -- LIM is unbacked, so the marker, the mailboxes and the trap block were not read\n";
    } else {
        out << " (marker " << hex(state.raw_ecc_marker) << ", extent " << hex(state.raw_ecc_extent);
        if (state.ecc == LimEccState::Primed) {
            if (state.ecc_primed_bytes.has_value()) {
                out << " -> " << (*state.ecc_primed_bytes / 1024) << " KiB primed";
            } else {
                out << " -> unrecorded, " << (kLimEccUnrecordedPrimeBytes / 1024) << " KiB assumed";
            }
        }
        out << ")\n";
        for (const auto& hart : state.harts) {
            out << "  hart " << hart.hart_id << "      : " << describe(hart) << "\n";
        }
        out << "  resident idle: ";
        if (state.resident_idle_live) {
            out << "live\n";
        } else if (!state.reset_released && std::all_of(state.harts.begin(), state.harts.end(), hart_resident_idle)) {
            out << "not live -- the mailboxes match, but the tile is held in reset, so they are stale LIM contents "
                   "from an earlier boot\n";
        } else {
            out << "not live\n";
        }
    }
    out << "  pll         : ";
    if (state.pll_mhz.has_value()) {
        out << *state.pll_mhz << " MHz (ARC telemetry)\n";
    } else {
        out << "not reported\n";
    }
    return out.str();
}

}  // namespace x280
