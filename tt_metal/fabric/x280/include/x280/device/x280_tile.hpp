// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One L2CPU tile: its LIM, its resident idle, and the images it runs.
//
// X280Runtime owns one stable object for every enabled physical tile. Looking
// up the same identity returns the same object; callers borrow it for no longer
// than the runtime lifetime.
//
// Every hazard here is a write a caller did not expect, so each call states its
// effect and const is the enforcement: a const method cannot reach a write.
//
//   read-only  id, inspect, reserved_lim, available_lim, read_noc_window,
//              healthy, last_failure
//   writes     boot preparation
//   one-way    boot with allow_reset_release on a tile held in reset
//
// Active images are submitted with X280Runtime::enqueue, which accepts only a
// healthy runtime-owned tile from that runtime, and restored with
// X280Runtime::cleanup. The health latch below is what a failed restoration
// leaves behind.

#pragma once

#include "x280/device/x280_active.hpp"
#include "x280/device/x280_noc_window.hpp"
#include "x280/device/x280_state.hpp"
#include "x280/device/x280_tile_state.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace x280 {

namespace detail {
class ChipImpl;
class RuntimeImpl;
class TileImpl;
class TileTestAccess;
}  // namespace detail

enum class BootOutcome {
    // Resident idle was cold-booted and is live.
    Booted,
    // Resident idle was already live; reset and the image were not touched.
    AlreadyRunning,
    // LIM has never been primed in this chip-reset lifetime, so a narrow write
    // would silently corrupt an uninitialised ECC line. Runtime opening owns
    // the chip-wide, one-way bootstrap.
    PrimeRequired,
    // WayEnable leaves no backed LIM. Left untouched, since expanding LIM
    // requires an ASIC reset.
    ResetRequiredNoLim,
    // The tile is out of reset but is not running the expected resident idle.
    // Not recoverable in software: harts cannot re-enter reset.
    ResetRequiredNoResidentIdle,
    // Dry run only: the image was validated, loaded and read back, and the
    // reset vectors were written and verified, but reset was left asserted.
    PreparedResetNotReleased,
    // A raw read came back all ones, or the transport failed. Never classified
    // as a state.
    InvalidObservation,
    // Anything else, with the reason in the diagnostic.
    Failed,
};

const char* to_string(BootOutcome outcome);

// True for the outcomes an operator resolves with an ASIC reset
// (tt-smi -r <device>).
bool reset_required(BootOutcome outcome);

struct BootRequest {
    // Path to the resident-idle image, supplied by the caller. The component
    // hard-codes no build-tree location.
    std::string idle_image_path;

    // Bounds the wait for the heartbeat and IDLE phase after a cold boot.
    std::chrono::milliseconds boot_timeout{3000};

    // Bounds ARC telemetry confirmation of the safe reset-release clock. Kept
    // separate from boot_timeout because the two wait on different hardware and
    // tune independently.
    std::chrono::milliseconds clock_confirm_timeout{4000};

    // Releasing L2CPU reset is one-way for the chip-reset lifetime, so a cold
    // boot is opt-in. With this false the tile is inspected and classified but
    // reset is never released.
    bool allow_reset_release = false;

    // Do everything a cold boot does up to, but not including, releasing reset:
    // validate the image, load it, read it back, write the reset vectors and
    // verify them, and clear the mailboxes.
    //
    // Exists because the step after it cannot be undone. The reset-vector
    // register file is the one address the component touches that lies outside
    // the static TLB window at LIM base, so this is how its reachability gets
    // established without betting a tile on it.
    bool dry_run = false;
};

struct BootResult {
    BootOutcome outcome = BootOutcome::Failed;

    // Actionable, and carries the last observed state for any wait that timed
    // out.
    std::string diagnostic;

    TileState last_state;

    bool ready() const { return outcome == BootOutcome::Booted || outcome == BootOutcome::AlreadyRunning; }
};

struct NocWindowResult {
    bool ok = false;
    std::string diagnostic;

    // What the entry read back as. Compare against the request to see what the
    // hardware actually latched.
    /** The entry's coordinate as the hardware holds it, not a resolver output. */
    x280_noc_coord_t observed_coord = {0, 0};
    NocWindow2M observed;
    // The four words as read, set whenever the entry was read even if it did
    // not decode; the fields decode() omits live only here.
    NocWindowWords words;
};

/**
 * Whether `window` was read and read back as a cleared entry: all four words
 * zero. Every field counts, not just the decoded ones, so a partial clear that
 * left a flag set is not a clear; a read that failed (`ok` false, words never
 * filled) is not one either.
 */
inline bool window_reads_cleared(const NocWindowResult& window) {
    return window.ok && window.words == NocWindowWords{};
}

class X280Tile {
public:
    X280Tile(const X280Tile&) = delete;
    X280Tile& operator=(const X280Tile&) = delete;
    X280Tile(X280Tile&&) = delete;
    X280Tile& operator=(X280Tile&&) = delete;
    ~X280Tile();

    const TileIdentity& id() const;

    // Pure inspection: re-reads the tile rather than returning a cached
    // snapshot, so a caller can verify that boot produced the state it
    // asked for. Never touches the clock, so an unclocked tile reports an
    // invalid observation.
    TileState inspect() const;

    // Loads resident idle when needed and optionally releases reset. A cold
    // boot establishes kSafePllMhz itself, in the same locked section that
    // proved every L2CPU is held in reset — the only window where moving the
    // chip-global clock disturbs nothing, and it closes as reset is released.
    // The runtime owner may establish the operating clock with
    // X280Runtime::set_pll once boot returns. See the PLL4 ownership rules in
    // x280_runtime.hpp, which apply here too.
    BootResult boot(const BootRequest& request);

    // Addresses the component owns. A caller placing a socket config buffer or
    // FIFO here corrupts a running image, and nothing detects it afterwards.
    LimRange reserved_lim() const;

    // What is left for a case, bounded by the backing WayEnable leaves in place.
    // Empty when the tile could not be observed.
    LimRange available_lim() const;

    // Bounded block I/O in available_lim(). Reads use Metal's 4-byte block
    // granularity. Writes require whole 64-byte cache lines so uninitialised
    // high-LIM ECC never takes a partial-line merge path.
    std::vector<uint8_t> read_available_lim(uint64_t address, size_t size) const;
    void write_available_lim(uint64_t address, std::span<const uint8_t> data);

    // Reads back one window without writing. Any index, since observing a
    // firmware-owned entry is safe and useful for diagnostics.
    NocWindowResult read_noc_window(uint32_t index) const;

    // False once the tile cannot be confirmed back in resident idle, from
    // either of two causes: a bounded restoration that did not return all four
    // harts, or a submission that failed after clearing the per-hart records,
    // which only resident idle's entry path republishes. Enqueue on an
    // unhealthy tile returns EnqueueOutcome::Unhealthy, and nothing in
    // software clears the latch.
    bool healthy() const;

    // The four-hart snapshot saved when health was lost. Outlives the session
    // that recorded it, which cleanup() reaps.
    std::optional<std::array<HartState, kHartsPerTile>> last_failure() const;

private:
    friend class X280Runtime;
    friend class detail::RuntimeImpl;
    friend class detail::TileTestAccess;
    X280Tile(detail::RuntimeImpl& runtime, detail::ChipImpl& chip, TileIdentity identity);

    std::unique_ptr<detail::TileImpl> impl_;
};

}  // namespace x280
