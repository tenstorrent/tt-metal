// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Private implementation of one runtime-owned active session.

#pragma once

#include "x280_tile_impl.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace x280::detail {

// The runtime owns every session, so this holds a bare TileImpl reference: the
// tile outlives the registry it is reached from.
class ActiveSessionImpl {
public:
    ActiveSessionImpl(TileImpl& tile, SessionId id, uint64_t run_value, std::chrono::milliseconds cleanup_timeout) :
        tile_(tile), id_(id), run_value_(run_value), cleanup_timeout_(cleanup_timeout) {}

    SessionId id() const { return id_; }
    const TileIdentity& tile() const { return tile_.id(); }
    uint64_t run_value() const { return run_value_; }

    std::array<HartState, kHartsPerTile> inspect() const;
    WaitResult wait_until_running(std::chrono::milliseconds timeout) const;
    WaitResult wait_until_result_ready(std::chrono::milliseconds timeout) const;

    // Reports through the returned result and never throws. The tile slot and
    // the health latch are released only here, so an escaping exception would
    // strand the tile with nothing left holding its id.
    CleanupResult cleanup(std::chrono::milliseconds timeout) noexcept;

    // The bound the runtime's destructor sweep restores this session with.
    std::chrono::milliseconds cleanup_timeout() const { return cleanup_timeout_; }

    // Both require RuntimeImpl::sessions_mutex_ to be held across the call: the
    // flag behind them is guarded by that mutex and not by this class, because
    // the claim has to be indivisible from the registry lookup that found this
    // session.
    //
    // claim_for_restoration returns false when a cleanup() already claimed it.
    // The winner is the only caller permitted to drive cleanup() and the only
    // one permitted to erase the registry entry.
    bool claim_for_restoration();
    bool claimed_for_restoration() const { return restoring_; }

private:
    enum class WaitTarget {
        Running,
        ResultReady,
    };

    CleanupResult restore(std::chrono::milliseconds timeout);

    template <typename Predicate>
    WaitResult wait_for(
        std::chrono::milliseconds timeout, Predicate predicate, WaitOutcome success, WaitTarget target) const;

    std::array<HartState, kHartsPerTile> inspect_best_effort() const noexcept;

    // Every path here leaves the tile unhealthy: a bounded restoration that did
    // not confirm resident idle cannot be retried in software. complete_session
    // reports only whether this session still held the slot, which a failure
    // outcome already tells the caller.
    CleanupResult fail_cleanup(
        CleanupOutcome outcome,
        const std::array<HartState, kHartsPerTile>& hartstates,
        std::chrono::milliseconds timeout,
        std::string error = {});

    TileImpl& tile_;
    SessionId id_;
    uint64_t run_value_;
    std::chrono::milliseconds cleanup_timeout_;

    // Set once a cleanup() has claimed this session, and never cleared: the
    // claimant erases the session from the registry instead. Guarded by
    // RuntimeImpl::sessions_mutex_, as the two accessors above document.
    bool restoring_ = false;
};

}  // namespace x280::detail
