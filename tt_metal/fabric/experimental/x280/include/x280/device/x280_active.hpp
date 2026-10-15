// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runtime-owned active-firmware submission and restoration.

#pragma once

#include "x280/device/x280_tile_state.hpp"

#include <array>
#include <chrono>
#include <compare>
#include <cstdint>
#include <memory>
#include <string>

namespace x280 {

namespace detail {
class ActiveSessionImpl;
class RuntimeImpl;
}  // namespace detail

struct ActiveRunRequest {
    std::string image_path;
    uint64_t run_value = 0;
    std::chrono::milliseconds cleanup_timeout{3000};

    // Address of a caller-owned argument block, told to every hart, or zero for
    // an image that takes none. Must lie wholly inside the target tile's
    // available_lim() and start on a cache line, because that is the only
    // granularity write_available_lim admits — so the address a case can
    // legally publish is one it can legally have written.
    //
    // Placement, not content. The runtime does not know the block's layout, so
    // an image that trusts the bytes without checking their freshness reads the
    // previous run's block, which LIM keeps.
    uint64_t args_addr = 0;
    uint64_t args_bytes = 0;
};

bool hart_running(const HartState& hart, uint64_t run_value);

// A correlated terminal success, which is what publishes the result word. That
// word is the application's payload, so readiness requires no value of it.
// Satisfied in either active phase: an image that finishes on its own commits
// and then advances to RETURNED_TO_IDLE, so readiness outlives RUNNING.
bool hart_result_ready(const HartState& hart, uint64_t run_value);

bool hart_returned(const HartState& hart, uint64_t run_value);
bool hart_resident_idle(const HartState& hart);
bool hart_idle(const HartState& hart, uint64_t run_value);
bool active_all_running(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value);
bool active_all_results_ready(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value);
bool active_all_returned(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value);
bool active_all_idle(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value);

// Whether every hart's result word echoes the run value, which readiness does
// not require. Only firmware specified to echo it can be asked this.
bool active_result_matches(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t run_value);

// Which of a hart's three commit words have landed, and whether a written
// payload is waiting behind one that has not. That difference is what separates
// a stale record from a real state.
//
// Each word guards a different payload, so each pending flag watches a
// different set: phase guards run_value and heartbeat, terminal_status guards
// result, and trap_valid guards the whole trap record bar itself. Firmware
// writes the final heartbeat alongside result, but heartbeat is already
// covered by the phase it committed earlier, so it is not a terminal payload.
struct HartCommit {
    bool phase = false;
    bool phase_payload_pending = false;
    bool terminal = false;
    bool terminal_payload_pending = false;
    bool trap = false;
    bool trap_payload_pending = false;
};

// A function rather than a field on HartState: it is pure derivation from the
// record beside it, so a stored copy would go stale the moment a caller edits
// one word of that record.
HartCommit commit_state(const HartState& hart);

std::string describe(const std::array<HartState, kHartsPerTile>& hartstates);

enum class WaitOutcome {
    Running,
    ResultReady,
    TimedOut,
    ResultMismatch,
    InvalidObservation,
    Failed,
};

const char* to_string(WaitOutcome outcome);

struct WaitResult {
    WaitOutcome outcome = WaitOutcome::Failed;
    std::array<HartState, kHartsPerTile> hartstates{};

    // The deadline the wait was given, so a TimedOut result is
    // self-describing.
    std::chrono::milliseconds timeout{0};

    // Transport or exception text, which has no value form. Empty unless the
    // outcome is Failed.
    std::string error;

    bool succeeded() const { return outcome == WaitOutcome::Running || outcome == WaitOutcome::ResultReady; }
};

std::string describe(const WaitResult& result);

enum class CleanupOutcome {
    ReturnedToIdle,
    StillRunningActiveFw,
    ReturnAcknowledgementFailed,
    IdleNotReArmed,
    InvalidObservation,
    // The id names no live execution: it was never issued, or a previous
    // cleanup() already reaped it.
    UnknownSession,
    AlreadyRestoring,
    Failed,
};

const char* to_string(CleanupOutcome outcome);

struct CleanupResult {
    CleanupOutcome outcome = CleanupOutcome::Failed;
    std::array<HartState, kHartsPerTile> hartstates{};

    std::chrono::milliseconds timeout{0};

    // Transport or exception text, which has no value form. Empty unless the
    // outcome is Failed or ReturnAcknowledgementFailed.
    std::string error;

    bool cleaned_up() const { return outcome == CleanupOutcome::ReturnedToIdle; }
};

std::string describe(const CleanupResult& result);

// Runtime-wide monotonic id naming one active-firmware execution the runtime
// owns. Zero is never issued, so a default-constructed id names no execution.
struct SessionId {
    uint64_t value = 0;
    TileKey tile;
    auto operator<=>(const SessionId&) const = default;
    explicit operator bool() const { return value != 0; }
};

// Observation of one execution the runtime owns. Cleanup is on X280Runtime,
// not this object, because the runtime owns both the session and its lifetime.
class X280ActiveSession {
public:
    // Non-copyable and non-movable, as X280Tile is: the runtime hands out
    // references.
    X280ActiveSession(const X280ActiveSession&) = delete;
    X280ActiveSession& operator=(const X280ActiveSession&) = delete;
    X280ActiveSession(X280ActiveSession&&) = delete;
    X280ActiveSession& operator=(X280ActiveSession&&) = delete;
    ~X280ActiveSession();

    SessionId id() const;
    const TileIdentity& tile() const;

    // The value every predicate in this header correlates against. Without it
    // an observation cannot be asked anything.
    uint64_t run_value() const;

    /**
     * @throws std::runtime_error if the caller-owned MeshDevice has been closed
     *         or dropped.
     */
    std::array<HartState, kHartsPerTile> inspect() const;

    /**
     * Report failure through WaitResult rather than throwing it. A non-positive
     * timeout is rejected before any I/O.
     *
     * WaitOutcome::ResultMismatch is reachable only from
     * wait_until_result_ready.
     */
    WaitResult wait_until_running(std::chrono::milliseconds timeout) const;
    WaitResult wait_until_result_ready(std::chrono::milliseconds timeout) const;

private:
    friend class detail::RuntimeImpl;
    explicit X280ActiveSession(std::unique_ptr<detail::ActiveSessionImpl> impl);

    std::unique_ptr<detail::ActiveSessionImpl> impl_;
};

enum class EnqueueOutcome {
    Enqueued,
    Busy,
    WrongRuntime,
    Unhealthy,
    NotResidentIdle,
    ImageRejected,
    // args_addr/args_bytes do not describe a cache-line-aligned block inside
    // the tile's available LIM. Rejected before any record is written, like
    // ImageRejected, so the slot is released and the tile stays healthy.
    ArgumentsRejected,
    InvalidObservation,
    JumpPublicationUncertain,
    Failed,
};

const char* to_string(EnqueueOutcome outcome);

struct EnqueueResult {
    EnqueueOutcome outcome = EnqueueOutcome::Failed;

    // Set whenever an execution was attached, including
    // JumpPublicationUncertain.
    SessionId session;
    std::array<HartState, kHartsPerTile> hartstates{};

    // Image rejection and transport text. Bounds violations and readback
    // mismatch offsets are genuinely textual, from validate_active_image() and
    // exceptions.
    std::string error;

    bool enqueued() const { return outcome == EnqueueOutcome::Enqueued; }
};

std::string describe(const EnqueueResult& result);

}  // namespace x280
