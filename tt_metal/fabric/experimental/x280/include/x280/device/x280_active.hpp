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
#include <optional>
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

    // Opt-in escalation. When the bounded cleanup of this session times out
    // (a hart does not answer STOP, or does not re-arm resident idle), the
    // runtime revives all four harts with an RNMI instead of latching the tile
    // unhealthy. The runtime's destructor sweep honours it too.
    bool revive_on_cleanup_timeout = false;
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
    // The session did not answer within the bound and its request opted in to
    // revive_on_cleanup_timeout: every hart was taken back to resident idle by
    // an RNMI. The tile is healthy; the image did not complete its contract.
    Revived,
};

const char* to_string(CleanupOutcome outcome);

// ---------------------------------------------------------------------------
// Revive: RNMI back to resident idle
// ---------------------------------------------------------------------------

inline constexpr uint32_t kAllHartsMask = (1u << kHartsPerTile) - 1;

struct ReviveRequest {
    // Bit h selects hart h. Must be all four harts while a session is live:
    // a session is a whole-tile execution.
    uint32_t hart_mask = kAllHartsMask;

    // Bounds the wait for the selected harts' acknowledgement, and separately
    // the wait for resident idle to re-arm afterwards. Must be positive.
    std::chrono::milliseconds timeout{100};
};

enum class ReviveOutcome {
    // Every selected hart took the RNMI and all four harts are resident idle.
    // The tile is healthy.
    Revived,
    // A selected hart did not take the RNMI within the bound: it is inside an
    // access that never completes (a NoC load to a target that does not
    // answer), or stopped by CEASE. Not recoverable in software; the tile is
    // unhealthy and needs an ASIC reset.
    NotResponding,
    // Every selected hart took the RNMI, but resident idle did not re-arm
    // within the bound on all four harts. The tile is unhealthy.
    NotReArmed,
    // The RNMI handler registers or the resident idle's RNMI marker do not
    // match: the tile was booted by an idle image without the handler, or by
    // another loader. Nothing was written.
    NotInstalled,
    // The tile is held in reset; there is nothing to revive.
    NotReleased,
    // A submission, a cleanup or another revive holds the tile, or an
    // exclusive (pinned-input) session is live. Nothing was written.
    Busy,
    // The tile belongs to another runtime.
    WrongRuntime,
    // A zero mask, bits above the tile's harts, a partial mask while a session
    // is live, or a non-positive timeout. Nothing was written.
    InvalidRequest,
    // A raw read came back all ones, or the transport failed before any write.
    InvalidObservation,
    Failed,
};

const char* to_string(ReviveOutcome outcome);

struct RevivedHart {
    uint32_t hart_id = 0;
    bool requested = false;
    // The handler ran: revive_count moved.
    bool acknowledged = false;
    // mnepc and mncause the handler recorded: where the hart was. The cause
    // carries X280_REVIVE_CAUSE_EXCEPTION for a fault inside the handler.
    uint64_t revive_epc = 0;
    uint64_t revive_cause = 0;
    // The hart's three records before the trigger, including the trap record
    // the handler clears, and after.
    HartState before;
    HartState after;
};

struct ReviveResult {
    ReviveOutcome outcome = ReviveOutcome::Failed;
    std::array<RevivedHart, kHartsPerTile> harts{};
    // The per-hart run-status word (cease / wfi bits) read before the trigger.
    uint32_t run_status_before = 0;
    uint32_t run_status_after = 0;
    std::chrono::milliseconds timeout{0};
    // Host trigger write -> the last selected hart's revive_count seen moved.
    std::chrono::microseconds trigger_to_ack{0};
    // Host trigger write -> the inspection that confirmed resident idle.
    std::chrono::microseconds trigger_to_idle{0};
    std::string error;

    bool revived() const { return outcome == ReviveOutcome::Revived; }
};

std::string describe(const ReviveResult& result);

struct CleanupResult {
    CleanupOutcome outcome = CleanupOutcome::Failed;
    std::array<HartState, kHartsPerTile> hartstates{};

    std::chrono::milliseconds timeout{0};

    // Transport or exception text, which has no value form. Empty unless the
    // outcome is Failed or ReturnAcknowledgementFailed.
    std::string error;

    // Set when the restoration escalated to a revive, whatever its outcome.
    std::optional<ReviveResult> revive;

    // The tile is back in resident idle and healthy, either through the
    // image's acknowledged return or through a revive. Callers that need to
    // know which read the outcome.
    bool cleaned_up() const { return outcome == CleanupOutcome::ReturnedToIdle || outcome == CleanupOutcome::Revived; }
};

std::string describe(const CleanupResult& result);

// Pure revive rules (no device I/O).

struct ReviveRequestCheck {
    bool ok = false;
    std::string reason;
};

// A non-zero mask within the tile's harts, a positive timeout, and the full
// mask when a session is live.
ReviveRequestCheck validate_revive_request(const ReviveRequest& request, bool session_live);

// The handler ran on this hart since `count_before` was read.
bool hart_revive_acknowledged(const HartState& hart, uint32_t count_before);

// Acknowledged, and resident idle re-armed on this hart.
bool hart_revived(const HartState& hart, uint32_t count_before);

// The trigger word with `mask` set or cleared, every other bit preserved.
uint32_t rnmi_trigger_set(uint32_t current, uint32_t mask);
uint32_t rnmi_trigger_clear(uint32_t current, uint32_t mask);

// The cleanup outcomes a revive_on_cleanup_timeout request escalates: a hart
// that did not answer STOP, or did not re-arm idle. Never an unreadable tile
// or a failed transport.
bool cleanup_escalates_to_revive(CleanupOutcome outcome);

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
