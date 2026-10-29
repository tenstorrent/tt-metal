// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runtime-owned active-image submission, observation, and restoration.

#include "x280_active_impl.hpp"

#include "x280/device/x280_deadline.hpp"
#include "x280/device/x280_state.hpp"
#include "x280_format.hpp"

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace x280::detail {

namespace {

constexpr std::chrono::milliseconds kPollInterval{5};

const HartState* first_invalid(const std::array<HartState, kHartsPerTile>& hartstates) {
    for (const auto& hart : hartstates) {
        if (!hart.observation_valid) {
            return &hart;
        }
    }
    return nullptr;
}

}  // namespace

void TileImpl::load_active_image(const std::vector<uint8_t>& image, std::optional<uint64_t> primed_bytes) {
    std::vector<uint8_t> padded = image;
    padded.resize((padded.size() + 3) & ~static_cast<size_t>(3), 0);

    const auto check =
        validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, padded.size(), primed_bytes, runtime_->lim_layout());
    if (!check.ok) {
        fail(check.reason);
    }

    const auto tile = core();
    transport().write_block(id().chip_id, tile, X280_ACTIVE_FW_LOAD_ADDR, padded.data(), padded.size());

    std::vector<uint8_t> readback(padded.size(), 0);
    transport().read_block(id().chip_id, tile, X280_ACTIVE_FW_LOAD_ADDR, readback.data(), readback.size());
    if (readback != padded) {
        size_t first = 0;
        while (first < padded.size() && readback[first] == padded[first]) {
            first++;
        }
        fail(
            "active image readback mismatch at LIM+" + hex(first) + ": " +
            format_readback_values(padded[first], readback[first]) +
            ". The jump was not published, so resident idle is still in control.");
    }
}

std::array<HartState, kHartsPerTile> TileImpl::snapshot_active_run() const {
    std::array<HartState, kHartsPerTile> hartstates{};
    try {
        hartstates = read_hart_states_unlocked();
    } catch (const std::exception& error) {
        for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
            hartstates[hart].hart_id = hart;
            hartstates[hart].invalid_reason =
                std::string("transport error while reading active state: ") + error.what();
        }
    }
    return hartstates;
}

std::array<uint64_t, kHartsPerTile> TileImpl::read_hart_phases() const {
    std::array<uint64_t, kHartsPerTile> phases{};
    const auto tile = core();
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        phases[hart] = transport().read_u64(
            id().chip_id,
            tile,
            X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) + hart * sizeof(x280_hart_state_t) +
                offsetof(x280_hart_state_t, phase));
    }
    return phases;
}

std::array<uint32_t, kHartsPerTile> TileImpl::read_hart_statuses() const {
    std::array<uint32_t, kHartsPerTile> statuses{};
    const auto tile = core();
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        transport().read_block(
            id().chip_id,
            tile,
            X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) + hart * sizeof(x280_hart_state_t) +
                offsetof(x280_hart_state_t, terminal_status),
            &statuses[hart],
            sizeof(statuses[hart]));
    }
    return statuses;
}

void TileImpl::clear_active_run_state() {
    const auto tile = core();
    const x280_hart_protocol_t protocol{};
    transport().write_block(id().chip_id, tile, X280_HART_PROTOCOL_BASE, &protocol, sizeof(protocol));
}

void TileImpl::publish_active_run_entry(
    uint64_t run_value, uint64_t entry_addr, uint64_t args_addr, uint64_t args_bytes) {
    const auto tile = core();
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        const uint64_t base =
            X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, control) + hart * sizeof(x280_hart_control_t);
        transport().write_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, run_value), run_value);
        transport().write_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, entry), entry_addr);
        transport().write_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, args), args_addr);
        transport().write_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, args_bytes), args_bytes);
        const uint64_t seen_run =
            transport().read_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, run_value));
        const uint64_t seen_entry =
            transport().read_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, entry));
        const uint64_t seen_args = transport().read_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, args));
        const uint64_t seen_args_bytes =
            transport().read_u64(id().chip_id, tile, base + offsetof(x280_hart_control_t, args_bytes));
        if (seen_run != run_value || seen_entry != entry_addr || seen_args != args_addr ||
            seen_args_bytes != args_bytes) {
            fail(
                "hart " + std::to_string(hart) + " run/entry/args/args_bytes readback mismatch: run " +
                format_readback_values(run_value, seen_run) + ", entry " +
                format_readback_values(entry_addr, seen_entry) + ", args " +
                format_readback_values(args_addr, seen_args) + ", args_bytes " +
                format_readback_values(args_bytes, seen_args_bytes) + ". No JUMP was published.");
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void TileImpl::publish_active_command(uint32_t hart, uint64_t command) {
    const auto tile = core();
    const uint64_t address = X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, control) +
                             hart * sizeof(x280_hart_control_t) + offsetof(x280_hart_control_t, command);
    transport().write_u64(id().chip_id, tile, address, command);
    if (command != X280_HART_COMMAND_JUMP) {
        const uint64_t seen = transport().read_u64(id().chip_id, tile, address);
        if (seen != command) {
            fail(
                "hart " + std::to_string(hart) +
                " command readback mismatch: " + format_readback_values(command, seen));
        }
    }
}

void TileImpl::publish_active_command_all(uint64_t command) {
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        publish_active_command(hart, command);
    }
}

SessionAttachOutcome TileImpl::attach_session(SessionId id) {
    if (!id) {
        fail("attach_session was given a default-constructed SessionId, which names no execution");
    }
    const std::lock_guard<std::mutex> lock(session_mutex_);
    if (!healthy_) {
        return SessionAttachOutcome::Unhealthy;
    }
    if (session_active_) {
        return SessionAttachOutcome::Busy;
    }

    session_active_ = true;
    active_session_id_ = id;
    return SessionAttachOutcome::Attached;
}

bool TileImpl::complete_session(SessionId id, bool restored, const std::array<HartState, kHartsPerTile>& hartstates) {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    if (!session_active_ || !id || id != active_session_id_) {
        return false;
    }

    session_active_ = false;
    active_session_id_ = SessionId{};
    if (!restored) {
        healthy_ = false;
        failure_hartstates_ = hartstates;
    }
    return true;
}

bool TileImpl::session_active() const {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    return session_active_;
}

bool TileImpl::healthy() const {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    return healthy_;
}

std::optional<std::array<HartState, kHartsPerTile>> TileImpl::failure_hartstates() const {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    return failure_hartstates_;
}

std::array<HartState, kHartsPerTile> ActiveSessionImpl::inspect() const {
    tile_.runtime_->require_mesh_live("active session inspect", &tile_.id());
    const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
    return tile_.snapshot_active_run();
}

WaitResult ActiveSessionImpl::wait_until_running(std::chrono::milliseconds timeout) const {
    return wait_for(
        timeout,
        [&](const std::array<HartState, kHartsPerTile>& hartstates) {
            return active_all_running(hartstates, run_value_);
        },
        WaitOutcome::Running,
        WaitTarget::Running);
}

WaitResult ActiveSessionImpl::wait_until_result_ready(std::chrono::milliseconds timeout) const {
    return wait_for(
        timeout,
        [&](const std::array<HartState, kHartsPerTile>& hartstates) {
            return active_all_results_ready(hartstates, run_value_);
        },
        WaitOutcome::ResultReady,
        WaitTarget::ResultReady);
}

CleanupResult ActiveSessionImpl::cleanup(std::chrono::milliseconds timeout) noexcept {
    try {
        return restore(timeout);
    } catch (const std::exception& error) {
        return fail_cleanup(
            CleanupOutcome::Failed,
            inspect_best_effort(),
            timeout,
            std::string("active restoration failed: ") + error.what());
    } catch (...) {
        return fail_cleanup(
            CleanupOutcome::Failed, inspect_best_effort(), timeout, "unknown active restoration failure");
    }
}

CleanupResult ActiveSessionImpl::restore(std::chrono::milliseconds timeout) {
    // Runtime cleanup validates caller input before consuming the session, and
    // enqueue validates the destructor's cleanup timeout.
    if (timeout.count() <= 0) {
        return fail_cleanup(
            CleanupOutcome::Failed,
            inspect_best_effort(),
            timeout,
            "cleanup timeout is " + std::to_string(timeout.count()) + " ms; it must be positive");
    }

    Deadline deadline{timeout};
    try {
        tile_.runtime_->require_mesh_live("active session cleanup", &tile_.id());
        {
            const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
            tile_.publish_active_command_all(X280_HART_COMMAND_STOP);
        }
    } catch (const std::exception& error) {
        return fail_cleanup(
            CleanupOutcome::Failed,
            inspect_best_effort(),
            timeout,
            std::string("could not publish active STOP: ") + error.what());
    }

    std::array<HartState, kHartsPerTile> hartstates{};
    const auto returned = poll_until(deadline, kPollInterval, [&] {
        const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
        const auto phases = tile_.read_hart_phases();
        if (!std::all_of(phases.begin(), phases.end(), [](uint64_t phase) {
                return phase == X280_HART_PHASE_RETURNED_TO_IDLE;
            })) {
            return false;
        }
        hartstates = tile_.snapshot_active_run();
        return first_invalid(hartstates) != nullptr || active_all_returned(hartstates, run_value_);
    });
    if (!returned.satisfied) {
        const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
        hartstates = tile_.snapshot_active_run();
    }
    if (first_invalid(hartstates) != nullptr) {
        return fail_cleanup(CleanupOutcome::InvalidObservation, hartstates, timeout);
    }
    if (!returned.satisfied) {
        return fail_cleanup(CleanupOutcome::StillRunningActiveFw, hartstates, timeout);
    }

    try {
        const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
        tile_.publish_active_command_all(X280_HART_COMMAND_RETURN_ACK);
    } catch (const std::exception& error) {
        return fail_cleanup(
            CleanupOutcome::ReturnAcknowledgementFailed,
            inspect_best_effort(),
            timeout,
            std::string("could not acknowledge active return: ") + error.what());
    }

    const auto rearmed = poll_until(deadline, kPollInterval, [&] {
        const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
        const auto phases = tile_.read_hart_phases();
        if (!std::all_of(phases.begin(), phases.end(), [](uint64_t phase) { return phase == X280_HART_PHASE_IDLE; })) {
            return false;
        }
        hartstates = tile_.snapshot_active_run();
        return first_invalid(hartstates) != nullptr || active_all_idle(hartstates, run_value_);
    });
    if (!rearmed.satisfied) {
        const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
        hartstates = tile_.snapshot_active_run();
    }
    if (first_invalid(hartstates) != nullptr) {
        return fail_cleanup(CleanupOutcome::InvalidObservation, hartstates, timeout);
    }
    if (!rearmed.satisfied) {
        return fail_cleanup(CleanupOutcome::IdleNotReArmed, hartstates, timeout);
    }

    CleanupResult result;
    result.outcome = CleanupOutcome::ReturnedToIdle;
    result.hartstates = hartstates;
    result.timeout = timeout;
    if (!tile_.complete_session(id_, true, hartstates)) {
        result.outcome = CleanupOutcome::Failed;
        result.error = "the tile slot no longer held this session";
    }
    return result;
}

template <typename Predicate>
WaitResult ActiveSessionImpl::wait_for(
    std::chrono::milliseconds timeout, Predicate predicate, WaitOutcome success, WaitTarget target) const {
    WaitResult result;
    result.timeout = timeout;
    if (timeout.count() <= 0) {
        result.error = "timeout is " + std::to_string(timeout.count()) + " ms; it must be positive";
        return result;
    }

    try {
        tile_.runtime_->require_mesh_live("active session wait", &tile_.id());
        Deadline deadline{timeout};
        const auto poll = poll_until(deadline, kPollInterval, [&] {
            const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
            bool committed = false;
            if (target == WaitTarget::ResultReady) {
                const auto statuses = tile_.read_hart_statuses();
                committed = std::all_of(statuses.begin(), statuses.end(), [](uint32_t status) {
                    return status == X280_HART_STATUS_SUCCESS;
                });
            } else {
                const auto phases = tile_.read_hart_phases();
                committed = std::all_of(phases.begin(), phases.end(), [](uint64_t phase) {
                    return phase == X280_HART_PHASE_RUNNING_ACTIVE_FW;
                });
            }
            if (!committed) {
                return false;
            }
            result.hartstates = tile_.snapshot_active_run();
            return first_invalid(result.hartstates) != nullptr || target == WaitTarget::ResultReady ||
                   predicate(result.hartstates);
        });
        if (!poll.satisfied) {
            const std::lock_guard<std::mutex> serialise(tile_.chip_->lock());
            result.hartstates = tile_.snapshot_active_run();
        }
        if (first_invalid(result.hartstates) != nullptr) {
            result.outcome = WaitOutcome::InvalidObservation;
        } else if (poll.satisfied && target == WaitTarget::ResultReady && !predicate(result.hartstates)) {
            // Every hart committed terminal success, but the observation did
            // not correlate with this run.
            result.outcome = WaitOutcome::ResultMismatch;
        } else if (poll.satisfied) {
            result.outcome = success;
        } else {
            result.outcome = WaitOutcome::TimedOut;
        }
    } catch (const std::exception& error) {
        result.outcome = WaitOutcome::Failed;
        result.error = error.what();
    }
    return result;
}

std::array<HartState, kHartsPerTile> ActiveSessionImpl::inspect_best_effort() const noexcept {
    try {
        return inspect();
    } catch (const std::exception& error) {
        std::array<HartState, kHartsPerTile> hartstates{};
        for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
            hartstates[hart].hart_id = hart;
            hartstates[hart].invalid_reason = error.what();
        }
        return hartstates;
    } catch (...) {
        std::array<HartState, kHartsPerTile> hartstates{};
        for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
            hartstates[hart].hart_id = hart;
            hartstates[hart].invalid_reason = "unknown active observation failure";
        }
        return hartstates;
    }
}

CleanupResult ActiveSessionImpl::fail_cleanup(
    CleanupOutcome outcome,
    const std::array<HartState, kHartsPerTile>& hartstates,
    std::chrono::milliseconds timeout,
    std::string error) {
    CleanupResult result;
    result.outcome = outcome;
    result.hartstates = hartstates;
    result.timeout = timeout;
    result.error = std::move(error);
    (void)tile_.complete_session(id_, false, hartstates);
    return result;
}

// ---------------------------------------------------------------------------
// The runtime-owned session registry
// ---------------------------------------------------------------------------

RuntimeImpl::~RuntimeImpl() {
    // The sweep is explicit rather than left to member destruction: the tiles
    // it restores and the health latches it writes are members too.
    std::map<TileKey, std::unique_ptr<X280ActiveSession>> live;
    {
        const std::lock_guard<std::mutex> lock(sessions_mutex_);
        live.swap(sessions_);
    }

    const bool transport_usable = mesh_live();
    for (const auto& entry : live) {
        auto& impl = *entry.second->impl_;
        const auto& tile = impl.tile();
        if (impl.claimed_for_restoration()) {
            log_warning(
                tt::LogMetal,
                "X280 active session {} on {} was still being restored as the runtime was destroyed; leaving it "
                "alone rather than restoring it twice. The tile may be left running active firmware until an ASIC "
                "reset.",
                entry.second->id().value,
                label_chip_l2cpu(tile.chip_id, tile.hw_index));
            continue;
        }
        if (!transport_usable) {
            log_warning(
                tt::LogMetal,
                "X280 abandoned active session {} on {}: the runtime is being destroyed and the caller-owned "
                "MeshDevice is already closed, so no STOP or RETURN_ACK can reach the tile. It is left running "
                "active firmware until an ASIC reset.",
                entry.second->id().value,
                label_chip_l2cpu(tile.chip_id, tile.hw_index));
            continue;
        }
        const auto restored = impl.cleanup(impl.cleanup_timeout());
        if (!restored.cleaned_up()) {
            // The one path whose whole purpose is to prevent a stranded tile,
            // so its failure is the case an operator most needs told: the
            // health latch and the observation die with this runtime.
            log_warning(
                tt::LogMetal,
                "X280 could not restore active session {} on {} during runtime destruction: {}. The tile is left "
                "running active firmware until an ASIC reset.",
                entry.second->id().value,
                label_chip_l2cpu(tile.chip_id, tile.hw_index),
                describe(restored));
        }
    }
}

bool ActiveSessionImpl::claim_for_restoration() {
    if (restoring_) {
        return false;
    }
    restoring_ = true;
    return true;
}

SessionId RuntimeImpl::mint_session_id(const TileKey& tile) {
    const std::lock_guard<std::mutex> lock(sessions_mutex_);
    uint64_t value = next_session_id_++;
    if (value == 0) {
        // Zero names no execution, so the counter steps over it on wrap. Values
        // still repeat after a full wrap, which nothing here prevents and which
        // 2^64 submissions within one mesh binding puts out of reach.
        value = next_session_id_++;
    }
    return SessionId{value, tile};
}

// The registry is keyed by tile, so a lookup has to confirm the entry it found
// is the one the caller named. Without this a handle to a reaped session would
// match the tile's current session and act on the wrong execution.
static bool names_session(
    const std::map<TileKey, std::unique_ptr<X280ActiveSession>>& sessions,
    const std::map<TileKey, std::unique_ptr<X280ActiveSession>>::const_iterator& entry,
    const SessionId& id) {
    return entry != sessions.end() && entry->second->id() == id;
}

std::vector<SessionId> RuntimeImpl::sessions() const {
    const std::lock_guard<std::mutex> lock(sessions_mutex_);
    std::vector<SessionId> ids;
    ids.reserve(sessions_.size());
    // Already grouped by tile: the map is ordered on the tile key.
    for (const auto& entry : sessions_) {
        ids.push_back(entry.second->id());
    }
    return ids;
}

X280ActiveSession& RuntimeImpl::session(SessionId id) {
    const std::lock_guard<std::mutex> lock(sessions_mutex_);
    const auto entry = sessions_.find(id.tile);
    if (!names_session(sessions_, entry, id)) {
        fail(
            "active session " + std::to_string(id.value) + " on " +
            label_chip_l2cpu(id.tile.chip_id, id.tile.tile_index) +
            " is not live: the id was never issued, or cleanup() already reaped it");
    }
    return *entry->second;
}

bool RuntimeImpl::session_live(SessionId id) const {
    const std::lock_guard<std::mutex> lock(sessions_mutex_);
    return names_session(sessions_, sessions_.find(id.tile), id);
}

CleanupResult RuntimeImpl::cleanup(SessionId id, std::chrono::milliseconds timeout) {
    // Validated before the session is consumed. A non-positive bound publishes
    // no STOP, so reaping first would spend the only id that can restore the
    // tile on a caller-side argument error. Deadline::remaining() returns zero
    // past expiry, which makes this a reachable call rather than a typo.
    if (timeout.count() <= 0) {
        CleanupResult result;
        result.outcome = CleanupOutcome::Failed;
        result.timeout = timeout;
        result.error = "cleanup timeout is " + std::to_string(timeout.count()) +
                       " ms; it must be positive. The session is still live.";
        return result;
    }

    // The session is claimed in place and its entry stays for the whole
    // restoration, so no STOP or RETURN_ACK is ever in flight for a tile this
    // map has no entry for. Claiming and finding are one critical section, so a
    // second concurrent cleanup() sees AlreadyRestoring instead of driving the
    // protocol into the same four harts twice.
    X280ActiveSession* claimed = nullptr;
    {
        const std::lock_guard<std::mutex> lock(sessions_mutex_);
        const auto entry = sessions_.find(id.tile);
        if (!names_session(sessions_, entry, id)) {
            CleanupResult result;
            result.outcome = CleanupOutcome::UnknownSession;
            result.timeout = timeout;
            return result;
        }
        if (!entry->second->impl_->claim_for_restoration()) {
            CleanupResult result;
            result.outcome = CleanupOutcome::AlreadyRestoring;
            result.timeout = timeout;
            return result;
        }
        claimed = entry->second.get();
    }

    // The restoration takes the chip lock and polls the tile, so it runs with
    // sessions_mutex_ released. Borrowing the session across it is safe because
    // only the caller that won the claim above ever erases the entry, and a
    // std::map node does not move.
    const auto result = claimed->impl_->cleanup(timeout);

    {
        const std::lock_guard<std::mutex> lock(sessions_mutex_);
        sessions_.erase(id.tile);
        if (exclusive_ == id.tile) {
            exclusive_.reset();
        }
    }
    return result;
}

EnqueueResult RuntimeImpl::enqueue(X280Tile& tile, const ActiveRunRequest& request, bool exclusive) {
    EnqueueResult result;
    auto& target = *tile.impl_;
    if (target.runtime_ != this) {
        result.outcome = EnqueueOutcome::WrongRuntime;
        return result;
    }
    if (!target.healthy()) {
        result.outcome = EnqueueOutcome::Unhealthy;
        return result;
    }
    // A cheap pre-check, so an already-busy tile costs no image validation,
    // load or readback. attach_session() below is the atomic commit.
    if (target.session_active()) {
        result.outcome = EnqueueOutcome::Busy;
        return result;
    }
    // The tile slot is released inside the restoration, before cleanup() regains
    // control to erase the registry entry, so there is a window where the slot
    // reads free while the entry is still present. Refusing here keeps a caller
    // that lands in it out of the write path entirely: reaching the registration
    // below and colliding there would mean the hart records had already been
    // cleared, latching a healthy tile unhealthy over transient bookkeeping.
    //
    // An exclusive submission is admitted only when no session is registered
    // or being submitted anywhere in the runtime, and refuses every other
    // admission until its session is reaped.
    bool registered = false;
    {
        const std::lock_guard<std::mutex> lock(sessions_mutex_);
        if (sessions_.contains(tile_key(target.id())) || exclusive_ ||
            (exclusive && (submitting_ > 0 || !sessions_.empty()))) {
            result.outcome = EnqueueOutcome::Busy;
            return result;
        }
        ++submitting_;
        if (exclusive) {
            exclusive_ = tile_key(target.id());
        }
    }
    struct Admission {
        RuntimeImpl& runtime;
        bool exclusive;
        const bool& registered;
        ~Admission() {
            const std::lock_guard<std::mutex> lock(runtime.sessions_mutex_);
            --runtime.submitting_;
            if (exclusive && !registered) {
                runtime.exclusive_.reset();
            }
        }
    } const admission{*this, exclusive, registered};

    if (request.cleanup_timeout.count() <= 0) {
        result.error = "cleanup_timeout is " + std::to_string(request.cleanup_timeout.count()) +
                       " ms; the runtime's destructor sweep requires a positive timeout";
        return result;
    }

    const SessionId id = mint_session_id(tile_key(target.id()));
    bool attached = false;
    bool records_cleared = false;

    try {
        target.runtime_->require_mesh_live("enqueue", &target.id());
        const std::lock_guard<std::mutex> serialise(target.chip_->lock());

        const auto state = target.inspect_state();
        if (!state.observation_valid) {
            result.outcome = EnqueueOutcome::InvalidObservation;
            result.error = state.invalid_reason;
            return result;
        }
        if (!state.resident_idle_live) {
            result.hartstates = target.snapshot_active_run();
            // No active-image bytes were written on either branch.
            result.outcome = first_invalid(result.hartstates) != nullptr ? EnqueueOutcome::InvalidObservation
                                                                         : EnqueueOutcome::NotResidentIdle;
            return result;
        }

        // Checked here rather than inside the image try below: it needs the
        // observed LIM extent but no device I/O, so it costs nothing and
        // rejects before the slot is reserved.
        const auto args_check =
            validate_argument_block(target.available_lim_unlocked(state), request.args_addr, request.args_bytes);
        if (!args_check.ok) {
            result.outcome = EnqueueOutcome::ArgumentsRejected;
            result.error = args_check.reason;
            return result;
        }

        // Reserved before anything is written, so a racing enqueue that loses
        // the slot loses it before it can zero the records or the run value of
        // the execution that won. The pre-check above only spares the
        // uncontended caller the image read; this is what makes it exclusive.
        const auto attachment = target.attach_session(id);
        if (attachment != SessionAttachOutcome::Attached) {
            result.outcome =
                attachment == SessionAttachOutcome::Unhealthy ? EnqueueOutcome::Unhealthy : EnqueueOutcome::Busy;
            return result;
        }
        attached = true;

        // The rejection path must reach the release below rather than return
        // from here, so the outcome is carried in a flag.
        bool image_ready = false;
        try {
            const auto image =
                detail::TileImpl::read_image_file(request.image_path, "active image", "ninja x280_images");
            const size_t padded_size = (image.size() + 3) & ~static_cast<size_t>(3);
            const auto image_check =
                validate_active_image(X280_ACTIVE_FW_LOAD_ADDR, padded_size, state.ecc_primed_bytes, lim_layout());
            if (!image_check.ok) {
                throw std::runtime_error("x280: " + image_check.reason);
            }
            const auto reserved = target.reserved_lim();
            if (!reserved.contains(X280_ACTIVE_FW_LOAD_ADDR, padded_size)) {
                throw std::runtime_error("x280: active image is outside the component's reserved LIM");
            }
            const LimRange backed{
                X280_IDLE_FW_LOAD_ADDR,
                X280_IDLE_FW_LOAD_ADDR + state.lim_backed_bytes,
            };
            if (!backed.contains(X280_ACTIVE_FW_LOAD_ADDR, padded_size)) {
                throw std::runtime_error("x280: active image exceeds currently backed LIM");
            }
            target.load_active_image(image, state.ecc_primed_bytes);
            image_ready = true;
        } catch (const std::exception& error) {
            result.outcome = EnqueueOutcome::ImageRejected;
            result.error = error.what();
        }

        if (image_ready) {
            // Past this point resident idle cannot be restored by any host
            // write. lim_idle.c publishes the four IDLE records once, before
            // its command poll loop, so clearing them leaves nothing that will
            // republish them.
            records_cleared = true;
            try {
                target.clear_active_run_state();
                target.publish_active_run_entry(
                    request.run_value, X280_ACTIVE_FW_LOAD_ADDR, request.args_addr, request.args_bytes);
            } catch (...) {
                // Snapshot while the chip lock is still held: this is the path
                // that latches the tile unhealthy, and the latch is worth
                // nothing without the records that explain it.
                result.hartstates = target.snapshot_active_run();
                throw;
            }
            result.hartstates = target.snapshot_active_run();

            // Registered before the first JUMP write, so a publication that
            // lands while the transport reports failure is already owned by the
            // registry and reachable by cleanup() and by the destructor sweep.
            // Lock order is chip then sessions; nothing takes them the other
            // way round, and no device I/O runs under sessions_mutex_.
            {
                const std::lock_guard<std::mutex> lock(sessions_mutex_);
                const auto inserted = sessions_.emplace(
                    id.tile,
                    std::unique_ptr<X280ActiveSession>(new X280ActiveSession(
                        std::make_unique<ActiveSessionImpl>(target, id, request.run_value, request.cleanup_timeout))));
                if (!inserted.second) {
                    fail(
                        "internal error: " + label_chip_l2cpu(id.tile.chip_id, id.tile.tile_index) +
                        " already has active session " + std::to_string(inserted.first->second->id().value) +
                        " registered while enqueueing session " + std::to_string(id.value));
                }
            }
            registered = true;
            // Set with registration, not after the JUMP loop: the id is how the
            // caller reaches a session the registry already owns, on every
            // outcome from here on.
            result.session = id;

            try {
                for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
                    target.publish_active_command(hart, X280_HART_COMMAND_JUMP);
                }
                result.outcome = EnqueueOutcome::Enqueued;
            } catch (const std::exception& error) {
                result.outcome = EnqueueOutcome::JumpPublicationUncertain;
                result.error = std::string("JUMP publication may have landed: ") + error.what();
            } catch (...) {
                result.outcome = EnqueueOutcome::JumpPublicationUncertain;
                result.error = "JUMP publication may have landed before an unknown transport failure";
            }
        }
    } catch (const std::exception& error) {
        result.outcome = EnqueueOutcome::Failed;
        result.error = error.what();
    } catch (...) {
        result.outcome = EnqueueOutcome::Failed;
        result.error = "unknown active enqueue failure";
    }

    if (attached && !registered) {
        // The only release for the one execution slot, so every path that
        // attached has to arrive here: no return may stand between
        // attach_session above and this block.
        //
        // No JUMP was published either way. Whether the tile is reusable turns
        // on how far the writes got: with the records intact resident idle is
        // still live and the slot is simply released, but once they are cleared
        // nothing republishes them, so the tile is latched unhealthy rather
        // than left reporting healthy while every later enqueue sees
        // NotResidentIdle.
        (void)target.complete_session(id, !records_cleared, result.hartstates);
    }
    return result;
}

}  // namespace x280::detail

namespace x280 {

X280ActiveSession::X280ActiveSession(std::unique_ptr<detail::ActiveSessionImpl> impl) : impl_(std::move(impl)) {}

X280ActiveSession::~X280ActiveSession() = default;

SessionId X280ActiveSession::id() const { return impl_->id(); }

const TileIdentity& X280ActiveSession::tile() const { return impl_->tile(); }

uint64_t X280ActiveSession::run_value() const { return impl_->run_value(); }

std::array<HartState, kHartsPerTile> X280ActiveSession::inspect() const { return impl_->inspect(); }

WaitResult X280ActiveSession::wait_until_running(std::chrono::milliseconds timeout) const {
    return impl_->wait_until_running(timeout);
}

WaitResult X280ActiveSession::wait_until_result_ready(std::chrono::milliseconds timeout) const {
    return impl_->wait_until_result_ready(timeout);
}

}  // namespace x280
