// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/packet_injection/common.hpp -- target-agnostic host stages of a
// packet injection run: the X280 arena, consumer launch and quiescence,
// collection, cleanup checks, and result checks. Each producer target's suite calls
// them in sequence around its own placement, kernels and producer readback.

#ifndef X280_SUITES_PACKET_INJECTION_COMMON_HPP
#define X280_SUITES_PACKET_INJECTION_COMMON_HPP

#include "harness/execution/execution.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/packet_injection/regions.hpp"
#include "harness/runner.hpp"
#include "tests/device/suites/packet_injection/records.hpp"

#include "kernels/packet_injection/packet_injection_abi.h"
#include "kernels/packet_injection/packet_injection_slot.hpp"
#include "rt/x280_noc_map.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace x280::suites::packet_injection {

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    run_nonce = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    if (run_nonce == 0u) {
        run_nonce = 1u;
    }
    return (0x50494E4AuLL << 32) | run_nonce;
}

// Renders a device failure record for a host diagnostic. All-zero means no
// failure, which PI_ERROR_NONE and PI_PHASE_INIT encode.
inline std::string describe_failure(const pi_failure_t& failure) {
    return "code=" + std::to_string(failure.code) + " phase=" + std::to_string(failure.phase) +
           " worker=" + std::to_string(failure.worker_index) + " actor=" + std::to_string(failure.actor);
}

// Reads the per-worker slot arrays out of the LIM arena.
//
// Records a failed or short read in `error` instead of throwing: every caller is
// already on a failure path and is either building a diagnostic or about to
// retain the run, and an exception raised here would replace both.
inline LimSlotDump read_lim_slots(x280::X280Tile& tile, const pi_layout_t& layout, uint32_t P) {
    const size_t produced_bytes = static_cast<size_t>(P) * sizeof(pi_counter_slot_t);
    const size_t ready_bytes = static_cast<size_t>(P) * sizeof(pi_ready_slot_t);
    const size_t consumer_bytes = static_cast<size_t>(P) * sizeof(pi_consumer_slot_t);

    LimSlotDump dump{};
    try {
        const auto produced =
            tile.read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.produced_offset, produced_bytes);
        const auto ready =
            tile.read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.ready_offset, ready_bytes);
        const auto slots = tile.read_available_lim(
            X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.consumer_slots_offset, consumer_bytes);
        if (produced.size() != produced_bytes || ready.size() != ready_bytes || slots.size() != consumer_bytes) {
            dump.error = "LIM slot read short: produced " + std::to_string(produced.size()) + "/" +
                         std::to_string(produced_bytes) + ", ready " + std::to_string(ready.size()) + "/" +
                         std::to_string(ready_bytes) + ", consumer " + std::to_string(slots.size()) + "/" +
                         std::to_string(consumer_bytes);
            return dump;
        }

        dump.produced.resize(P);
        dump.ready.resize(P);
        dump.slots.resize(P);
        for (uint32_t w = 0; w < P; ++w) {
            std::memcpy(
                &dump.produced[w],
                produced.data() + static_cast<size_t>(w) * sizeof(pi_counter_slot_t) +
                    offsetof(pi_counter_slot_t, value),
                sizeof(uint32_t));
            std::memcpy(
                &dump.ready[w],
                ready.data() + static_cast<size_t>(w) * sizeof(pi_ready_slot_t) + offsetof(pi_ready_slot_t, value),
                sizeof(uint32_t));
            std::memcpy(
                &dump.slots[w],
                slots.data() + static_cast<size_t>(w) * sizeof(pi_consumer_slot_t),
                sizeof(pi_consumer_slot_t));
        }
    } catch (const std::exception& e) {
        dump.produced.clear();
        dump.ready.clear();
        dump.slots.clear();
        dump.error = std::string("LIM slot read failed: ") + e.what();
    }
    return dump;
}

// Renders a dump raw, one line per worker, or its error.
inline std::string describe_lim_slots(const LimSlotDump& dump, const pi_layout_t& layout) {
    if (!dump.ok()) {
        return dump.error.empty() ? "LIM slots were not read" : dump.error;
    }
    std::ostringstream out;
    out << "LIM slots (produced_offset " << layout.produced_offset << ", ready_offset " << layout.ready_offset
        << ", consumer_slots_offset " << layout.consumer_slots_offset << "):";
    for (size_t w = 0; w < dump.produced.size(); ++w) {
        out << "\n  worker " << w << ": produced " << dump.produced[w] << " ready " << dump.ready[w] << " consumed "
            << dump.slots[w].consumed << " credited " << dump.slots[w].credited << " phase_done "
            << dump.slots[w].phase_done << " warmup_done " << dump.slots[w].warmup_done;
    }
    return out.str();
}

// For a caller that needs only the rendered form.
inline std::string describe_lim_slots(x280::X280Tile& tile, const pi_layout_t& layout, uint32_t P) {
    return describe_lim_slots(read_lim_slots(tile, layout, P), layout);
}

[[noreturn]] inline void retain_failed_run(const pi_failure_t& failure) {
    std::cerr << "FATAL: Packet injection run entered unrecoverable state (code=" << failure.code
              << ", phase=" << failure.phase << ", worker=" << failure.worker_index << ", actor=" << failure.actor
              << "). Retaining process to prevent device reset or resource corruption.\n";
    while (true) {
        std::this_thread::sleep_for(std::chrono::hours(24));
    }
}

/**
 * Zeroes the whole packet arena on `tile`, then writes `xargs` at the arena's
 * args offset and `tile_coords` (the release selector's translated coordinate
 * per ordered tile) at its tile-coordinate offset. Must precede the consumer launch. Throws if the
 * layout's table span cannot hold every record, or on a LIM write failure.
 */
inline void prime_lim_arena(
    x280::X280Tile& tile,
    const pi_layout_t& L,
    const pi_x280_args_t& xargs,
    const std::vector<x280::TranslatedCoord>& tile_coords) {
    const uint32_t T = static_cast<uint32_t>(tile_coords.size());

    // Prime the whole arena, then write args and the tile coordinates. The args
    // record must land at the arena base, and that same address is published as
    // the hart args block at launch: the compact schema transmits no arena base, so
    // X280 takes the base from its args pointer and rejects the run if the two
    // disagree. Zeroing after this point would erase them.
    std::vector<uint8_t> zeros(L.arena_bytes, 0);
    tile.write_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE, zeros);

    std::vector<uint8_t> args_bytes(sizeof(pi_x280_args_t));
    std::memcpy(args_bytes.data(), &xargs, sizeof(pi_x280_args_t));
    tile.write_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.args_offset, args_bytes);

    // Written as the whole span the layout reserves for the table rather than as
    // 8*T bytes: write_available_lim admits only whole cache lines, and 8*T is a
    // multiple of 64 only when T is a multiple of 8. Sizing from the layout
    // keeps the write aligned by construction, and the tail beyond the T records
    // stays zero.
    const size_t coord_records_bytes = static_cast<size_t>(T) * sizeof(x280_noc_coord_t);
    if (L.tile_coords_bytes < coord_records_bytes) {
        throw std::runtime_error(
            "layout reserves " + std::to_string(L.tile_coords_bytes) +
            " bytes for the tile coordinate table, too few for the " + std::to_string(coord_records_bytes) +
            " bytes of records at T=" + std::to_string(T));
    }
    std::vector<uint8_t> coord_bytes(static_cast<size_t>(L.tile_coords_bytes), 0);
    for (uint32_t t = 0; t < T; ++t) {
        const x280_noc_coord_t record = tile_coords[t].noc();
        std::memcpy(coord_bytes.data() + t * sizeof(record), &record, sizeof(record));
    }
    tile.write_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.tile_coords_offset, coord_bytes);
}

/** Grace given to a recovery cleanup, and the poll window it extends to. */
inline constexpr auto kRecoveryGrace = std::chrono::milliseconds(30000);

/**
 * The X280 consumer session of one run and what polling it has observed.
 *
 * start_recovery() runs a cleanup thread that holds references into this
 * object and into the Fixture, so it is neither copyable nor movable and the
 * Fixture must outlive it. The destructor joins a still-joinable `recovery`, so
 * an exception unwinding past this object waits for that cleanup, bounded by
 * kRecoveryGrace, and reaches the caller's handler instead of terminating.
 */
struct ConsumerRun {
    x280::SessionId session_id{};
    std::chrono::steady_clock::time_point normal_deadline{};

    std::thread recovery;
    std::optional<x280::CleanupResult> recovery_cleanup;
    bool recovery_started = false;
    std::chrono::steady_clock::time_point recovery_deadline = std::chrono::steady_clock::time_point::max();

    bool quiescent_seen = false;
    bool argument_rejection = false;
    uint64_t rejected_result = PI_ERROR_NONE;
    uint32_t failed_worker = UINT32_MAX;
    pi_run_result_t pre_ack_run_res{};

    ConsumerRun() = default;
    ConsumerRun(const ConsumerRun&) = delete;
    ConsumerRun& operator=(const ConsumerRun&) = delete;
    ~ConsumerRun() {
        if (recovery.joinable()) {
            recovery.join();
        }
    }

    /** The host deadline in force: the recovery grace once recovery started. */
    std::chrono::steady_clock::time_point deadline() const {
        return recovery_started ? recovery_deadline : normal_deadline;
    }
};

/**
 * Enqueues the X280 consumer image described by `launch` on the fixture's tile
 * and records its session in `run`. Throws if the runtime refuses it, after
 * cleaning up any session the refused enqueue created.
 */
inline void enqueue_consumer(
    x280::harness::session::Fixture& f, const x280::harness::execution::X280Launch& launch, ConsumerRun& run) {
    const auto outcome = x280::harness::execution::enqueue_x280(f.handle(), launch);
    if (!outcome.enqueued()) {
        throw std::runtime_error("X280 packet image enqueue failed: " + x280::describe(outcome.enqueue));
    }
    run.session_id = outcome.enqueue.session;
}

/**
 * Starts the recovery cleanup of `run`'s session on a background thread, with
 * kRecoveryGrace, and moves the deadline to its end. A second call does
 * nothing.
 */
inline void start_recovery(x280::harness::session::Fixture& f, ConsumerRun& run) {
    if (run.recovery_started) {
        return;
    }
    run.recovery_started = true;
    run.recovery_deadline = std::chrono::steady_clock::now() + kRecoveryGrace;
    run.recovery =
        std::thread([&f, &run]() { run.recovery_cleanup = f.runtime->cleanup(run.session_id, kRecoveryGrace); });
}

/**
 * Polls the run result until X280 reports data quiescence, or until every
 * producer has drained after X280 rejected its arguments, under
 * `normal_deadline` (extended by recovery once a producer reports failure or
 * the deadline passes). On return exactly one of `run.quiescent_seen` and
 * `run.argument_rejection` holds, and a quiescent export record has been
 * matched against `abi_tag`, `run_value` and `P`.
 *
 * Throws on a short LIM read, on an X280 failure with no collectable export,
 * or on a mismatched export record. Never returns when neither is reached: the
 * run is retained with the device's last failure.
 */
inline void await_quiescence(
    x280::harness::session::Fixture& f,
    const pi_layout_t& L,
    uint32_t P,
    uint32_t abi_tag,
    uint64_t run_value,
    std::chrono::steady_clock::time_point normal_deadline,
    ConsumerRun& run) {
    run.normal_deadline = normal_deadline;
    pi_run_result_t& pre_ack_run_res = run.pre_ack_run_res;

    // Poll for the quiescence that authorizes collection, under host deadline
    while (std::chrono::steady_clock::now() < run.deadline()) {
        auto raw_res = f.tile->read_available_lim(
            X280_LIM_PACKET_INJECTION_ARENA_BASE + L.run_result_offset, sizeof(pi_run_result_t));
        if (raw_res.size() != sizeof(pi_run_result_t)) {
            throw std::runtime_error(
                "short read of run result: expected " + std::to_string(sizeof(pi_run_result_t)) + " got " +
                std::to_string(raw_res.size()));
        }
        {
            std::memcpy(&pre_ack_run_res, raw_res.data(), sizeof(pi_run_result_t));
            if (pre_ack_run_res.data_quiescent == 1u) {
                run.quiescent_seen = true;
                break;
            }
        }
        const auto harts = f.runtime->session(run.session_id).inspect();
        if (x280::active_all_results_ready(harts, run_value)) {
            run.rejected_result = harts[0].state.result;
            if (run.rejected_result != PI_ERROR_BAD_ARGS && run.rejected_result != PI_ERROR_MODE_MISMATCH &&
                run.rejected_result != PI_ERROR_BAD_PLACEMENT && run.rejected_result != PI_ERROR_CAPACITY) {
                // X280 failed first and the producers have not drained, so its
                // own error is the only thing that explains the run. This path
                // collects nothing, so it is also the only chance to see the LIM
                // slots for this failure: once the retained process is killed the
                // tile is left out of reset and cannot be reopened to read them.
                throw std::runtime_error(
                    "X280 returned without a collectable export: hart result " + std::to_string(run.rejected_result) +
                    ", run result state " + std::to_string(pre_ack_run_res.state) + " failure " +
                    describe_failure(pre_ack_run_res.failure) + ", data_quiescent " +
                    std::to_string(pre_ack_run_res.data_quiescent) + "; " + describe_lim_slots(*f.tile, L, P));
            }
            const size_t ready_bytes = static_cast<size_t>(P) * sizeof(pi_ready_slot_t);
            auto ready = f.tile->read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.ready_offset, ready_bytes);
            if (ready.size() != ready_bytes) {
                throw std::runtime_error(
                    "short read of producer readiness: expected " + std::to_string(ready_bytes) + " got " +
                    std::to_string(ready.size()));
            }
            bool drained = true;
            for (uint32_t w = 0; w < P; ++w) {
                uint32_t value = 0;
                std::memcpy(&value, ready.data() + static_cast<size_t>(w) * sizeof(pi_ready_slot_t), sizeof(value));
                drained &= value == PI_READY_DONE || value == PI_READY_FAILED;
            }
            if (drained) {
                run.argument_rejection = true;
                break;
            }
        }

        const size_t ready_bytes = static_cast<size_t>(P) * sizeof(pi_ready_slot_t);
        auto ready = f.tile->read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.ready_offset, ready_bytes);
        if (ready.size() != ready_bytes) {
            throw std::runtime_error(
                "short read of producer completion state: expected " + std::to_string(ready_bytes) + " got " +
                std::to_string(ready.size()));
        }
        for (uint32_t w = 0; w < P; ++w) {
            uint32_t value = 0u;
            std::memcpy(&value, ready.data() + static_cast<size_t>(w) * sizeof(pi_ready_slot_t), sizeof(value));
            if (value == PI_READY_FAILED) {
                run.failed_worker = w;
                start_recovery(f, run);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    if (!run.quiescent_seen && !run.argument_rejection) {
        if (!run.recovery_started) {
            start_recovery(f, run);
            while (std::chrono::steady_clock::now() < run.recovery_deadline) {
                auto raw_res = f.tile->read_available_lim(
                    X280_LIM_PACKET_INJECTION_ARENA_BASE + L.run_result_offset, sizeof(pi_run_result_t));
                if (raw_res.size() != sizeof(pi_run_result_t)) {
                    throw std::runtime_error("short read of run result during recovery");
                }
                std::memcpy(&pre_ack_run_res, raw_res.data(), sizeof(pi_run_result_t));
                if (pre_ack_run_res.data_quiescent == 1u) {
                    run.quiescent_seen = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    if (!run.quiescent_seen && !run.argument_rejection) {
        // Retain the device's own last-published failure where it has one: a
        // generic host timeout would discard the only record of the cause.
        const pi_failure_t f_fail =
            pre_ack_run_res.failure.code != PI_ERROR_NONE ? pre_ack_run_res.failure
            : run.failed_worker != UINT32_MAX
                ? pi_failure_t{PI_ERROR_PEER_FAILED, PI_PHASE_WAIT_MEASURE, run.failed_worker, PI_ACTOR_HOST}
                : pi_failure_t{PI_ERROR_TIMEOUT, PI_PHASE_EXPORT, UINT32_MAX, PI_ACTOR_HOST};
        std::cerr << "Host deadline reached without device quiescence: run result state " << pre_ack_run_res.state
                  << " failure " << describe_failure(pre_ack_run_res.failure) << ", data_quiescent "
                  << pre_ack_run_res.data_quiescent << '\n'
                  << describe_lim_slots(*f.tile, L, P) << '\n';
        retain_failed_run(f_fail);
    }

    if (!run.argument_rejection && (pre_ack_run_res.abi_tag != abi_tag || pre_ack_run_res.run_value != run_value ||
                                    pre_ack_run_res.worker_count != P)) {
        std::ostringstream why;
        why << "invalid export record ahead of collection: abi_tag 0x" << std::hex << pre_ack_run_res.abi_tag
            << " expected 0x" << abi_tag << std::dec << ", run_value " << pre_ack_run_res.run_value << " expected "
            << run_value << ", worker_count " << pre_ack_run_res.worker_count << " expected " << P << ", state "
            << pre_ack_run_res.state << " failure " << describe_failure(pre_ack_run_res.failure);
        throw std::runtime_error(why.str());
    }
}

/**
 * Runs `collect` on its own thread and waits for it until `run`'s host
 * deadline. `collect` is the target's readback once its producers' workload
 * has finished -- a Finish that may block without bound, then producer state
 * -- and writes only state the caller owns.
 *
 * Returns once `collect` returns, and rethrows what it threw after joining it.
 * When the deadline passes first, retains the run with a host WAIT_COLLECT
 * timeout and never returns, so `collect` keeps running against caller state
 * that the retention keeps alive. It never returns or throws with the thread
 * still joinable.
 */
template <typename Collect>
inline void collect_within_deadline(const ConsumerRun& run, Collect&& collect) {
    std::mutex m;
    std::condition_variable cv;
    bool finished = false;
    std::exception_ptr err = nullptr;
    std::thread collector([&]() {
        try {
            collect();
            {
                std::lock_guard<std::mutex> lk(m);
                finished = true;
            }
            cv.notify_one();
        } catch (...) {
            std::lock_guard<std::mutex> lk(m);
            err = std::current_exception();
            finished = true;
            cv.notify_one();
        }
    });

    {
        std::unique_lock<std::mutex> lk(m);
        const auto deadline = run.deadline();
        const auto now = std::chrono::steady_clock::now();
        const auto remaining = deadline > now ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                                              : std::chrono::milliseconds(0);
        if (!cv.wait_for(lk, remaining, [&]() { return finished; })) {
            lk.unlock();  // Release lock before infinite retention to prevent deadlock
            retain_failed_run(pi_failure_t{PI_ERROR_TIMEOUT, PI_PHASE_WAIT_COLLECT, UINT32_MAX, PI_ACTOR_HOST});
        }
    }
    collector.join();

    if (err) {
        std::rethrow_exception(err);
    }
}

/**
 * Cleans up a run whose X280 consumer rejected its arguments, sets
 * `resources_safe` once the tile is back in resident idle, and always throws:
 * the cleanup failure if cleanup did not return the tile healthy, otherwise
 * the rejection itself.
 */
[[noreturn]] inline void reject_arguments(
    x280::harness::session::Fixture& f, const ConsumerRun& run, bool& resources_safe) {
    const auto clean = f.runtime->cleanup(run.session_id, std::chrono::milliseconds(5000));
    if (!clean.cleaned_up() || !f.tile->healthy()) {
        throw std::runtime_error("argument rejection cleanup failed: " + x280::describe(clean));
    }
    resources_safe = true;
    throw std::runtime_error(
        "X280 rejected benchmark arguments: hart result " + std::to_string(run.rejected_result) +
        ", run result failure " + describe_failure(run.pre_ack_run_res.failure));
}

/**
 * The latency export as 64-bit samples, tiled by `L.latency_sample_stride` per
 * worker; empty outside PI_MODE_LATENCY or when the layout reserves none.
 * Throws on a short read.
 */
inline std::vector<uint64_t> read_latency_export(x280::X280Tile& tile, uint32_t mode, const pi_layout_t& L) {
    std::vector<uint64_t> export_samples;
    if (mode == PI_MODE_LATENCY && L.latency_export_bytes > 0) {
        auto raw_exp = tile.read_available_lim(
            X280_LIM_PACKET_INJECTION_ARENA_BASE + L.latency_export_offset, L.latency_export_bytes);
        if (raw_exp.size() != L.latency_export_bytes) {
            throw std::runtime_error(
                "short read of latency export samples: expected " + std::to_string(L.latency_export_bytes) + " got " +
                std::to_string(raw_exp.size()));
        }
        export_samples.resize(L.latency_export_bytes / sizeof(uint64_t));
        std::memcpy(export_samples.data(), raw_exp.data(), L.latency_export_bytes);
    }
    return export_samples;
}

/** The P X280-owned consumer slots, in worker order. Throws on a short read. */
inline std::vector<pi_consumer_slot_t> read_consumer_slots(x280::X280Tile& tile, const pi_layout_t& L, uint32_t P) {
    std::vector<pi_consumer_slot_t> consumer_slots(P);
    const size_t consumer_bytes = static_cast<size_t>(P) * sizeof(pi_consumer_slot_t);
    auto raw_slots =
        tile.read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.consumer_slots_offset, consumer_bytes);
    if (raw_slots.size() != consumer_bytes) {
        throw std::runtime_error(
            "short read of consumer slots: expected " + std::to_string(consumer_bytes) + " got " +
            std::to_string(raw_slots.size()));
    }
    std::memcpy(consumer_slots.data(), raw_slots.data(), consumer_bytes);
    return consumer_slots;
}

/**
 * Rejects worker `w`'s producer result and consumer slot when the result is
 * from another run or not quiescent, or when any count exceeds the requested
 * W or N. Throws naming the worker.
 */
inline void check_worker_counts(
    const x280::harness::regions::packet_injection::Config& cfg,
    uint32_t w,
    const pi_producer_result_t& result,
    const pi_consumer_slot_t& slot,
    uint32_t nonce) {
    if (result.run_nonce != nonce) {
        throw std::runtime_error(
            "stale producer result for worker " + std::to_string(w) + ": run_nonce " +
            std::to_string(result.run_nonce) + " expected " + std::to_string(nonce));
    }
    if (result.quiescent != 1 || result.measured_completed > cfg.N || result.warmup_completed > cfg.W ||
        result.latency_count > cfg.N || result.stall_count > cfg.N || result.rtt_count > cfg.N ||
        result.measured_counter_writes > cfg.N) {
        throw std::runtime_error("invalid producer result/count for worker " + std::to_string(w));
    }
    if (slot.phase_done > cfg.N || slot.warmup_done > cfg.W) {
        throw std::runtime_error(
            "invalid consumer slot count for worker " + std::to_string(w) + ": phase_done " +
            std::to_string(slot.phase_done) + " warmup_done " + std::to_string(slot.warmup_done) + " against N " +
            std::to_string(cfg.N) + " W " + std::to_string(cfg.W));
    }
}

/**
 * Acknowledges collection by writing `run_value` into the host control line,
 * which releases X280 into cleanup. Call only after every producer result and
 * the export have been read.
 */
inline void acknowledge_collection(x280::X280Tile& tile, const pi_layout_t& L, uint64_t run_value) {
    // Acknowledge collection with the active run value. The line is written
    // field-wise: pi_host_control_t holds a volatile member and is not
    // trivially copyable.
    std::vector<uint8_t> hctrl_bytes(sizeof(pi_host_control_t), 0);
    std::memcpy(hctrl_bytes.data() + offsetof(pi_host_control_t, collect_ack_run_value), &run_value, sizeof(run_value));
    tile.write_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.host_control_offset, hctrl_bytes);
}

/** The consumer's terminal hart states and its verified-clean run result. */
struct ConsumerTerminal {
    std::array<x280::HartState, x280::kHartsPerTile> harts{};
    pi_run_result_t run_res{};
};

/**
 * Waits for the acknowledged consumer to finish (joining the recovery cleanup
 * when one started), verifies from the run result that X280 cleared its
 * windows, restored its cache and stayed quiescent for this run, reads each
 * window a `mode` run opens back cleared (a test failure, not a throw), then returns
 * the tile to resident idle and sets `resources_safe`. Throws on any failed
 * wait, cleanup or check.
 */
inline ConsumerTerminal finish_consumer(
    x280::harness::session::Fixture& f,
    ConsumerRun& run,
    const pi_layout_t& L,
    uint32_t mode,
    uint32_t abi_tag,
    uint64_t run_value,
    uint32_t P,
    bool& resources_safe) {
    ConsumerTerminal terminal_state{};
    auto& terminal_harts = terminal_state.harts;
    if (run.recovery_started) {
        run.recovery.join();
        if (!run.recovery_cleanup || !run.recovery_cleanup->cleaned_up() || !f.tile->healthy()) {
            throw std::runtime_error(
                "X280 recovery cleanup failed: " +
                (run.recovery_cleanup ? x280::describe(*run.recovery_cleanup) : std::string("no cleanup result")));
        }
        terminal_harts = run.recovery_cleanup->hartstates;
        resources_safe = true;
    } else {
        const auto terminal =
            f.runtime->session(run.session_id).wait_until_result_ready(std::chrono::milliseconds(5000));
        if (terminal.outcome != x280::WaitOutcome::ResultReady) {
            throw std::runtime_error("X280 terminal state unknown: " + x280::describe(terminal));
        }
        terminal_harts = terminal.hartstates;
    }

    // Verify device cleanup before authorizing the active harness to reclaim it.
    pi_run_result_t& run_res = terminal_state.run_res;
    auto raw_terminal_res =
        f.tile->read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.run_result_offset, sizeof(pi_run_result_t));
    if (raw_terminal_res.size() != sizeof(pi_run_result_t)) {
        throw std::runtime_error(
            "short read of terminal run_result: expected " + std::to_string(sizeof(pi_run_result_t)) + " got " +
            std::to_string(raw_terminal_res.size()));
    }
    std::memcpy(&run_res, raw_terminal_res.data(), sizeof(pi_run_result_t));

    if (run_res.abi_tag != abi_tag || run_res.run_value != run_value || run_res.worker_count != P ||
        run_res.windows_cleared != 1 || run_res.cache_restored != 1 || run_res.data_quiescent != 1) {
        std::ostringstream why;
        why << "X280 cleanup is incomplete or unverified: abi_tag 0x" << std::hex << run_res.abi_tag << " expected 0x"
            << abi_tag << std::dec << ", run_value " << run_res.run_value << " expected " << run_value
            << ", worker_count " << run_res.worker_count << " expected " << P << ", windows_cleared "
            << run_res.windows_cleared << " cache_restored " << run_res.cache_restored << " data_quiescent "
            << run_res.data_quiescent << ", state " << run_res.state << " failure "
            << describe_failure(run_res.failure);
        throw std::runtime_error(why.str());
    }
    for (const uint32_t slot : x280::harness::regions::packet_injection::consumer_window_slots(mode, L.tile_count)) {
        x280::harness::execution::expect_window_cleared(*f.tile, slot);
    }
    if (!run.recovery_started) {
        const auto clean_res = f.runtime->cleanup(run.session_id, std::chrono::milliseconds(5000));
        if (!clean_res.cleaned_up() || !f.tile->healthy()) {
            throw std::runtime_error("active harness cleanup failed: " + x280::describe(clean_res));
        }
        resources_safe = true;
    }
    return terminal_state;
}

/**
 * For a run X280 reports successful: requires the consumer total to be P*N and
 * every worker to have completed N measured requests cleanly, with producer
 * and consumer counts agreeing, and in ceil(N/D) to N produced-counter writes,
 * each publishing at most D = min(B, S) headers. P is `producer_res.size()`,
 * which `consumer_slots` must match. Throws naming the first disagreement.
 */
inline void check_successful_run(
    const x280::harness::regions::packet_injection::Config& cfg,
    const pi_run_result_t& run_res,
    const std::vector<pi_producer_result_t>& producer_res,
    const std::vector<pi_consumer_slot_t>& consumer_slots) {
    if (consumer_slots.size() != producer_res.size()) {
        throw std::runtime_error(
            "successful-run check given " + std::to_string(producer_res.size()) + " producer results but " +
            std::to_string(consumer_slots.size()) + " consumer slots");
    }
    const uint32_t P = static_cast<uint32_t>(producer_res.size());
    if (run_res.measured_completed != uint64_t(P) * cfg.N) {
        throw std::runtime_error("consumer total does not match requested workload");
    }
    const uint64_t D = std::min(cfg.B, cfg.S);
    const uint64_t min_counter_writes = cfg.N / D + (cfg.N % D != 0u ? 1u : 0u);
    for (uint32_t w = 0; w < P; ++w) {
        if (producer_res[w].measured_completed != cfg.N || producer_res[w].phase != PI_PHASE_DONE ||
            producer_res[w].failure.code != PI_ERROR_NONE ||
            (cfg.mode == PI_MODE_LATENCY && producer_res[w].latency_count != cfg.N)) {
            throw std::runtime_error("incomplete successful worker " + std::to_string(w));
        }
        if (producer_res[w].measured_counter_writes < min_counter_writes ||
            producer_res[w].measured_counter_writes > cfg.N) {
            throw std::runtime_error(
                "worker " + std::to_string(w) + " issued " + std::to_string(producer_res[w].measured_counter_writes) +
                " produced-counter writes for N " + std::to_string(cfg.N) + " at D " + std::to_string(D) +
                ", outside [" + std::to_string(min_counter_writes) + ", " + std::to_string(cfg.N) + "]");
        }
        // Producer and consumer counted the same traffic independently.
        if (consumer_slots[w].phase_done != producer_res[w].measured_completed ||
            consumer_slots[w].warmup_done != producer_res[w].warmup_completed) {
            throw std::runtime_error(
                "worker " + std::to_string(w) + " progress cross-check failed: consumer measured " +
                std::to_string(consumer_slots[w].phase_done) + " warmup " +
                std::to_string(consumer_slots[w].warmup_done) + ", producer measured " +
                std::to_string(producer_res[w].measured_completed) + " warmup " +
                std::to_string(producer_res[w].warmup_completed));
        }
    }
}

/**
 * Requires X280's export byte count to match the consumer progress read back,
 * one slot per worker in `consumer_slots`, and to fit the layout's export
 * region. Throws with both sides otherwise.
 */
inline void check_export_accounting(
    const x280::harness::regions::packet_injection::Config& cfg,
    const pi_layout_t& L,
    const pi_run_result_t& run_res,
    const std::vector<pi_consumer_slot_t>& consumer_slots) {
    // X280 accounts its export as 8 bytes per completed measured request, so it
    // must agree with the consumer progress the host read back and stay inside
    // the region the layout reserved. A run that never reached the measured
    // phase exports nothing even though `phase_done` is non-zero, because each
    // phase boundary rearms that counter and it still holds warmup progress.
    // The retained failure phase is what separates the two cases: every failure
    // recorded before the measured phase begins carries a phase below
    // PI_PHASE_MEASURE, and the first failure is the one kept.
    const bool measured_started = run_res.failure.code == PI_ERROR_NONE || run_res.failure.phase >= PI_PHASE_MEASURE;
    uint64_t exported_bytes = 0;
    if (cfg.mode == PI_MODE_LATENCY && measured_started) {
        for (const pi_consumer_slot_t& slot : consumer_slots) {
            exported_bytes += 8ULL * slot.phase_done;
        }
    }
    if (run_res.export_valid_bytes != exported_bytes || run_res.export_valid_bytes > L.latency_export_bytes) {
        throw std::runtime_error(
            "X280 export accounting (" + std::to_string(run_res.export_valid_bytes) +
            " bytes) disagrees with consumer progress (" + std::to_string(exported_bytes) + " bytes, measured phase " +
            (measured_started ? "started" : "not started") + ", failure phase " +
            std::to_string(run_res.failure.phase) + ") or exceeds the reserved export region (" +
            std::to_string(L.latency_export_bytes) + " bytes)");
    }
}

/**
 * The first worker whose throughput wall or CPU interval is invalid, or
 * `producer_res.size()` when every one is valid. Each interval is one
 * producer's own; nothing is compared across workers.
 */
inline uint32_t first_invalid_interval(const std::vector<pi_producer_result_t>& producer_res) {
    const uint32_t P = static_cast<uint32_t>(producer_res.size());
    for (uint32_t w = 0; w < P; ++w) {
        if (!x280::harness::output::throughput_interval_valid(
                producer_res[w].throughput_start, producer_res[w].throughput_end) ||
            !x280::harness::output::throughput_interval_valid(producer_res[w].cpu_start, producer_res[w].cpu_end)) {
            return w;
        }
    }
    return P;
}

/** Both aggregate endpoint pairs of `run_res`, for a host diagnostic. */
inline std::string describe_aggregate_interval(const pi_run_result_t& run_res) {
    return "mtime " + std::to_string(run_res.aggregate_start_mtime) + ".." +
           std::to_string(run_res.aggregate_end_mtime) + ", rdcycle " +
           std::to_string(run_res.aggregate_start_rdcycle) + ".." + std::to_string(run_res.aggregate_end_rdcycle);
}

}  // namespace x280::suites::packet_injection

#endif  // X280_SUITES_PACKET_INJECTION_COMMON_HPP
