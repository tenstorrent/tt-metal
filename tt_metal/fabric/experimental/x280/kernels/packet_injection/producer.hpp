// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/packet_injection/producer.hpp
//
// Producer ring loop, phase orchestration, credit observation and state
// machine, shared by every producer target through its Transport.

#ifndef X280_KERNELS_PACKET_INJECTION_PRODUCER_HPP
#define X280_KERNELS_PACKET_INJECTION_PRODUCER_HPP

#include <stdint.h>
#include <stddef.h>

#include "packet_injection_abi.h"
#include "packet_injection_slot.hpp"
#include "../shared/core_sync_barrier.hpp"
#include "../shared/core_sync_transport_metal.hpp"
#include "../../include/rt/measurement.h"
#include "../../include/x280_address_map.h"

namespace packet_injection {

#if !defined(PACKET_INJECTION_MODE) || !defined(PACKET_INJECTION_INCLUDE_CREATION)
#error "PACKET_INJECTION_MODE and PACKET_INJECTION_INCLUDE_CREATION must be explicitly defined"
#endif

#if (PACKET_INJECTION_MODE != 1) && (PACKET_INJECTION_MODE != 2) && (PACKET_INJECTION_MODE != 3)
#error "PACKET_INJECTION_MODE must be 1 (LATENCY), 2 (THROUGHPUT), or 3 (BACKPRESSURE)"
#endif

#if (PACKET_INJECTION_INCLUDE_CREATION != 0) && (PACKET_INJECTION_INCLUDE_CREATION != 1)
#error "PACKET_INJECTION_INCLUDE_CREATION must be 0 or 1"
#endif

#if (PACKET_INJECTION_MODE == 2) && (PACKET_INJECTION_INCLUDE_CREATION == 1)
#error "Throughput mode requires PACKET_INJECTION_INCLUDE_CREATION=0"
#endif

// X280_NOC_COORD_MAX states the bound; this is the half that makes it
// checkable, and only a device build sees the NoC parameters to check it against.
#if defined(NOC_ADDR_NODE_ID_BITS)
static_assert(
    X280_NOC_COORD_MAX == (1u << NOC_ADDR_NODE_ID_BITS) - 1u,
    "peer coordinate bound must match the NoC address node-id field");
#endif

static_assert(
    PI_TIMER_MAX_ATTEMPTS > 0u && PI_TIMER_MAX_ATTEMPTS <= 1024u,
    "live-timer retry budget must be non-zero and bounded");

// A NoC write carries a 32-bit destination, and pi_compute_layout() bounds
// arena_bytes by PI_ARENA_BYTES, so every LIM address below narrows safely.
static_assert(
    X280_LIM_PACKET_INJECTION_ARENA_BASE + PI_ARENA_BYTES <= 0x100000000ULL,
    "the packet arena must end inside a 32-bit NoC destination");

/* Image-local worker binding, populated only after the complete packet layout
 * and peer destination have been validated. */
struct PacketWorkerSyncConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = false;
    static inline uint32_t kLocalReleaseAddr = 0u;
    static inline core_sync::MetalWordArrivalTarget kArrivalTarget{};
};

/**
 * One packet-injection producer: runs warmup, the core-sync release and the
 * measured phase against its own SPSC ring, then publishes its result.
 *
 * `Transport` is the target binding, all static and all inlined:
 *
 * - `kNoc`: the NoC index every write and barrier names.
 * - `kRisc`: the RISC index pi_worker_risc_index() must return for this
 *   image's worker; checked only in a device build.
 * - `kL1SpanLimit`: bytes of core-local L1 a worker's tile span must end
 *   within.
 * - `kActor`: the pi_actor_t recorded in a failure this producer publishes.
 * - `prepare_header(header, dst_chip, dst_mesh)`: readies the staged 96-byte
 *   header before each request is issued.
 * - `invalidate_cache()`, `barrier(noc, budget)`, `issue_header_write(...)`,
 *   `issue_u32_write(...)`: L1 visibility and NoC issue/drain; `barrier`
 *   returns false once `budget` ticks of the `read_timer` clock pass.
 * - `read_timer(max_attempts, ticks)`: the native wall counter that
 *   `timeout_ticks` and the throughput interval are measured in; returns zero
 *   when the read cannot be made.
 * - `read_cpu(ticks)`: native per-RISC cycles; returns zero when unsupported.
 *
 * `SyncTransport` is the core_sync transport the post-warmup barrier runs on.
 */
template <uint32_t Mode, uint32_t IncludeCreation, typename Transport, typename SyncTransport>
class Producer {
public:
    /**
     * Binds one worker. Its L1 records come from pi_worker_l1_base() and its
     * LIM destinations from the arena base plus a pi_layout_t offset; nothing
     * is dereferenced until run() has validated the args.
     *
     * `host_l1_override` replaces the derived L1 slice address, so a host test
     * can place the slice in its own buffer, and defaults to zero, meaning the
     * derived slice the firmware owns.
     *
     * The arena base takes no such override on purpose. These destinations are
     * uint32_t values handed to Transport:: writes and are never dereferenced
     * here, so a mock asserts on the address directly and needs no relocation
     * to do it; a host buffer pointer would not even fit one.
     */
    explicit Producer(const pi_producer_args_t& args, uintptr_t host_l1_override = 0) :
        args_(args), host_l1_override_(host_l1_override) {}

    pi_error_t run() {
        pi_error_t err = derive_l1_spans();
        if (err != PI_ERROR_NONE) {
            return err;
        }

        // Identity is in L1 from here on, so every later failure is legible to
        // the host even when it cannot be announced over the NoC.
        initialize_l1_state();

        err = derive_peer_spans();
        if (err != PI_ERROR_NONE) {
            // The ready slot this would announce on is the thing in question.
            record_failure(err, PI_PHASE_INIT);
            return err;
        }

        err = validate_controls();
        if (err != PI_ERROR_NONE) {
            fail_and_publish(err, PI_PHASE_INIT);
            return err;
        }

        err = run_phase<false>(W_);
        if (err != PI_ERROR_NONE) {
            fail_and_publish(err, PI_PHASE_WARMUP);
            return err;
        }

        const core_sync::Outcome sync =
            core_sync::core_sync_barrier<PacketWorkerSyncConfig, SyncTransport>::sync(0u, 0u, args_.run_nonce);
        if (sync != core_sync::Outcome::Success) {
            err = sync == core_sync::Outcome::StopRequested ? PI_ERROR_STOPPED : PI_ERROR_LAYOUT_INVARIANT;
            fail_and_publish(err, PI_PHASE_WAIT_MEASURE);
            return err;
        }

        err = check_control();
        if (err != PI_ERROR_NONE) {
            fail_and_publish(err, PI_PHASE_WAIT_MEASURE);
            return err;
        }

        // Earliest point at which the measured phase has been granted, and
        // still before any measured header exists.
        if constexpr (Mode == (uint32_t)PI_MODE_THROUGHPUT) {
            if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &throughput_start_) ||
                !Transport::read_cpu(&cpu_start_)) {
                fail_and_publish(PI_ERROR_TIMER_RETRY_EXHAUSTED, PI_PHASE_MEASURE);
                return PI_ERROR_TIMER_RETRY_EXHAUSTED;
            }
        }

        err = run_phase<true>(N_);
        if (err != PI_ERROR_NONE) {
            fail_and_publish(err, PI_PHASE_MEASURE);
            return err;
        }

        publish_final_results();
        if (!publish_ready(PI_READY_DONE)) {
            record_failure(PI_ERROR_TIMEOUT, PI_PHASE_DRAIN);
            return PI_ERROR_TIMEOUT;
        }
        return PI_ERROR_NONE;
    }

    pi_error_t publish_pending() {
        if (pending_ == 0u) {
            return PI_ERROR_NONE;
        }
        // Orders the counter behind the batch's slot writes, so the consumer
        // cannot see produced == r_ before the data, and drains the previous
        // publication before its source word is rewritten below. That holds
        // only if the L2CPU applies accepted writes to LIM in arrival order
        // (PI_ASSUMED_INBOUND).
        if (!drain_writes()) {
            return PI_ERROR_TIMEOUT;
        }
        *stable_produced_source_ = r_;
        Transport::issue_u32_write(
            worker_base_ + PI_L1_PRODUCED_SRC_OFFSET, args_.peer.x, args_.peer.y, produced_dst_, Transport::kNoc);
        p_ = r_;
        pending_ = 0u;
        counter_writes_ += 1;
        return PI_ERROR_NONE;
    }

    pi_error_t observe_credit(uint32_t& next, uint32_t& jump) {
        Transport::invalidate_cache();
        next = *consumed_mirror_;
        uint32_t occupied;
        if (!credit_pair_ok(r_, c_seen_, UINT32_MAX, S_, &occupied) ||
            !credit_pair_ok(next, c_seen_, UINT32_MAX, occupied, &jump)) {
            return PI_ERROR_COUNTER_DISTANCE;
        }
        return PI_ERROR_NONE;
    }

    pi_error_t retire(uint32_t next, uint32_t jump, bool emit_rtt, uint64_t optional_end) {
        if (jump == 0u) {
            return PI_ERROR_NONE;
        }
        if (retired_ + (uint64_t)jump > issued_) {
            return PI_ERROR_PHASE_OVERRUN;
        }

        if constexpr (Mode == 3) {
            if (emit_rtt) {
                if (rtt_count_ + (uint64_t)jump > N_) {
                    return PI_ERROR_CAPACITY;
                }
                for (uint32_t j = 0; j < jump; ++j) {
                    uint32_t start_idx = (c_seen_ + j) & (S_ - 1u);
                    const auto orig_start = starts_[start_idx];
                    uint64_t d = 0;
                    uint64_t cpu = 0;
                    if (!measurement_delta(orig_start.wall, optional_end, &d) ||
                        !measurement_delta(orig_start.cpu, cpu_retirement_end_, &cpu) || d > UINT32_MAX ||
                        cpu > UINT32_MAX) {
                        return PI_ERROR_TIMER_DELTA;
                    }
                    rtts_[rtt_count_++] = {static_cast<uint32_t>(d), static_cast<uint32_t>(cpu)};
                    pi_error_t err = amortized_safety_check();
                    if (err != PI_ERROR_NONE) {
                        return err;
                    }
                }
            }
        } else {
            (void)emit_rtt;
            (void)optional_end;
        }

        retired_ += (uint64_t)jump;
        c_seen_ = next;
        return PI_ERROR_NONE;
    }

    template <bool Measured>
    pi_error_t acquire_space_for_current_request() {
        if (pi_counter_distance(r_, c_seen_) < S_) {
            return PI_ERROR_NONE;
        }
        if (pi_counter_distance(r_, c_seen_) != S_) {
            return PI_ERROR_COUNTER_DISTANCE;
        }

        pi_error_t err = publish_pending();
        if (err != PI_ERROR_NONE) {
            return err;
        }

        uint32_t next = 0;
        uint32_t jump = 0;
        err = observe_credit(next, jump);
        if (err != PI_ERROR_NONE) {
            return err;
        }

        if (jump != 0u) {
            return retire(next, jump, false, 0);
        }

        uint64_t stall_start = 0;
        uint64_t cpu_stall_start = 0;
        if constexpr (Mode == 3 && Measured) {
            if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &stall_start) || !Transport::read_cpu(&cpu_stall_start)) {
                return PI_ERROR_TIMER_RETRY_EXHAUSTED;
            }
        }
        // Counted on entry, where the credit has just been seen absent, so the
        // increment overlaps the wait instead of delaying the next issue.
        if constexpr (Mode == (uint32_t)PI_MODE_THROUGHPUT && Measured) {
            stall_count_ += 1;
        }

        while (true) {
            Transport::invalidate_cache();
            next = *consumed_mirror_;
            if (next != c_seen_) {
                uint64_t end = 0;
                if constexpr (Mode == 3 && Measured) {
                    if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &end) ||
                        !Transport::read_cpu(&cpu_retirement_end_)) {
                        return PI_ERROR_TIMER_RETRY_EXHAUSTED;
                    }
                }
                uint32_t occupied;
                if (!credit_pair_ok(r_, c_seen_, UINT32_MAX, S_, &occupied) ||
                    !credit_pair_ok(next, c_seen_, UINT32_MAX, occupied, &jump)) {
                    return PI_ERROR_COUNTER_DISTANCE;
                }

                if constexpr (Mode == 3 && Measured) {
                    if (stall_count_ >= N_) {
                        return PI_ERROR_CAPACITY;
                    }
                    uint64_t stall_duration = 0;
                    uint64_t cpu_duration = 0;
                    if (!measurement_delta(stall_start, end, &stall_duration) ||
                        !measurement_delta(cpu_stall_start, cpu_retirement_end_, &cpu_duration) ||
                        stall_duration > UINT32_MAX || cpu_duration > UINT32_MAX) {
                        return PI_ERROR_TIMER_DELTA;
                    }
                    stalls_[stall_count_++] = {
                        static_cast<uint32_t>(stall_duration), static_cast<uint32_t>(cpu_duration)};
                    return retire(next, jump, true, end);
                } else {
                    return retire(next, jump, false, 0);
                }
            }
            err = amortized_safety_check();
            if (err != PI_ERROR_NONE) {
                return err;
            }
        }
    }

    template <bool Measured>
    pi_error_t run_phase(uint64_t limit) {
        issued_ = 0;
        retired_ = 0;
        if constexpr (Measured) {
            measured_started_ = true;
            stall_count_ = 0;
            rtt_count_ = 0;
            counter_writes_ = 0;
        }
        pi_error_t err = reset_phase_deadline();
        if (err != PI_ERROR_NONE) {
            return err;
        }
        if constexpr (Measured) {
            err = check_control();
            if (err != PI_ERROR_NONE) {
                return err;
            }
        }

        while (issued_ < limit) {
            if (pending_ >= D_) {
                return PI_ERROR_CAPACITY;
            }
            if (pi_counter_distance(r_, c_seen_) == S_ && pending_ != 0) {
                return PI_ERROR_COUNTER_DISTANCE;
            }
            pi_slot_t* source = &staging_[pending_];
            const uint32_t source_l1_addr = worker_base_ + PI_L1_STAGING_OFFSET + pending_ * PI_SLOT_BYTES;

            uint64_t pending_start = 0;
            uint64_t cpu_pending_start = 0;
            if constexpr ((Mode == 1 || Mode == 3) && IncludeCreation == 1) {
                if (Mode == 1 || Measured) {
                    if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &pending_start) ||
                        (Mode == 3 && !Transport::read_cpu(&cpu_pending_start))) {
                        return PI_ERROR_TIMER_RETRY_EXHAUSTED;
                    }
                }
            }

            Transport::prepare_header(&source->header, args_.dst_chip, args_.dst_mesh);

            if constexpr ((Mode == 1 || Mode == 3) && IncludeCreation == 0) {
                if (Mode == 1 || Measured) {
                    if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &pending_start) ||
                        (Mode == 3 && !Transport::read_cpu(&cpu_pending_start))) {
                        return PI_ERROR_TIMER_RETRY_EXHAUSTED;
                    }
                }
            }

            if constexpr (Mode == 1) {
                source->start = pending_start;
            }

            err = acquire_space_for_current_request<Measured>();
            if (err != PI_ERROR_NONE) {
                return err;
            }

            if constexpr (Mode == 3) {
                if (Measured) {
                    starts_[r_ & (S_ - 1u)] = {pending_start, cpu_pending_start};
                }
            }

            uint32_t slot_lim_addr = ring_base_ + (r_ & (S_ - 1u)) * PI_SLOT_BYTES;
            Transport::issue_header_write(source_l1_addr, args_.peer.x, args_.peer.y, slot_lim_addr, Transport::kNoc);

            r_ += 1u;
            pending_ += 1u;
            issued_ += 1u;

            if (pending_ == B_ || pi_counter_distance(r_, c_seen_) == S_ || issued_ == limit) {
                err = publish_pending();
                if (err != PI_ERROR_NONE) {
                    return err;
                }
            }

            err = amortized_safety_check();
            if (err != PI_ERROR_NONE) {
                return err;
            }
        }

        err = publish_pending();
        if (err != PI_ERROR_NONE) {
            return err;
        }
        if (!drain_writes()) {
            return PI_ERROR_TIMEOUT;
        }

        while (c_seen_ != r_) {
            uint32_t next = 0;
            uint32_t jump = 0;
            pi_error_t err = observe_credit(next, jump);
            if (err != PI_ERROR_NONE) {
                return err;
            }
            err = retire(next, jump, false, 0);
            if (err != PI_ERROR_NONE) {
                return err;
            }
            err = amortized_safety_check();
            if (err != PI_ERROR_NONE) {
                return err;
            }
        }

        // Every measured credit has now been observed; nothing after this point
        // belongs to the injection window.
        if constexpr (Mode == (uint32_t)PI_MODE_THROUGHPUT && Measured) {
            if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &throughput_end_) || !Transport::read_cpu(&cpu_end_)) {
                return PI_ERROR_TIMER_RETRY_EXHAUSTED;
            }
        }

        if (issued_ != limit || retired_ != limit || r_ != p_ || r_ != c_seen_) {
            return PI_ERROR_PHASE_OVERRUN;
        }

        return PI_ERROR_NONE;
    }

private:
    // Accepts the launch and turns it into every L1 span this producer touches.
    // Returns before a single one is dereferenced when any input is rejected.
    pi_error_t derive_l1_spans() {
        if (!pi_abi_tag_matches(args_.abi_tag, Mode, IncludeCreation)) {
            return PI_ERROR_MODE_MISMATCH;
        }
        const pi_layout_check_t check = pi_compute_layout(
            args_.worker_count, args_.tile_count, args_.S, args_.B, args_.N, Mode, args_.l1_tile_base, &layout_);
        if (check.code != PI_LAYOUT_OK) {
            return pi_layout_code_to_error(check.code);
        }
#if defined(__riscv)
        // The slice this image owns is the one its worker index maps to, so a
        // disagreement means the launch handed it another worker's args.
        if (pi_worker_risc_index(&layout_, args_.worker_index) != Transport::kRisc) {
            return PI_ERROR_BAD_PLACEMENT;
        }
#endif
        if (layout_.l1_tile_base + layout_.l1_tile_bytes > Transport::kL1SpanLimit) {
            return PI_ERROR_CAPACITY;
        }

        S_ = args_.S;
        B_ = args_.B;
        D_ = layout_.staging_depth;
        W_ = args_.W;
        N_ = args_.N;

        worker_base_ = pi_worker_l1_base(&layout_, args_.worker_index);
        const uintptr_t root = host_l1_override_ != 0 ? host_l1_override_ : (uintptr_t)worker_base_;
        consumed_mirror_ = (volatile uint32_t*)(root + PI_L1_CONSUMED_OFFSET);
        control_line_ = (volatile uint32_t*)(root + PI_L1_CONTROL_OFFSET);
        stable_produced_source_ = (volatile uint32_t*)(root + PI_L1_PRODUCED_SRC_OFFSET);
        stable_ready_source_ = (volatile uint32_t*)(root + PI_L1_READY_SRC_OFFSET);
        result_ = (pi_producer_result_t*)(root + PI_L1_RESULT_OFFSET);
        staging_ = (pi_slot_t*)(root + PI_L1_STAGING_OFFSET);
        starts_ = (pi_local_stamp_t*)(root + layout_.l1_starts_offset);
        stalls_ = (pi_local_sample_t*)(root + layout_.l1_stalls_offset);
        rtts_ = (pi_local_sample_t*)(root + layout_.l1_rtts_offset);

        return PI_ERROR_NONE;
    }

    // Accepts the peer, then turns the layout into the three LIM destinations.
    // Every one is bounded before the first NoC write uses it.
    pi_error_t derive_peer_spans() {
        if (args_.worker_index >= layout_.worker_count) {
            return PI_ERROR_BAD_ARGS;
        }
        const uint64_t arena_end = X280_LIM_PACKET_INJECTION_ARENA_BASE + layout_.arena_bytes;
        const uint64_t produced = X280_LIM_PACKET_INJECTION_ARENA_BASE + layout_.produced_offset +
                                  (uint64_t)args_.worker_index * sizeof(pi_counter_slot_t);
        const uint64_t ready = X280_LIM_PACKET_INJECTION_ARENA_BASE + layout_.ready_offset +
                               (uint64_t)args_.worker_index * sizeof(pi_ready_slot_t);
        const uint64_t ring = X280_LIM_PACKET_INJECTION_ARENA_BASE + layout_.rings_offset +
                              (uint64_t)args_.worker_index * layout_.ring_stride;
        if (produced + sizeof(pi_counter_slot_t) > arena_end || ready + sizeof(pi_ready_slot_t) > arena_end ||
            ring + layout_.ring_stride > arena_end) {
            return PI_ERROR_CAPACITY;
        }
        if (!x280_noc_coord_valid(args_.peer)) {
            return PI_ERROR_BAD_PLACEMENT;
        }

        produced_dst_ = (uint32_t)produced;
        ready_dst_ = (uint32_t)ready;
        ring_base_ = (uint32_t)ring;
        PacketWorkerSyncConfig::kLocalReleaseAddr = layout_.l1_tile_base + layout_.l1_release_offset;
        PacketWorkerSyncConfig::kArrivalTarget.noc = args_.peer;
        PacketWorkerSyncConfig::kArrivalTarget.ready_base =
            (uint32_t)(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout_.sync_arrivals_offset);
        PacketWorkerSyncConfig::kArrivalTarget.worker_index = args_.worker_index;
        return PI_ERROR_NONE;
    }

    pi_error_t validate_controls() {
        if (W_ > UINT64_MAX - N_) {
            return PI_ERROR_BAD_ARGS;
        }
        if (args_.poll_cadence == 0u || args_.poll_cadence > (1u << 20) ||
            (args_.poll_cadence & (args_.poll_cadence - 1u)) != 0u) {
            return PI_ERROR_BAD_ARGS;
        }
        if (args_.timeout_ticks == 0u || args_.timeout_ticks >= (1ULL << 63)) {
            return PI_ERROR_BAD_ARGS;
        }
        return PI_ERROR_NONE;
    }

    void initialize_l1_state() {
        r_ = 0u;
        p_ = 0u;
        c_seen_ = 0u;
        pending_ = 0u;
        issued_ = 0;
        retired_ = 0;
        stall_count_ = 0;
        rtt_count_ = 0;
        safety_work_ = 0u;

        // Host initializes consumed_mirror and control_line; firmware does not re-zero them.
        *stable_produced_source_ = 0u;
        *stable_ready_source_ = PI_READY_NONE;

        // Initialize staging slots
        for (uint32_t d = 0; d < D_; ++d) {
            staging_[d].start = 0;
            for (size_t i = 0; i < 24; ++i) {
                staging_[d].padding[i] = 0;
            }
        }

        // Initialize producer result
        result_->abi_tag = PI_ABI_TAG(Mode, IncludeCreation);
        result_->worker_index = args_.worker_index;
        result_->phase = PI_PHASE_INIT;
        result_->quiescent = 0u;
        result_->run_nonce = args_.run_nonce;
        result_->reserved_progress = 0u;
        result_->published_progress = 0u;
        result_->consumed_progress = 0u;
        result_->warmup_completed = 0u;
        result_->measured_completed = 0u;
        result_->latency_count = 0u;
        result_->stall_count = 0u;
        result_->rtt_count = 0u;
        result_->throughput_start = 0u;
        result_->throughput_end = 0u;
        result_->measured_counter_writes = 0u;
        result_->failure = pi_failure_t{0, 0, 0, 0};
    }

    /**
     * Every wait for this producer's own NoC writes, bounded by the phase
     * timeout so a destination that never acknowledges fails the run rather
     * than hanging the producer. False on expiry.
     */
    bool drain_writes() { return Transport::barrier(Transport::kNoc, args_.timeout_ticks); }

    // False when a wait expired: the word may not have been announced, and a
    // source still in flight is never rewritten.
    bool publish_ready(pi_ready_t state) {
        // Zero means derive_peer_spans() has not run, so there is no slot to
        // announce on; a real destination is at least the arena base plus 192.
        if (ready_dst_ == 0u) {
            return true;
        }
        if (!drain_writes()) {
            return false;
        }
        *stable_ready_source_ = (uint32_t)state;
        Transport::issue_u32_write(
            worker_base_ + PI_L1_READY_SRC_OFFSET, args_.peer.x, args_.peer.y, ready_dst_, Transport::kNoc);
        return drain_writes();
    }

    // Reads the host-owned RUN/STOP word and rejects every other value.
    pi_error_t check_control() {
        Transport::invalidate_cache();
        const uint32_t control = *control_line_;
        if (control == PI_CONTROL_RUN) {
            return PI_ERROR_NONE;
        }
        if (control == PI_CONTROL_STOP) {
            return PI_ERROR_STOPPED;
        }
        return PI_ERROR_BAD_ARGS;
    }

    pi_error_t reset_phase_deadline() {
        safety_work_ = 0u;
        deadline_origin_ = 0;
        if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &deadline_origin_)) {
            return PI_ERROR_TIMER_RETRY_EXHAUSTED;
        }
        return PI_ERROR_NONE;
    }

    pi_error_t amortized_safety_check() {
        safety_work_ += 1u;
        if (safety_work_ < args_.poll_cadence) {
            return PI_ERROR_NONE;
        }
        safety_work_ = 0u;

        const pi_error_t control = check_control();
        if (control != PI_ERROR_NONE) {
            return control;
        }

        uint64_t now = 0;
        if (!Transport::read_timer(PI_TIMER_MAX_ATTEMPTS, &now)) {
            return PI_ERROR_TIMER_RETRY_EXHAUSTED;
        }
        uint64_t elapsed = now - deadline_origin_;
        if (elapsed >= args_.timeout_ticks) {
            return PI_ERROR_TIMEOUT;
        }
        return PI_ERROR_NONE;
    }

    // Publish only initialized per-phase counts, including on a drained failure.
    void publish_final_results(pi_phase_t phase = PI_PHASE_DONE) {
        result_->phase = phase;
        result_->warmup_completed = measured_started_ ? W_ : retired_;
        result_->measured_completed = measured_started_ ? retired_ : 0;
        if constexpr (Mode == 1) {
            result_->latency_count = measured_started_ ? retired_ : 0;
        } else if constexpr (Mode == 2) {
            result_->stall_count = measured_started_ ? stall_count_ : 0;
        } else if constexpr (Mode == 3) {
            result_->stall_count = measured_started_ ? stall_count_ : 0;
            result_->rtt_count = measured_started_ ? rtt_count_ : 0;
        }
        result_->reserved_progress = r_;
        result_->published_progress = p_;
        result_->consumed_progress = c_seen_;
        result_->throughput_start = throughput_start_;
        result_->throughput_end = throughput_end_;
        result_->cpu_start = cpu_start_;
        result_->cpu_end = cpu_end_;
        result_->measured_counter_writes = measured_started_ ? counter_writes_ : 0;
        result_->quiescent = 1u;
    }

    void record_failure(pi_error_t err, pi_phase_t phase) {
        result_->failure =
            pi_failure_t{(uint32_t)err, (uint32_t)phase, args_.worker_index, (uint32_t)Transport::kActor};
        publish_final_results(PI_PHASE_FAILED);
    }

    // A write wait that expires here leaves the failure recorded in L1 only:
    // announcing it would rewrite a source still in flight.
    void fail_and_publish(pi_error_t err, pi_phase_t phase) {
        const bool drained = drain_writes();
        record_failure(err, phase);
        if (drained) {
            (void)publish_ready(PI_READY_FAILED);
        }
    }

    const pi_producer_args_t& args_;
    uintptr_t host_l1_override_;
    // Left uninitialized: derive_spans() populates it through pi_compute_layout()
    // before anything reads it, and returns on failure without a reader.
    pi_layout_t layout_;

    uint32_t S_ = 0u;
    uint32_t B_ = 0u;
    uint32_t D_ = 0u;
    uint64_t W_ = 0;
    uint64_t N_ = 0;

    uint32_t worker_base_ = 0u;
    uint32_t produced_dst_ = 0u;
    uint32_t ready_dst_ = 0u;
    uint32_t ring_base_ = 0u;

    volatile uint32_t* consumed_mirror_ = nullptr;
    volatile uint32_t* control_line_ = nullptr;
    volatile uint32_t* stable_produced_source_ = nullptr;
    volatile uint32_t* stable_ready_source_ = nullptr;
    pi_producer_result_t* result_ = nullptr;
    pi_slot_t* staging_ = nullptr;
    pi_local_stamp_t* starts_ = nullptr;
    pi_local_sample_t* stalls_ = nullptr;
    pi_local_sample_t* rtts_ = nullptr;

    uint32_t r_ = 0u;
    uint32_t p_ = 0u;
    uint32_t c_seen_ = 0u;
    uint32_t pending_ = 0u;
    uint64_t issued_ = 0;
    uint64_t retired_ = 0;
    uint64_t stall_count_ = 0;
    uint64_t rtt_count_ = 0;
    uint64_t counter_writes_ = 0;
    uint64_t throughput_start_ = 0;
    uint64_t throughput_end_ = 0;
    uint64_t cpu_start_ = 0;
    uint64_t cpu_end_ = 0;
    uint64_t cpu_retirement_end_ = 0;
    bool measured_started_ = false;

    uint32_t safety_work_ = 0u;
    uint64_t deadline_origin_ = 0;
};

}  // namespace packet_injection

#endif  // X280_KERNELS_PACKET_INJECTION_PRODUCER_HPP
