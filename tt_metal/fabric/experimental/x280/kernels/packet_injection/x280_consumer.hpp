// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/packet_injection/x280_consumer.hpp
//
// Consumer-specific ring loop, fixed TLB window management, remote timer
// retargeting, sample store/export, and host synchronization for X280.

#ifndef X280_KERNELS_PACKET_INJECTION_X280_CONSUMER_HPP
#define X280_KERNELS_PACKET_INJECTION_X280_CONSUMER_HPP

#include <stdint.h>
#include <stddef.h>

#include "packet_injection_abi.h"
#include "packet_injection_slot.hpp"
#include "../shared/core_sync_barrier.hpp"
#include "../shared/core_sync_transport_x280.hpp"
#include "../shared/tensix_live_timer.h"
#include "x280_address_map.h"
#include "rt/x280_pinned_input.h"
#include "rt/x280_noc_map.h"

#if defined(__riscv)
#include "noc.h"
#include "../shared/x280_counter_window.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "rt/measurement.h"
#include "rt/x280_measurement.h"
#endif

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

static_assert(
    PI_MAX_TILES <= X280_TLB_PACKET_INJECTION_CREDIT_COUNT,
    "one credit window per tile must fit the packet-injection TLB reservation");
static_assert(
    PI_ARENA_BYTES == X280_LIM_PACKET_INJECTION_ARENA_BYTES,
    "the ABI arena limit must match the LIM arena the host hands over");

static core_sync::WordArrival packet_arrivals[PI_MAX_WORKERS]{};
static core_sync::X280Group packet_groups[1]{};
static core_sync::X280Endpoint packet_endpoints[PI_MAX_TILES]{};

/* Image-local director roster, bound after argument and coordinate validation
 * and retained through normal or recovery release. */
struct PacketDirectorSyncConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = true;
    static inline core_sync::X280Roster kRoster{};
};

struct DeviceX280Transport {
    static inline void fence_full() {
#if defined(__riscv)
        x280_fence_iorw();
#endif
    }

    static inline void fence_load() {
#if defined(__riscv)
        x280_fence_ir();
#endif
    }

    static inline uint64_t mtime() {
#if defined(__riscv)
        return x280_mtime_fenced();
#else
        return 0;
#endif
    }

    /**
     * This hart's raw `rdcycle`, in core cycles at the running PLL, fenced
     * like mtime(). Only a delta between two reads on hart 0 is meaningful.
     * Zero off-target.
     */
    static inline uint64_t rdcycle() {
#if defined(__riscv)
        return x280_rdcycle_fenced();
#else
        return 0;
#endif
    }

    static inline int check_stop(uint64_t hartid, uint64_t run_value) {
#if defined(__riscv)
        return x280_hart_stop_requested(hartid, run_value);
#else
        (void)hartid;
        (void)run_value;
        return 0;
#endif
    }

    /** Base of the writable pinned sample region; the reserved cache way. */
    static inline uint64_t pinned_base() { return X280_PINNED_INPUT_BASE; }

    /** Base of the packet LIM arena every derived address is measured from. */
    static inline uint64_t arena_base() { return X280_LIM_PACKET_INJECTION_ARENA_BASE; }

    /* Opens the tile's credit window, a counter window (x280_counter_window.h).
     * -1 for a refused or unverified window. The pointer is unused (window_base()
     * derives it), and an unverified window stays programmed: the consumer
     * retains it (cleanup_windows). */
    static inline int configure_credit_window(uint32_t index, uint32_t noc_x, uint32_t noc_y) {
#if defined(__riscv)
        volatile void* ptr = nullptr;
        return x280_counter_window_open(index, x280_noc_coord_t{noc_x, noc_y}, &ptr) == NOC_WINDOW_OPEN ? 0 : -1;
#else
        (void)index;
        (void)noc_x;
        (void)noc_y;
        return 0;
#endif
    }

    static inline int configure_timer_window(uint32_t index, uint32_t noc_x, uint32_t noc_y) {
#if defined(__riscv)
        volatile void* ptr = nullptr;
        return noc_open_window_2m(
                   index,
                   x280_noc_coord_t{noc_x, noc_y},
                   TENSIX_TIMER_NOC_PAGE_ADDR,
                   NOC_ORDER_STRICT,
                   NOC_VC_DYNAMIC,
                   &ptr) == NOC_WINDOW_OPEN
                   ? 0
                   : -1;
#else
        (void)index;
        (void)noc_x;
        (void)noc_y;
        return 0;
#endif
    }

    static inline int clear_window(uint32_t index) {
#if defined(__riscv)
        return noc_clear_tlb_2m(index);
#else
        (void)index;
        return 0;
#endif
    }

    static inline volatile void* window_base(uint32_t index) {
#if defined(__riscv)
        return (volatile void*)(NOC_2M_WINDOW_BASE + (uint64_t)index * NOC_2M_WINDOW_STRIDE);
#else
        (void)index;
        return (volatile void*)0;
#endif
    }

    static inline uint64_t prepare_pinned(
        uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region) {
#if defined(__riscv)
        return x280_pinned_input_prepare(hartid, run_value, region);
#else
        (void)hartid;
        (void)run_value;
        (void)region;
        return X280_PINNED_INPUT_RESULT_PREPARED;
#endif
    }

    /**
     * Admission code for the baseline pinned-cache policy, which is what an
     * image that reserves no sample region requires. X280_PINNED_INPUT_ADMITTED
     * means the way masks are the baseline ones; any other code means the
     * configuration is wrong, including a prepared mask a prior run left behind.
     */
    static inline uint32_t check_pinned_baseline() {
#if defined(__riscv)
        return x280_pinned_input_check_policy((const x280_pinned_input_region_t*)0, X280_PINNED_INPUT_BASELINE).code;
#else
        return (uint32_t)X280_PINNED_INPUT_ADMITTED;
#endif
    }

    static inline uint64_t teardown_pinned(
        uint64_t hartid, uint64_t run_value, const x280_pinned_input_region_t* region) {
#if defined(__riscv)
        return x280_pinned_input_teardown(hartid, run_value, region);
#else
        (void)hartid;
        (void)run_value;
        (void)region;
        return X280_PINNED_INPUT_RESULT_TORN_DOWN;
#endif
    }

    /* x280_counter_store(): the slot loads before it are performed first; it
     * returns once the store is accepted, not delivered. */
    static inline void counter_store(volatile uint32_t* addr, uint32_t val) {
#if defined(__riscv)
        x280_counter_store(addr, val);
#else
        *addr = val;
#endif
    }
};

template <
    uint32_t Mode = PACKET_INJECTION_MODE,
    uint32_t IncludeCreation = PACKET_INJECTION_INCLUDE_CREATION,
    typename Transport = DeviceX280Transport,
    typename SyncTransport = core_sync::X280Transport<>>
class X280Consumer {
public:
    /**
     * Binds one run. Every LIM address is derived as Transport::arena_base()
     * plus a pi_layout_t offset, and `args` must be the record the layout
     * reserves at that base; nothing is dereferenced until run() has
     * validated both.
     */
    X280Consumer(uint64_t hartid, uint64_t run_value, const pi_x280_args_t& args) :
        hartid_(hartid),
        run_value_(run_value),
        args_(args),
        arena_base_(Transport::arena_base()),
        P_(args.worker_count),
        T_(args.tile_count),
        S_(args.S),
        C_(args.C),
        K_(args.K),
        W_(args.W),
        N_(args.N) {}

    pi_error_t run() {
        pi_error_t err = validate_args();
        if (err != PI_ERROR_NONE) {
            // Do NOT dereference unverified run_result_ or slots_ pointers!
            return err;
        }

        initialize_state_and_results();

        err = configure_windows();
        if (err != PI_ERROR_NONE) {
            return fail_drain_and_cleanup(err, PI_PHASE_INIT, UINT32_MAX);
        }

        if constexpr (Mode == 1) {
            const x280_pinned_input_region_t region = pinned_region();
            uint64_t prep_res = Transport::prepare_pinned(hartid_, run_value_, &region);
            if (prep_res != X280_PINNED_INPUT_RESULT_PREPARED) {
                pin_policy_uncertain_ = true;
                return fail_drain_and_cleanup(PI_ERROR_PIN_POLICY, PI_PHASE_INIT, UINT32_MAX);
            }
            pinned_prepared_ = true;
        } else {
            // No sample region is reserved outside latency mode, so the only
            // requirement is that the cache still carries the baseline masks.
            // Checked here so a stale prepared mask fails the run rather than
            // starving the measured phase into a timeout.
            const uint32_t admission = Transport::check_pinned_baseline();
            if (admission != (uint32_t)X280_PINNED_INPUT_ADMITTED) {
                return fail_drain_and_cleanup(pin_admission_error(admission), PI_PHASE_INIT, UINT32_MAX);
            }
        }

        begin_phase(W_);

        err = consume_phase<false>(W_);
        if (err != PI_ERROR_NONE) {
            return fail_drain_and_cleanup(err, PI_PHASE_WARMUP, first_failed_worker_);
        }
        flush_all_credits();

        for (uint32_t w = 0; w < P_; ++w) {
            if (slots_[w].consumed != slots_[w].credited) {
                return fail_drain_and_cleanup(PI_ERROR_COUNTER_DISTANCE, PI_PHASE_WAIT_MEASURE, w);
            }
            slots_[w].warmup_done = slots_[w].phase_done;
        }

        const core_sync::Outcome sync = core_sync::core_sync_barrier<PacketDirectorSyncConfig, SyncTransport>::sync(
            hartid_, run_value_, args_.run_nonce);
        if (sync != core_sync::Outcome::Success) {
            err = sync == core_sync::Outcome::StopRequested ? PI_ERROR_STOPPED : PI_ERROR_LAYOUT_INVARIANT;
            return fail_drain_and_cleanup(err, PI_PHASE_WAIT_MEASURE, first_failed_worker_);
        }
        // Nested inside the mtime pair, so the cycle interval never spans
        // more than the wall interval.
        aggregate_start_mtime_ = Transport::mtime();
        aggregate_start_rdcycle_ = Transport::rdcycle();
        normal_release_issued_ = true;

        begin_phase(N_);
        measured_started_ = true;

        err = consume_phase<true>(N_);
        if (err != PI_ERROR_NONE) {
            return fail_drain_and_cleanup(err, PI_PHASE_MEASURE, first_failed_worker_);
        }
        flush_all_credits();
        aggregate_end_rdcycle_ = Transport::rdcycle();
        aggregate_end_mtime_ = Transport::mtime();

        err = wait_all_producers_complete();
        if (err != PI_ERROR_NONE) {
            return fail_drain_and_cleanup(err, PI_PHASE_DRAIN, first_failed_worker_);
        }

        for (uint32_t w = 0; w < P_; ++w) {
            uint32_t final_prod = ordered_produced_load(w);
            if (final_prod != slots_[w].consumed || slots_[w].consumed != slots_[w].credited) {
                return fail_drain_and_cleanup(PI_ERROR_COUNTER_DISTANCE, PI_PHASE_DRAIN, w);
            }
        }

        return collect_and_cleanup();
    }

private:
    // One 2 MiB TLB window spans a tile's L1, so a credit or control write past
    // it resolves through the next window and lands on a different tile.
    static constexpr uint64_t kCreditWindowBytes = 0x200000ULL;

    // A code outside the admission enum names no cause, which is what
    // PI_ERROR_PIN_POLICY means; it still fails the run.
    static pi_error_t pin_admission_error(uint32_t admission) {
        switch (admission) {
            case (uint32_t)X280_PINNED_INPUT_WRONG_WAYENABLE: return PI_ERROR_PIN_WAYENABLE;
            case (uint32_t)X280_PINNED_INPUT_WRONG_WAYMASK: return PI_ERROR_PIN_WAYMASK;
            case (uint32_t)X280_PINNED_INPUT_WRONG_SHAPE: return PI_ERROR_PIN_SHAPE;
            default: return PI_ERROR_PIN_POLICY;
        }
    }

    x280_pinned_input_region_t pinned_region() const {
        return x280_pinned_input_region_t{
            .base = Transport::pinned_base(),
            .bytes = layout_.latency_export_bytes,
            .kind = static_cast<uint32_t>(X280_PINNED_INPUT_WRITABLE_SAMPLES),
            .reserved = 0,
            .reserved_tail = 0,
        };
    }

    pi_error_t validate_args() {
        if (hartid_ != 0) {
            return PI_ERROR_BAD_ARGS;
        }
        if (!pi_abi_tag_matches(args_.abi_tag, Mode, IncludeCreation)) {
            return PI_ERROR_MODE_MISMATCH;
        }
        // Zero is the unacknowledged value of collect_ack_run_value.
        if (run_value_ == 0) {
            return PI_ERROR_BAD_ARGS;
        }
        if (arena_base_ == 0 || (arena_base_ & 63u) != 0) {
            return PI_ERROR_BAD_ARGS;
        }
        if (P_ == 0 || P_ > PI_MAX_WORKERS || T_ == 0 || T_ > PI_MAX_TILES) {
            return PI_ERROR_BAD_PLACEMENT;
        }
        if (P_ != T_ && P_ != 2 * T_) {
            return PI_ERROR_BAD_PLACEMENT;
        }
        if (S_ == 0 || (S_ & (S_ - 1u)) != 0 || S_ >= 0x80000000u) {
            return PI_ERROR_BAD_ARGS;
        }
        if (C_ == 0 || K_ == 0 || C_ > K_ || args_.B == 0 || N_ == 0 || W_ > UINT64_MAX - N_ ||
            W_ + N_ > UINT64_MAX / P_ || args_.poll_cadence == 0 || args_.poll_cadence > (1u << 20) ||
            (args_.poll_cadence & (args_.poll_cadence - 1)) != 0 || args_.x280_timeout_ticks == 0 ||
            args_.x280_timeout_ticks >= (1ULL << 63) || (args_.l1_tile_base & 63u) != 0) {
            return PI_ERROR_BAD_ARGS;
        }
        if ((args_.inflight != 1u && args_.inflight != 2u && args_.inflight != 4u) ||
            (Mode == 1 && args_.inflight != 1u)) {
            return PI_ERROR_BAD_ARGS;
        }

        const pi_layout_check_t check = pi_compute_layout(P_, T_, S_, args_.B, N_, Mode, args_.l1_tile_base, &layout_);
        if (check.code != PI_LAYOUT_OK) {
            return pi_layout_code_to_error(check.code);
        }
        // x280_hart_args() is any cache-line-aligned block in available LIM, so
        // where the args were published is a claim to check, never a source of
        // the arena base: the layout reserves the arena's first record for
        // them, and args placed anywhere else mean every derived address is
        // measured from the wrong origin. Checked on every path, so a
        // relocated arena has to mirror the real one.
        if ((uint64_t)(uintptr_t)&args_ != arena_base_ + layout_.args_offset) {
            return PI_ERROR_BAD_ARGS;
        }
        // pi_compute_layout() places every LIM span inside [0, arena_bytes),
        // so bounding that one extent bounds all of them.
        if (layout_.arena_bytes > PI_ARENA_BYTES || arena_base_ > (uint64_t)UINTPTR_MAX - layout_.arena_bytes) {
            return PI_ERROR_CAPACITY;
        }
        if ((uint64_t)layout_.l1_tile_base + (uint64_t)layout_.l1_tile_bytes > kCreditWindowBytes) {
            return PI_ERROR_CAPACITY;
        }

        host_control_ = (volatile pi_host_control_t*)(uintptr_t)(arena_base_ + layout_.host_control_offset);
        run_result_ = (pi_run_result_t*)(uintptr_t)(arena_base_ + layout_.run_result_offset);
        tile_coords_ = (const x280_noc_coord_t*)(uintptr_t)(arena_base_ + layout_.tile_coords_offset);
        slots_ = (pi_consumer_slot_t*)(uintptr_t)(arena_base_ + layout_.consumer_slots_offset);
        if constexpr (Mode == 1) {
            pinned_samples_ = (uint64_t*)(uintptr_t)Transport::pinned_base();
            export_samples_ = (uint64_t*)(uintptr_t)(arena_base_ + layout_.latency_export_offset);
        }

        // Each tile owns one credit window, so two tiles at one coordinate
        // would aim two windows at the same L1 and lose one tile's credits.
        for (uint32_t t = 0; t < T_; ++t) {
            const x280_noc_coord_t c = tile_coords_[t];
            if (!x280_noc_coord_valid(c)) {
                return PI_ERROR_BAD_PLACEMENT;
            }
            for (uint32_t u = 0; u < t; ++u) {
                if (tile_coords_[u].x == c.x && tile_coords_[u].y == c.y) {
                    return PI_ERROR_BAD_PLACEMENT;
                }
            }
        }

        // The host selected the release; this only checks the group is the
        // tile table's: distinct tiles, all inside it, as many as it releases.
        const x280_noc_rect_t& rect = args_.release_rect;
        if (args_.release_num_dests == 0u) {
            if (rect.start.x != 0u || rect.start.y != 0u || rect.end.x != 0u || rect.end.y != 0u) {
                return PI_ERROR_BAD_ARGS;
            }
        } else {
            if (args_.release_num_dests != T_ || !x280_noc_rect_valid(rect)) {
                return PI_ERROR_BAD_ARGS;
            }
            for (uint32_t t = 0; t < T_; ++t) {
                if (!x280_noc_rect_contains(rect, tile_coords_[t])) {
                    return PI_ERROR_BAD_ARGS;
                }
            }
        }

        const uint32_t arrivals_base = (uint32_t)(arena_base_ + layout_.sync_arrivals_offset);
        for (uint32_t w = 0; w < P_; ++w) {
            packet_arrivals[w] = core_sync::WordArrival{core_sync_slot_addr(arrivals_base, w), w};
        }

        const uint32_t release_addr = layout_.l1_tile_base + layout_.l1_release_offset;
        uint32_t group_count = 0u;
        uint32_t endpoint_count = 0u;
        if (args_.release_num_dests != 0u) {
            packet_groups[0] = core_sync::X280Group{rect, release_addr, args_.release_num_dests};
            group_count = 1u;
        } else {
            // Endpoints, not one-tile groups: the transport releases a group by
            // broadcast whatever its size, and every tile but Tensix drops a
            // broadcast.
            for (uint32_t t = 0; t < T_; ++t) {
                packet_endpoints[t] = core_sync::X280Endpoint{tile_coords_[t], release_addr, 0u};
            }
            endpoint_count = T_;
        }
        PacketDirectorSyncConfig::kRoster = core_sync::X280Roster{
            .endpoints = packet_endpoints,
            .endpoint_count = endpoint_count,
            .counter_addr = 0u,
            .counter_arrivals = 0u,
            .word_arrivals = packet_arrivals,
            .word_arrival_count = P_,
            .groups = packet_groups,
            .group_count = group_count};

        return PI_ERROR_NONE;
    }

    void initialize_state_and_results() {
        for (uint32_t w = 0; w < P_; ++w) {
            slots_[w].consumed = 0u;
            slots_[w].credited = 0u;
            slots_[w].phase_done = 0;
            slots_[w].warmup_done = 0;
        }

        run_result_->abi_tag = PI_ABI_TAG(Mode, IncludeCreation);
        run_result_->state = PI_PHASE_INIT;
        run_result_->run_value = run_value_;
        run_result_->measured_completed = 0;
        run_result_->export_valid_bytes = 0u;
        run_result_->failure = pi_failure_t{0, 0, 0, 0};
        run_result_->data_quiescent = 0u;
        run_result_->windows_cleared = 0u;
        run_result_->cache_restored = 0u;
        run_result_->worker_count = P_;
        run_result_->aggregate_start_mtime = 0;
        run_result_->aggregate_end_mtime = 0;
        run_result_->aggregate_start_rdcycle = 0;
        run_result_->aggregate_end_rdcycle = 0;
        run_result_->empty_visits = 0;
        const core_sync::X280Roster& roster = PacketDirectorSyncConfig::kRoster;
        run_result_->released_group_dests = roster.group_count != 0u ? roster.groups[0].num_dests : 0u;
        run_result_->released_endpoints = roster.endpoint_count;

        host_control_->collect_ack_run_value = 0u;

        current_timer_tile_ = UINT32_MAX;
    }

    pi_error_t configure_windows() {
        for (uint32_t t = 0; t < T_; ++t) {
            uint32_t tlb_idx = X280_TLB_PACKET_INJECTION_CREDIT_BASE + t;
            int rc = Transport::configure_credit_window(tlb_idx, tile_coords_[t].x, tile_coords_[t].y);
            if (rc != 0) {
                windows_uncertain_ = true;
                return PI_ERROR_WINDOW_CONFIG;
            }
            tiles_configured_ = t + 1u;
        }
        return PI_ERROR_NONE;
    }

    // Clear only verified, completed mappings; retain failed readbacks as owned.
    // A window noc_open_window_2m could not verify stays programmed: it is not
    // proven this run's, since the four harts share the tile's config block.
    pi_error_t cleanup_windows() {
        if (windows_uncertain_) {
            return PI_ERROR_WINDOW_CONFIG;
        }
        while (tiles_configured_ != 0) {
            if (Transport::clear_window(X280_TLB_PACKET_INJECTION_CREDIT_BASE + tiles_configured_ - 1u) != 0) {
                return PI_ERROR_WINDOW_CLEAR;
            }
            tiles_configured_ -= 1u;
        }
        if (current_timer_tile_ != UINT32_MAX) {
            if (Transport::clear_window(X280_TLB_PACKET_INJECTION_TIMER) != 0) {
                return PI_ERROR_WINDOW_CLEAR;
            }
            current_timer_tile_ = UINT32_MAX;
        }
        return PI_ERROR_NONE;
    }

    // A partial preparation or failed restoration does not relinquish the way.
    pi_error_t cleanup_pinned() {
        if (pin_policy_uncertain_) {
            return PI_ERROR_PIN_POLICY;
        }
        if constexpr (Mode == 1) {
            if (pinned_prepared_) {
                const x280_pinned_input_region_t region = pinned_region();
                if (Transport::teardown_pinned(hartid_, run_value_, &region) != X280_PINNED_INPUT_RESULT_TORN_DOWN) {
                    return PI_ERROR_PIN_POLICY;
                }
                pinned_prepared_ = false;
            }
        }
        return PI_ERROR_NONE;
    }

    pi_error_t ensure_timer_window(uint32_t tile_index) {
        if constexpr (Mode == 1) {
            if (current_timer_tile_ == tile_index) {
                return PI_ERROR_NONE;
            }
            Transport::fence_full();
            int rc = Transport::configure_timer_window(
                X280_TLB_PACKET_INJECTION_TIMER, tile_coords_[tile_index].x, tile_coords_[tile_index].y);
            if (rc != 0) {
                windows_uncertain_ = true;
                return PI_ERROR_WINDOW_CONFIG;
            }
            Transport::fence_full();
            current_timer_tile_ = tile_index;
            return PI_ERROR_NONE;
        } else {
            (void)tile_index;
            return PI_ERROR_NONE;
        }
    }

    /** Fenced load of worker `worker_index`'s word in the per-worker LIM array at `base`. */
    static uint32_t ordered_counter_load(uint64_t base, uint32_t worker_index, uint64_t stride) {
        Transport::fence_load();
        return *(const volatile uint32_t*)(uintptr_t)(base + (uint64_t)worker_index * stride);
    }

    uint32_t ordered_produced_load(uint32_t worker_index) {
        return ordered_counter_load(arena_base_ + layout_.produced_offset, worker_index, sizeof(pi_counter_slot_t));
    }

    uint32_t ordered_ready_load(uint32_t worker_index) {
        return ordered_counter_load(arena_base_ + layout_.ready_offset, worker_index, sizeof(pi_ready_slot_t));
    }

    /** The worker's credit word through its tile's window, or nullptr when that window is unverified. */
    volatile uint32_t* credit_register(uint32_t worker_index, uint32_t tile) const {
        if (tile >= tiles_configured_) {
            return nullptr;
        }
        uint32_t tlb_idx = X280_TLB_PACKET_INJECTION_CREDIT_BASE + tile;
        uint64_t win_addr = (uint64_t)(uintptr_t)Transport::window_base(tlb_idx);
        return (volatile uint32_t*)(uintptr_t)(win_addr + pi_worker_l1_base(&layout_, worker_index) +
                                               PI_L1_CONSUMED_OFFSET);
    }

    void return_credit(uint32_t worker_index, uint32_t tile) {
        if (slots_[worker_index].consumed == slots_[worker_index].credited) {
            return;
        }
        volatile uint32_t* credit_reg = credit_register(worker_index, tile);
        if (credit_reg == nullptr) {
            return;
        }

        Transport::counter_store(credit_reg, slots_[worker_index].consumed);
        slots_[worker_index].credited = slots_[worker_index].consumed;
    }

    void flush_all_credits() {
        for (uint32_t w = 0; w < P_; ++w) {
            return_credit(w, pi_worker_tile_index(&layout_, w));
        }
    }

    // Writes one control value through every verified worker window in worker order.
    void broadcast_control(pi_control_t control) {
        for (uint32_t w = 0; w < P_; ++w) {
            uint32_t tile = pi_worker_tile_index(&layout_, w);
            if (tile >= tiles_configured_) {
                continue;
            }

            uint32_t tlb_idx = X280_TLB_PACKET_INJECTION_CREDIT_BASE + tile;
            uint64_t win_addr = (uint64_t)(uintptr_t)Transport::window_base(tlb_idx);
            volatile uint32_t* control_reg =
                (volatile uint32_t*)(uintptr_t)(win_addr + pi_worker_l1_base(&layout_, w) + PI_L1_CONTROL_OFFSET);
            Transport::counter_store(control_reg, (uint32_t)control);
        }
    }

    void begin_phase(uint64_t limit) {
        for (uint32_t w = 0; w < P_; ++w) {
            slots_[w].phase_done = 0;
        }
        remaining_total_ = (uint64_t)P_ * limit;
        total_measured_done_ = 0;
        safety_work_ = 0u;
        phase_start_mtime_ = Transport::mtime();
    }

    template <bool Measured>
    pi_error_t consume_phase(uint64_t limit) {
        if constexpr (Mode != 1) {
            if (args_.inflight == 2u) {
                return consume_phase_in_flight<Measured, 2>(limit);
            }
            if (args_.inflight == 4u) {
                return consume_phase_in_flight<Measured, 4>(limit);
            }
        }
        return consume_phase_serial<Measured>(limit);
    }

    /** One phase, visiting one worker at a time in index order (inflight 1). */
    template <bool Measured>
    pi_error_t consume_phase_serial(uint64_t limit) {
        phase_start_mtime_ = Transport::mtime();

        // The timed-loop rule (transport_bandwidth_frontend_cxx.hpp): the phase
        // works on register copies of the consumer's state, which lives on the
        // uncached LIM stack and arena, and stores them back before anything
        // reads them through the object.
        const uint32_t P = P_;
        const uint32_t K = K_;
        const uint32_t C = C_;
        const uint32_t S = S_;
        const uint32_t poll_cadence = args_.poll_cadence;
        pi_consumer_slot_t* const slots = slots_;
        const uint64_t produced_base = arena_base_ + layout_.produced_offset;
        const uint64_t rings_base = arena_base_ + layout_.rings_offset;
        const uint64_t ring_stride = layout_.ring_stride;
        uint64_t remaining = remaining_total_;
        uint64_t measured_done = total_measured_done_;
        uint64_t empty_visits = empty_visits_;
        uint32_t safety_work = 0u;
        auto store_back = [&] {
            remaining_total_ = remaining;
            total_measured_done_ = measured_done;
            empty_visits_ = empty_visits;
            safety_work_ = safety_work;
        };
        // amortized_safety_check() on the register counter.
        auto poll = [&](uint32_t w) {
            if (++safety_work < poll_cadence) {
                return PI_ERROR_NONE;
            }
            safety_work = 0u;
            return safety_check(true, w);
        };

        while (remaining != 0) {
            for (uint32_t w = 0; w < P; ++w) {
                pi_error_t err = poll(w);
                if (err != PI_ERROR_NONE) {
                    store_back();
                    return err;
                }

                pi_consumer_slot_t& state = slots[w];
                uint64_t phase_done = state.phase_done;
                if (phase_done == limit) {
                    continue;
                }

                uint32_t snapshot = ordered_counter_load(produced_base, w, sizeof(pi_counter_slot_t));
                uint32_t consumed = state.consumed;
                uint32_t available;
                if (!credit_pair_ok(snapshot, consumed, UINT32_MAX, S, &available)) {
                    store_back();
                    first_failed_worker_ = w;
                    return PI_ERROR_COUNTER_DISTANCE;
                }
                if ((uint64_t)available > limit - phase_done) {
                    store_back();
                    first_failed_worker_ = w;
                    return PI_ERROR_PHASE_OVERRUN;
                }
                if (available == 0u) {
                    if (Measured) {
                        empty_visits += 1;
                    }
                    continue;
                }

                // The snapshot above is related to the slot loads below only by
                // a control dependency, which RVWMO does not order, and the
                // slot address derives from `consumed` rather than from the
                // snapshot, so there is no address dependency either. One fence
                // here orders the counter load against every slot load of this
                // visit; the producer has already settled arrival order.
                Transport::fence_load();

                uint32_t visit_count = available < K ? available : K;
                const uint32_t tile = pi_worker_tile_index(&layout_, w);

                uint64_t timer_low_addr = 0;
                uint64_t* worker_samples = nullptr;
                if constexpr (Mode == 1) {
                    err = ensure_timer_window(tile);
                    if (err != PI_ERROR_NONE) {
                        store_back();
                        first_failed_worker_ = w;
                        return err;
                    }
                    timer_low_addr = (uint64_t)(uintptr_t)Transport::window_base(X280_TLB_PACKET_INJECTION_TIMER) +
                                     TENSIX_TIMER_LOW_OFFSET;
                    worker_samples = pinned_samples_ + (uint64_t)w * N_;
                }

                const uint64_t ring_base = rings_base + (uint64_t)w * ring_stride;
                volatile uint32_t* const credit_reg = credit_register(w, tile);
                uint32_t credited = state.credited;
                auto store_slot = [&] {
                    state.consumed = consumed;
                    state.credited = credited;
                    state.phase_done = phase_done;
                };

                for (uint32_t i = 0; i < visit_count; ++i) {
                    uint32_t slot_idx = consumed & (S - 1u);
                    const pi_slot_t* slot =
                        (const pi_slot_t*)(uintptr_t)(ring_base + (uint64_t)slot_idx * PI_SLOT_BYTES);

                    // Four volatile destination-ID byte loads at offsets 83, 84, 85, 86
                    const volatile uint8_t* p = (volatile const uint8_t*)slot;
                    uint8_t b0 = p[83];
                    uint8_t b1 = p[84];
                    uint8_t b2 = p[85];
                    uint8_t b3 = p[86];
                    __asm__ volatile("" ::"r"(b0), "r"(b1), "r"(b2), "r"(b3));

                    if constexpr (Mode == 1) {
                        uint64_t start = slot->start;
                        uint64_t end = 0;
                        if (!tensix_live_timer_read((uintptr_t)timer_low_addr, PI_TIMER_MAX_ATTEMPTS, &end)) {
                            store_slot();
                            store_back();
                            first_failed_worker_ = w;
                            return PI_ERROR_TIMER_RETRY_EXHAUSTED;
                        }
                        uint64_t d = 0;
                        if (!measurement_delta(start, end, &d)) {
                            store_slot();
                            store_back();
                            first_failed_worker_ = w;
                            return PI_ERROR_TIMER_DELTA;
                        }
                        if (Measured) {
                            worker_samples[phase_done] = d;
                        }
                    }

                    consumed = (uint32_t)(consumed + 1u);
                    phase_done += 1;
                    remaining -= 1;
                    if (Measured) {
                        measured_done += 1;
                    }

                    // return_credit() on the register copies. Marked likely:
                    // GCC otherwise treats the branch as cold and spills
                    // credit_reg to the stack, one LIM load per credit.
                    if (__builtin_expect(pi_counter_distance(consumed, credited) == C && credit_reg != nullptr, 1)) {
                        Transport::counter_store(credit_reg, consumed);
                        credited = consumed;
                    }

                    err = poll(w);
                    if (err != PI_ERROR_NONE) {
                        store_slot();
                        store_back();
                        return err;
                    }
                }
                // return_credit() on the register copies.
                if (consumed != credited && credit_reg != nullptr) {
                    Transport::counter_store(credit_reg, consumed);
                    credited = consumed;
                }
                store_slot();
            }
        }
        store_back();

        for (uint32_t w = 0; w < P_; ++w) {
            if (slots_[w].phase_done != limit) {
                first_failed_worker_ = w;
                return PI_ERROR_PHASE_OVERRUN;
            }
        }
        return PI_ERROR_NONE;
    }

    /**
     * One phase, visiting the workers D at a time in index order. A pass
     * issues the D workers' produced loads together, then takes one header
     * from every visit with headers left, issuing all their slot loads before
     * using any, until each visit (at most K headers) ends. Credits, safety
     * polls and error attribution are per worker, as in consume_phase_serial.
     */
    template <bool Measured, uint32_t D>
    pi_error_t consume_phase_in_flight(uint64_t limit) {
        static_assert(D > 1 && Mode != 1, "the latency timer window follows one tile at a time");
        phase_start_mtime_ = Transport::mtime();

        // The timed-loop rule (transport_bandwidth_frontend_cxx.hpp): the phase
        // works on register copies of the consumer's state, which lives on the
        // uncached LIM stack and arena, and stores them back before anything
        // reads them through the object.
        const uint32_t P = P_;
        const uint32_t K = K_;
        const uint32_t C = C_;
        const uint32_t S = S_;
        const uint32_t poll_cadence = args_.poll_cadence;
        pi_consumer_slot_t* const slots = slots_;
        const uint64_t produced_base = arena_base_ + layout_.produced_offset;
        const uint64_t rings_base = arena_base_ + layout_.rings_offset;
        const uint64_t ring_stride = layout_.ring_stride;
        uint64_t remaining = remaining_total_;
        uint64_t measured_done = total_measured_done_;
        uint64_t empty_visits = empty_visits_;
        uint32_t safety_work = 0u;
        auto store_back = [&]() __attribute__((always_inline)) {
            remaining_total_ = remaining;
            total_measured_done_ = measured_done;
            empty_visits_ = empty_visits;
            safety_work_ = safety_work;
        };
        // amortized_safety_check() on the register counter.
        auto poll = [&](uint32_t w) __attribute__((always_inline)) {
            if (++safety_work < poll_cadence) {
                return PI_ERROR_NONE;
            }
            safety_work = 0u;
            return safety_check(true, w);
        };

        // Worker w0 + j's visit in this pass, on register copies of its slot.
        // Every loop over j runs D times and is unrolled, so these stay in
        // registers rather than in an array on the stack. The slot's
        // phase_done is stored when the visit starts, already counting the
        // visit's headers, and corrected on an error return.
        struct Visit {
            uint32_t consumed;
            uint32_t end;  // `consumed` once the visit has taken its headers
            uint32_t credited;
            volatile uint32_t* credit_reg;
        };
        Visit visits[D] = {};
        uint32_t w0 = 0u;
        uint32_t taken = 0u;  // bit j: worker w0 + j's visit took headers
        // At a visit's end only consumed and credited change; an error return
        // also takes the untaken headers back out of phase_done.
        auto store_visits = [&](bool error) __attribute__((always_inline)) {
#pragma GCC unroll 4
            for (uint32_t j = 0; j < D; ++j) {
                if ((taken >> j) & 1u) {
                    pi_consumer_slot_t& state = slots[w0 + j];
                    state.consumed = visits[j].consumed;
                    state.credited = visits[j].credited;
                    if (error) {
                        state.phase_done -= visits[j].end - visits[j].consumed;
                    }
                }
            }
        };

        while (remaining != 0) {
            for (w0 = 0u; w0 < P; w0 += D) {
                uint32_t live = 0u;
                uint64_t phases[D] = {};
#pragma GCC unroll 4
                for (uint32_t j = 0; j < D; ++j) {
                    if (w0 + j < P) {
                        pi_error_t err = poll(w0 + j);
                        if (err != PI_ERROR_NONE) {
                            store_back();
                            return err;
                        }
                        phases[j] = slots[w0 + j].phase_done;
                        if (phases[j] != limit) {
                            live |= 1u << j;
                        }
                    }
                }
                if (live == 0u) {
                    continue;
                }

                uint32_t snapshots[D] = {};
                Transport::fence_load();
#pragma GCC unroll 4
                for (uint32_t j = 0; j < D; ++j) {
                    if ((live >> j) & 1u) {
                        snapshots[j] =
                            *(const volatile uint32_t*)(uintptr_t)(produced_base +
                                                                   (uint64_t)(w0 + j) * sizeof(pi_counter_slot_t));
                    }
                }

                taken = 0u;
#pragma GCC unroll 4
                for (uint32_t j = 0; j < D; ++j) {
                    if (((live >> j) & 1u) == 0u) {
                        continue;
                    }
                    const uint32_t w = w0 + j;
                    pi_consumer_slot_t& state = slots[w];
                    const uint32_t consumed = state.consumed;
                    const uint64_t phase_done = phases[j];
                    uint32_t available;
                    if (!credit_pair_ok(snapshots[j], consumed, UINT32_MAX, S, &available)) {
                        store_visits(true);
                        store_back();
                        first_failed_worker_ = w;
                        return PI_ERROR_COUNTER_DISTANCE;
                    }
                    if ((uint64_t)available > limit - phase_done) {
                        store_visits(true);
                        store_back();
                        first_failed_worker_ = w;
                        return PI_ERROR_PHASE_OVERRUN;
                    }
                    if (available == 0u) {
                        if (Measured) {
                            empty_visits += 1;
                        }
                        continue;
                    }
                    const uint32_t visit_count = available < K ? available : K;
                    state.phase_done = phase_done + visit_count;
                    visits[j] = Visit{
                        consumed,
                        (uint32_t)(consumed + visit_count),
                        state.credited,
                        credit_register(w, pi_worker_tile_index(&layout_, w))};
                    taken |= 1u << j;
                }
                if (taken == 0u) {
                    continue;
                }

                // The snapshots above are related to the slot loads below only
                // by a control dependency, which RVWMO does not order, and a
                // slot address derives from `consumed` rather than from the
                // snapshot, so there is no address dependency either. One fence
                // here orders the counter loads against every slot load of this
                // pass; the producer has already settled arrival order.
                Transport::fence_load();

                const uint64_t pass_rings = rings_base + (uint64_t)w0 * ring_stride;
                for (uint32_t active = taken; active != 0u;) {
                    // Four volatile destination-ID byte loads at offsets 83, 84,
                    // 85, 86 per header, every header's issued before any is used.
                    uint32_t ids[D][4];
#pragma GCC unroll 4
                    for (uint32_t j = 0; j < D; ++j) {
                        // No instruction: only marks the registers defined on
                        // the path that loads nothing into them.
                        __asm__("" : "=r"(ids[j][0]), "=r"(ids[j][1]), "=r"(ids[j][2]), "=r"(ids[j][3]));
                        if ((active >> j) & 1u) {
                            const volatile uint8_t* p =
                                (const volatile uint8_t*)(uintptr_t)(pass_rings + (uint64_t)j * ring_stride +
                                                                     (uint64_t)(visits[j].consumed & (S - 1u)) *
                                                                         PI_SLOT_BYTES);
                            ids[j][0] = p[83];
                            ids[j][1] = p[84];
                            ids[j][2] = p[85];
                            ids[j][3] = p[86];
                        }
                    }
#pragma GCC unroll 4
                    for (uint32_t j = 0; j < D; ++j) {
                        if (((active >> j) & 1u) == 0u) {
                            continue;
                        }
                        Visit& v = visits[j];
                        __asm__ volatile("" ::"r"(ids[j][0]), "r"(ids[j][1]), "r"(ids[j][2]), "r"(ids[j][3]));

                        v.consumed = (uint32_t)(v.consumed + 1u);
                        if (v.consumed == v.end) {
                            active &= ~(1u << j);
                        }
                        remaining -= 1;
                        if (Measured) {
                            measured_done += 1;
                        }

                        // return_credit() on the register copies. Marked likely:
                        // GCC otherwise treats the branch as cold and spills
                        // credit_reg to the stack, one LIM load per credit.
                        if (__builtin_expect(
                                pi_counter_distance(v.consumed, v.credited) == C && v.credit_reg != nullptr, 1)) {
                            Transport::counter_store(v.credit_reg, v.consumed);
                            v.credited = v.consumed;
                        }

                        const pi_error_t err = poll(w0 + j);
                        if (err != PI_ERROR_NONE) {
                            store_visits(true);
                            store_back();
                            return err;
                        }
                    }
                }
#pragma GCC unroll 4
                for (uint32_t j = 0; j < D; ++j) {
                    // return_credit() on the register copies.
                    Visit& v = visits[j];
                    if (((taken >> j) & 1u) && v.consumed != v.credited && v.credit_reg != nullptr) {
                        Transport::counter_store(v.credit_reg, v.consumed);
                        v.credited = v.consumed;
                    }
                }
                store_visits(false);
                taken = 0u;
            }
        }
        store_back();

        for (uint32_t w = 0; w < P_; ++w) {
            if (slots_[w].phase_done != limit) {
                first_failed_worker_ = w;
                return PI_ERROR_PHASE_OVERRUN;
            }
        }
        return PI_ERROR_NONE;
    }

    // Waits for every producer's terminal DONE word and attributes FAILED to its worker.
    pi_error_t wait_all_producers_complete() {
        phase_start_mtime_ = Transport::mtime();
        safety_work_ = 0u;

        for (uint32_t w = 0; w < P_; ++w) {
            while (true) {
                uint32_t val = ordered_ready_load(w);
                if (val == (uint32_t)PI_READY_DONE) {
                    break;
                }
                if (val == (uint32_t)PI_READY_FAILED) {
                    first_failed_worker_ = w;
                    return PI_ERROR_PEER_FAILED;
                }
                pi_error_t err = amortized_safety_check(true, w);
                if (err != PI_ERROR_NONE) {
                    return err;
                }
            }
        }
        return PI_ERROR_NONE;
    }

    // Retain the active image after deadline/STOP until a correlated ACK arrives.
    void wait_for_collect_ack() {
        phase_start_mtime_ = Transport::mtime();
        safety_work_ = 0u;

        while (true) {
            Transport::fence_load();
            if (host_control_->collect_ack_run_value == run_value_) {
                return;
            }
            pi_error_t err = amortized_safety_check(false);
            if (err != PI_ERROR_NONE) {
                record_failure(err, PI_PHASE_WAIT_COLLECT, UINT32_MAX);
            }
        }
    }

    /**
     * Polls the stop request, the phase deadline, and peer completion mirrors,
     * doing the work at most once per `poll_cadence` calls.
     *
     * Attributes every error it returns and owns `first_failed_worker_` for
     * them: a peer failure names the peer that failed, a deadline or a stop
     * names `blocked_on`. A caller must not set that field for these errors.
     *
     * @param check_peers false where the caller waits on no peer.
     * @param blocked_on  worker the caller is waiting for, or UINT32_MAX when
     *                    it waits for none.
     */
    pi_error_t amortized_safety_check(bool check_peers = true, uint32_t blocked_on = UINT32_MAX) {
        safety_work_ += 1u;
        if (safety_work_ < args_.poll_cadence) {
            return PI_ERROR_NONE;
        }
        safety_work_ = 0u;
        return safety_check(check_peers, blocked_on);
    }

    /** amortized_safety_check()'s checks, done on every call; same attribution contract. */
    pi_error_t safety_check(bool check_peers, uint32_t blocked_on) {
        if (Transport::check_stop(hartid_, run_value_)) {
            first_failed_worker_ = blocked_on;
            return PI_ERROR_STOPPED;
        }

        uint64_t now = Transport::mtime();
        if (now - phase_start_mtime_ >= args_.x280_timeout_ticks) {
            first_failed_worker_ = blocked_on;
            return PI_ERROR_TIMEOUT;
        }

        // Scan completion mirrors for peer failures.
        for (uint32_t w = 0; check_peers && w < P_; ++w) {
            uint32_t r = ordered_ready_load(w);
            if (r == (uint32_t)PI_READY_FAILED) {
                first_failed_worker_ = w;
                return PI_ERROR_PEER_FAILED;
            }
        }
        return PI_ERROR_NONE;
    }

    // Preserve the first failure; later recovery errors remain visible in cleanup flags.
    void record_failure(pi_error_t err, pi_phase_t phase, uint32_t worker_idx) {
        if (run_result_->failure.code != PI_ERROR_NONE) {
            return;
        }
        run_result_->state = PI_PHASE_FAILED;
        run_result_->failure = pi_failure_t{(uint32_t)err, (uint32_t)phase, worker_idx, (uint32_t)PI_ACTOR_X280};
        Transport::fence_full();
    }

    // Called only after every producer has published drained data. Export per-worker
    // measured prefixes while pinned, then wait for ACK before any verified cleanup.
    pi_error_t collect_and_cleanup() {
        phase_start_mtime_ = Transport::mtime();
        safety_work_ = 0;
        run_result_->measured_completed = total_measured_done_;
        run_result_->export_valid_bytes = 0;
        run_result_->aggregate_start_mtime = aggregate_start_mtime_;
        run_result_->aggregate_end_mtime = aggregate_end_mtime_;
        run_result_->aggregate_start_rdcycle = aggregate_start_rdcycle_;
        run_result_->aggregate_end_rdcycle = aggregate_end_rdcycle_;
        run_result_->empty_visits = empty_visits_;
        Transport::fence_full();
        if constexpr (Mode == 1) {
            for (uint32_t w = 0; w < P_; ++w) {
                const uint64_t count = measured_started_ ? slots_[w].phase_done : 0;
                const uint64_t base = (uint64_t)w * N_;
                for (uint64_t i = 0; i < count; ++i) {
                    export_samples_[base + i] = pinned_samples_[base + i];
                    const auto err = amortized_safety_check(false, w);
                    if (err != PI_ERROR_NONE) {
                        record_failure(err, PI_PHASE_EXPORT, w);
                    }
                }
                run_result_->export_valid_bytes += 8 * count;
            }
        }
        Transport::fence_full();
        run_result_->data_quiescent = 1;
        Transport::fence_full();
        wait_for_collect_ack();

        auto err = cleanup_pinned();
        if (err == PI_ERROR_NONE) {
            run_result_->cache_restored = 1;
            err = cleanup_windows();
            if (err == PI_ERROR_NONE) {
                run_result_->windows_cleared = 1;
            }
        }
        if (err != PI_ERROR_NONE) {
            record_failure(err, PI_PHASE_CLEANUP, UINT32_MAX);
        }
        const bool success = run_result_->failure.code == PI_ERROR_NONE;
        run_result_->state = success ? PI_PHASE_DONE : PI_PHASE_FAILED;
        Transport::fence_full();
        return static_cast<pi_error_t>(run_result_->failure.code);
    }

    pi_error_t fail_drain_and_cleanup(pi_error_t err, pi_phase_t phase, uint32_t worker_idx) {
        record_failure(err, phase, worker_idx);
        broadcast_control(PI_CONTROL_STOP);

        bool recovery_release_ok = true;
        if (!normal_release_issued_) {
            const auto& roster = PacketDirectorSyncConfig::kRoster;
            recovery_release_ok = SyncTransport::validate_roster(roster) &&
                                  SyncTransport::notify_group(roster, core_sync_release_token(args_.run_nonce));
        }

        // Bounded drain wait for all producers to reach DONE or FAILED
        phase_start_mtime_ = Transport::mtime();
        safety_work_ = 0u;
        bool all_drained = true;

        for (uint32_t w = 0; w < P_; ++w) {
            while (true) {
                uint32_t val = ordered_ready_load(w);
                if (val == (uint32_t)PI_READY_DONE || val == (uint32_t)PI_READY_FAILED) {
                    break;
                }
                uint64_t now = Transport::mtime();
                if (now - phase_start_mtime_ >= args_.x280_timeout_ticks) {
                    all_drained = false;
                    break;
                }
            }
            if (!all_drained) {
                break;
            }
        }

        if (all_drained && recovery_release_ok && counters_quiescent()) {
            return collect_and_cleanup();
        } else {
            // Producers stuck / unproven: do NOT unpin or clear windows!
            run_result_->data_quiescent = 0u;
        }

        Transport::fence_full();
        return err;
    }

    // True only when every published packet is consumed and its credit returned.
    bool counters_quiescent() {
        for (uint32_t w = 0; w < P_; ++w) {
            const uint32_t produced = ordered_produced_load(w);
            if (produced != slots_[w].consumed || slots_[w].consumed != slots_[w].credited) {
                return false;
            }
        }
        return true;
    }

    uint64_t hartid_;
    uint64_t run_value_;
    const pi_x280_args_t& args_;
    uint64_t arena_base_;
    uint32_t P_;
    uint32_t T_;
    uint32_t S_;
    uint32_t C_;
    uint32_t K_;
    uint64_t W_;
    uint64_t N_;

    // Left uninitialized: pi_compute_layout() writes every field or none, and
    // nothing reads it before that call succeeds. A zero-initializer here
    // becomes a memset call this freestanding image cannot link.
    pi_layout_t layout_;
    const x280_noc_coord_t* tile_coords_ = nullptr;
    pi_consumer_slot_t* slots_ = nullptr;
    pi_run_result_t* run_result_ = nullptr;
    volatile pi_host_control_t* host_control_ = nullptr;
    uint64_t* pinned_samples_ = nullptr;
    uint64_t* export_samples_ = nullptr;

    uint32_t tiles_configured_ = 0u;
    uint32_t current_timer_tile_ = UINT32_MAX;
    bool pinned_prepared_ = false;
    bool pin_policy_uncertain_ = false;
    bool windows_uncertain_ = false;
    bool measured_started_ = false;
    bool normal_release_issued_ = false;

    uint64_t remaining_total_ = 0;
    uint64_t total_measured_done_ = 0;
    uint64_t empty_visits_ = 0;
    uint64_t aggregate_start_mtime_ = 0;
    uint64_t aggregate_end_mtime_ = 0;
    uint64_t aggregate_start_rdcycle_ = 0;
    uint64_t aggregate_end_rdcycle_ = 0;
    uint32_t first_failed_worker_ = UINT32_MAX;

    uint32_t safety_work_ = 0u;
    uint64_t phase_start_mtime_ = 0;
};

#if X280_CORE_SYNC_X280_TRANSPORT
inline pi_error_t run_consumer(uint64_t hartid, uint64_t run_value, const pi_x280_args_t& args) {
    X280Consumer<
        PACKET_INJECTION_MODE,
        PACKET_INJECTION_INCLUDE_CREATION,
        DeviceX280Transport,
        core_sync::X280Transport<>>
        consumer(hartid, run_value, args);
    return consumer.run();
}
#endif

}  // namespace packet_injection

#endif  // X280_KERNELS_PACKET_INJECTION_X280_CONSUMER_HPP
