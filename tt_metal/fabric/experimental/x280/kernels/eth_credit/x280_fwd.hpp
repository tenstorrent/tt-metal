// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/eth_credit/x280_fwd.hpp
//
// The X280 forwarder: one hart per p, sending through tx_fifo[p] on the local
// E0 and draining in[p], in one loop. W1 goes through a strict TLB window onto
// the Ethernet tile, whose store completes only when the NoC write has; W2 and
// R1 go through a counter window onto the same tile (x280_counter_window.h).

#ifndef X280_KERNELS_ETH_CREDIT_X280_FWD_HPP
#define X280_KERNELS_ETH_CREDIT_X280_FWD_HPP

#include <stddef.h>
#include <stdint.h>

#include "eth_credit_abi.h"
#include "eth_credit_header.hpp"
#include "../shared/core_sync_barrier.hpp"
#include "../shared/core_sync_transport_x280.hpp"
#include "rt/x280_noc_map.h"
#include "x280_address_map.h"

#if defined(__riscv) && __riscv_xlen == 64
#define X280_ETH_CREDIT_FWD 1
#include "noc.h"
#include "../shared/x280_counter_window.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "rt/x280_measurement.h"
#else
#define X280_ETH_CREDIT_FWD 0
#endif

/* Nonzero builds the measurement modes the args flags select
 * (EC_FWD_FLAG_LEGS_ALONE, EC_FWD_FLAG_ROUND_TRIP). The benchmark image leaves
 * them out, so its code is unchanged by them. */
#ifndef EC_FWD_MEASURE
#define EC_FWD_MEASURE 0
#endif

static_assert(
    X280_TLB_ETH_CREDIT_PER_HART == 2u,
    "each forwarder hart needs two windows onto its Ethernet tile: strict for W1, a counter window for W2 and R1");
static_assert(
    X280_TLB_ETH_CREDIT_COUNT == EC_X280_HARTS * X280_TLB_ETH_CREDIT_PER_HART, "the reservation covers every hart");

#if X280_ETH_CREDIT_FWD

namespace eth_credit {

/* The director's roster arrays, written only by the director hart and borrowed
 * by the transport for the whole barrier. */
inline core_sync::X280Endpoint g_release_endpoints[EC_BARRIER_MAX_WORKERS]{};
inline core_sync::WordArrival g_arrivals[EC_BARRIER_MAX_WORKERS]{};

struct DirectorSync {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = true;
    static inline core_sync::X280Roster kRoster{};
};

/* One config per hart: harts run concurrently and a worker config is written
 * before its sync, so a shared one would be overwritten by another hart. */
template <uint32_t Hart>
struct WorkerSync {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = false;
    static inline uint32_t kLocalReleaseAddr = 0u;
    static inline core_sync::X280ArrivalTarget kArrivalTarget{};
};

class Forwarder {
public:
    /**
     * Binds one hart's run. Nothing is dereferenced beyond `args` until run()
     * has validated it; `run_value` is the lifecycle generation whose STOP
     * ends every wait.
     */
    Forwarder(uint64_t hartid, uint64_t run_value, const ec_x280_fwd_args_t& args) :
        hartid_(hartid), run_value_(run_value), a_(args), s_(args.shape) {}

    /** Runs the forwarder to DONE or FAILED. @return the terminal error. */
    ec_error_t run() {
        /* A record that fails these is not this hart's valid record, so there
         * is no area to hold a result; the hart's result word carries the
         * BAD_ARGS. */
        if (!validate_record()) {
            return EC_ERROR_BAD_ARGS;
        }
        initialize();
        if (!validate_window() || !configure_windows()) {
            return fail(EC_ERROR_BAD_ARGS, EC_CONSTRAINT_ETH_BASE);
        }
        if (!validate_controls()) {
            return fail(EC_ERROR_BAD_ARGS, EC_CONSTRAINT_ARGS);
        }
        const core_sync::Outcome sync = synchronize();
        if (sync != core_sync::Outcome::Success) {
            return fail(
                sync == core_sync::Outcome::StopRequested ? EC_ERROR_STALL_TIMEOUT : EC_ERROR_BAD_ARGS,
                EC_SITE_BARRIER);
        }
        ec_error_t err;
        if constexpr (EC_FWD_MEASURE) {
            if ((a_.flags & EC_FWD_FLAG_LEGS_ALONE) != 0u) {
                err = leg_alone();
            } else if ((a_.flags & EC_FWD_FLAG_ROUND_TRIP) != 0u) {
                result_->stalls[3].site = EC_SITE_FWD_ROUND_TRIP;
                err = loop<true>();
            } else {
                err = loop();
            }
        } else {
            err = loop();
        }
        if (err != EC_ERROR_NONE) {
            return err;
        }
        finish(ec_status_word(EC_STATE_DONE, kernel_id(), EC_ERROR_NONE, 0u));
        return EC_ERROR_NONE;
    }

private:
    static constexpr uint32_t kUnit = EC_UNIT_BYTES;
    static constexpr uint64_t kWindowBytes = NOC_2M_WINDOW_STRIDE;
    static constexpr uint32_t kHeaderWords = EC_HEADER_BYTES / sizeof(uint64_t);

    uint16_t kernel_id() const { return static_cast<uint16_t>(a_.kernel_id); }
    bool outbound() const { return (a_.flags & EC_FWD_FLAG_OUTBOUND_ACTIVE) != 0u; }
    bool inbound() const { return (a_.flags & EC_FWD_FLAG_INBOUND_ACTIVE) != 0u; }
    bool director() const { return (a_.flags & EC_FWD_FLAG_DIRECTOR) != 0u; }

    volatile uint32_t* lim_word(uint32_t offset) const {
        return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(area_ + offset));
    }
    volatile uint8_t* eth_window(uint32_t l1_addr) const { return window_ + l1_addr; }
    uint32_t ordered_load(uint32_t offset) const {
        x280_fence_ir();
        return *lim_word(offset);
    }

    /* Identity, shape and placement: what locating this hart's area needs. */
    bool validate_record() {
        if (a_.magic != EC_ABI_MAGIC || a_.p != hartid_ || ec_kernel_role(a_.kernel_id) != EC_ROLE_X280_FWD ||
            ec_kernel_index(a_.kernel_id) != a_.p) {
            return false;
        }
        if (ec_validate_shape(&s_).code != EC_CHECK_OK || a_.p >= s_.P ||
            ec_compute_x280_layout(&s_, &lim_).code != EC_CHECK_OK ||
            ec_compute_eth_layout(&s_, outbound() ? 1 : 0, inbound() ? 1 : 0, &eth_).code != EC_CHECK_OK) {
            return false;
        }
        /* The args block is the layout's first record, so where it was
         * published is a claim about lim_base to check, as packet_injection
         * checks its arena. */
        const uint64_t args_addr = a_.lim_base + lim_.args_offset + a_.p * sizeof(ec_x280_fwd_args_t);
        if (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&a_)) != args_addr ||
            (a_.lim_base % EC_LIM_ALIGN) != 0u) {
            return false;
        }
        area_ = a_.lim_base + lim_.fwd_offset + static_cast<uint64_t>(a_.p) * lim_.fwd_stride;
        const ec_bounds_t map{0u, 0u, X280_LIM_BASE, X280_LIM_END};
        ec_bounds_t bounds{};
        if (ec_narrow_lim(&map, a_.lim_end, &bounds).code != EC_CHECK_OK ||
            !ec_range_ok(area_, lim_.area_bytes, bounds.lim_lo, bounds.lim_hi)) {
            return false;
        }
        result_ = reinterpret_cast<volatile ec_result_t*>(static_cast<uintptr_t>(area_ + lim_.result));
        eth_tx_base_ = a_.eth_ctrl_base + eth_.tx_offset;
        return true;
    }

    /* The Ethernet tile's coordinate, this hart's window, and the control
     * area and tx_fifo inside the window. */
    bool validate_window() const {
        if (!x280_noc_coord_valid(a_.eth) || a_.tlb_base < X280_TLB_ETH_CREDIT_BASE ||
            a_.tlb_base + X280_TLB_ETH_CREDIT_PER_HART > X280_TLB_ETH_CREDIT_BASE + X280_TLB_ETH_CREDIT_COUNT) {
            return false;
        }
        const uint64_t eth_end = static_cast<uint64_t>(a_.eth_ctrl_base) + eth_.total_bytes;
        return (a_.eth_ctrl_base % EC_ETH_CONTROL_ALIGN) == 0u && eth_end <= kWindowBytes;
    }

    /* Everything else: the run's identity, the barrier role, and the
     * scalar controls. */
    bool validate_controls() const {
        return run_value_ != 0u && a_.run_nonce != 0u && a_.director_p < s_.P &&
               director() == (a_.p == a_.director_p) && ec_is_pow2(a_.poll_cadence) &&
               a_.poll_cadence <= EC_MAX_POLL_CADENCE && a_.stall_budget_ticks != 0u &&
               a_.stall_budget_ticks < (1ull << 63) && (outbound() || a_.send_count == 0u) &&
               (inbound() || a_.recv_count == 0u);
    }

    void initialize() {
        volatile uint32_t* r = reinterpret_cast<volatile uint32_t*>(result_);
        for (uint32_t i = 0; i < EC_RESULT_BYTES / sizeof(uint32_t); ++i) {
            r[i] = 0u;
        }
        result_->magic = EC_RESULT_MAGIC;
        result_->kernel_id = a_.kernel_id;
        const uint32_t sites[EC_MAX_STALL_SITES] = {
            EC_SITE_FWD_OUT_FULL, EC_SITE_FWD_W1_STORE, EC_SITE_FWD_IN_EMPTY, EC_SITE_NONE};
        for (uint32_t i = 0; i < EC_MAX_STALL_SITES; ++i) {
            result_->stalls[i].site = sites[i];
        }
        *lim_word(lim_.wr_src) = 0u;
        *lim_word(lim_.rd_src) = 0u;
        *lim_word(lim_.seq_expected) = 0u;
        volatile fs_hybrid_mesh_packet_header_t* staging =
            reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(static_cast<uintptr_t>(area_ + lim_.staging));
        ec_header_seed(staging, a_.dst_chip, a_.dst_mesh, a_.p, ec_payload_bytes(&s_), 0u);
        x280_fence_ow();
    }

    bool configure_windows() {
        volatile void* base = nullptr;
        const noc_window_status_t opened =
            noc_open_window_2m(a_.tlb_base, a_.eth, 0u, NOC_ORDER_STRICT, NOC_VC_DYNAMIC, &base);
        /* An unverified window is kept so finish() clears it: both slots are
         * this forwarder hart's own reservation (fwd_tlb_base), so a mapping
         * left in one is a standing path no other owner relies on. */
        window_ = static_cast<volatile uint8_t*>(base);
        if (opened != NOC_WINDOW_OPEN) {
            return false;
        }
        volatile void* counter = nullptr;
        const noc_window_status_t counter_opened = x280_counter_window_open(a_.tlb_base + 1u, a_.eth, &counter);
        counter_window_ = static_cast<volatile uint8_t*>(counter);
        return counter_opened == NOC_WINDOW_OPEN;
    }

    void clear_windows() {
        if (window_ != nullptr) {
            (void)noc_clear_tlb_2m(a_.tlb_base);
            window_ = nullptr;
        }
        if (counter_window_ != nullptr) {
            (void)noc_clear_tlb_2m(a_.tlb_base + 1u);
            counter_window_ = nullptr;
        }
    }

    template <uint32_t Hart>
    core_sync::Outcome worker_sync() {
        WorkerSync<Hart>::kLocalReleaseAddr = static_cast<uint32_t>(area_ + lim_.release);
        WorkerSync<Hart>::kArrivalTarget = core_sync::X280ArrivalTarget{
            {0u, 0u}, static_cast<uint32_t>(a_.lim_base + lim_.ready_offset), ec_barrier_index_fwd(a_.p), 1u};
        return core_sync::core_sync_barrier<WorkerSync<Hart>, core_sync::X280Transport<>>::sync(
            hartid_, run_value_, a_.run_nonce);
    }

    core_sync::Outcome synchronize() {
        if (!director()) {
            switch (hartid_) {
                case 0: return worker_sync<0>();
                case 1: return worker_sync<1>();
                case 2: return worker_sync<2>();
                case 3: return worker_sync<3>();
                default: return core_sync::Outcome::ConfigError;
            }
        }
        const uint32_t ready = static_cast<uint32_t>(a_.lim_base + lim_.ready_offset);
        /* Every participant both arrives and is released. */
        uint32_t n = 0u;
        const auto add = [&](uint32_t index, core_sync::X280Endpoint release) {
            g_arrivals[n] = core_sync::WordArrival{core_sync_slot_addr(ready, index), index};
            g_release_endpoints[n++] = release;
        };
        add(ec_barrier_index_e0(), {{a_.eth.x, a_.eth.y}, a_.eth_ctrl_base + eth_.e0_release, 0u});
        if (inbound()) {
            add(ec_barrier_index_e1(), {{a_.eth.x, a_.eth.y}, a_.eth_ctrl_base + eth_.e1_release, 0u});
        }
        for (uint32_t q = 0; q < s_.P; ++q) {
            if (q != a_.p) {
                const uint64_t other = a_.lim_base + lim_.fwd_offset + static_cast<uint64_t>(q) * lim_.fwd_stride;
                add(ec_barrier_index_fwd(q), {{0u, 0u}, static_cast<uint32_t>(other + lim_.release), 1u});
            }
        }
        DirectorSync::kRoster = core_sync::X280Roster{g_release_endpoints, n, 0u, 0u, g_arrivals, n, nullptr, 0u};
        return core_sync::core_sync_barrier<DirectorSync, core_sync::X280Transport<>>::sync(
            hartid_, run_value_, a_.run_nonce);
    }

    void finish(const ec_status_t& s) {
        result_->status.origin = s.origin;
        result_->status.code = s.code;
        result_->status.detail = s.detail;
        x280_fence_ow();
        result_->status.state = s.state;
        x280_fence_ow();
        clear_windows();
    }

    /* Records FAILED in this hart's result. Neighbours learn nothing and exit
     * on their own stall budgets; the host attributes the failure. */
    ec_error_t fail(ec_error_t code, uint32_t detail) {
        finish(ec_status_word(EC_STATE_FAILED, kernel_id(), code, detail));
        return code;
    }

    /* What the loop reads and updates on every pass, built once by loop() and
     * passed down by reference to always_inline steps, so it stays in
     * registers under the timed-loop rule (transport_bandwidth_frontend_cxx.hpp):
     * the Forwarder itself is on the uncached LIM stack. */
    struct Hot {
        uint32_t wr;
        uint32_t rd;
        uint32_t sent;
        uint32_t consumed;
        uint32_t safety_work;
        uint64_t last_send;
        uint64_t last_recv;

        uint32_t send_count;
        uint32_t recv_count;
        uint32_t n_link;
        uint32_t n_deliver;
        uint32_t batch;
        uint32_t slot_size;
        uint32_t P;
        uint32_t p;
        uint32_t poll_cadence;
        uint64_t stall_budget_ticks;
        volatile ec_result_t* result;
        const volatile uint32_t* rd_sh;
        const volatile uint32_t* wr_sh;
        volatile uint32_t* wr_src;
        volatile uint32_t* rd_src;
        volatile uint32_t* seq_expected;
        volatile fs_hybrid_mesh_packet_header_t* staging;
        uint64_t in_slots;
        /* tx_fifo[p] inside the strict window, and the two shadows inside the
         * counter window, both onto the Ethernet tile. */
        volatile uint8_t* tx_fifo;
        volatile uint32_t* e0_wr_sh;
        volatile uint32_t* e1_out_rd_sh;
    };

    Hot hot() const {
        return Hot{
            .wr = 0u,
            .rd = 0u,
            .sent = 0u,
            .consumed = 0u,
            .safety_work = 0u,
            .last_send = 0u,
            .last_recv = 0u,
            .send_count = a_.send_count,
            .recv_count = a_.recv_count,
            .n_link = s_.n_link,
            .n_deliver = s_.n_deliver,
            .batch = s_.batch,
            .slot_size = s_.slot_size,
            .P = s_.P,
            .p = a_.p,
            .poll_cadence = a_.poll_cadence,
            .stall_budget_ticks = a_.stall_budget_ticks,
            .result = result_,
            .rd_sh = lim_word(lim_.rd_sh),
            .wr_sh = lim_word(lim_.wr_sh),
            .wr_src = lim_word(lim_.wr_src),
            .rd_src = lim_word(lim_.rd_src),
            .seq_expected = lim_word(lim_.seq_expected),
            .staging = reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(
                static_cast<uintptr_t>(area_ + lim_.staging)),
            .in_slots = area_ + lim_.in_slots,
            .tx_fifo = eth_window(eth_tx_base_ + a_.p * eth_.fifo_stride),
            .e0_wr_sh =
                reinterpret_cast<volatile uint32_t*>(counter_window_ + a_.eth_ctrl_base + eth_.e0_wr_sh + a_.p * kUnit),
            .e1_out_rd_sh = reinterpret_cast<volatile uint32_t*>(
                counter_window_ + a_.eth_ctrl_base + eth_.e1_out_rd_sh + a_.p * kUnit),
        };
    }

    /* Polled at the cadence: host STOP and the stall budget. A timeout, or a
     * host STOP, names the unfinished side that has gone longest without
     * progress. */
    __attribute__((always_inline)) ec_error_t safety_check(Hot& h) {
        if (++h.safety_work < h.poll_cadence) {
            return EC_ERROR_NONE;
        }
        h.safety_work = 0u;
        const bool sending = h.sent < h.send_count;
        const bool receiving = h.consumed < h.recv_count;
        const bool send_oldest = sending && (!receiving || h.last_send <= h.last_recv);
        const uint64_t oldest = send_oldest ? h.last_send : h.last_recv;
        if (x280_hart_stop_requested(hartid_, run_value_) || x280_mtime_fenced() - oldest >= h.stall_budget_ticks) {
            const uint32_t site = send_oldest ? EC_SITE_FWD_OUT_FULL : EC_SITE_FWD_IN_EMPTY;
            result_->failure.site = site;
            result_->failure.local_ptr = send_oldest ? h.wr : h.rd;
            result_->failure.shadow_ptr = ordered_load(send_oldest ? lim_.rd_sh : lim_.wr_sh);
            return fail(EC_ERROR_STALL_TIMEOUT, site);
        }
        return EC_ERROR_NONE;
    }

    /* Stamps the window at `now`, after a batch of `batch_packets` sent or
     * received; the first stamp keeps that count as first_batch_packets. */
    __attribute__((always_inline)) static void stamp(Hot& h, uint64_t now, uint32_t batch_packets) {
        if (h.result->first_ts == 0u) {
            h.result->first_ts = now;
            h.result->first_batch_packets = batch_packets;
        }
        h.result->last_ts = now;
    }

    __attribute__((always_inline)) static void stall(Hot& h, uint32_t index, uint64_t start, uint64_t end) {
        h.result->stalls[index].count += 1u;
        h.result->stalls[index].ticks = ec_stall_add(h.result->stalls[index].ticks, start, end);
    }

    ec_error_t distance_fail(uint32_t local, uint32_t shadow, uint32_t site) {
        result_->failure.local_ptr = local;
        result_->failure.shadow_ptr = shadow;
        result_->failure.site = site;
        return fail(EC_ERROR_COUNTER_DISTANCE, site);
    }

    /* EC_FWD_FLAG_ROUND_TRIP's state: the one packet being timed, from the W2
     * that publishes it to the first R3 that covers it. */
    struct RoundTrip {
        bool timing;
        uint32_t sent;      /* h.sent once that W2 has published it */
        uint64_t published; /* mtime after that W2 */
    };

    /* W1 for `k` slots from h.wr, then W2. Sets *progressed when it sends; with
     * no credit it returns EC_ERROR_NONE and leaves *progressed unset. With
     * TimeRoundTrip, adds the round trip of the packet `rt` times to stalls[3]
     * when R3 first covers it, and starts timing the last packet of a W2 when
     * none is being timed. */
    template <bool TimeRoundTrip = false>
    __attribute__((always_inline)) ec_error_t send_step(Hot& h, RoundTrip& rt, bool* progressed) {
        x280_fence_ir();
        const uint32_t rd = *h.rd_sh;
        if (!ec_ptr_distance_ok(h.wr, rd, h.n_link)) {
            return distance_fail(h.wr, rd, EC_SITE_FWD_OUT_FULL);
        }
        if constexpr (TimeRoundTrip) {
            /* R3 covers the packets sent less those still in tx_fifo[p]. */
            if (rt.timing && h.sent - ec_ptr_used(h.wr, rd, h.n_link) >= rt.sent) {
                stall(h, 3u, rt.published, x280_mtime_fenced());
                rt.timing = false;
            }
        }
        uint32_t k = ec_ptr_credit(h.wr, rd, h.n_link);
        k = k < h.batch ? k : h.batch;
        k = k < h.send_count - h.sent ? k : h.send_count - h.sent;
        if (k == 0u) {
            return EC_ERROR_NONE;
        }
        const uint64_t t0 = x280_mtime_fenced();
        const volatile uint64_t* src = reinterpret_cast<const volatile uint64_t*>(h.staging);
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t slot = ec_ptr_slot(ec_ptr_advance(h.wr, j, h.n_link), h.n_link);
            ec_header_set_seq(h.staging, h.sent + j);
            x280_fence_iorw();
            volatile uint64_t* dst = reinterpret_cast<volatile uint64_t*>(h.tx_fifo + slot * h.slot_size);
            for (uint32_t i = 0; i < kHeaderWords; ++i) {
                dst[i] = src[i];
            }
        }
        x280_fence_iorw();
        const uint64_t t1 = x280_mtime_fenced();
        stall(h, 1u, t0, t1);
        h.wr = ec_ptr_advance(h.wr, k, h.n_link);
        *h.wr_src = h.wr;
        x280_counter_store(h.e0_wr_sh, h.wr);
        h.sent += k;
        if constexpr (TimeRoundTrip) {
            if (!rt.timing) {
                rt = RoundTrip{true, h.sent, x280_mtime_fenced()};
            }
        }
        h.result->packets_out[h.p] = h.sent;
        stamp(h, t1, k);
        *progressed = true;
        return EC_ERROR_NONE;
    }

    __attribute__((always_inline)) ec_error_t check_header(Hot& h, uint32_t slot) {
        const volatile fs_hybrid_mesh_packet_header_t* hdr =
            reinterpret_cast<const volatile fs_hybrid_mesh_packet_header_t*>(
                static_cast<uintptr_t>(h.in_slots + slot * EC_HEADER_BYTES));
        const uint32_t ch = fs_src_ch_id(hdr);
        const uint32_t seq = ec_header_seq(hdr);
        const uint32_t expected = *h.seq_expected;
        /* Each inbound FIFO has one source, so the channel is checked against
         * P before it names anything, and then against this forwarder. */
        if (ch >= h.P || ch != h.p || seq != expected) {
            result_->failure.p = ch;
            result_->failure.expected = expected;
            result_->failure.seen = seq;
            return fail(EC_ERROR_BAD_SEQUENCE, expected);
        }
        *h.seq_expected = expected + 1u;
        return EC_ERROR_NONE;
    }

    __attribute__((always_inline)) static void publish_rd(Hot& h) {
        *h.rd_src = h.rd;
        x280_counter_store(h.e1_out_rd_sh, h.rd);
    }

    /* Consumes up to BATCH headers from in[p], then R1. */
    __attribute__((always_inline)) ec_error_t receive_step(Hot& h, bool* progressed) {
        x280_fence_ir();
        const uint32_t w = *h.wr_sh;
        if (!ec_ptr_distance_ok(w, h.rd, h.n_deliver)) {
            return distance_fail(h.rd, w, EC_SITE_FWD_IN_EMPTY);
        }
        uint32_t k = ec_ptr_used(w, h.rd, h.n_deliver);
        k = k < h.batch ? k : h.batch;
        k = k < h.recv_count - h.consumed ? k : h.recv_count - h.consumed;
        if (k == 0u) {
            return EC_ERROR_NONE;
        }
        /* The pointer load and the slot loads it covers are ordered only by a
         * control dependency, which RVWMO does not honour. */
        x280_fence_ir();
        for (uint32_t j = 0; j < k; ++j) {
            const ec_error_t err = check_header(h, ec_ptr_slot(ec_ptr_advance(h.rd, j, h.n_deliver), h.n_deliver));
            if (err != EC_ERROR_NONE) {
                return err;
            }
        }
        h.rd = ec_ptr_advance(h.rd, k, h.n_deliver);
        publish_rd(h);
        h.consumed += k;
        h.result->packets_in[h.p] = h.consumed;
        stamp(h, x280_mtime_fenced(), k);
        *progressed = true;
        return EC_ERROR_NONE;
    }

    template <bool TimeRoundTrip = false>
    ec_error_t loop() {
        Hot h = hot();
        RoundTrip rt{false, 0u, 0u};
        h.last_send = x280_mtime_fenced();
        h.last_recv = h.last_send;
        while (h.sent < h.send_count || h.consumed < h.recv_count) {
            const uint64_t t0 = x280_mtime_fenced();
            bool sent_progress = false;
            bool recv_progress = false;
            if (h.sent < h.send_count) {
                const ec_error_t err = send_step<TimeRoundTrip>(h, rt, &sent_progress);
                if (err != EC_ERROR_NONE) {
                    return err;
                }
                if (sent_progress) {
                    h.last_send = x280_mtime_fenced();
                } else {
                    stall(h, 0u, t0, x280_mtime_fenced());
                }
            }
            if (h.consumed < h.recv_count) {
                const ec_error_t err = receive_step(h, &recv_progress);
                if (err != EC_ERROR_NONE) {
                    return err;
                }
                if (recv_progress) {
                    h.last_recv = x280_mtime_fenced();
                } else {
                    stall(h, 2u, t0, x280_mtime_fenced());
                }
            }
            const ec_error_t err = safety_check(h);
            if (err != EC_ERROR_NONE) {
                return err;
            }
        }
        return EC_ERROR_NONE;
    }

    /* W2's and R1's store on this build: x280_counter_store() through the
     * counter window, as send_step() and publish_rd() store. */
    static void pointer_store(volatile uint32_t* dst, uint32_t value) { x280_counter_store(dst, value); }

    /* `step(i)` for i in [0, n), counted in packets_out, the window opening at
     * the stamp after step 0 as every window opens after its first batch. */
    template <typename Step>
    __attribute__((always_inline)) void time_alone(uint32_t n, Step step) {
        step(0u);
        x280_fence_iorw();
        const uint64_t first = x280_mtime_fenced();
        for (uint32_t i = 1u; i < n; ++i) {
            step(i);
        }
        x280_fence_iorw();
        const uint64_t last = x280_mtime_fenced();
        result_->first_ts = first;
        result_->first_batch_packets = 1u;
        result_->last_ts = last;
        result_->packets_out[a_.p] = n;
    }

    /* The leg its EC_FWD_FLAG_*_ALONE flag names, alone, send_count times back
     * to back in a loop whose operands stay in registers: W1 is one header
     * into tx_fifo[p] as send_step() writes it, W2 and R1 one pointer_store()
     * each into E0's and E1's shadow. Nothing else runs. */
    ec_error_t leg_alone() {
        const uint32_t n = a_.send_count;
        if ((a_.flags & EC_FWD_FLAG_W1_ALONE) != 0u) {
            volatile fs_hybrid_mesh_packet_header_t* const staging =
                reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(
                    static_cast<uintptr_t>(area_ + lim_.staging));
            const volatile uint64_t* const src = reinterpret_cast<const volatile uint64_t*>(staging);
            volatile uint8_t* const tx = eth_window(eth_tx_base_ + a_.p * eth_.fifo_stride);
            const uint32_t mask = s_.n_link - 1u;
            const uint32_t slot_size = s_.slot_size;
            time_alone(n, [=](uint32_t i) __attribute__((always_inline)) {
                ec_header_set_seq(staging, i);
                x280_fence_iorw();
                volatile uint64_t* dst = reinterpret_cast<volatile uint64_t*>(tx + (i & mask) * slot_size);
                for (uint32_t w = 0; w < kHeaderWords; ++w) {
                    dst[w] = src[w];
                }
            });
            return EC_ERROR_NONE;
        }
        const bool w2 = (a_.flags & EC_FWD_FLAG_W2_ALONE) != 0u;
        volatile uint32_t* const dst = reinterpret_cast<volatile uint32_t*>(
            counter_window_ + a_.eth_ctrl_base + (w2 ? eth_.e0_wr_sh : eth_.e1_out_rd_sh) + a_.p * kUnit);
        const uint32_t mask = ec_ptr_mask(w2 ? s_.n_link : s_.n_deliver);
        time_alone(n, [=](uint32_t i) __attribute__((always_inline)) { pointer_store(dst, (i + 1u) & mask); });
        return EC_ERROR_NONE;
    }

    uint64_t hartid_;
    uint64_t run_value_;
    const ec_x280_fwd_args_t& a_;
    ec_shape_t s_;
    /* Written by the layout calls before any read; a zero-initializer becomes
     * a memset call this freestanding image cannot link. */
    ec_x280_layout_t lim_;
    ec_eth_layout_t eth_;
    uint64_t area_ = 0u;
    volatile ec_result_t* result_ = nullptr;
    /* E0's tx_fifo region, at the layout's tx_offset past eth_ctrl_base. */
    uint32_t eth_tx_base_ = 0u;
    /* The verified strict window onto the Ethernet tile, at its L1 address 0. */
    volatile uint8_t* window_ = nullptr;
    /* The counter window onto the same tile and address, for W2 and R1. */
    volatile uint8_t* counter_window_ = nullptr;
};

/**
 * Runs hart `hartid` as the forwarder its record in the args block names.
 * @return EC_ERROR_NONE for a hart with no record, as it has nothing to run.
 */
inline ec_error_t run_forwarder(uint64_t hartid, uint64_t run_value, const ec_x280_fwd_args_t* block) {
    if (hartid >= EC_X280_HARTS || block[hartid].magic == 0u) {
        return EC_ERROR_NONE;
    }
    Forwarder fwd(hartid, run_value, block[hartid]);
    return fwd.run();
}

}  // namespace eth_credit

#endif  // X280_ETH_CREDIT_FWD

#endif  // X280_KERNELS_ETH_CREDIT_X280_FWD_HPP
