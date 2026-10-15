// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/eth_credit/e1_kernel.cpp -- ERISC1, the inbound role.
//
// Reads pkts[p] (a register read, no fence), delivers each rx[p] header into
// forwarder p's in[p] (W5), waits for those writes' acknowledgement, publishes
// the new write pointer (W6), and returns the freed rx[p] slots to the peer E0
// over TXQ 1 (R2).

#include <stdint.h>

#include "eth_credit_header.hpp"
#include "metal_common.hpp"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

/* Bring-up modes, each off in the benchmark build. EC_E1_SINK_LOCAL consumes
 * rx here, checking each header's sequence number and channel, and returns
 * R2 without delivering, so the Ethernet hop runs alone; EC_E1_SOURCE_SELF
 * fills rx here, with every slot always full, and delivers it without reading
 * pkts or returning R2, so the receive side (W5, W6, R1) runs alone;
 * EC_E1_R2_AFTER_CONSUME returns R2 for a packet only once the receiving
 * forwarder's R1 shows it consumed, not once W5 is acknowledged, so the
 * sender's R3 covers the remote consume: a harness that times the end-to-end
 * credit's round trip on the as-built path; EC_TAIL_MARKER checks each slot's
 * last 16 bytes against the marker E0 stamped. */
#ifndef EC_E1_SINK_LOCAL
#define EC_E1_SINK_LOCAL 0
#endif
#ifndef EC_E1_SOURCE_SELF
#define EC_E1_SOURCE_SELF 0
#endif
#ifndef EC_E1_R2_AFTER_CONSUME
#define EC_E1_R2_AFTER_CONSUME 0
#endif
#ifndef EC_TAIL_MARKER
#define EC_TAIL_MARKER 0
#endif

static_assert(
    !EC_TAIL_MARKER || EC_SLOT_SIZE >= EC_HEADER_BYTES + EC_TAIL_MARKER_BYTES,
    "the tail marker needs a payload of at least 16 bytes");
static_assert(
    !(EC_E1_SOURCE_SELF && (EC_E1_SINK_LOCAL || EC_TAIL_MARKER)),
    "a local source neither sinks locally nor receives a marker over the link");
static_assert(
    !(EC_E1_R2_AFTER_CONSUME && (EC_E1_SINK_LOCAL || EC_E1_SOURCE_SELF)),
    "R2 after the consume needs packets from the link delivered to the forwarder");

namespace {

using namespace eth_credit::metal;

constexpr uint32_t kTxq = 1u;
constexpr uint32_t kSlotBytes = EC_SLOT_SIZE;
constexpr uint32_t kSites[EC_MAX_STALL_SITES] = {
    EC_SITE_E1_LINK_EMPTY, EC_SITE_E1_IN_FULL, EC_SITE_E1_NOC_BARRIER, EC_SITE_E1_TXQ1_BUSY};

class E1 {
public:
    explicit E1(const ec_e1_args_t& a) : a_(a) {}

    void run() {
        if (!bind()) {
            return;
        }
        initialize();
        /* TXQ 1 is touched only after release: E0 programs both queues before
         * it arrives. */
        const core_sync::Outcome sync = barrier<ProgrammableCoreType::ACTIVE_ETH>(
            a_.erisc.fwd,
            a_.erisc.barrier_ready_base,
            ec_barrier_index_e1(),
            a_.erisc.ctrl_base + eth_.e1_release,
            barrier_cap(),
            a_.erisc.run_nonce);
        if (sync != core_sync::Outcome::Success) {
            fail(barrier_error(sync), EC_SITE_BARRIER);
            return;
        }
        if (!loop()) {
            return;
        }
        result_.finish(ec_status_word(EC_STATE_DONE, a_.erisc.kernel_id, EC_ERROR_NONE, 0u));
    }

private:
    uint64_t budget() const { return ec_u64(a_.erisc.stall_budget_lo, a_.erisc.stall_budget_hi); }
    uint64_t barrier_cap() const { return ec_u64(a_.erisc.barrier_cap_lo, a_.erisc.barrier_cap_hi); }
    uint32_t unit(uint32_t base, uint32_t p) const { return a_.erisc.ctrl_base + base + p * kUnit; }

    /* Validates in the order each check makes the next action safe, as E0's
     * bind() does: an unbounded ctrl_base leaves no result, and every later
     * rejection is BAD_ARGS in its own result. The rx region's offset depends on whether
     * this tile also sends, so ec_check_e1_args bounds it by alignment and
     * slot fit rather than against one offset. */
    bool bind() {
        if (ec_compute_eth_layout(&kShape, 1, 1, &eth_).code != EC_CHECK_OK ||
            ec_compute_x280_layout(&kShape, &lim_).code != EC_CHECK_OK ||
            ec_check_erisc_ctrl(a_.erisc.ctrl_base, &eth_, &kBounds).code != EC_CHECK_OK) {
            return false;
        }
        result_.init(a_.erisc.ctrl_base + eth_.e1_result, a_.erisc.kernel_id, kSites);
        const ec_check_t args = ec_check_e1_args(&a_, &kShape, &kBounds);
        if (args.code != EC_CHECK_OK) {
            result_.bad_args(args);
            fail(EC_ERROR_BAD_ARGS, args.constraint);
            return false;
        }
        return true;
    }

    void initialize() {
        for (uint32_t p = 0; p < EC_P; ++p) {
            set_local_word(unit(eth_.e1_crd_src, p), 0u);
            set_local_word(unit(eth_.e1_out_wr_src, p), 0u);
        }
    }

    bool wait_txq(uint64_t* stalled) { return bounded_txq_idle(kTxq, budget(), stalled); }

    /* Records FAILED in this kernel's result. Neighbours learn nothing and
     * exit on their own stall budgets; the host attributes the failure. */
    void fail(uint32_t code, uint32_t detail) {
        result_.finish(ec_status_word(EC_STATE_FAILED, a_.erisc.kernel_id, code, detail));
    }

    /* The stall budget per channel and link state. A timeout names the
     * lowest-indexed unfinished channel past its budget, and the site it was
     * stalled at. */
    bool safety_check() {
        if (++safety_work_ < a_.erisc.poll_cadence) {
            return true;
        }
        safety_work_ = 0u;
        if (!link_up()) {
            fail(EC_ERROR_LINK_DOWN, 0u);
            return false;
        }
        const uint64_t t = now();
        for (uint32_t p = 0; p < EC_P; ++p) {
            if (!channel_done(p) && t - last_progress_[p] >= budget()) {
                result_.failure(p, 0u, 0u, site_[p], link_rd_[p], deliver_wr_[p]);
                fail(EC_ERROR_STALL_TIMEOUT, site_[p]);
                return false;
            }
        }
        return true;
    }

    bool timeout(uint32_t p, uint32_t site) {
        result_.failure(p, 0u, 0u, site, link_rd_[p], deliver_wr_[p]);
        fail(EC_ERROR_STALL_TIMEOUT, site);
        return false;
    }

    /* EC_E1_SINK_LOCAL checks the header E0 filled; EC_TAIL_MARKER checks the
     * tail E0 stamped. A mismatch is BAD_SEQUENCE naming the packet expected
     * and the ordinal or sequence found. */
    bool check_slot(uint32_t p, uint32_t slot_addr, uint32_t ordinal) {
        if constexpr (EC_E1_SINK_LOCAL) {
            const auto* h = reinterpret_cast<const volatile fs_hybrid_mesh_packet_header_t*>(slot_addr);
            const uint32_t seq = ec_header_seq(h);
            const uint32_t ch = fs_src_ch_id(h);
            if (seq != ordinal || ch != p) {
                result_.failure(ch, ordinal, seq, EC_SITE_NONE, link_rd_[p], 0u);
                fail(EC_ERROR_BAD_SEQUENCE, ordinal);
                return false;
            }
        }
        if constexpr (EC_TAIL_MARKER) {
            uint32_t m[4];
            ec_tail_marker(p, ordinal, m);
            const volatile tt_l1_ptr uint32_t* tail =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(slot_addr + kSlotBytes - EC_TAIL_MARKER_BYTES);
            if (tail[0] != m[0] || tail[1] != m[1] || tail[2] != m[2] || tail[3] != m[3]) {
                result_.failure(p, ordinal, tail[1], EC_SITE_NONE, link_rd_[p], tail[0]);
                fail(EC_ERROR_BAD_SEQUENCE, ordinal);
                return false;
            }
        }
        return true;
    }

    /* One delivery turn for p: up to BATCH packets, W5, barrier, W6, then R2. */
    bool deliver(uint32_t p, uint32_t* progressed, uint32_t* site) {
        const uint32_t pkts =
            EC_E1_SOURCE_SELF ? EC_N_LINK
                              : static_cast<uint32_t>(get_ptr_val(static_cast<uint8_t>(a_.erisc.stream_id[p])));
        if (pkts > EC_N_LINK) {
            result_.failure(p, 0u, 0u, EC_SITE_NONE, pkts, link_rd_[p]);
            fail(EC_ERROR_COUNTER_DISTANCE, p);
            return false;
        }
        if (pkts == 0u) {
            *site = EC_SITE_E1_LINK_EMPTY;
            return true;
        }
        const uint32_t frd = EC_E1_SINK_LOCAL ? deliver_wr_[p] : local_word(unit(eth_.e1_out_rd_sh, p));
        if (!ec_ptr_distance_ok(deliver_wr_[p], frd, EC_N_DELIVER)) {
            result_.failure(p, 0u, 0u, EC_SITE_NONE, deliver_wr_[p], frd);
            fail(EC_ERROR_COUNTER_DISTANCE, p);
            return false;
        }
        if constexpr (!EC_E1_SINK_LOCAL) {
            /* R1 only advances, and by no more than E1 had delivered when it
             * last read it; anything else is an older R1 landing after a newer
             * one. */
            uint32_t advance;
            if (!credit_pair_ok(
                    frd,
                    frd_seen_[p],
                    ec_ptr_mask(EC_N_DELIVER),
                    ec_ptr_used(deliver_wr_[p], frd_seen_[p], EC_N_DELIVER),
                    &advance)) {
                result_.failure(p, 0u, 0u, EC_SITE_NONE, frd_seen_[p], frd);
                fail(EC_ERROR_COUNTER_DISTANCE, p);
                return false;
            }
            frd_seen_[p] = frd;
        }
        const uint32_t free = ec_ptr_credit(deliver_wr_[p], frd, EC_N_DELIVER);
        uint32_t k = pkts < free ? pkts : free;
        k = k < EC_BATCH ? k : EC_BATCH;
        k = k < a_.erisc.count[p] - received_[p] ? k : a_.erisc.count[p] - received_[p];
        if (k == 0u) {
            *site = EC_SITE_E1_IN_FULL;
            return true;
        }
        if constexpr (EC_E1_SINK_LOCAL || EC_TAIL_MARKER) {
            /* The slots were written by the RX queue, which the L0 cache does
             * not observe. */
            invalidate_l1_cache();
        }
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t src = a_.rx_base + p * eth_.fifo_stride + ec_ptr_slot(link_rd_[p], EC_N_LINK) * kSlotBytes;
            const uint32_t i = ec_ptr_slot(deliver_wr_[p], EC_N_DELIVER);
            if (!check_slot(p, src, received_[p] + j)) {
                return false;
            }
            if constexpr (EC_E1_SOURCE_SELF) {
                /* The fields the forwarder checks. */
                ec_header_mark(
                    reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(src), p, received_[p] + j);
            }
            if constexpr (!EC_E1_SINK_LOCAL) {
                noc_async_write(
                    src,
                    get_noc_addr(
                        a_.erisc.fwd.x, a_.erisc.fwd.y, a_.erisc.fwd_area[p] + lim_.in_slots + i * EC_HEADER_BYTES),
                    EC_HEADER_BYTES);
            }
            if constexpr (!EC_E1_SOURCE_SELF) {
                increment_local_update_ptr_val(static_cast<uint8_t>(a_.erisc.stream_id[p]), -1);
            }
            link_rd_[p] = ec_ptr_advance(link_rd_[p], 1u, EC_N_LINK);
            deliver_wr_[p] = ec_ptr_advance(deliver_wr_[p], 1u, EC_N_DELIVER);
        }
        /* W5's headers and W6's pointer reach the forwarder's LIM on one NoC
         * and one static VC. That such writes are delivered to AXI in order is
         * item 17 of docs/measurement.md "Unconfirmed hardware assumptions": stated
         * by the NoC specification, not confirmed on silicon.
         * W6 still follows W5's acknowledgement, so it is issued only after W5
         * was accepted at the L2CPU: that W5 is then visible to X280 loads from
         * LIM rests on the L2CPU applying accepted writes to LIM in arrival
         * order, an owner assumption (docs/measurement.md "Unconfirmed hardware
         * assumptions", item 2),
         * since NoC/Counters.md defines the ack for "L1 or register space"
         * only. */
        if constexpr (!EC_E1_SINK_LOCAL) {
            const uint64_t t0 = now();
            bool waited = false;
            if (!metal_wait::write_barrier(noc_index, budget(), &waited)) {
                return timeout(p, EC_SITE_E1_NOC_BARRIER);
            }
            if (waited) {
                result_.stall(2u, t0, now());
            }

            set_local_word(unit(eth_.e1_out_wr_src, p), deliver_wr_[p]);
            noc_async_write(
                unit(eth_.e1_out_wr_src, p),
                get_noc_addr(a_.erisc.fwd.x, a_.erisc.fwd.y, a_.erisc.fwd_area[p] + lim_.wr_sh),
                kUnit);
        }

        if constexpr (!EC_E1_SOURCE_SELF && !EC_E1_R2_AFTER_CONSUME) {
            if (!send_r2(p, link_rd_[p])) {
                return false;
            }
        }
        received_[p] += k;
        result_.packets_in(p, received_[p]);
        result_.packets_out(p, received_[p]);
        result_.stamp(now(), k);
        *progressed = 1u;
        return true;
    }

    /* R2: link read pointer `rd` to the peer E0 over TXQ 1. A resend of R2
     * rereads crd_src, which only ever holds the same or a newer read pointer,
     * so overwriting it while TXQ 1 may resend is safe. */
    bool send_r2(uint32_t p, uint32_t rd) {
        uint64_t busy = 0u;
        if (!wait_txq(&busy)) {
            return timeout(p, EC_SITE_E1_TXQ1_BUSY);
        }
        set_local_word(unit(eth_.e1_crd_src, p), rd);
        internal_::eth_send_packet_bytes_unsafe(
            kTxq, unit(eth_.e1_crd_src, p), a_.peer_ctrl_base + eth_.e0_crd + p * kUnit, kUnit);
        if (busy != 0u) {
            result_.stall(3u, busy);
        }
        return true;
    }

    /* EC_E1_R2_AFTER_CONSUME: R2 for the packets R1 shows consumed since the
     * last, as the link read pointer of that count. */
    bool return_consumed(uint32_t p, uint32_t* progressed) {
        const uint32_t frd = local_word(unit(eth_.e1_out_rd_sh, p));
        if (!ec_ptr_distance_ok(deliver_wr_[p], frd, EC_N_DELIVER)) {
            result_.failure(p, 0u, 0u, EC_SITE_NONE, deliver_wr_[p], frd);
            fail(EC_ERROR_COUNTER_DISTANCE, p);
            return false;
        }
        /* deliver()'s advance check: R1 is posted, so an older R1 can land
         * after a newer one, and would otherwise send R2 backwards. */
        uint32_t advance;
        if (!credit_pair_ok(
                frd,
                frd_seen_[p],
                ec_ptr_mask(EC_N_DELIVER),
                ec_ptr_used(deliver_wr_[p], frd_seen_[p], EC_N_DELIVER),
                &advance)) {
            result_.failure(p, 0u, 0u, EC_SITE_NONE, frd_seen_[p], frd);
            fail(EC_ERROR_COUNTER_DISTANCE, p);
            return false;
        }
        frd_seen_[p] = frd;
        const uint32_t consumed = received_[p] - ec_ptr_used(deliver_wr_[p], frd, EC_N_DELIVER);
        if (consumed == credited_[p]) {
            return true;
        }
        if (!send_r2(p, consumed & ec_ptr_mask(EC_N_LINK))) {
            return false;
        }
        credited_[p] = consumed;
        *progressed = 1u;
        return true;
    }

    /* Every packet received and, with EC_E1_R2_AFTER_CONSUME, credited. */
    bool channel_done(uint32_t p) const {
        return received_[p] == a_.erisc.count[p] && (!EC_E1_R2_AFTER_CONSUME || credited_[p] == received_[p]);
    }

    bool finished() const {
        for (uint32_t p = 0; p < EC_P; ++p) {
            if (!channel_done(p)) {
                return false;
            }
        }
        return true;
    }

    /* A pass in which no channel progressed is one stall, charged to the site
     * of the channel stalled longest. */
    bool loop() {
        for (uint32_t p = 0; p < EC_P; ++p) {
            last_progress_[p] = now();
        }
        while (!finished()) {
            const uint64_t t0 = now();
            bool any = false;
            uint32_t oldest = EC_P;
            for (uint32_t p = 0; p < EC_P; ++p) {
                if (channel_done(p)) {
                    continue;
                }
                uint32_t progressed = 0u;
                uint32_t site = EC_SITE_E1_LINK_EMPTY;
                if constexpr (EC_E1_R2_AFTER_CONSUME) {
                    if (!return_consumed(p, &progressed)) {
                        return false;
                    }
                }
                if (received_[p] < a_.erisc.count[p] && !deliver(p, &progressed, &site)) {
                    return false;
                }
                if (progressed != 0u) {
                    last_progress_[p] = now();
                    any = true;
                } else {
                    site_[p] = site;
                    if (oldest == EC_P || last_progress_[p] < last_progress_[oldest]) {
                        oldest = p;
                    }
                }
            }
            if (!any && oldest != EC_P) {
                result_.stall(site_[oldest] == EC_SITE_E1_IN_FULL ? 1u : 0u, t0, now());
            }
            if (!safety_check()) {
                return false;
            }
        }
        uint64_t ignored = 0u;
        if (!metal_wait::write_barrier(noc_index, budget()) || !wait_txq(&ignored)) {
            fail(EC_ERROR_STALL_TIMEOUT, EC_SITE_DRAIN);
            return false;
        }
        return true;
    }

    const ec_e1_args_t& a_;
    ec_eth_layout_t eth_{};
    ec_x280_layout_t lim_{};
    Result result_;
    uint32_t link_rd_[EC_P] = {};
    uint32_t deliver_wr_[EC_P] = {};
    uint32_t received_[EC_P] = {};
    /* EC_E1_R2_AFTER_CONSUME: the packets R2 has returned. */
    uint32_t credited_[EC_P] = {};
    /* The last R1 value deliver() accepted. */
    uint32_t frd_seen_[EC_P] = {};
    uint64_t last_progress_[EC_P] = {};
    uint32_t site_[EC_P] = {};
    uint32_t safety_work_ = 0u;
};

}  // namespace

void kernel_main() {
    ec_e1_args_t args;
    read_args(args);
    E1 e1(args);
    e1.run();
}
