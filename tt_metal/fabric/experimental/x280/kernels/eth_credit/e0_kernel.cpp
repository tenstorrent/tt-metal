// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/eth_credit/e0_kernel.cpp -- ERISC0, the outbound role.
//
// Sends each forwarder's tx_fifo[p] slots over TXQ 0 into the peer's rx[p]
// (W3) with a +1 on the peer's pkts[p] stream counter behind each (W4), and
// forwards the peer E1's read pointer crd[p] to forwarder p unchanged (R3).
// Runs on both tiles in every run: an idle outbound direction still resets its
// tile's counters, enables packet mode, handshakes and joins the barrier, then
// exits.

#include <stdint.h>

#include "eth_credit_header.hpp"
#include "metal_common.hpp"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"

/* Bring-up modes, each off in the benchmark build. EC_E0_SINK_LOCAL consumes
 * tx_fifo here and returns R3 from E0's own read pointer, so the NoC hop runs
 * alone; EC_E0_SOURCE_SELF fills tx_fifo here, so the Ethernet hop runs
 * without forwarders; EC_TAIL_MARKER stamps each slot's last 16 bytes for E1
 * to check; EC_E0_COUNTER_FIRST_DELAY > 0 sends W4 that many ticks before its
 * W3, the admission test's positive control. */
#ifndef EC_E0_SINK_LOCAL
#define EC_E0_SINK_LOCAL 0
#endif
#ifndef EC_E0_SOURCE_SELF
#define EC_E0_SOURCE_SELF 0
#endif
#ifndef EC_TAIL_MARKER
#define EC_TAIL_MARKER 0
#endif
#ifndef EC_E0_COUNTER_FIRST_DELAY
#define EC_E0_COUNTER_FIRST_DELAY 0
#endif

static_assert(!(EC_E0_SINK_LOCAL && EC_E0_SOURCE_SELF), "E0 cannot both source and sink locally");
static_assert(
    !(EC_E0_SINK_LOCAL && (EC_TAIL_MARKER || EC_E0_COUNTER_FIRST_DELAY)),
    "a local sink sends nothing over the link to mark or reorder");
static_assert(
    !EC_TAIL_MARKER || EC_SLOT_SIZE >= EC_HEADER_BYTES + EC_TAIL_MARKER_BYTES,
    "the tail marker needs a payload of at least 16 bytes, or it overwrites the header");

namespace {

using namespace eth_credit::metal;

constexpr uint32_t kTxq = 0u;
constexpr uint32_t kSlotBytes = EC_SLOT_SIZE;
constexpr uint32_t kSites[EC_MAX_STALL_SITES] = {
    EC_SITE_E0_INPUTS_EMPTY, EC_SITE_E0_LINK_CREDITS, EC_SITE_E0_TXQ0_BUSY, EC_SITE_NONE};

class E0 {
public:
    explicit E0(const ec_e0_args_t& a) : a_(a) {}

    void run() {
        if (!bind()) {
            return;
        }
        initialize();
        if (!reset_stream_counters()) {
            return;
        }

        eth_enable_packet_mode(kTxq);
        if (!bounded_handshake(
                a_.erisc.ctrl_base + eth_.handshake, (a_.erisc.flags & EC_E0_FLAG_HANDSHAKE_SENDER) != 0u, budget())) {
            result_.failure(0u, 0u, 0u, EC_SITE_HANDSHAKE, 0u, 0u);
            fail(EC_ERROR_STALL_TIMEOUT, EC_SITE_HANDSHAKE);
            return;
        }

        const core_sync::Outcome sync = barrier<ProgrammableCoreType::ACTIVE_ETH>(
            a_.erisc.fwd,
            a_.erisc.barrier_ready_base,
            ec_barrier_index_e0(),
            a_.erisc.ctrl_base + eth_.e0_release,
            barrier_cap(),
            a_.erisc.run_nonce);
        if (sync != core_sync::Outcome::Success) {
            fail(barrier_error(sync), EC_SITE_BARRIER);
            return;
        }
        if ((a_.erisc.flags & EC_E0_FLAG_OUTBOUND_ACTIVE) == 0u) {
            result_.finish(ec_status_word(EC_STATE_DONE, a_.erisc.kernel_id, EC_ERROR_NONE, 0u));
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

    /* Validates in the order each check makes the next action safe. With
     * ctrl_base unbounded nothing may be written, so the kernel leaves no
     * result and the host attributes the zero magic to its arguments; every
     * later rejection is BAD_ARGS in its own result. */
    bool bind() {
        if (ec_compute_eth_layout(&kShape, 1, 1, &eth_).code != EC_CHECK_OK ||
            ec_compute_x280_layout(&kShape, &lim_).code != EC_CHECK_OK ||
            ec_check_erisc_ctrl(a_.erisc.ctrl_base, &eth_, &kBounds).code != EC_CHECK_OK) {
            return false;
        }
        result_.init(a_.erisc.ctrl_base + eth_.e0_result, a_.erisc.kernel_id, kSites);
        tx_base_ = a_.erisc.ctrl_base + eth_.tx_offset;
        const ec_check_t args = ec_check_e0_args(&a_, &kShape, &kBounds);
        if (args.code != EC_CHECK_OK) {
            result_.bad_args(args);
            fail(EC_ERROR_BAD_ARGS, args.constraint);
            return false;
        }
        return true;
    }

    void initialize() {
        for (uint32_t p = 0; p < EC_P; ++p) {
            set_local_word(unit(eth_.e0_rd_src, p), 0u);
        }
    }

    /* Zeroes this tile's pkts[p] counters, as the production router does in
     * its kernel (init_ptr_val, fabric_erisc_router.cpp), because no host path
     * writes an Ethernet tile's stream registers. It precedes the handshake,
     * and the peer E0 sends nothing until that handshake completes, so no W4
     * can land ahead of it; E1 reads the counters only after release. */
    bool reset_stream_counters() {
        for (uint32_t p = 0; p < EC_P; ++p) {
            init_ptr_val(a_.local_stream_id[p], 0);
        }
        /* A counter left nonzero hangs two-ERISC operation (tt-metal PR
         * #32985); before the handshake it is quiescent, so read it back. */
        for (uint32_t p = 0; p < EC_P; ++p) {
            const int32_t v = get_ptr_val(static_cast<uint8_t>(a_.local_stream_id[p]));
            if (v != 0) {
                result_.failure(
                    EC_CONSTRAINT_STREAM, a_.local_stream_id[p], static_cast<uint32_t>(v), EC_SITE_NONE, 0u, 0u);
                fail(EC_ERROR_BAD_ARGS, EC_CONSTRAINT_STREAM);
                return false;
            }
        }
        return true;
    }

    bool wait_txq(uint64_t* stalled) { return bounded_txq_idle(kTxq, budget(), stalled); }

    /* Records FAILED in this kernel's result. Neighbours learn nothing and
     * exit on their own stall budgets; the host attributes the failure. */
    void fail(uint32_t code, uint32_t detail) {
        result_.finish(ec_status_word(EC_STATE_FAILED, a_.erisc.kernel_id, code, detail));
    }

    bool channel_done(uint32_t p) const { return forwarded_[p] == a_.erisc.count[p] && last_crd_[p] == sent_[p]; }

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
                result_.failure(p, 0u, 0u, site_[p], sent_[p], credit_pointer(p));
                fail(EC_ERROR_STALL_TIMEOUT, site_[p]);
                return false;
            }
        }
        return true;
    }

    bool distance_fail(uint32_t p, uint32_t local, uint32_t shadow) {
        result_.failure(p, 0u, 0u, EC_SITE_NONE, local, shadow);
        fail(EC_ERROR_COUNTER_DISTANCE, p);
        return false;
    }

    uint32_t credit_pointer(uint32_t p) const {
        return EC_E0_SINK_LOCAL ? local_crd_[p] : local_word(unit(eth_.e0_crd, p));
    }

    /* The source's write pointer: the forwarder's W2, or under
     * EC_E0_SOURCE_SELF every packet still to send that the credit `crd`
     * admits, as a forwarder would publish. */
    uint32_t source_pointer(uint32_t p, uint32_t crd) const {
        if (EC_E0_SOURCE_SELF) {
            const uint32_t left = a_.erisc.count[p] - forwarded_[p];
            const uint32_t room = ec_ptr_credit(sent_[p], crd, EC_N_LINK);
            return ec_ptr_advance(sent_[p], left < room ? left : room, EC_N_LINK);
        }
        return local_word(unit(eth_.e0_wr_sh, p));
    }

    /* EC_E0_SINK_LOCAL: the slot the forwarder wrote before its W2 must hold
     * this packet's sequence number and channel. */
    bool check_local_slot(uint32_t p, uint32_t slot_addr) {
        invalidate_l1_cache();
        const auto* h = reinterpret_cast<const volatile fs_hybrid_mesh_packet_header_t*>(slot_addr);
        const uint32_t seq = ec_header_seq(h);
        const uint32_t ch = fs_src_ch_id(h);
        if (seq != forwarded_[p] || ch != p) {
            result_.failure(ch, forwarded_[p], seq, EC_SITE_NONE, sent_[p], 0u);
            fail(EC_ERROR_BAD_SEQUENCE, forwarded_[p]);
            return false;
        }
        return true;
    }

    /* EC_E0_SOURCE_SELF: writes the fields E1's local sink checks. */
    void fill_slot(uint32_t p, uint32_t slot_addr, uint32_t ordinal) {
        auto* h = reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(slot_addr);
        ec_header_set_seq(h, ordinal);
        fs_set_src_ch_id(h, static_cast<uint8_t>(p));
    }

    void stamp_marker(uint32_t p, uint32_t slot_addr, uint32_t ordinal) {
        uint32_t m[4];
        ec_tail_marker(p, ordinal, m);
        volatile tt_l1_ptr uint32_t* tail =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + kSlotBytes - EC_TAIL_MARKER_BYTES);
        for (uint32_t i = 0; i < 4u; ++i) {
            tail[i] = m[i];
        }
    }

    bool txq_timeout(uint32_t p, uint32_t wr) {
        result_.failure(p, 0u, 0u, EC_SITE_E0_TXQ0_BUSY, sent_[p], wr);
        fail(EC_ERROR_STALL_TIMEOUT, EC_SITE_E0_TXQ0_BUSY);
        return false;
    }

    /* W3 then W4 for up to BATCH slots of tx_fifo[p], the router's
     * send_next_data sequence: wait, data, wait, counter. E1 takes W4 as
     * proof that W3's slot is in its L1. Whether a TT-link MMIO write lands
     * only after an earlier TT-link L1 write on the same TX queue has
     * committed is an open hardware question (docs/measurement.md "Unconfirmed hardware assumptions",
     * item 1): Wormhole orders no packets of different kinds, and Blackhole's
     * RX counters move before L1 commit (EthernetTxRx.md). */
    bool send(uint32_t p, uint32_t* progressed, uint32_t* site) {
        const uint32_t crd = credit_pointer(p);
        const uint32_t wr = source_pointer(p, crd);
        if (!ec_ptr_distance_ok(wr, crd, EC_N_LINK) || !ec_ptr_distance_ok(sent_[p], crd, EC_N_LINK) ||
            !ec_ptr_distance_ok(wr, sent_[p], EC_N_LINK)) {
            return distance_fail(p, sent_[p], wr);
        }
        const uint32_t avail = ec_ptr_used(wr, sent_[p], EC_N_LINK);
        const uint32_t credit = ec_ptr_credit(sent_[p], crd, EC_N_LINK);
        uint32_t k = avail < credit ? avail : credit;
        k = k < EC_BATCH ? k : EC_BATCH;
        k = k < a_.erisc.count[p] - forwarded_[p] ? k : a_.erisc.count[p] - forwarded_[p];
        if (k == 0u) {
            *site = ec_e0_idle_site(credit);
            return true;
        }
        uint64_t busy = 0u;
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t off = p * eth_.fifo_stride + ec_ptr_slot(sent_[p], EC_N_LINK) * kSlotBytes;
            const uint32_t ordinal = forwarded_[p] + j;
            if constexpr (EC_E0_SINK_LOCAL) {
                if (!check_local_slot(p, tx_base_ + off)) {
                    return false;
                }
                forwarded_[p] += 1u;
                local_crd_[p] = ec_ptr_advance(local_crd_[p], 1u, EC_N_LINK);
                sent_[p] = ec_ptr_advance(sent_[p], 1u, EC_N_LINK);
                continue;
            }
            if constexpr (EC_E0_SOURCE_SELF) {
                fill_slot(p, tx_base_ + off, ordinal);
            }
            if constexpr (EC_TAIL_MARKER) {
                stamp_marker(p, tx_base_ + off, ordinal);
            }
            if constexpr (EC_E0_COUNTER_FIRST_DELAY != 0) {
                if (!wait_txq(&busy)) {
                    return txq_timeout(p, wr);
                }
                remote_update_ptr_val<kTxq>(a_.erisc.stream_id[p], 1);
                const uint64_t t0 = now();
                while (now() - t0 < static_cast<uint64_t>(EC_E0_COUNTER_FIRST_DELAY)) {
                }
            }
            if (!wait_txq(&busy)) {
                return txq_timeout(p, wr);
            }
            internal_::eth_send_packet_bytes_unsafe(kTxq, tx_base_ + off, a_.peer_rx_base + off, kSlotBytes);
            if constexpr (EC_E0_COUNTER_FIRST_DELAY == 0) {
                if (!wait_txq(&busy)) {
                    return txq_timeout(p, wr);
                }
                remote_update_ptr_val<kTxq>(a_.erisc.stream_id[p], 1);
            }
            sent_[p] = ec_ptr_advance(sent_[p], 1u, EC_N_LINK);
        }
        if (busy != 0u) {
            result_.stall(2u, busy);
        }
        if constexpr (!EC_E0_SINK_LOCAL) {
            forwarded_[p] += k;
        }
        result_.packets_out(p, forwarded_[p]);
        result_.stamp(now());
        *progressed = 1u;
        return true;
    }

    /* R3: crd[p] to forwarder p, unchanged, whenever it has moved. */
    bool forward_credit(uint32_t p, uint32_t* progressed) {
        const uint32_t crd = credit_pointer(p);
        /* Equality means no movement: crd stays within N_LINK of sent_ (the
         * distance check in send()), so it cannot advance a whole 2N between
         * two polls and come back to the value last forwarded. */
        if (crd == last_crd_[p]) {
            return true;
        }
        if (!ec_ptr_distance_ok(sent_[p], crd, EC_N_LINK)) {
            return distance_fail(p, sent_[p], crd);
        }
        if constexpr (EC_E0_SOURCE_SELF) {
            last_crd_[p] = crd;
            *progressed = 1u;
            return true;
        }
        if (!metal_wait::writes_sent(noc_index, budget())) {
            fail(EC_ERROR_STALL_TIMEOUT, EC_SITE_E0_R3_SOURCE);
            return false;
        }
        set_local_word(unit(eth_.e0_rd_src, p), crd);
        noc_async_write(
            unit(eth_.e0_rd_src, p),
            get_noc_addr(a_.erisc.fwd.x, a_.erisc.fwd.y, a_.erisc.fwd_area[p] + lim_.rd_sh),
            kUnit);
        last_crd_[p] = crd;
        *progressed = 1u;
        return true;
    }

    bool finished() const {
        for (uint32_t p = 0; p < EC_P; ++p) {
            if (!channel_done(p)) {
                return false;
            }
        }
        return true;
    }

    /* Round-robin over p until every packet is sent and every R2 has come
     * back and been forwarded as R3. A pass in which no channel progressed is
     * one stall, charged to the site of the channel stalled longest. */
    bool loop() {
        for (uint32_t p = 0; p < EC_P; ++p) {
            last_progress_[p] = now();
        }
        while (!finished()) {
            const uint64_t t0 = now();
            bool any = false;
            uint32_t oldest = EC_P;
            for (uint32_t p = 0; p < EC_P; ++p) {
                uint32_t progressed = 0u;
                uint32_t site = EC_SITE_DRAIN;
                if (forwarded_[p] < a_.erisc.count[p] && !send(p, &progressed, &site)) {
                    return false;
                }
                if (!forward_credit(p, &progressed)) {
                    return false;
                }
                if (progressed != 0u) {
                    last_progress_[p] = now();
                    any = true;
                } else if (!channel_done(p)) {
                    site_[p] = site;
                    if (oldest == EC_P || last_progress_[p] < last_progress_[oldest]) {
                        oldest = p;
                    }
                }
            }
            if (!any && oldest != EC_P) {
                if (site_[oldest] == EC_SITE_E0_INPUTS_EMPTY) {
                    result_.stall(0u, t0, now());
                } else if (site_[oldest] == EC_SITE_E0_LINK_CREDITS) {
                    result_.stall(1u, t0, now());
                }
            }
            if (!safety_check()) {
                return false;
            }
        }
        if (!metal_wait::write_barrier(noc_index, budget())) {
            fail(EC_ERROR_STALL_TIMEOUT, EC_SITE_DRAIN);
            return false;
        }
        return true;
    }

    const ec_e0_args_t& a_;
    ec_eth_layout_t eth_{};
    /* This tile's tx_fifo region, at the layout's tx_offset past ctrl_base. */
    uint32_t tx_base_ = 0u;
    ec_x280_layout_t lim_{};
    Result result_;
    uint32_t sent_[EC_P] = {};
    uint32_t forwarded_[EC_P] = {};
    uint32_t last_crd_[EC_P] = {};
    uint32_t local_crd_[EC_P] = {};
    uint64_t last_progress_[EC_P] = {};
    uint32_t site_[EC_P] = {};
    uint32_t safety_work_ = 0u;
};

}  // namespace

void kernel_main() {
    ec_e0_args_t args;
    read_args(args);
    E0 e0(args);
    e0.run();
}
