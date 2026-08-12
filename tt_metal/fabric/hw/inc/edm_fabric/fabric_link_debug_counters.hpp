// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dev_mem_map.h"
#include "internal/risc_attribs.h"

// ─── Fabric ERISC router packet counters (DIAGNOSTIC -- not for merge) ─────────
//
// Per-link, per-ERISC packet accounting: how many packets went onto the wire, how many
// came off it, and a HISTOGRAM of both broken down by NocSendType. Lives in ERISC L1 so
// it survives a hang; read it post-mortem with exalens (read_fabric_counters.py).
//
// ── Why a histogram ───────────────────────────────────────────────────────────
// Earlier layouts counted only "fused seminc" vs "not", so a packet that went missing
// could not be attributed to a traffic class. tx_hist/rx_hist count every NocSendType
// separately on both sides, so a per-link, per-type send/receive comparison is possible
// and a deficit can be pinned to the class it belongs to.
//
// ── Why the counters are split per ERISC ──────────────────────────────────────
// erisc0 and erisc1 share one eth core. They used to increment the SAME words with
// non-atomic read-modify-writes, so a lost update was indistinguishable from a lost
// packet -- and every result in this investigation is an off-by-one. Each ERISC now owns
// a private bank indexed by MY_ERISC_ID. A deficit that survives this is real. In
// practice the split also showed erisc0 only sends and erisc1 only receives.
//
// ── Where this lives, and the standing hazard ─────────────────────────────────
// The 160-byte FabricTelemetry struct is:
//     +0    .. +16   StaticInfo    (written even with telemetry off -- do not use)
//     +16   .. +128  DynamicInfo   <-- both banks live here
//     +128  .. +132  postcode
//     +132  .. +160  scratch[7]
// That leaves 112 bytes for two banks, so a bank is capped at 56 bytes / 14 words. It
// cannot grow; adding a field means reclaiming one. The banks squat on DynamicInfo,
// which is only written when fabric telemetry is compiled in -- confirm
// TT_METAL_FABRIC_TELEMETRY is OFF before trusting a run.
//
// HAZARD: initialize_fabric_telemetry() memsets all 160 bytes UNCONDITIONALLY and both
// ERISCs call it. If one ERISC's memset lands after the other has started counting, the
// earlier bank is wiped and restarts from zero -- a huge deficit, not an off-by-one, so
// compare against peers and discard any implausibly low bank.
//
// FIELD OFFSETS ARE FROZEN -- read_fabric_counters.py indexes by word. Append only.

namespace tt::tt_fabric::debug {

constexpr uint32_t kFabricLinkDbgMagic = 0x00C0FFEE;

// Low word of the SDPA R3 receive semaphore -- the row-axis reduction round whose
// semaphore every observed hang is stuck on. Allocation-dependent: if a run reports
// fused_inc_r3 == 0 everywhere while fused_inc_total is large, this constant is stale.
constexpr uint32_t kR3SemaphoreAddrLow = 213440;

// NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC; static_asserted in fabric_erisc_router.cpp.
constexpr uint8_t kNocFusedUnicastAtomicInc = 3;

// NocSendType is 3 bits, so 8 buckets. Packed two per word to fit the 14-word budget.
constexpr uint32_t kNumSendTypes = 8;
constexpr uint32_t kHistWords = kNumSendTypes / 2;

struct FabricLinkCounters {
    uint32_t tx;               // w0  packets handed to the eth hardware (at the wire)
    uint32_t rx;               // w1  packets consumed from the eth link
    uint32_t magic;            // w2  kFabricLinkDbgMagic
    // w3 RECLAIMED. It held fused_inc_r3_issued, the "after noc_semaphore_inc was issued"
    // count. Retired because r3_entered == r3_issued EXACTLY in all six runs r70-r75, and the
    // failure is now established to be BEFORE dispatch (the packet never reaches the receive
    // switch at all), so that gap is no longer where anything happens. fused_inc_r3 (w4)
    // survives, so R3 accounting is unaffected. If a future run ever needs the post-issue
    // count back, take it from forward_hist's per-type breakdown instead -- but note that
    // collapsing forward_hist reintroduces the forwarded-as-lost confounder that caused two
    // wrong withdrawals here, so prefer almost anything else.
    //
    // Now: L1 address of the packet header the dispatch gate is refusing. ONE word buys the
    // whole header post-mortem -- hop_index, both branch offsets, all 35 route_buffer bytes,
    // dst_start_chip_id/mesh_id, send type -- which is far more than a few route bytes would.
    // Safe to read at leisure: the slot cannot be reused while the router is wedged on it.
    uint32_t block_hdr_addr;          // w3  0 = never blocked
    uint32_t fused_inc_r3;            // w4  R3 incs counted on ENTERING the fused case
    // w5 is SHARED by role. The telemetry region is exactly full (+16..+160 for two 72-byte
    // banks) so there is no word left to append, and these two uses can never collide:
    // tx_r3 is only written while sending, block_state only while receiving, and erisc0
    // only sends while erisc1 only receives (verified: 0 of 1536 banks across r70-r72 had
    // both tx>0 and rx>0). The reader labels w5 by role and FLAGS any bank that does both,
    // because that is the one condition under which this word becomes meaningless.
    union {
        uint32_t tx_r3;        // w5, SEND bank:    R3 semincs classified at send
        uint32_t block_state;  // w5, RECEIVE bank: why the head-of-line packet won't dispatch
    };
    uint32_t tx_hist[kHistWords];       // w6-w9   sent,                       by NocSendType
    uint32_t rx_hist[kHistWords];       // w10-w13 arrived and TERMINATED here
    // w14-w17 WERE forward_hist[4], a per-NocSendType breakdown of packets passed onward.
    // Collapsed to a single total to make room for the overrun hunt. Justification: forwarding
    // is now MEASURED to happen on exactly 5 cores in this configuration, all carrying only
    // `ainc` traffic, and never on a core that blocks or loses packets. The total is all
    // `unaccounted` needs, so the classification self-check is unaffected.
    //
    // COST, STATED PLAINLY: per-type comparison (tx_hist[T] vs arrived[T]) is no longer
    // possible on a FORWARDING receiver, because arrived[T] = rx_hist[T] + forward_hist[T] and
    // the second term is gone. That is the exact confounder that caused two wrong withdrawals
    // in this investigation, so analyze_fabric_counters.py now REFUSES to report per-type
    // deficits for any edge whose receiver has forward_total > 0, rather than reporting a
    // wrong one. Restore forward_hist if per-type forwarding attribution is ever needed again.
    uint32_t forward_total;  // w14  arrived and PASSED ONWARD, all types
    uint32_t guard_state;    // w15  role-split overrun detector, see record_oversized_send /
                             //      record_arrival_check
    uint32_t max_payload;    // w16  role-split: largest payload_size_bytes seen (send bank) or
                             //      seen arriving (receive bank). Context for how far out of
                             //      range a bad size is.
    uint32_t spare_w17;      // w17  reserved
};

// ── Why forward_hist exists, and where it has to be counted ───────────────────
// A packet consumed off the link either TERMINATES on this chip or is PASSED ONWARD to the
// next hop. Originally only termination was classified, so a forwarded packet bumped rx but
// no histogram bucket -- and comparing tx_hist[T] at the sender against rx_hist[T] at the
// receiver made every forwarded packet look like a lost one. That artifact produced two
// wrong conclusions before it was spotted, and left a one-packet ambiguity on every failing
// link which made "was the lost packet a seminc?" unanswerable.
//
// Getting the counter onto the right branch took two attempts, so the branch structure is
// worth stating exactly. From receiver_forward_packet (fabric_erisc_router.cpp, one
// overload for 1D and one for 2D):
//
//   TERMINATES HERE   execute_chip_unicast_to_local_chip -> ..._impl -> record_rx_type
//                     execute_chip_unicast_to_relay                 -> record_rx_type
//        Two mechanisms for the SAME outcome, selected by udm_mode in
//        forward_to_local_destination: write into this chip directly, or hand the whole
//        packet to a tensix relay core which does the write. Either way the packet stops
//        here, so both belong in rx_hist.
//        NOTE: UDM_MODE reads 0 on all 120 generated router variants in this
//        configuration, so the relay arm is compiled out -- its single call site sits
//        inside `if constexpr (udm_mode)`. Only the local-chip arm is live here.
//
//   PASSED ONWARD     forward_payload_to_downstream_edm             -> record_forward_type
//        36 call sites across the two overloads, which is why the counter lives inside the
//        function rather than at its callers.
//
// The first attempt put the forward counter on execute_chip_unicast_to_relay, whose name
// reads like forwarding but which is the UDM-mode LOCAL delivery path. Since UDM_MODE is 0
// here that branch is compiled out, so the counter recorded NOTHING -- not merely packets
// rx_hist already covered -- and the real forward path stayed uncounted. The original
// artifact survived intact, behind a field that claimed to have fixed it.
//
// That failure mode is worth remembering because it is invisible: a double-counting
// mistake would have shown up as a NEGATIVE residual in the self-check below, whereas
// counting zero is indistinguishable from a correctly-idle forward path.
//
// SELF-CHECK, and its one exception:
//     rx == sum(rx_hist) + sum(forward_hist)
// holds for unicast traffic, where every arriving packet lands in exactly one bin. A
// MULTICAST packet can be written locally AND forwarded (the WRITE_AND_FORWARD_* routing
// cases), and can fan out to more than one direction, landing in several bins at once. So
// the guaranteed form is
//     rx <= sum(rx_hist) + sum(forward_hist)
// and the reader prints the signed residual: POSITIVE means a receive path exists that is
// still unclassified and per-type comparisons are not yet trustworthy; NEGATIVE means
// multicast fan-out, not duplicated packets. This workload is unicast apart from a single
// observed NOC_MULTICAST_ATOMIC_INC, so treat either sign as something to explain.

constexpr uint32_t kFabricLinkBankStride = sizeof(FabricLinkCounters);
static_assert(kFabricLinkBankStride == 72, "layout is frozen -- readers index by word");

// Bank 0 at +16, bank 1 at +88. Together they run to +160, the end of the telemetry
// region, which means they now also cover the postcode word (+128) and scratch[7] (+132).
// Verified unused: nothing in the fabric router writes either, and the only access to the
// whole region is initialize_fabric_telemetry()'s memset, which runs before the counters
// are initialised. If a future build starts using the postcode, it lands inside bank 1 and
// these readings become void -- so re-check that before trusting a run on new firmware.
constexpr uint32_t kFabricLinkCountersBase = MEM_AERISC_FABRIC_TELEMETRY_BASE + 16;

static_assert(kFabricLinkCountersBase == 458208, "base moved -- update BASE in read_fabric_counters.py");
static_assert(
    kFabricLinkCountersBase + 2 * kFabricLinkBankStride <= MEM_AERISC_FABRIC_TELEMETRY_BASE + 160,
    "banks overrun the end of the fabric telemetry region");

inline volatile tt_l1_ptr FabricLinkCounters* fabric_link_counters_bank(uint32_t erisc_id) {
    return reinterpret_cast<volatile tt_l1_ptr FabricLinkCounters*>(
        kFabricLinkCountersBase + erisc_id * kFabricLinkBankStride);
}

// MY_ERISC_ID comes from fabric_erisc_router_ct_args.hpp, which must be included first.
inline volatile tt_l1_ptr FabricLinkCounters* fabric_link_counters() { return fabric_link_counters_bank(MY_ERISC_ID); }

// Bump one 16-bit histogram lane. SATURATES rather than wraps: counts reach ~40k in a long
// run and 16 bits holds 65535, so a saturated 0xFFFF is a visible "this bucket overflowed,
// do not trust it" marker instead of a silently wrong small number.
inline void hist_bump(volatile tt_l1_ptr uint32_t* hist, uint8_t type) {
    volatile tt_l1_ptr uint32_t* w = hist + ((type & (kNumSendTypes - 1)) >> 1);
    const uint32_t v = *w;
    if (type & 1) {
        const uint32_t hi = v >> 16;
        if (hi != 0xFFFFu) {
            *w = (v & 0x0000FFFFu) | ((hi + 1) << 16);
        }
    } else {
        const uint32_t lo = v & 0xFFFFu;
        if (lo != 0xFFFFu) {
            *w = (v & 0xFFFF0000u) | (lo + 1);
        }
    }
}

template <typename PacketHeaderT>
inline bool is_r3_fused_inc(volatile PacketHeaderT* hdr) {
    if (static_cast<uint8_t>(hdr->get_noc_send_type()) != kNocFusedUnicastAtomicInc) {
        return false;
    }
    return static_cast<uint32_t>(hdr->command_fields.unicast_seminc_fused.semaphore_noc_address & 0xFFFFFFFF) ==
           kR3SemaphoreAddrLow;
}

// Classify a packet about to be sent. Called where the header is still guaranteed intact
// -- BEFORE update_packet_header_before_eth_send, which rewrites routing fields.
// Deliberately does NOT bump tx: tx is counted at the wire (record_tx_wire) so the two can
// be compared. tx > sum(tx_hist) would mean a packet reached the hardware without being
// classified; tx < sum(tx_hist) would mean one was classified but never sent.
template <typename PacketHeaderT>
inline void record_tx_type(volatile PacketHeaderT* hdr) {
    auto* c = fabric_link_counters();
    const uint8_t t = static_cast<uint8_t>(hdr->get_noc_send_type());
    hist_bump(c->tx_hist, t);
    if (t == kNocFusedUnicastAtomicInc &&
        static_cast<uint32_t>(hdr->command_fields.unicast_seminc_fused.semaphore_noc_address & 0xFFFFFFFF) ==
            kR3SemaphoreAddrLow) {
        c->tx_r3++;
    }
}

// Count a packet at the last software point before it goes out: immediately ahead of
// eth_send_packet_bytes_unsafe. This is the closest thing to "left the chip" that software
// can observe.
inline void record_tx_wire() { fabric_link_counters()->tx++; }

// Classify a received packet, BEFORE the receive switch's
// `if (noc_send_type > NOC_SEND_TYPE_LAST) __builtin_unreachable();` hint. That hint makes
// types 5..7 undefined behaviour -- the switch may match nothing and execute nothing while
// the packet still counted as received -- so this is the only place the evidence survives.
//
// NOTE: this fires for packets that TERMINATE on this chip, by either delivery mechanism --
// see the branch table above. Packets passed onward to the next hop go to
// record_forward_type instead, never here.
inline void record_rx_type(uint8_t noc_send_type) { hist_bump(fabric_link_counters()->rx_hist, noc_send_type); }

// The other outcome: this packet arrived and is being passed onward to the next hop rather
// than terminating here. Counting it is what stops a forwarded packet from looking like a
// lost one in the per-type comparison. Called from inside forward_payload_to_downstream_edm,
// ahead of the point where the header's routing fields are rewritten for the next hop.
inline void record_forward_type(uint8_t) { fabric_link_counters()->forward_total++; }

// ─── The overrun hunt (w15/w16, role-split) ───────────────────────────────────
// Slots are laid out [header | payload] with the header AT the slot base, and the wire
// transfer is one contiguous block of header+payload starting at the destination slot base.
// So an oversized transfer runs FORWARD off the end of slot N and lands on slot N+1's HEADER.
// Reaching it at all requires
//     payload_size_bytes > CHANNEL_BUFFER_SIZE
// which makes the check below necessary AND sufficient: if payload ever appears in a header
// region because of a transfer overrun, this must fire. If it never fires while headers keep
// turning up full of bfloat16, the overrun theory is dead.
//
// It also catches the corruption propagating: if a slot's header is already payload, then the
// payload_size_bytes read out of it is float bits too, and almost certainly huge.
//
// SEND bank packing (w15):
//   bits 0..15   last offending payload_size_bytes
//   bits 16..23  saturating count of oversized sends
//   bits 24..27  sender channel index of the last one -- identifies WHICH client/op sent it
inline void record_oversized_send(uint32_t payload_size_bytes, uint32_t max_size, uint8_t ch) {
    auto* c = fabric_link_counters();
    if (payload_size_bytes > c->max_payload) {
        c->max_payload = payload_size_bytes;
    }
    if (payload_size_bytes <= max_size) {
        return;
    }
    const uint32_t prev = c->guard_state;
    uint32_t count = (prev >> 16) & 0xFFu;
    if (count != 0xFFu) {
        count++;
    }
    c->guard_state = (static_cast<uint32_t>(ch & 0xFu) << 24) | (count << 16) |
                     (payload_size_bytes & 0xFFFFu);
}

// RECEIVE bank packing (w15): validate the header the moment the packet comes off the link,
// BEFORE the dispatch gate. This is the temporal discriminator -- it separates "arrived
// already corrupt" (so the damage happened at or before the send) from "arrived clean and was
// clobbered while waiting" (so a later transfer overran it in place).
//   bits 0..15   last bad hop_index (saturated at 0xFFFF)
//   bits 16..23  saturating count of bad arrivals
//   bit  24      hop_index was >= route buffer size
//   bit  25      payload_size_bytes exceeded the slot size
inline void record_arrival_check(
    uint32_t hop_index, uint32_t payload_size_bytes, uint32_t route_buffer_size, uint32_t max_size) {
    auto* c = fabric_link_counters();
    if (payload_size_bytes > c->max_payload) {
        c->max_payload = payload_size_bytes;
    }
    const bool bad_hop = hop_index >= route_buffer_size;
    const bool bad_size = payload_size_bytes > max_size;
    if (!bad_hop && !bad_size) {
        return;
    }
    const uint32_t prev = c->guard_state;
    uint32_t count = (prev >> 16) & 0xFFu;
    if (count != 0xFFu) {
        count++;
    }
    uint32_t flags = 0;
    if (bad_hop) {
        flags |= 1u << 24;
    }
    if (bad_size) {
        flags |= 1u << 25;
    }
    c->guard_state = flags | (count << 16) | (hop_index > 0xFFFFu ? 0xFFFFu : hop_index);
}

// Count a packet consumed off the ethernet link, before any dispatch decision.
inline void record_rx_wire() { fabric_link_counters()->rx++; }

// ─── Why the head-of-line packet did not dispatch ─────────────────────────────
// The receiver buffer is drained strictly in order: run_receiver_channel_step_impl looks at
// the single slot at wr_sent_counter and only advances it inside the success branch. So one
// packet that never satisfies the dispatch gate blocks every packet behind it forever, which
// is what a persistent unaccounted > 0 measures (unaccounted == ack_counter - wr_sent_counter).
// This records WHICH arm of the gate is refusing, and the hop command of the packet it is
// refusing, so the blocker is named rather than inferred.
//
// Packed into one word:
//   bits 0..3    hop_cmd of the blocked packet (the 4-bit Mesh command field)
//   bit 4        the gate returned false. NOT "a downstream direction was full": with
//                hop_cmd == NOOP it is false BECAUSE of the NOOP case, so this bit is
//                ENTAILED by bit 7 and is not independent evidence of backpressure.
//   bit 5        trid_flushed was false
//   bit 6        hop_cmd had bits ABOVE the 4-bit field -- such a value matches no case in
//                the dispatch switch, whose default is __builtin_unreachable()
//   bit 7        hop_cmd == NOOP. Called out separately because it is a SILENT permanent
//                stall in this configuration: `case NOOP` only sets ret_val under
//                z_router_enabled (0 here), and the NOOP -> recompute_path rescue in
//                get_cmd_with_mesh_boundary_adjustment is compiled out because both
//                is_inter/intramesh_router_on_edge are 0. Verified across all 120 variants.
//   bits 8..13   hop_index, saturated at 63. The encoder NOOP-pads the route buffer from the
//                end of the route to the end of the buffer, so a NOOP read is not corrupt
//                data -- it means hop_index reached the padding, i.e. the packet travelled
//                past its own route. hop_index vs the route contents is what separates
//                "overshot the destination" from "route never written".
//   bit 14       hop_index > 63, i.e. far outside a 35-byte route buffer: index corruption
//                rather than a plain overshoot.
//   bits 15..31  saturating count of blocked polls. Only "saturated vs small" carries
//                meaning: a healthy router sees momentary backpressure and leaves a small
//                count; a permanent stall pins it at max.
//
// Everything except the count is OVERWRITTEN each time rather than histogrammed: at a hang
// the router spins on the same blocked slot forever, so the last value written IS the stuck
// packet's.
constexpr uint32_t kBlockCountMax = 0x1FFFFu;  // 17 bits, bits 15..31

inline void record_dispatch_block(
    uint32_t hop_cmd, uint32_t hop_index, bool space_ok, bool trid_ok, uint32_t hdr_addr) {
    auto* c = fabric_link_counters();
    volatile tt_l1_ptr uint32_t* w = &c->block_state;
    uint32_t count = *w >> 15;
    if (count != kBlockCountMax) {
        count++;
    }
    uint32_t flags = hop_cmd & 0xFu;
    if (!space_ok) {
        flags |= 1u << 4;
    }
    if (!trid_ok) {
        flags |= 1u << 5;
    }
    if (hop_cmd > 0xFu) {
        flags |= 1u << 6;
    }
    if (hop_cmd == 0u) {
        flags |= 1u << 7;
    }
    if (hop_index > 63u) {
        flags |= 1u << 14;
        flags |= 63u << 8;
    } else {
        flags |= hop_index << 8;
    }
    *w = (count << 15) | flags;
    // The header itself is read post-mortem from this address; the slot cannot be recycled
    // while the router is still refusing it.
    c->block_hdr_addr = hdr_addr;
}

// STAGE 4 -- the receive switch has ENTERED the fused-atomic-inc case for this packet.
// Note this is not "the increment happened": the payload write has been issued but
// noc_semaphore_inc has not. Pair with record_fused_inc_issued below.
//
// fused_inc_total (all semaphores, entering the case) was dropped to make room. It is not
// lost: rx_hist[3] counts the same packets a few instructions earlier, with no branch
// between, and the two matched exactly in r57 (8445 == 8445).
inline void record_fused_inc(uint32_t sem_lo) {
    if (sem_lo == kR3SemaphoreAddrLow) {
        fabric_link_counters()->fused_inc_r3++;
    }
}

// STAGE 5 -- RETIRED, now a no-op. It counted R3 incs after noc_semaphore_inc was issued, in
// w3, which has been reclaimed for block_hdr_addr. Kept as an empty inline so the call site
// in the receive switch does not have to be touched (and so this can be restored in one edit
// if a future run needs it). r3_entered == r3_issued held in all six runs r70-r75, so this
// counter has never once disagreed with the one that remains.
inline void record_fused_inc_issued(uint32_t) {}

// Zero this ERISC's bank once per boot, magic-guarded, so counts are cumulative.
// Magic is written LAST so a reader seeing it is guaranteed a fully zeroed bank.
inline void fabric_link_counters_init() {
    auto* counters = fabric_link_counters();
    if (counters->magic == kFabricLinkDbgMagic) {
        return;
    }
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counters);
    constexpr uint32_t num_words = sizeof(FabricLinkCounters) / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
        words[i] = 0;
    }
    counters->magic = kFabricLinkDbgMagic;
}

}  // namespace tt::tt_fabric::debug
