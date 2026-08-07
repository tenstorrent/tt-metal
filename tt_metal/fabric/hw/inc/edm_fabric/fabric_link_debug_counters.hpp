// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dev_mem_map.h"
#include "internal/risc_attribs.h"

// ─── Fabric ERISC router packet counters (DIAGNOSTIC -- not for merge) ─────────
//
// Counts eth packets pushed onto and consumed from the link, and what the receive
// path did with each one. Lives in ERISC L1 so it survives a hang: read it
// post-mortem off the wedged device with exalens (read_fabric_counters.py).
//
// ── Why the counters are split per ERISC ──────────────────────────────────────
// erisc0 and erisc1 share one eth core. The original layout had them incrementing
// the SAME words with non-atomic read-modify-writes, so a lost update was
// indistinguishable from a lost packet -- and every result in this investigation is
// an off-by-one or off-by-two. Each ERISC now owns a private bank indexed by
// MY_ERISC_ID, which removes the race. A deficit that survives this is real.
//
// ── Where this lives, and the standing hazard ─────────────────────────────────
// The 160-byte FabricTelemetry struct is:
//     +0    .. +16   StaticInfo
//     +16   .. +128  DynamicInfo   <-- both banks live here
//     +128  .. +132  postcode
//     +132  .. +160  scratch[7]
// The banks squat on DynamicInfo, which is only written when fabric telemetry is
// compiled in. Confirm TT_METAL_FABRIC_TELEMETRY is OFF before trusting a run --
// with it on, the bandwidth and heartbeat fields overlap these counters exactly.
//
// HAZARD: initialize_fabric_telemetry() memsets all 160 bytes UNCONDITIONALLY and
// both ERISCs call it. If one ERISC's memset lands after the other has started
// counting, the earlier bank is wiped and restarts from zero -- a huge deficit, not
// an off-by-one, so compare against peers and discard any implausibly low bank.
//
// FIELD OFFSETS ARE FROZEN -- read_fabric_counters.py indexes by word. Append only.

namespace tt::tt_fabric::debug {

constexpr uint32_t kFabricLinkDbgMagic = 0x00C0FFEE;

// Low word of the SDPA R3 receive semaphore. Allocation-dependent: if a run reports
// fused_inc_r3 == 0 everywhere while last_fused_sem is nonzero, this is stale.
constexpr uint32_t kR3SemaphoreAddrLow = 213440;

// NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC; static_asserted in fabric_erisc_router.cpp.
constexpr uint8_t kNocFusedUnicastAtomicInc = 3;

struct FabricLinkCounters {
    uint32_t tx;               // w0  packets pushed onto the eth link
    uint32_t rx;               // w1  packets consumed from the eth link
    uint32_t txq;              // w2  packets handed to the eth TXQ hardware
    uint32_t tx_r3;            // w3  ... of tx, those carrying an R3 seminc
    uint32_t txq_r3;           // w4  ... of txq, those carrying an R3 seminc
    uint32_t fused_inc_total;  // w5  fused write+atomic-inc executed locally (any sem)
    uint32_t fused_inc_r3;     // w6  ... of those, targeting the R3 semaphore
    uint32_t magic;            // w7  kFabricLinkDbgMagic
    uint32_t last_fused_sem;   // w8  low word of the last fused-inc semaphore address
    uint32_t rx_type_mask;     // w9  bit N set if NocSendType N was received
    uint32_t rx_not_fused;     // w10 received packets that were NOT a fused atomic inc
    uint32_t first_bad;        // w11 (first anomalous noc_send_type << 24) | count(24b)
    uint32_t first_bad_sem;    // w12 sem addr low of the first non-R3 fused inc seen
    uint32_t first_bad_hdr;    // w13 first raw header word of the first anomalous packet
};

constexpr uint32_t kFabricLinkBankStride = sizeof(FabricLinkCounters);
static_assert(kFabricLinkBankStride == 56, "layout is frozen -- readers index by word");

// Bank 0 at +16, bank 1 at +72; both end before the postcode word at +128.
constexpr uint32_t kFabricLinkCountersBase = MEM_AERISC_FABRIC_TELEMETRY_BASE + 16;

static_assert(kFabricLinkCountersBase == 458208, "base moved -- update BASE in read_fabric_counters.py");
static_assert(
    kFabricLinkCountersBase + 2 * kFabricLinkBankStride <= MEM_AERISC_FABRIC_TELEMETRY_BASE + 128,
    "banks overrun into the fabric postcode word");

inline volatile tt_l1_ptr FabricLinkCounters* fabric_link_counters_bank(uint32_t erisc_id) {
    return reinterpret_cast<volatile tt_l1_ptr FabricLinkCounters*>(
        kFabricLinkCountersBase + erisc_id * kFabricLinkBankStride);
}

// MY_ERISC_ID comes from fabric_erisc_router_ct_args.hpp, which must be included first.
inline volatile tt_l1_ptr FabricLinkCounters* fabric_link_counters() { return fabric_link_counters_bank(MY_ERISC_ID); }

template <typename PacketHeaderT>
inline bool is_r3_fused_inc(volatile PacketHeaderT* hdr) {
    if (static_cast<uint8_t>(hdr->get_noc_send_type()) != kNocFusedUnicastAtomicInc) {
        return false;
    }
    return static_cast<uint32_t>(hdr->command_fields.unicast_seminc_fused.semaphore_noc_address & 0xFFFFFFFF) ==
           kR3SemaphoreAddrLow;
}

// Record what the receive path was handed, BEFORE the switch's
// `if (noc_send_type > NOC_SEND_TYPE_LAST) __builtin_unreachable();` hint.
//
// NOC_SEND_TYPE_LAST is NOC_UNICAST_SCATTER_WRITE (4), but the enum also defines
// NOC_MULTICAST_WRITE (5), NOC_MULTICAST_ATOMIC_INC (6) and NOC_UNICAST_READ (7),
// and the packet-header API can produce all three. So a legitimately-typed packet --
// or a garbled header -- can land in undefined behaviour, fall through the switch
// executing nothing, and still be counted as received. That is exactly the "arrived
// but never acted on" signature. After the hint the compiler may assume it cannot
// happen, so this is the only place the evidence survives.
inline void record_rx_type(uint8_t noc_send_type, uint32_t hdr_word0) {
    auto* c = fabric_link_counters();
    c->rx_type_mask |= (1u << (noc_send_type & 0x1F));
    if (noc_send_type != kNocFusedUnicastAtomicInc) {
        c->rx_not_fused++;
        uint32_t prev = c->first_bad;
        uint32_t count = (prev & 0x00FFFFFF) + 1;
        if ((prev & 0x00FFFFFF) == 0) {
            c->first_bad_hdr = hdr_word0;
        }
        c->first_bad = (static_cast<uint32_t>(noc_send_type) << 24) | (count & 0x00FFFFFF);
    }
}

// Record a locally-executed fused inc; latch the first whose semaphore is not R3 --
// "what replaced it" needs the actual address, not just a count.
inline void record_fused_inc(uint32_t sem_lo) {
    auto* c = fabric_link_counters();
    c->fused_inc_total++;
    c->last_fused_sem = sem_lo;
    if (sem_lo == kR3SemaphoreAddrLow) {
        c->fused_inc_r3++;
    } else if (c->first_bad_sem == 0) {
        c->first_bad_sem = sem_lo;
    }
}

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
