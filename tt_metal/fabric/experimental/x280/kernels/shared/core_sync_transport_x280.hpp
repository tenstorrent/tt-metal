// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_transport_x280.hpp -- X280 L2CPU transport policy
// for core_sync_barrier.
#ifndef X280_CORE_SYNC_TRANSPORT_X280_HPP
#define X280_CORE_SYNC_TRANSPORT_X280_HPP

#include <stdint.h>

#include "core_sync_arrival.hpp"
#include "core_sync_layout.h"
#include "core_sync_x280_roster.h"

/* Pure macros, so the reservation bounds are visible to a host build that only
 * names the transport type. */
#include "../../include/rt/x280_noc_map.h"

/* The operations below are L2CPU-tile machinery: the LIM-resident hart protocol
 * and the 2 MiB outbound TLB windows. Tensix, ERISC and DRISC cores are RISC-V
 * as well, so __riscv does not separate them; the register width does, as those
 * cores are RV32. rt/x280_pinned_input.h discriminates the same way.
 *
 * A transport for a Tensix-class core is not a matter of substituting addresses.
 * It needs NoC operations that tile has (a semaphore write, not a TLB window),
 * and its release poll must invalidate the L1 cache on every iteration: a peer's
 * NoC write does not invalidate Blackhole L1, so a plain load spins on a stale
 * word forever. Until such a transport exists, an enabled config on one of those
 * cores is a compile error rather than a barrier that silently passes.
 */
#if defined(__riscv) && __riscv_xlen == 64
#define X280_CORE_SYNC_X280_TRANSPORT 1
#include "noc.h"
#include "rt/x280_idle.h"
#else
#define X280_CORE_SYNC_X280_TRANSPORT 0
#endif

namespace core_sync {

using X280Endpoint = ::core_sync_x280_endpoint_t;

/* Corners in the order the window takes them: data flows from start to end, and
 * coordinate translation does not change that direction
 * (tt-isa-documentation, BlackholeA0/NoC/Coordinates.md:24). A one-tile
 * rectangle is valid. num_dests is destination metadata, not a count of the
 * local participants one release word wakes -- two harts on one tile may poll
 * the same word. */
using X280Group = ::core_sync_x280_group_t;

struct X280Roster {
    const X280Endpoint* endpoints = nullptr;
    uint32_t endpoint_count = 0;

    /* Tier 1. counter_addr is the director-local address of the shared arrival
     * counter and counter_arrivals is how many workers will increment it. An
     * X280 director cannot host a counter (see X280Transport::kMayHostCounter),
     * so a non-zero counter_arrivals here is refused rather than served. */
    uint32_t counter_addr = 0;
    uint32_t counter_arrivals = 0;

    /* Tier 2. Caller-owned; must outlive wait_for_arrivals. */
    const WordArrival* word_arrivals = nullptr;
    uint32_t word_arrival_count = 0;

    /* An ordered, caller-owned array of Tensix rectangles, each released by one
     * broadcast, plus how many there are. The array is borrowed and must outlive
     * notify_group; a null pointer is allowed only when group_count is zero. A
     * tile a broadcast reaches must not also appear in endpoints, or it is
     * released twice; groups must not overlap each other's tiles. The transport
     * validates these structural constraints but neither allocates nor canonicalizes
     * the topology; the caller proves that its bounded coordinates answer and that
     * its release targets cover every participant.
     * Last, so the positional forms that predate it still name the same fields. */
    const X280Group* groups = nullptr;
    uint32_t group_count = 0;
};

/* Distinct from X280Endpoint: the release path has no worker index, and a
 * field the primitive ignores is a field that goes stale. A target naming one
 * slot with another's index writes the wrong token to the right slot, which the
 * director waits on forever, so the slot address is derived. */
using X280ArrivalTarget = ::core_sync_x280_arrival_target_t;

struct X280WaitContext {
    uint64_t hartid;
    uint64_t run_value;
};

/* TlbSlot is this participant's own outbound window, not a property of anything
 * it reaches. The four harts of an L2CPU tile share one TLB configuration block
 * at a fixed peripheral-port address, so two harts given the same slot overwrite
 * each other's aim mid-sequence, and the collision is silent -- the loser's
 * window still resolves, just at the wrong tile. Every hart that participates
 * concurrently must therefore be given a distinct slot, which is the host's to
 * choose and to state in the config. */
template <uint32_t TlbSlot = X280_TLB_CORE_SYNC_BASE>
struct X280Transport {
    /* One remote slot, by NoC coordinates and address. same_tile selects a
     * direct store over a windowed write. */
    using Endpoint = X280Endpoint;

    /* The endpoints a director releases, and the arrival storage it polls. */
    using Roster = X280Roster;

    /* The Tensix rectangle one broadcast reaches, plus its landing address. */
    using Group = X280Group;

    /* Where this core's arrival is written. */
    using ArrivalTarget = X280ArrivalTarget;

    /* An X280 cannot issue a NoC atomic: System Port is uncached with no atomics
     * per PMA (x280/include/noc.h:25). It announces arrival by writing a word. */
    static constexpr ArrivalKind kArrivalKind = ArrivalKind::Word;

    /* An X280 cannot receive one either, so it cannot hold an arrival counter.
     * Atomics.md, "NoC Atomic Requests": "Software can instruct the NIU to
     * perform certain atomic operations against L1 of Tensix tiles and Ethernet
     * tiles (atomic operations cannot be performed against MMIO addresses, nor
     * DRAM addresses, nor can they be translated to or from PCIe
     * transactions)." MemoryMap.md, NOC_CTRL: "NOC_CMD_AT requires that the
     * request be destined for the L1 of a Tensix tile or Ethernet tile". Black
     * Hole NOC Specification 5.2 "NOC operations", p. 8: "NOC assumes that the
     * atomic operations are implemented in the L1 memory block. There is no
     * atomic logic in NOC itself. Atomics are not supported for AXI endpoints."
     * The L2CPU inbound path terminates at an AXI endpoint -- NOC Specification
     * 5.4.2, p. 10: "The NIU is similar, except it translates NOC packets into
     * AXI transactions rather than L1 memory and register accesses"; L2CPU Spec
     * 3.4, p. 16: "The entire front port address space of the L2CPU is visible
     * to the inbound noc address." The behaviour of such a packet is
     * unspecified; it is not documented to fail cleanly. */
    static constexpr bool kMayHostCounter = false;

    /* Identifies the polling hart and the run generation whose STOP ends the
     * wait, so a poll cannot be ended by a STOP left from an earlier session. */
    using WaitContext = X280WaitContext;

    /* False on a build that is not RV64, where the operations below do not
     * exist. The barrier refuses an enabled config against such a transport. */
    static constexpr bool kHasTransport = (X280_CORE_SYNC_X280_TRANSPORT != 0);

    /* The L2CPU system port originates a broadcast through the same 2 MiB window
     * the transport already programs for unicast. TLBWindows.md,
     * "Configuration": "mcast ... true causes (x_start, y_start) through
     * (x_end, y_end) to specify a rectangle of target Tensix tiles, whereas
     * false causes (x_end, y_end) to specify a single tile target". Only Tensix
     * tiles are reachable that way, so ERISC, DRISC and other L2CPU
     * participants still take one notify_write each. */
    static constexpr bool kHasMulticast = true;

    static constexpr bool kMayDirect = true;

    static constexpr uint32_t kTlbSlot = TlbSlot;

    static_assert(
        TlbSlot >= X280_TLB_CORE_SYNC_BASE && TlbSlot < X280_TLB_CORE_SYNC_BASE + X280_TLB_CORE_SYNC_COUNT,
        "a core_sync TLB slot must fall inside the core_sync reservation");

    /* How many arrival primitives the director must observe: the shared counter
     * plus every private word. This is configured arrivals only, and an X280
     * cannot host a counter, so counter_arrivals is refused elsewhere rather
     * than served -- but it is still counted here, where the barrier's
     * nonzero-arrival gate consults it before any transport write. */
    static inline uint64_t arrival_count(const Roster& roster) {
        return static_cast<uint64_t>(roster.counter_arrivals) + roster.word_arrival_count;
    }

    /* Whether the director has at least one release operation to issue: a group
     * broadcast or an exceptional endpoint. It does not infer participant count
     * -- one group can wake several local participants -- only whether release
     * would write anything. */
    static inline bool has_release_targets(const Roster& roster) {
        return roster.group_count != 0u || roster.endpoint_count != 0u;
    }

    /* Validates each borrowed array pointer/count pair and every target before
     * the director waits or releases anything. This proves structural safety only:
     * bounded coordinates can still name a harvested tile, which the topology owner must reject.
     * Returns false without reading an array whose pointer is null. */
    static inline bool validate_roster(const Roster& roster) {
        if (roster.counter_arrivals > 0u || !word_arrivals_valid(roster.word_arrivals, roster.word_arrival_count) ||
            (roster.group_count > 0u && roster.groups == nullptr) ||
            (roster.endpoint_count > 0u && roster.endpoints == nullptr)) {
            return false;
        }

        for (uint32_t i = 0u; i < roster.group_count; ++i) {
            const Group& group = roster.groups[i];
            if (group.addr == 0u || (group.addr & 3u) != 0u || !x280_noc_rect_valid(group.rect) ||
                group.num_dests == 0u ||
                group.num_dests >
                    (group.rect.end.x - group.rect.start.x + 1u) * (group.rect.end.y - group.rect.start.y + 1u)) {
                return false;
            }
            for (uint32_t j = 0u; j < i; ++j) {
                const x280_noc_rect_t& previous = roster.groups[j].rect;
                if (group.rect.start.x <= previous.end.x && previous.start.x <= group.rect.end.x &&
                    group.rect.start.y <= previous.end.y && previous.start.y <= group.rect.end.y) {
                    return false;
                }
            }
        }

        for (uint32_t i = 0u; i < roster.endpoint_count; ++i) {
            const Endpoint& endpoint = roster.endpoints[i];
            if (endpoint.addr == 0u || (endpoint.addr & 3u) != 0u || endpoint.same_tile > 1u ||
                (!endpoint.same_tile && !x280_noc_coord_valid(endpoint.noc))) {
                return false;
            }
            for (uint32_t j = 0u; j < i; ++j) {
                if (endpoint == roster.endpoints[j]) {
                    return false;
                }
            }
            if (!endpoint.same_tile) {
                for (uint32_t j = 0u; j < roster.group_count; ++j) {
                    if (x280_noc_rect_contains(roster.groups[j].rect, endpoint.noc)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

#if X280_CORE_SYNC_X280_TRANSPORT

    /* Writes value to ep. Returns once the write is issued and retired from
     * the window, not once it is delivered: the receiver observes delivery by
     * polling for its token. Carries release traffic only; arrivals go through
     * notify_arrival. A same-tile peer takes a direct LIM store; any other takes
     * a posted write through kTlbSlot, whose window is cleared before return, so
     * the slot is free for the next call. The write may still be in flight
     * then. That it keeps its old target rests on Noc2AxiDesignSpec, which
     * captures the window's sideband bits on the AXI request at acceptance; no
     * L2CPU document states it.
     *
     * Returns false when the window was refused -- an out-of-range coordinate or
     * slot. A peer that has simply not arrived is not a failure and is not
     * reported here; that is what wait() is for. The caller owns ep's validity:
     * a well-formed coordinate naming a harvested tile hangs rather than
     * faulting. */
    static inline bool notify_write(const Endpoint& ep, uint32_t value) {
        if (ep.same_tile) {
            volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(ep.addr);
            *dst = value;
            x280_fence_ow();
            return true;
        }
        noc_tlb_2m_t programmed;
        volatile void* window = noc_configure_tlb_2m_checked(
            kTlbSlot, ep.noc.x, ep.noc.y, ep.addr, NOC_ORDER_POSTED, NOC_VC_DYNAMIC, &programmed);
        if (window == nullptr) {
            return false;
        }
        *reinterpret_cast<volatile uint32_t*>(window) = value;
        unordered_read_back(window);
        (void)noc_clear_tlb_2m(kTlbSlot);
        return true;
    }

    /* Writes value to every tile of the Tensix rectangle g in one packet
     * through kTlbSlot. Returns once the write is issued and retired from the
     * window, as notify_write does; the receivers observe delivery by polling.
     *
     * The window is programmed with mcast set, the rectangle in
     * x_start/y_start/x_end/y_end, and num_destinations_override zero, which is
     * permitted because this call programs neither keep/skip masking nor an
     * exclusion, and hardware then substitutes the count (TLBWindows.md,
     * "Configuration"). Whether a non-zero override is honoured under posted
     * writes is not documented; this path does not take it. An acknowledged
     * multicast would need g.num_dests as an explicit override, because the
     * same section does not say whether the substituted count excludes the
     * opted-out tiles a rectangle may span.
     *
     * No read follows the store: a read through a broadcast window is out of
     * contract (BlackholeA0/NoC/MemoryMap.md, NOC_CMD_BRCST_PACKET "should
     * always be false for read requests"). The window is retired before return.
     *
     * Returns false when the window was refused -- an out-of-range coordinate or
     * slot. A destination that has not yet begun polling is not a failure: the
     * write latches. The caller owns g's validity; a well-formed rectangle
     * containing a tile that does not answer hangs rather than faulting. */
    static inline bool notify_multicast(const Group& g, uint32_t value) {
        noc_tlb_2m_t programmed;
        volatile void* window = noc_configure_tlb_2m_mcast(
            kTlbSlot,
            g.rect.start.x,
            g.rect.start.y,
            g.rect.end.x,
            g.rect.end.y,
            g.addr,
            NOC_ORDER_POSTED,
            /*multicast_en=*/1,
            /*num_destinations_override=*/0u,
            NOC_VC_DYNAMIC,
            &programmed);
        if (window == nullptr) {
            return false;
        }
        *reinterpret_cast<volatile uint32_t*>(window) = value;
        (void)noc_clear_tlb_2m(kTlbSlot);
        return true;
    }

    /* Writes value to every group's rectangle in one broadcast each, in array
     * order, then to every exceptional endpoint in turn, stopping at the first
     * refusal. Each group reuses kTlbSlot: the window is retired before the next
     * group programs it. The endpoints are sequential because one window carries
     * them all: the release is issued to roster.endpoints[0] a round trip before
     * the last, and that skew is the cost of a member no broadcast reaches.
     *
     * Returns false without writing anything when validation fails. A caller
     * going through core_sync_barrier has already performed the same preflight
     * before waiting for arrivals. */
    static inline bool notify_group(const Roster& roster, uint32_t value) {
        if (!validate_roster(roster)) {
            return false;
        }
        for (uint32_t i = 0u; i < roster.group_count; ++i) {
            const Group& g = roster.groups[i];
            if (!notify_multicast(g, value)) {
                return false;
            }
        }
        for (uint32_t i = 0u; i < roster.endpoint_count; ++i) {
            if (!notify_write(roster.endpoints[i], value)) {
                return false;
            }
        }
        return true;
    }

    /* Spins on local_addr until it holds expected, or until the host publishes
     * STOP for ctx.run_value. Returns false in the latter case, and a caller
     * that receives it must unwind to its return from main rather than proceed:
     * the counterparty has left and nothing will arrive. */
    static inline bool wait(uint32_t local_addr, uint32_t expected, const WaitContext& ctx) {
        const volatile uint32_t* slot = reinterpret_cast<volatile const uint32_t*>(local_addr);
        return x280_hart_wait_word_or_stop(ctx.hartid, ctx.run_value, slot, expected) != 0;
    }

    /* Announces this core's arrival by writing its ready token to the
     * director's word, over the same path notify_write uses.
     *
     * Returns false only on a transport failure the caller cannot retry past --
     * a window the coordinates were refused for. A director that has not yet
     * begun waiting is not a failure: the word latches and is observed whenever
     * the director looks. */
    static inline bool notify_arrival(const ArrivalTarget& t, uint32_t run_nonce) {
        const uint32_t slot = core_sync_slot_addr(t.ready_base, t.worker_index);
        const Endpoint ep{t.noc, slot, t.same_tile};
        return notify_write(ep, core_sync_ready_token(run_nonce, t.worker_index));
    }

    /* Returns once every word in roster holds its worker's ready token for
     * run_nonce. Returns false when the wait was abandoned on host STOP, exactly
     * as wait() does, and the caller must unwind rather than proceed.
     *
     * Also returns false, without polling anything, when the roster asks this
     * core to host an arrival counter: an X280 cannot receive the inbound NoC
     * atomic that would carry it (see kMayHostCounter). The refusal is here
     * rather than at compile time because the roster reaches this function as a
     * reference, so counter_arrivals is not a constant expression at this point;
     * make_core_sync_roster rejects the same shape at the host boundary, where
     * it can name the offending value. */
    static inline bool wait_for_arrivals(const Roster& roster, uint32_t run_nonce, const WaitContext& ctx) {
        static_assert(!kMayHostCounter, "this refusal is written against an X280 that cannot host a counter");
        if (roster.counter_arrivals > 0u) {
            return false;
        }
        for (uint32_t i = 0u; i < roster.word_arrival_count; ++i) {
            const uint32_t expected = core_sync_ready_token(run_nonce, roster.word_arrivals[i].worker_index);
            if (!wait(roster.word_arrivals[i].addr, expected, ctx)) {
                return false;
            }
        }
        return true;
    }

private:
    /* One fenced load through a unicast window, its value not inspected. Under
     * the posted ordering mode a read may overtake the write before it
     * (WormholeB0/PCIExpressTile/TLBs.md, write then read: "Reordering
     * possible"; TLBWindows.md calls the L2CPU bits functionally identical), so
     * it proves nothing about delivery. It is kept so the unicast release keeps
     * the timing its recorded characterization measured. Never call it on a
     * broadcast window: a read there is out of contract
     * (BlackholeA0/NoC/MemoryMap.md, NOC_CMD_BRCST_PACKET). */
    static inline __attribute__((always_inline)) void unordered_read_back(volatile void* window) {
        x280_fence_iorw();
        const uint32_t observed = *reinterpret_cast<volatile uint32_t*>(window);
        __asm__ volatile("" : : "r"(observed) : "memory");
        x280_fence_iorw();
    }

#endif /* X280_CORE_SYNC_X280_TRANSPORT */
};

}  // namespace core_sync

#endif /* X280_CORE_SYNC_TRANSPORT_X280_HPP */
