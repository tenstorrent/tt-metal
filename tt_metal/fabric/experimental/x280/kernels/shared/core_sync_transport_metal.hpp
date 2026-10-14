// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_transport_metal.hpp -- tt-metal hardware transport
// policy for core_sync_barrier.
#ifndef X280_CORE_SYNC_TRANSPORT_METAL_HPP
#define X280_CORE_SYNC_TRANSPORT_METAL_HPP

#include <stdint.h>
#include <cstddef>
#include <type_traits>

#include "core_sync_arrival.hpp"
#include "core_sync_layout.h"
#include "../../include/x280_noc_coord.h"

#if defined(__riscv) && __riscv_xlen == 32
#define X280_CORE_SYNC_METAL_TRANSPORT 1
#include "api/dataflow/dataflow_api.h"
#else
#define X280_CORE_SYNC_METAL_TRANSPORT 0
#endif

#if !X280_CORE_SYNC_METAL_TRANSPORT
#ifndef PROGRAMMABLE_CORE_TYPE_DEFINED
#define PROGRAMMABLE_CORE_TYPE_DEFINED
enum ProgrammableCoreType {
    TENSIX = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH = 2,
    DRAM = 3,
    COUNT = 4,
};
#endif
#endif

namespace core_sync {

/* Remote core synchronization target. noc is the virtual (translated) NoC
 * coordinate. semaphore_id is a 0..7 hardware semaphore index (when
 * is_semaphore is true) or a direct local L1 byte address (when is_semaphore
 * is false). core_type is the destination core's type, used for address
 * translation and routing policy. */
struct MetalEndpoint {
    x280_noc_coord_t noc;
    uint32_t semaphore_id;
    ProgrammableCoreType core_type;
    bool is_semaphore;

    constexpr MetalEndpoint() :
        noc{0u, 0u}, semaphore_id(0), core_type(ProgrammableCoreType::TENSIX), is_semaphore(false) {}

    /* core_type has no default: it is what resolves semaphore_id to an address
     * on the destination, and a wrong family names a real but unrelated
     * location. */
    constexpr MetalEndpoint(x280_noc_coord_t c, uint32_t sem_id, ProgrammableCoreType ct, bool is_sem = true) :
        noc(c), semaphore_id(sem_id), core_type(ct), is_semaphore(is_sem) {}

    /* A raw L1 destination. core_type reaches nothing here but DRISC routing,
     * semaphore_id being an address already, so TENSIX stands for "no tag" and
     * is the right default for a family with no honest ProgrammableCoreType --
     * an L2CPU hart. */
    static constexpr MetalEndpoint raw_l1(
        x280_noc_coord_t c, uint32_t l1_addr, ProgrammableCoreType ct = ProgrammableCoreType::TENSIX) {
        return MetalEndpoint(c, l1_addr, ct, false);
    }

    bool operator==(const MetalEndpoint& o) const {
        return noc.x == o.noc.x && noc.y == o.noc.y && semaphore_id == o.semaphore_id && core_type == o.core_type &&
               is_semaphore == o.is_semaphore;
    }
    bool operator!=(const MetalEndpoint& o) const { return !(*this == o); }
};

/* A rectangle of Tensix cores reached by one broadcast, the semaphore the
 * release token lands on, and how many cores the packet is delivered to.
 *
 * rect runs from its start to its end corner in virtual (translated) NoC
 * coordinates; notify_multicast swaps the corners on NoC 1. semaphore_id is a
 * hardware semaphore index, identical on every destination, which is what makes
 * one address serve the whole rectangle.
 *
 * num_dests counts the destinations the sender is targeting and EXCLUDES the
 * sender itself, which the non-loopback multicast does not write
 * (dataflow_api.h:904-907). It is not derivable from the corners: whether the
 * director sits inside the rectangle is the host's knowledge, not the
 * transport's. Zero means there is no rectangle to release. */
struct MetalGroup {
    x280_noc_rect_t rect{};
    uint32_t semaphore_id = 0;
    uint32_t num_dests = 0;
};

/* Release roster for director notification, and the arrival storage the
 * director polls. endpoints points to an array of count destination endpoints
 * owned by caller; must remain valid for duration of notify_group. It holds
 * only members outside grid: a core the broadcast reaches must not also be
 * written, or it is released twice. */
struct MetalRoster {
    const MetalEndpoint* endpoints = nullptr;
    uint32_t count = 0;

    /* Tier 1. counter_addr is the director-local address of the shared arrival
     * counter; counter_arrivals is how many workers will increment it. Zero
     * arrivals means no counter is consulted and counter_addr is not read. */
    uint32_t counter_addr = 0;
    uint32_t counter_arrivals = 0;

    /* Tier 2. Caller-owned; must outlive wait_for_arrivals. */
    const WordArrival* word_arrivals = nullptr;
    uint32_t word_arrival_count = 0;

    /* The Tensix rectangle one broadcast releases. A grid with num_dests zero
     * leaves the corners unread and no broadcast is issued. Last, so the
     * positional forms that predate it still name the same fields. */
    MetalGroup grid{};
};

/* The director's arrival counter, as a worker reaches it. noc is the
 * director's virtual (translated) NoC coordinate, counter_semaphore_id its hardware
 * semaphore index, and core_type the DIRECTOR's core type, which selects the
 * semaphore address space and the DRISC tag.
 *
 * core_type carries an initializer only because TENSIX is the zero value an
 * aggregate initializer would supply anyway; naming the director's family is
 * the caller's obligation, not this struct's default. */
struct MetalCounterArrivalTarget {
    x280_noc_coord_t noc{};
    uint32_t counter_semaphore_id = 0;
    ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX;
};

/* Where this worker's ready word lives in the director's memory, and the index
 * whose ready token that word must hold. noc is the director's virtual
 * (translated) NoC coordinate; ready_base is the director-local byte address of its
 * ready table, and the slot is core_sync_slot_addr(ready_base, worker_index).
 *
 * The base is carried rather than the resolved slot: one field holding the
 * address and another the index that derives it are free to disagree, and a
 * target naming one slot with another's index writes the wrong token to the
 * right slot, which the director waits on forever.
 *
 * There is no core_type. A word slot is a raw director-local address, never a
 * semaphore, so the only thing core_type would decide is the DRISC tag -- and a
 * DRISC is refused as director on kMayDirect, so a word arrival never addresses
 * one. */
struct MetalWordArrivalTarget {
    x280_noc_coord_t noc{};
    uint32_t ready_base = 0;
    uint32_t worker_index = 0;
};

/* Bounded wait context carrying an iteration cap for polling. */
struct MetalWaitContext {
    static constexpr uint64_t kDefaultIterationCap = 100000000ULL;

    uint64_t hartid = 0;
    uint64_t run_value = 0;
    uint64_t iteration_cap = kDefaultIterationCap;

    constexpr MetalWaitContext() = default;
    /** The form core_sync_barrier constructs: rv becomes the iteration cap,
     * or kDefaultIterationCap when it is zero. A Metal participant's poll bound
     * is therefore the run_value its caller passes to sync(). */
    constexpr MetalWaitContext(uint64_t h, uint64_t rv) :
        hartid(h), run_value(rv), iteration_cap(rv != 0 ? rv : kDefaultIterationCap) {}
    constexpr MetalWaitContext(uint64_t h, uint64_t rv, uint64_t cap) : hartid(h), run_value(rv), iteration_cap(cap) {}
};

/* Hardware transport policy using tt-metal NoC semaphore operations for
 * core_sync_barrier. Parameterized by this core's own ProgrammableCoreType and
 * by the arrival primitive it originates. Directing is permitted for TENSIX and
 * ETH cores; DRAM (DRISC) directing is refused at compile time because
 * coordinating requires stream mode, which disables Tensix-to-GDDR through that
 * endpoint.
 *
 * Arrival is Counter by default, which is what a Tensix or Ethernet worker takes
 * against a director that can receive a NoC atomic. Word is what it takes
 * against one that cannot -- an X280 director, whose inbound path terminates at
 * an AXI endpoint (see X280Transport::kMayHostCounter for the citations). */
template <ProgrammableCoreType CoreType, ArrivalKind Arrival = ArrivalKind::Counter>
struct MetalTransport {
    using Endpoint = MetalEndpoint;
    using Roster = MetalRoster;
    using WaitContext = MetalWaitContext;
    using Group = MetalGroup;

    /* Where this core announces arrival, in the shape the chosen primitive
     * needs: the director's counter semaphore, or the director's ready table
     * and this worker's index in it. */
    using ArrivalTarget =
        std::conditional_t<Arrival == ArrivalKind::Word, MetalWordArrivalTarget, MetalCounterArrivalTarget>;

    static constexpr bool kHasTransport = (X280_CORE_SYNC_METAL_TRANSPORT != 0);

    /* A Tensix or Ethernet core originates a broadcast over the Tensix grid; a
     * DRISC core does not. An ethernet kernel calling noc_semaphore_set_multicast
     * directly is what settles the ETH arms
     * (tt-metal/tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer_eth.cpp:90-91). */
    static constexpr bool kHasMulticast =
        (CoreType == ProgrammableCoreType::TENSIX || CoreType == ProgrammableCoreType::ACTIVE_ETH ||
         CoreType == ProgrammableCoreType::IDLE_ETH);
    static constexpr bool kMayDirect = (CoreType != ProgrammableCoreType::DRAM);

    /* A Tensix or Ethernet core can issue a NoC atomic, so it may announce
     * arrival by incrementing the director's shared counter. It falls back to a
     * private word when the director cannot receive one. */
    static constexpr ArrivalKind kArrivalKind = Arrival;

    /* The core family this transport is written for. A release endpoint built
     * for one of its workers resolves a semaphore id against this family's
     * base, so a roster group must carry it rather than assume Tensix. */
    static constexpr ProgrammableCoreType kCoreType = CoreType;

    /* DRISC release words use caller-owned L1 because Metal does not
     * initialize DRAM-core semaphores. */
    static constexpr bool kUsesSemaphoreRelease = (CoreType != ProgrammableCoreType::DRAM);

    /* Atomics.md, "NoC Atomic Requests": the NIU performs atomics "against L1 of
     * Tensix tiles and Ethernet tiles (atomic operations cannot be performed
     * against MMIO addresses, nor DRAM addresses ...)". A DRISC core is refused
     * as director on kMayDirect before this is consulted; the constant records
     * the inbound restriction rather than restating the outbound one. */
    static constexpr bool kMayHostCounter = (CoreType != ProgrammableCoreType::DRAM);

    /* How many arrival primitives the director must observe: the shared counter
     * plus every private word. Configured arrivals only. */
    static inline uint64_t arrival_count(const Roster& roster) {
        return static_cast<uint64_t>(roster.counter_arrivals) + roster.word_arrival_count;
    }

    /* Whether the director has at least one release operation to issue: the
     * Tensix rectangle or an endpoint outside it. It does not infer participant
     * count, only whether release would write anything. */
    static inline bool has_release_targets(const Roster& roster) {
        return roster.grid.num_dests != 0u || roster.count != 0u;
    }

    /* Validates borrowed roster pointer/count pairs and locally checkable target
     * fields before the director waits or releases anything. Device placement remains
     * the topology owner's responsibility. */
    static inline bool validate_roster(const Roster& roster) {
        if ((roster.count > 0u && roster.endpoints == nullptr) ||
            !word_arrivals_valid(roster.word_arrivals, roster.word_arrival_count) ||
            (roster.counter_arrivals > 0u && (roster.counter_addr == 0u || (roster.counter_addr & 3u) != 0u))) {
            return false;
        }
        if (roster.grid.num_dests > 0u &&
            (!x280_noc_rect_valid(roster.grid.rect) || roster.grid.semaphore_id > 7u || !kHasMulticast)) {
            return false;
        }
        for (uint32_t i = 0u; i < roster.count; ++i) {
            if (!x280_noc_coord_valid(roster.endpoints[i].noc)) {
                return false;
            }
        }
        return true;
    }

#if X280_CORE_SYNC_METAL_TRANSPORT

    /* Resolves a hardware semaphore index to the destination core's local L1
     * byte address. core_type selects the address space; an unrecognised type is
     * treated as TENSIX, matching Metal's own default placement. */
    static inline uint32_t semaphore_addr(ProgrammableCoreType core_type, uint32_t semaphore_id) {
        switch (core_type) {
            case ProgrammableCoreType::ACTIVE_ETH:
                return static_cast<uint32_t>(get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(semaphore_id));
            case ProgrammableCoreType::IDLE_ETH:
                return static_cast<uint32_t>(get_semaphore<ProgrammableCoreType::IDLE_ETH>(semaphore_id));
            case ProgrammableCoreType::DRAM:
                return static_cast<uint32_t>(get_semaphore<ProgrammableCoreType::DRAM>(semaphore_id));
            case ProgrammableCoreType::TENSIX:
            default: return static_cast<uint32_t>(get_semaphore<ProgrammableCoreType::TENSIX>(semaphore_id));
        }
    }

    /* Delivers value to ep and blocks until the write is acknowledged: for a
     * Tensix or ERISC destination the value is then in its L1
     * (NoC/Counters.md); for an L2CPU destination it was accepted at the
     * L2CPU, and a write issued after the ack reaches LIM after it by owner
     * assumption (MEASUREMENT.md "Unconfirmed hardware assumptions", item 2). Carries release traffic and
     * notify_arrival's Word arrivals. A DRISC core-sync participant is in
     * stream mode, where experimental/drisc_mode.h requires an untagged local
     * L1 address; the 0x2000000000 tag is for NOC2AXI mode. Delivery cannot
     * fail: underlying tt-metal NoC operations provide no error status, and an
     * unmapped destination hangs rather than reporting an error. Returns true
     * on completion. */
    static inline bool notify_write(const Endpoint& ep, uint32_t value) {
        const uint32_t dest_l1_addr = ep.is_semaphore ? semaphore_addr(ep.core_type, ep.semaphore_id) : ep.semaphore_id;

        const uint64_t dst_noc_addr = get_noc_addr(ep.noc.x, ep.noc.y, dest_l1_addr);

        uint32_t src_l1_addr = noc_get_interim_inline_value_addr(noc_index, dst_noc_addr);
        volatile tt_l1_ptr uint32_t* interim_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1_addr);
        *interim_ptr = value;

        noc_semaphore_set_remote(src_l1_addr, dst_noc_addr);
        noc_async_write_barrier();
        return true;
    }

    /* Announces this core's arrival to the director, by the primitive Arrival
     * names.
     *
     * Under Counter, adds one to the director's arrival counter. The increment
     * carries no identity: the counter records how many workers arrived, not
     * which, which is what lets one 4-byte counter serve a roster of any size.
     * run_nonce is not read on this arm -- freshness for a counter comes from
     * the host zeroing it before launch, not from the value written; there is no
     * value to choose, an increment being the only operation the NIU offers
     * here.
     *
     * Under Word, writes core_sync_ready_token(run_nonce, worker_index) to this
     * worker's slot in the director's ready table, over the same path
     * notify_write uses. run_nonce IS read on this arm: the token is what
     * distinguishes this run's arrival from a word left by the last.
     *
     * The destination is a raw director-local address, so the endpoint's
     * core_type reaches nothing but the DRISC tag, and a DRISC is refused as
     * director on kMayDirect before any worker addresses one.
     *
     * Returns true on the Counter arm: the underlying NoC atomic reports no
     * status, and a director that has not yet begun waiting is not a failure,
     * the increment being observed whenever the director looks. Returns what
     * notify_write returns on the Word arm, which is likewise true on a delivery
     * that completed. */
    static inline bool notify_arrival(const ArrivalTarget& target, uint32_t run_nonce) {
        if constexpr (Arrival == ArrivalKind::Counter) {
            static_assert(
                CoreType != ProgrammableCoreType::DRAM, "a DRISC worker must announce arrival through a private word");
            (void)run_nonce;
            uint64_t dst_noc_addr =
                get_noc_addr(target.noc.x, target.noc.y, semaphore_addr(target.core_type, target.counter_semaphore_id));
            if (target.core_type == ProgrammableCoreType::DRAM) {
                dst_noc_addr += 0x2000000000ull;
            }
            noc_semaphore_inc(dst_noc_addr, 1);
            return true;
        } else {
            const Endpoint slot =
                Endpoint::raw_l1(target.noc, core_sync_slot_addr(target.ready_base, target.worker_index));
            return notify_write(slot, core_sync_ready_token(run_nonce, target.worker_index));
        }
    }

    /* Delivers value to every core of the Tensix rectangle g in one packet, and
     * blocks until the write completes and is observable by them.
     *
     * Declared only when kHasMulticast: a core that cannot broadcast has no such
     * operation rather than one that reports a delivery it did not make.
     *
     * The token lands on g.semaphore_id, resolved in the TENSIX semaphore space
     * because the destinations are Tensix whatever this core is. g.num_dests is
     * passed through as the host computed it and is not re-derived from the
     * corners.
     *
     * The sender is NOT written to, whether or not it lies inside the rectangle
     * (dataflow_api.h:904-907); a director does not consume its own release
     * token, so the loopback variant is never used here. The rectangle must
     * contain at least one core other than the sender, which is what
     * num_dests > 0 states.
     *
     * Owns the corner order, which is NoC-dependent, and the transaction-ID
     * reset a multicast batch leaves behind. Returns true: as with notify_write,
     * the underlying NoC operations carry no error status and a misaimed
     * rectangle hangs rather than reporting. The caller owns g's validity. */
    template <bool M = kHasMulticast, typename = typename std::enable_if<M>::type>
    static inline bool notify_multicast(const Group& g, uint32_t value) {
        const uint32_t sem = semaphore_addr(ProgrammableCoreType::TENSIX, g.semaphore_id);

        /* Corner order is NoC-dependent: forward on NOC0, reversed on NOC1.
         * dynamic_noc_writer_eth.cpp:38-44. Wrong order gives an empty or
         * misaimed rectangle. */
        const uint64_t dst =
            (noc_index == 0)
                ? get_noc_multicast_addr(g.rect.start.x, g.rect.start.y, g.rect.end.x, g.rect.end.y, sem, noc_index)
                : get_noc_multicast_addr(g.rect.end.x, g.rect.end.y, g.rect.start.x, g.rect.start.y, sem, noc_index);

        const uint32_t src_l1_addr = noc_get_interim_inline_value_addr(noc_index, dst);
        volatile tt_l1_ptr uint32_t* interim_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1_addr);
        *interim_ptr = value;

        noc_semaphore_set_multicast(src_l1_addr, dst, g.num_dests, false, noc_index);
        noc_async_write_barrier();

        /* Multicasts consume transaction IDs; the unicast path does not.
         * dynamic_noc_writer_eth.cpp:106. */
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        return true;
    }

    /* Delivers value to every endpoint in roster: one broadcast over the Tensix
     * rectangle when this core can originate one and the rectangle is non-empty,
     * then one notify_write per endpoint outside it. Refuses compilation if
     * CoreType is DRAM.
     *
     * Returns false when a rectangle is asked of a core that cannot broadcast:
     * those destinations would otherwise go unreleased and the director would
     * report success. Returns true once every member has been written. */
    static inline bool notify_group(const Roster& roster, uint32_t value) {
        static_assert(
            kMayDirect,
            "coordinating requires initiating, which requires stream mode, which "
            "disables Tensix-to-GDDR through that endpoint, and NIU_CFG_0 persists "
            "across runs");
        if (!validate_roster(roster)) {
            return false;
        }
        if constexpr (kHasMulticast) {
            if (roster.grid.num_dests > 0u) {
                if (!notify_multicast(roster.grid, value)) {
                    return false;
                }
            }
        } else {
            if (roster.grid.num_dests > 0u) {
                return false;
            }
        }
        for (uint32_t i = 0u; i < roster.count; ++i) {
            if (!notify_write(roster.endpoints[i], value)) {
                return false;
            }
        }
        return true;
    }

    /* Spins on local_addr until satisfied() accepts the word read there, or
     * until ctx.iteration_cap is exhausted. Invalidates L1 cache on each
     * iteration: a peer's NoC write does not invalidate Blackhole L1, so a plain
     * load spins on a stale word forever. Returns false on expiry, so no caller
     * of this file can spin without bound. */
    template <typename Satisfied>
    static inline bool bounded_poll(uint32_t local_addr, const WaitContext& ctx, Satisfied satisfied) {
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_addr);
        for (uint64_t i = 0; i < ctx.iteration_cap; ++i) {
            invalidate_l1_cache();
            if (satisfied(*ptr)) {
                return true;
            }
        }
        return false;
    }

    /* Spins on local_addr until expected is observed or ctx.iteration_cap is
     * exhausted. Returns true on arrival, false on expiry. */
    static inline bool wait(uint32_t local_addr, uint32_t expected, const WaitContext& ctx) {
        return bounded_poll(local_addr, ctx, [expected](uint32_t observed) { return observed == expected; });
    }

    /* Returns once every participant has arrived: the counter at
     * roster.counter_addr has reached roster.counter_arrivals AND every word in
     * roster.word_arrivals holds its worker's ready token for run_nonce. A zero
     * counter_arrivals leaves counter_addr unread.
     *
     * Returns false when the wait was abandoned on an expired iteration cap,
     * exactly as wait() does, and the caller must unwind rather than proceed.
     *
     * The counter test is >=, not ==: the counter only rises, and an equality
     * test can be missed when two increments land between polls. */
    static inline bool wait_for_arrivals(const Roster& roster, uint32_t run_nonce, const WaitContext& ctx) {
        if (roster.counter_arrivals > 0u) {
            const uint32_t needed = roster.counter_arrivals;
            if (!bounded_poll(roster.counter_addr, ctx, [needed](uint32_t observed) { return observed >= needed; })) {
                return false;
            }
        }
        for (uint32_t i = 0u; i < roster.word_arrival_count; ++i) {
            const uint32_t expected = core_sync_ready_token(run_nonce, roster.word_arrivals[i].worker_index);
            if (!wait(roster.word_arrivals[i].addr, expected, ctx)) {
                return false;
            }
        }
        return true;
    }

#endif
};

}  // namespace core_sync

#endif /* X280_CORE_SYNC_TRANSPORT_METAL_HPP */
