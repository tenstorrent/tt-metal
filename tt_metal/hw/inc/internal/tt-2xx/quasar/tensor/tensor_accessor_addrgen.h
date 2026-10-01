// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the Quasar overlay address generator (overlay/addrgen_api.hpp) behind TensorAccessor, deriving banking
// parameters from the active ATT map instead of hardcoded endpoint IDs (see address_generators.md and the
// quasar_aether_2x3_att_config.h map).
//
// The address generator only *produces* addresses here. Each one is popped back to the RISC-V and handed to the
// ordinary NoC V3 APIs, which program and issue the transaction exactly as they do for software-computed addresses.
// Nothing in this header writes command buffer registers, so it cannot perturb the transaction state the NoC APIs
// rely on. Pushing straight into the command buffer (push_src + issue) is a later step.
//
// Only builds under NOC_ATT_ENABLED: the banking recipes below are ATT endpoint-selector encoding
// (selector << endpoint_shift), which has no meaning under the default V2/XY backend.

#pragma once

#ifndef NOC_ATT_ENABLED
#error "tensor_accessor_addrgen.h requires the ATT address backend (NOC_ATT_ENABLED); see address_generators.md"
#endif

#include <cstdint>
#include <type_traits>

#include "internal/tt-2xx/quasar/noc/att/att_config.h"
#include "internal/tt-2xx/quasar/noc_address_backend.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/addrgen_state.hpp"

namespace tt_addrgen {

// Inner-loop end bound. X_END is an absolute address compared against the running inner address, so it must exceed
// any local address a walk can reach (the caller bounds each walk by page count, not by this).
inline constexpr uint64_t kInnerEndSentinel = uint64_t{1} << 48;

template <bool IsDram>
constexpr const noc_att::Window& interleaved_window() {
    return noc_att::map_window(ACTIVE_ATT_MAP, IsDram ? noc_att::WindowClass::Dram : noc_att::WindowClass::Worker);
}

// The bank shift lives in a MISC register shared by an address generator's two sides, and a walk's registers are
// restored into whichever slot is free, so every walk on a core must use the same shift. DRAM and worker windows do on
// the current maps (26); a map where they differ would need per-slot bookkeeping here.
namespace att_check {
constexpr const noc_att::Window& dram = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Dram);
constexpr const noc_att::Window& worker = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Worker);
static_assert(
    noc_att::is_no_window(dram) || noc_att::is_no_window(worker) || dram.endpoint_shift == worker.endpoint_shift,
    "walks share one bank shift per address generator; DRAM and worker windows must use the same endpoint_shift");
}  // namespace att_check

// ATT selector of interleaved bank `bank`, resolved exactly as the software path resolves it (DRAM bank ->
// dram_selectors[], L1 bank -> l1_bank_to_noc_xy -> worker selector).
template <bool IsDram>
inline __attribute__((always_inline)) uint32_t interleaved_bank_selector(uint32_t bank) {
    return interleaved_window<IsDram>().selector(noc_address_backend::bank_address<IsDram>(bank, 0, noc_index));
}

// Whether a single BankingConfig can walk this device's interleaved banks in page-id order: bank i's ATT selector
// must be (bank 0's selector + i) and no per-bank offset may apply (the hardware adds only bank << endpoint_shift;
// software also adds bank_to_{dram,l1}_offset[bank]). The device assumes it can -- that is the intended configuration.
// Bring-up (a device whose allocator shuffles L1 banks, or a new ATT map) checks this on the host and builds the
// kernel with TT_TA_ADDRGEN_INTERLEAVED_{DRAM,L1}_SW, which sends that memory's interleaved transfers to software.
template <bool IsDram>
inline constexpr bool interleaved_walkable =
#if defined(TT_TA_ADDRGEN_INTERLEAVED_DRAM_SW)
    IsDram ? false :
#endif
#if defined(TT_TA_ADDRGEN_INTERLEAVED_L1_SW)
    !IsDram ? false
            :
#endif
            true;

// Programs addrgen_1's source side for a page-id-order sequential walk of an interleaved DRAM or L1
// TensorAccessor, starting at page 0. BANK_INNER: for page_id = 0, 1, 2, ..., the bank id cycles through
// every bank once before the inner (per-bank) offset advances -- the same bank_offset_index / bank_index
// split InterleavedAddrGen<IsDram>::get_noc_addr computes in software.
//
// Requires interleaved_walkable<IsDram>; only bank 0's selector is read here.
//
// Deliberately does not call setup_src_base_start_addrgen(): that writes the *command buffer's* SRC_BASE
// register, which would outlive this walk and leak into every later software read on the same command
// buffer. The window's compare bits are OR'd in by pop_src_noc_addr_interleaved() instead.
template <bool IsDram, typename TensorAccessorT>
inline __attribute__((always_inline)) void configure_addrgen_src_interleaved(const TensorAccessorT& acc) {
    constexpr uint32_t num_banks = IsDram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const noc_att::Window& window = interleaved_window<IsDram>();
    const uint32_t first_selector = interleaved_bank_selector<IsDram>(0);

    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_src_banking_addrgen<overlay::ADDRGEN_1>(overlay::BankingConfig{
        .endpoint_id_shift = window.endpoint_shift,
        .size = num_banks,
        .skip = 1,
        .base = first_selector,
        // BANK_CURRENT is relative to BANK_BASE (bank = base + current), not an absolute selector:
        // on emu-quasar-2x3, base=2/current=2 produced selector 4 (the dispatch tile). 0 = start at bank 0.
        .current = 0,
        .bank_order = overlay::BANK_INNER,
    });
    // Inner loop starts at this tensor's per-bank base offset and advances by one aligned page each time
    // the bank loop wraps. The end is a sentinel: the caller controls how many addresses it pops.
    overlay::setup_src_inner_loop_addrgen<overlay::ADDRGEN_1>(
        /*stride=*/acc.get_aligned_page_size(),
        /*end=*/kInnerEndSentinel,
        /*start=*/acc.get_bank_base_address());
}

// Pops the next address from addrgen_1 (returning it and advancing the walk) and composes the full ATT
// NoC operand: window compare bits | (selector << endpoint_shift) | local offset. Usable directly as the
// src_noc_addr of noc_async_read().
template <bool IsDram>
inline __attribute__((always_inline)) uint64_t pop_src_noc_addr_interleaved() {
    return interleaved_window<IsDram>().compare | overlay::pop_src_addrgen<overlay::ADDRGEN_1>();
}

// ============================================================================
// Walkers behind tensor_accessor::transfer_noc_addr() (api/tensor/transfer_noc_addr.h)
// ============================================================================
//
// A walker is one address-generator source side programmed to walk one accessor's pages in page-id order. Each DM
// core has two: the source side of addrgen_0 and of addrgen_1. (Using one side per addrgen gives each walker its own
// MISC register -- bank_offset in MISC is shared by an addrgen's two sides, see addrgen_api.hpp.) Transfer addresses
// are popped back to the RISC-V and handed to the ordinary NoC APIs, never pushed into a command buffer, so which
// command buffer an addrgen is paired with doesn't matter here.
//
// A walker remembers which accessor it walks, the index the hardware is positioned at, and the index where its current
// programming stops being valid (run_end). A transfer at or ahead of the hardware inside the run pops, skipping forward
// in hardware over any gap and following a repeating stride (pop_index). Only a transfer behind the hardware (a
// re-read, a backward jump, two streams alternating on one accessor) or past the run re-seeks in software -- correct,
// just not the fast path. A third accessor evicts the least recently used walker.
//
// Recipes:
//   Interleaved: one programming walks the whole tensor. BANK_INNER cycles the bank every page and the inner loop
//     advances one page per bank wrap, as InterleavedAddrGen computes in software. Needs the device's interleaved
//     banks to be an ascending stride-1 ATT selector run with no per-bank offset (interleaved_walkable).
//   Sharded, cross-bank (plan_cross_bank): when the shards along the innermost split dimension sit on banks with
//     ascending stride-1 selectors at the same bank-local address, one BANK_MIDDLE programming walks a segment in one
//     shard, the same segment in the next shard's bank, ..., then steps the outer loop to the next row (or, for a
//     split outermost dimension such as HEIGHT / round-robin, to the next shard slot in each bank). A WIDTH tensor or
//     a BLOCK shard band is then one seek, not one per page.
//   Sharded, fallback (any rank, any distribution, DRAM or L1): software resolves the requested page's address and
//     the length of the run of following page ids that are contiguous in that same bank (contiguous_run); the
//     address generator walks the run with a single-bank BankingConfig and a page-stride inner loop.
//   shard_pages() and ShardView have their own walker kinds (WalkKind) -- see their sections at the end.
//
// Must be included after api/tensor/tensor_accessor.h (it is, via transfer_noc_addr.h).

template <typename T>
struct is_tensor_accessor : std::false_type {};
template <typename DSpecT>
struct is_tensor_accessor<TensorAccessor<DSpecT>> : std::true_type {};

// Layouts with a hardware recipe: every TensorAccessor (interleaved and sharded).
template <typename Accessor>
inline constexpr bool has_hw_recipe = is_tensor_accessor<Accessor>::value;

// What a walker steps through. One accessor can have a walker of each kind and direction; they're separate walks.
enum class WalkKind : uint8_t {
    Pages,       // global page ids (TensorAccessor / PageView / wrapper / pages())
    ShardBases,  // shard ids -> each shard's base address (ShardView)
    ShardPages,  // pages of one shard in storage order (shard_pages())
};

using tensor_accessor::TransferDir;

// Hardware slots: each address generator has a source and a destination side, so a DM core has four. A read walks on a
// source side and a write on a destination side, so reads and writes never compete for slots:
//   slot = side * 2 + generator:  reads -> 0, 1 (source sides)   writes -> 2, 3 (destination sides)
inline constexpr uint32_t kNumSlots = 4;
inline constexpr uint32_t kSlotsPerDir = 2;
constexpr uint32_t slot_generator(uint32_t slot) { return slot & 1u; }
constexpr overlay::Side slot_side(uint32_t slot) { return slot >> 1 ? overlay::Side::Dest : overlay::Side::Src; }
constexpr uint32_t first_slot(TransferDir dir) { return dir == TransferDir::Write ? kSlotsPerDir : 0; }

// Walker records, 80 bytes each: 8-byte fields first, small ones last. They are thread-local, and thread-local storage
// shares a DM core's 8 KB with its stack -- see the budget below.
struct Walker {
    uint64_t hi_bits = 0;    // window compare bits OR'd onto every pop (the addrgen produces selector | local)
    uint64_t last_addr = 0;  // ShardBases: the base popped for `shard` (ShardView transfers repeat it with offsets)
    uint32_t key = 0;        // which accessor this walks (walk_key); 0 = free record
    uint32_t bank_base = 0;  // guards against a new accessor reusing a dead one's address
    uint32_t page_size = 0;
    uint32_t next = 0;        // index the hardware is positioned at: page id, shard id, or page-in-shard (by kind)
    uint32_t run_end = 0;     // first index the current programming does not cover
    uint32_t last = 0;        // index of the previous pop (stride learning, see pop_index)
    uint32_t last_delta = 0;  // last - the pop before it (0 = none yet)
    uint32_t shard = 0;       // ShardPages: the shard being walked. ShardBases: the shard whose base was popped last
    uint32_t last_use = 0;    // LRU stamp
    // The walk's programming minus its start position (WalkProgram), kept so a spill only has to read back the
    // position (overlay::save_position_addrgen) and a restore writes the rest from here. Narrowed to fit the record:
    // bank-local strides and ends fit 32 bits and the bank registers are 8 bits; a programming that doesn't fit is
    // not restorable and is dropped instead of parked. The outer loop's end is always kInnerEndSentinel.
    uint32_t inner_stride = 0;
    uint32_t inner_end = 0;  // 0 = kInnerEndSentinel
    uint32_t outer_stride = 0;
    uint8_t bank_first = 0;  // BANK_BASE: first bank's endpoint id
    uint8_t bank_size = 0;
    uint8_t bank_skip = 0;
    uint8_t bank_shift = 0;
    uint8_t bank_order = 0;
    bool restorable = false;
    WalkKind kind = WalkKind::Pages;
    TransferDir dir = TransferDir::Read;
    bool has_prev = false;  // `last` is valid
    bool has_base = false;  // ShardBases: last_addr is valid
    uint8_t slot1 = 0;      // 1 + hardware slot the walk occupies; 0 = not resident
    uint8_t parked1 = 0;    // 1 + entry of parked_pos holding the walk's position; 0 = none
    bool resident() const { return slot1 != 0; }
    uint32_t slot() const { return slot1 - 1u; }
};

// More walks than slots are kept as records; when a walk loses its slot its position is parked and its registers are
// written back (programming from the record, position from the pool) when it is next used, instead of re-seeking in
// software. Parked positions live in a small shared pool (not in the records), since at most a couple of walks are
// parked at a time; if the pool is full the walk is simply forgotten and
// re-seeks if used again. TT_TA_ADDRGEN_NO_SPILL parks nothing (an evicted walk is forgotten), for comparison.
inline constexpr uint32_t kNumWalks = 4;  // e.g. two tensors read and two written, or three read
#if defined(TT_TA_ADDRGEN_NO_SPILL)
inline constexpr uint32_t kNumParked = 0;
#else
inline constexpr uint32_t kNumParked = 2;
#endif
inline constexpr uint32_t kParkedStorage = kNumParked > 0 ? kNumParked : 1;

// All walker state is zero-initialized on purpose: it is thread_local, and only zero-initialized TLS (.tbss) is known
// to start fresh for every kernel launch. A nonzero initializer (.tdata) let a previous kernel's walks leak into the
// next.
inline thread_local Walker walkers[kNumWalks];
inline thread_local overlay::AddrgenPosition parked_pos[kParkedStorage];
inline thread_local uint8_t parked_owner1[kParkedStorage];  // 1 + walk record parked there; 0 = free
inline thread_local uint8_t slot_walk1[kNumSlots];          // 1 + walk record occupying each slot; 0 = free
inline thread_local uint8_t generator_ready[2];             // this kernel already reset generator g
inline thread_local uint32_t walker_clock;

// A DM core's 8 KB of thread-local storage + stack already holds ~6.6 KB of DFB/CB interface state. At 640 bytes of
// walker state the sharded path overflowed the remaining stack and hung; ~380 bytes (this) runs the whole suite.
inline constexpr uint32_t kWalkerTlsBytes = sizeof(walkers) + sizeof(parked_pos) + sizeof(parked_owner1) +
                                            sizeof(slot_walk1) + sizeof(generator_ready) + sizeof(walker_clock);
static_assert(kWalkerTlsBytes <= 400, "walker state eats into the DM stack (TLS and stack share 8 KB); keep it small");

// Walk identity. An accessor built from a Metal 2.0 binding token carries its binding id in its type (Paul's
// DistributionSpec BindingId -- a base-address runtime-arg offset, so small), a compile-time constant that also makes
// copies of the accessor (e.g. the one an iterator holds) share one walk. Other accessors are identified by address
// (an L1 address, so it fits 32 bits and never has the top bit set).
template <typename Accessor>
inline uint32_t walk_key(const Accessor& acc) {
    if constexpr (Accessor::DSpec::binding_id != tensor_accessor::NO_BINDING_ID) {
        return (uint32_t{1} << 31) | Accessor::DSpec::binding_id;
    } else {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&acc));
    }
}

// One programming of a walker's source side. Loop semantics (checked by AddrgenLoopProbe): the inner loop counts from
// inner_start in steps of inner_stride and wraps to 0 at inner_end; each wrap carries into the next loop out (the bank
// loop under BANK_MIDDLE, the outer loop under BANK_INNER once the banks wrap). The outer loop keeps outer_start as
// its base and adds outer_stride per carry. Address = inner + outer + (bank << endpoint_shift).
struct WalkProgram {
    overlay::BankingConfig banking;
    uint64_t inner_stride = 0;  // one page (page walks) or one shard (shard-base walks)
    uint64_t inner_start = 0;
    uint64_t inner_end = kInnerEndSentinel;  // default: never wraps
    uint64_t outer_start = 0;
    uint64_t outer_stride = 0;
};

// Run f(generator, side) for a slot: the generator and side are immediates in each RoCC instruction (template arguments
// here), so a slot chosen at run time dispatches to one of four instantiations.
template <typename F>
inline __attribute__((always_inline)) auto with_slot(uint32_t slot, F&& f) {
    using overlay::AddrGen;
    using overlay::Side;
    switch (slot) {
        case 0:
            return f(std::integral_constant<AddrGen, overlay::ADDRGEN_0>{}, std::integral_constant<Side, Side::Src>{});
        case 1:
            return f(std::integral_constant<AddrGen, overlay::ADDRGEN_1>{}, std::integral_constant<Side, Side::Src>{});
        case 2:
            return f(std::integral_constant<AddrGen, overlay::ADDRGEN_0>{}, std::integral_constant<Side, Side::Dest>{});
        default:
            return f(std::integral_constant<AddrGen, overlay::ADDRGEN_1>{}, std::integral_constant<Side, Side::Dest>{});
    }
}

// A generator is reset once, the first time this kernel uses either of its sides, to clear whatever an earlier kernel
// left (face size and the like, which no walk programs). Never again: a reset clears both sides, and the other side may
// hold a live walk. Every register a walk depends on is written when it is programmed or restored.
template <overlay::AddrGen G>
inline __attribute__((always_inline)) void ensure_generator_reset() {
    if (!generator_ready[G]) {
        overlay::reset_addrgen<G>();
        generator_ready[G] = 1;
    }
}

// Program a walk into its slot and record the programming for restore_slot. Every field is written each time, so
// nothing from a previous programming (of this or another recipe) leaks in.
inline void program_walker(Walker& w, const WalkProgram& prog) {
    const overlay::BankingConfig& b = prog.banking;
    w.inner_stride = static_cast<uint32_t>(prog.inner_stride);
    w.inner_end = prog.inner_end == kInnerEndSentinel ? 0 : static_cast<uint32_t>(prog.inner_end);
    w.outer_stride = static_cast<uint32_t>(prog.outer_stride);
    w.bank_first = static_cast<uint8_t>(b.base);
    w.bank_size = static_cast<uint8_t>(b.size);
    w.bank_skip = static_cast<uint8_t>(b.skip);
    w.bank_shift = static_cast<uint8_t>(b.endpoint_id_shift);
    w.bank_order = static_cast<uint8_t>(b.bank_order);
    w.restorable = (prog.inner_stride >> 32) == 0 && (prog.outer_stride >> 32) == 0 &&
                   (prog.inner_end == kInnerEndSentinel || (prog.inner_end != 0 && (prog.inner_end >> 32) == 0)) &&
                   ((b.base | b.size | b.skip | b.endpoint_id_shift) >> 8) == 0;
    with_slot(w.slot(), [&](auto g, auto side) {
        constexpr overlay::AddrGen G = decltype(g)::value;
        constexpr overlay::Side S = decltype(side)::value;
        ensure_generator_reset<G>();
        overlay::setup_banking_addrgen<G, S>(prog.banking);
        overlay::setup_inner_loop_addrgen<G, S>(prog.inner_stride, prog.inner_end, prog.inner_start);
        overlay::setup_outer_loop_addrgen<G, S>(prog.outer_stride, kInnerEndSentinel, prog.outer_start);
    });
}

// Returns the walker's current address and advances it by `amount` addresses (the hardware's skip; 1 = next).
inline uint64_t pop_walker(uint32_t slot, uint32_t amount) {
    return with_slot(slot, [&](auto g, auto side) {
        return overlay::pop_addrgen<decltype(g)::value, decltype(side)::value>(amount);
    });
}

// What one transfer cost the walker, for the test instrumentation (TransferStats in transfer_noc_addr.h).
struct PopInfo {
    bool seeked = false;    // the walk was (re)programmed in software
    bool skipped = false;   // the hardware skipped forward over untransferred indices
    bool restored = false;  // a parked walk was written back into an address generator
};

inline void save_slot(uint32_t slot, overlay::AddrgenPosition& out) {
    with_slot(slot, [&](auto g, auto side) {
        overlay::save_position_addrgen<decltype(g)::value, decltype(side)::value>(out);
    });
}

// Write walk w back into `slot`: its programming from the record, its position from `pos`.
inline void restore_slot(uint32_t slot, const Walker& w, const overlay::AddrgenPosition& pos) {
    const overlay::AddrgenProgram prog{
        .banking =
            {
                .endpoint_id_shift = w.bank_shift,
                .size = w.bank_size,
                .skip = w.bank_skip,
                .base = w.bank_first,
                .bank_order = static_cast<overlay::bank_order_e>(w.bank_order),
            },
        .inner_stride = w.inner_stride,
        .inner_end = w.inner_end == 0 ? kInnerEndSentinel : w.inner_end,
        .outer_stride = w.outer_stride,
        .outer_end = kInnerEndSentinel,
    };
    with_slot(slot, [&](auto g, auto side) {
        overlay::restore_addrgen<decltype(g)::value, decltype(side)::value>(prog, pos);
    });
}

// Forget a walk entirely: free its slot and parked registers.
inline void drop_walk(uint32_t wi) {
    Walker& w = walkers[wi];
    if (w.resident()) {
        slot_walk1[w.slot()] = 0;
    }
    if (w.parked1 != 0) {
        parked_owner1[w.parked1 - 1] = 0;
    }
    w = Walker{};
}

// Take the walk occupying `slot` off the hardware: park its position (or forget it if there is no room to, or if its
// programming didn't fit the record).
inline void spill_slot(uint32_t slot) {
    if (slot_walk1[slot] == 0) {
        return;
    }
    const uint32_t wi = slot_walk1[slot] - 1;
    if (!walkers[wi].restorable) {
        drop_walk(wi);
        return;
    }
    uint32_t entry = kNumParked;
    for (uint32_t i = 0; i < kNumParked; ++i) {
        if (parked_owner1[i] == 0) {
            entry = i;
            break;
        }
    }
    if (entry == kNumParked) {
        drop_walk(wi);
        return;
    }
    save_slot(slot, parked_pos[entry]);
    parked_owner1[entry] = static_cast<uint8_t>(wi + 1);
    Walker& w = walkers[wi];
    w.parked1 = static_cast<uint8_t>(entry + 1);
    w.slot1 = 0;
    slot_walk1[slot] = 0;
}

// Slot for a walk that needs one in direction `dir`: a free one, else the one whose walk was used least recently.
inline uint32_t take_slot(TransferDir dir) {
    const uint32_t first = first_slot(dir);
    for (uint32_t slot = first; slot < first + kSlotsPerDir; ++slot) {
        if (slot_walk1[slot] == 0) {
            return slot;
        }
    }
    const uint32_t pick =
        walkers[slot_walk1[first + 1] - 1].last_use < walkers[slot_walk1[first] - 1].last_use ? first + 1 : first;
    spill_slot(pick);
    return pick;
}

// Pops index `index` from walker `slot`, positioned at w.next <= index (< w.run_end). A forward gap is skipped in
// hardware -- one pop that discards index - w.next addresses -- instead of re-seeking. Once the kernel's stride has
// repeated (index - last == the previous delta), each pop also advances by that stride, so a steady strided walk (every
// Nth page, e.g. work split across DMs) costs one pop per transfer and discards nothing. A stride is only trusted after
// it repeats, so one jump (a block edge) doesn't make the next sequential access land behind the hardware.
inline uint64_t pop_index(Walker& w, uint32_t slot, uint32_t index, PopInfo& info) {
    info.skipped = index > w.next;
    if (info.skipped) {
        (void)pop_walker(slot, index - w.next);
    }
    const uint32_t delta = (w.has_prev && index > w.last) ? index - w.last : 0;
    const uint32_t step = (delta != 0 && delta == w.last_delta) ? delta : 1;
    w.last_delta = delta;
    w.last = index;
    w.has_prev = true;
    w.next = index + step;
    return w.hi_bits | pop_walker(slot, step);
}

// The seek routines below are the slow path, and they are deliberately not inlined. Every tensor binding is its own
// type (Paul's BindingId), so each accessor gets its own copy of these; inlined, the copies' locals (plans, programs)
// would all sit in the calling kernel's single stack frame at once -- a three-tensor sharded kernel used ~750 bytes of
// frame that way, on a DM core whose stack has ~1 KB. As calls, only one seek's locals exist at a time.
#define TT_TA_SEEK_NOINLINE __attribute__((noinline))

// Interleaved seek: the next pop is page `page_id`; the programming covers every later page.
template <bool IsDram>
TT_TA_SEEK_NOINLINE inline void seek_interleaved(Walker& w, uint32_t page_id) {
    constexpr uint32_t num_banks = IsDram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const noc_att::Window& window = interleaved_window<IsDram>();
    const WalkProgram prog{
        .banking =
            {
                .endpoint_id_shift = window.endpoint_shift,
                .size = num_banks,
                .skip = 1,
                .base = interleaved_bank_selector<IsDram>(0),
                .current = page_id % num_banks,  // relative to base (see configure_addrgen_src_interleaved)
                .bank_order = overlay::BANK_INNER,
            },
        .inner_stride = w.page_size,
        .inner_start = w.bank_base + static_cast<uint64_t>(page_id / num_banks) * w.page_size,
    };
    program_walker(w, prog);
    w.run_end = UINT32_MAX;
    w.hi_bits = window.compare;
}

// Number of page ids starting at `page_id` whose pages sit back to back (one aligned page apart) in the bank that
// holds `page_id`. Pages stay contiguous along the innermost dimension up to the end of the shard (or tensor) in that
// dimension; they continue into the next-outer dimension only while the shard spans the whole tensor in every inner
// dimension (otherwise the next page id is in another shard, or the shard's padding intervenes).
template <typename Accessor>
inline uint32_t contiguous_run(const Accessor& acc, uint32_t page_id) {
    const auto& ds = acc.dspec();
    uint32_t rest = page_id;
    uint32_t inner_volume = 1;  // pages per step of dimension i
    uint32_t inner_flat = 0;    // page_id's offset within the dimensions inside i
    for (int i = static_cast<int>(ds.rank()) - 1; i >= 0; --i) {
        const uint32_t extent = ds.tensor_shape()[i];
        const uint32_t shard = ds.shard_shape()[i];
        const uint32_t coord = rest % extent;
        rest /= extent;
        if (shard != extent || i == 0) {
            const uint32_t to_shard_end = shard - (coord % shard);
            const uint32_t to_tensor_end = extent - coord;
            return (to_shard_end < to_tensor_end ? to_shard_end : to_tensor_end) * inner_volume - inner_flat;
        }
        inner_flat += coord * inner_volume;
        inner_volume *= extent;
    }
    return 1;  // unreachable: i == 0 always returns
}

// Program a walk that starts at full NoC address `addr` and steps `stride` bytes within that one bank. Returns the
// window bits to OR onto each pop (the addrgen produces only selector << endpoint_shift | local).
TT_TA_SEEK_NOINLINE inline uint64_t seek_single_bank(Walker& w, uint64_t addr, uint64_t stride) {
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr));
    const uint32_t selector = window.selector(addr);
    const uint64_t local = window.local_address(addr);
    const WalkProgram prog{
        .banking =
            {
                .endpoint_id_shift = window.endpoint_shift,
                .size = 1,
                .skip = 1,
                .base = selector,
                .current = 0,
                .bank_order = overlay::BANK_INNER,
            },
        .inner_stride = stride,
        .inner_start = local,
    };
    program_walker(w, prog);
    return addr & ~(local | (static_cast<uint64_t>(selector) << window.endpoint_shift));
}

// Geometry of `page_id` in a sharded accessor, in pages. k is the innermost dimension whose shard doesn't span the
// tensor (every dimension inside k is whole in each shard, so a "segment" -- one step of dimension k across the shard
// with everything inside it -- is contiguous in the bank).
struct ShardedPosition {
    int k = -1;                 // -1: one shard spans the whole tensor
    uint32_t inner_volume = 1;  // pages per step of dimension k
    uint32_t inner_flat = 0;    // page_id's offset within the dimensions inside k
    uint32_t coord_k = 0;
    uint32_t shard_k = 0;
    uint32_t extent_k = 0;
    uint32_t outer_rest = 0;  // page_id / (extent_k * inner_volume): the index over dimensions outside k
};

template <typename Accessor>
inline ShardedPosition sharded_position(const Accessor& acc, uint32_t page_id) {
    const auto& ds = acc.dspec();
    ShardedPosition pos;
    uint32_t rest = page_id;
    for (int i = static_cast<int>(ds.rank()) - 1; i >= 0; --i) {
        const uint32_t extent = ds.tensor_shape()[i];
        const uint32_t shard = ds.shard_shape()[i];
        const uint32_t coord = rest % extent;
        rest /= extent;
        if (shard != extent) {
            pos.k = i;
            pos.coord_k = coord;
            pos.shard_k = shard;
            pos.extent_k = extent;
            pos.outer_rest = rest;
            return pos;
        }
        pos.inner_flat += coord * pos.inner_volume;
        pos.inner_volume *= extent;
    }
    return pos;
}

// Number of segment rows from the current one to the end of the current shard band: dimensions outside k, walked
// outward while each is whole in the shard (so its rows keep following in the bank), stopping at the first that isn't.
template <typename Accessor>
inline uint32_t rows_left_in_band(const Accessor& acc, const ShardedPosition& pos) {
    const auto& ds = acc.dspec();
    uint32_t rest = pos.outer_rest;
    uint32_t volume = 1;
    uint32_t flat = 0;
    for (int i = pos.k - 1; i >= 0; --i) {
        const uint32_t extent = ds.tensor_shape()[i];
        const uint32_t shard = ds.shard_shape()[i];
        const uint32_t coord = rest % extent;
        rest /= extent;
        if (shard != extent || i == 0) {
            const uint32_t to_shard_end = shard - (coord % shard);
            const uint32_t to_tensor_end = extent - coord;
            return (to_shard_end < to_tensor_end ? to_shard_end : to_tensor_end) * volume - flat;
        }
        flat += coord * volume;
        volume *= extent;
    }
    return 1;  // k == 0: the band is a single row
}

struct CrossBankPlan {
    WalkProgram prog;  // inner: within a segment; outer: bank-local row (k >= 1) or slot (k == 0) address
    uint64_t hi_bits;
    uint32_t run;  // pages this programming covers from page_id
};

// Whether the shards of page_id's band can be walked by one BANK_MIDDLE programming: >= 2 whole shards along k, on
// banks with ascending stride-1 selectors in one ATT window, all at the same bank-local address for the same row.
// k == 0 (e.g. HEIGHT, round-robin): the band is every shard, and a bank wrap moves to the next slot in each bank.
template <typename Accessor>
inline bool plan_cross_bank(const Accessor& acc, uint32_t page_id, uint8_t noc, CrossBankPlan& plan) {
    const ShardedPosition pos = sharded_position(acc, page_id);
    if (pos.k < 0 || pos.extent_k % pos.shard_k != 0) {
        return false;  // single shard, or a ragged last shard along k (its segment is shorter)
    }
    const uint32_t shards_k = pos.extent_k / pos.shard_k;
    const uint32_t num_banks = acc.dspec().num_banks();
    const uint32_t band_banks = pos.k == 0 ? (shards_k < num_banks ? shards_k : num_banks) : shards_k;
    if (band_banks < 2) {
        return false;
    }
    const uint32_t page_size = acc.get_aligned_page_size();
    const uint32_t segment = pos.shard_k * pos.inner_volume;  // pages
    const uint32_t row_start = page_id - pos.coord_k * pos.inner_volume - pos.inner_flat;

    // Shard j of the band starts its copy of this row at page row_start + j * segment (k >= 1); for k == 0 the
    // row is the tensor and shard j starts at page j * segment.
    const uint32_t band_origin = pos.k == 0 ? 0 : row_start;
    const uint64_t addr0 = acc.get_noc_addr(band_origin, 0, noc);
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr0));
    const uint32_t selector0 = window.selector(addr0);
    const uint64_t local0 = window.local_address(addr0);
    const uint64_t hi_bits = addr0 & ~(local0 | (static_cast<uint64_t>(selector0) << window.endpoint_shift));
    for (uint32_t j = 1; j < band_banks; ++j) {
        const uint64_t addr = acc.get_noc_addr(band_origin + j * segment, 0, noc);
        if (window.local_address(addr) != local0 || window.selector(addr) != selector0 + j ||
            (addr & ~(local0 | (static_cast<uint64_t>(selector0 + j) << window.endpoint_shift))) != hi_bits) {
            return false;
        }
    }
    const uint32_t shard_j = pos.coord_k / pos.shard_k;
    uint64_t outer_start = local0;
    if (pos.k == 0) {
        // Round-robin slots: shard band_banks lands back on the first bank, one shard further in. Verify once.
        if (shards_k > band_banks) {
            const uint64_t wrap = acc.get_noc_addr(band_banks * segment, 0, noc);
            if (window.selector(wrap) != selector0 ||
                window.local_address(wrap) != local0 + static_cast<uint64_t>(segment) * page_size) {
                return false;
            }
        }
        outer_start += static_cast<uint64_t>(shard_j / band_banks) * segment * page_size;
    }
    plan.prog.banking = overlay::BankingConfig{
        .endpoint_id_shift = window.endpoint_shift,
        .size = band_banks,
        .skip = 1,
        .base = selector0,
        .current = shard_j % band_banks,
        .bank_order = overlay::BANK_MIDDLE,
    };
    plan.prog.inner_stride = page_size;
    plan.prog.inner_start =
        static_cast<uint64_t>((pos.coord_k % pos.shard_k) * pos.inner_volume + pos.inner_flat) * page_size;
    plan.prog.inner_end = static_cast<uint64_t>(segment) * page_size;
    plan.prog.outer_start = outer_start;
    plan.prog.outer_stride = static_cast<uint64_t>(segment) * page_size;
    plan.hi_bits = hi_bits;
    const uint32_t row_pages = pos.extent_k * pos.inner_volume;
    plan.run = pos.k == 0 ? acc.dspec().tensor_volume() - page_id
                          : rows_left_in_band(acc, pos) * row_pages - (page_id - row_start);
    return true;
}

// Sharded seek. Prefer one BANK_MIDDLE programming across the shards of page_id's band (plan_cross_bank); otherwise
// resolve page_id in software and let the address generator walk its contiguous run in that one bank.
template <typename Accessor>
TT_TA_SEEK_NOINLINE inline void seek_sharded(const Accessor& acc, Walker& w, uint32_t page_id, uint8_t noc) {
    CrossBankPlan plan;
    if (plan_cross_bank(acc, page_id, noc, plan)) {
        program_walker(w, plan.prog);
        w.run_end = page_id + plan.run;
        w.hi_bits = plan.hi_bits;
        return;
    }
    w.hi_bits = seek_single_bank(w, acc.get_noc_addr(page_id, 0, noc), w.page_size);
    w.run_end = page_id + contiguous_run(acc, page_id);
}

// Find this accessor's walk in direction `dir` and make sure it occupies a hardware slot. Returns the walk.
//   - resident: nothing to do;
//   - parked: take a slot (parking that slot's walk) and write the walk back -- no software seek;
//   - new (or forgotten): take a free record, else discard the least recently used one; `claimed` tells the caller to
//     seek.
inline Walker& acquire_walker(
    uint32_t key,
    TransferDir dir,
    WalkKind kind,
    uint32_t bank_base,
    uint32_t page_size,
    bool& claimed,
    PopInfo& info) {
    uint32_t victim = 0;  // first free record, else the least recently used
    bool have_free = false;
    for (uint32_t wi = 0; wi < kNumWalks; ++wi) {
        Walker& w = walkers[wi];
        if (w.key == key && w.dir == dir && w.kind == kind && w.bank_base == bank_base && w.page_size == page_size) {
            claimed = false;
            if (!w.resident()) {
                // Whatever loses its slot to this walk is parked in another entry, or forgotten if the pool is full;
                // this walk's own entry stays put until its registers are written back (no stack copy of them).
                const uint32_t slot = take_slot(dir);
                restore_slot(slot, w, parked_pos[w.parked1 - 1]);
                parked_owner1[w.parked1 - 1] = 0;
                w.parked1 = 0;
                w.slot1 = static_cast<uint8_t>(slot + 1);
                slot_walk1[slot] = static_cast<uint8_t>(wi + 1);
                info.restored = true;
            }
            return w;
        }
        if (!have_free) {
            if (w.key == 0) {
                victim = wi;
                have_free = true;
            } else if (wi == 0 || w.last_use < walkers[victim].last_use) {
                victim = wi;
            }
        }
    }
    if (!have_free) {
        drop_walk(victim);
    }
    const uint32_t slot = take_slot(dir);
    Walker& w = walkers[victim];
    w = Walker{};
    w.key = key;
    w.dir = dir;
    w.kind = kind;
    w.bank_base = bank_base;
    w.page_size = page_size;
    w.slot1 = static_cast<uint8_t>(slot + 1);
    slot_walk1[slot] = static_cast<uint8_t>(victim + 1);
    claimed = true;
    return w;
}

// Hardware transfer address of `page_id` (+ offset) for an accessor with a hardware recipe. Returns false (and leaves
// `out` untouched) when this device's tables don't fit the recipe; the caller then uses the software address.
template <TransferDir Dir, typename Accessor>
inline bool try_transfer_noc_addr(
    const Accessor& acc, uint32_t page_id, uint32_t offset, uint8_t noc, uint64_t& out, PopInfo& info) {
    static_assert(has_hw_recipe<Accessor>);
    constexpr bool is_interleaved = Accessor::DSpec::is_interleaved;
    if constexpr (is_interleaved) {
        if constexpr (!interleaved_walkable<Accessor::DSpec::is_dram>) {
            return false;
        }
    }
    bool claimed = false;
    Walker& w = acquire_walker(
        walk_key(acc), Dir, WalkKind::Pages, acc.get_bank_base_address(), acc.get_aligned_page_size(), claimed, info);
    const uint32_t slot = w.slot();
    // Only a page behind the hardware, or past what the programming covers, needs software; a page ahead is skipped to.
    info.seeked = claimed || page_id < w.next || page_id >= w.run_end;
    if (info.seeked) {
        if constexpr (is_interleaved) {
            seek_interleaved<Accessor::DSpec::is_dram>(w, page_id);
        } else {
            seek_sharded(acc, w, page_id, noc);
        }
        w.next = page_id;
    }
    w.last_use = ++walker_clock;
    out = pop_index(w, slot, page_id, info) + offset;
    return true;
}

// ---- shard_pages(): pages of one shard, in storage order ----
//
// A shard's pages are consecutive in its bank (bank_page_offset = shard_in_bank * shard_volume + page_in_shard), so
// one single-bank walk covers the whole shard. Padding pages the iterator skips are skipped in hardware too.
template <TransferDir Dir, typename Accessor>
inline bool try_transfer_shard_page_noc_addr(
    const Accessor& acc,
    uint32_t shard_id,
    uint32_t page_in_shard,
    uint32_t offset,
    uint8_t noc,
    uint64_t& out,
    PopInfo& info) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    bool claimed = false;
    const uint32_t page_size = acc.get_aligned_page_size();
    Walker& w =
        acquire_walker(walk_key(acc), Dir, WalkKind::ShardPages, acc.get_bank_base_address(), page_size, claimed, info);
    const uint32_t slot = w.slot();
    info.seeked = claimed || w.shard != shard_id || page_in_shard < w.next || page_in_shard >= w.run_end;
    if (info.seeked) {
        w.hi_bits = seek_single_bank(w, acc.get_shard_noc_addr(shard_id, page_in_shard * page_size, noc), page_size);
        w.shard = shard_id;
        w.run_end = acc.dspec().shard_volume();
        w.next = page_in_shard;
        w.has_prev = false;  // a new shard's page indices restart
    }
    w.last_use = ++walker_clock;
    out = pop_index(w, slot, page_in_shard, info) + offset;
    return true;
}

// ---- ShardView: shard ids -> shard base addresses ----
//
// Round-robin puts shard s in bank s % N at slot s / N, so when the banks' selectors ascend by 1 at one bank-local
// address, shard bases are the interleaved pattern with a whole shard as the "page": BANK_INNER over the N banks, and
// the inner loop steps one shard volume each time the banks wrap. Otherwise (shard-contiguous distribution, or banks
// the recipe can't walk) each shard gets a single-address seek.
//
// A ShardView transfer usually moves part of a shard (an offset into it), so consecutive transfers often name the same
// shard; those reuse the base popped for it instead of popping again.
template <typename Accessor>
TT_TA_SEEK_NOINLINE inline bool plan_shard_bases(
    const Accessor& acc, uint32_t shard_id, uint8_t noc, WalkProgram& prog, uint64_t& hi_bits) {
    const auto& ds = acc.dspec();
    const uint32_t num_shards = ds.num_shards();
    const uint32_t num_banks = ds.num_banks();
    const uint32_t banks = num_shards < num_banks ? num_shards : num_banks;
    if (banks < 2) {
        return false;
    }
    const uint64_t shard_bytes = static_cast<uint64_t>(ds.shard_volume()) * acc.get_aligned_page_size();
    const uint64_t addr0 = acc.get_shard_noc_addr(0, 0, noc);
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr0));
    const uint32_t selector0 = window.selector(addr0);
    const uint64_t local0 = window.local_address(addr0);
    hi_bits = addr0 & ~(local0 | (static_cast<uint64_t>(selector0) << window.endpoint_shift));
    for (uint32_t j = 1; j < banks; ++j) {
        const uint64_t addr = acc.get_shard_noc_addr(j, 0, noc);
        if (window.local_address(addr) != local0 || window.selector(addr) != selector0 + j ||
            (addr & ~(local0 | (static_cast<uint64_t>(selector0 + j) << window.endpoint_shift))) != hi_bits) {
            return false;
        }
    }
    if (num_shards > banks) {
        // Shard `banks` must wrap back to the first bank, one shard further in (round-robin, not shard-contiguous).
        const uint64_t wrap = acc.get_shard_noc_addr(banks, 0, noc);
        if (window.selector(wrap) != selector0 || window.local_address(wrap) != local0 + shard_bytes) {
            return false;
        }
    }
    prog = WalkProgram{
        .banking =
            {
                .endpoint_id_shift = window.endpoint_shift,
                .size = banks,
                .skip = 1,
                .base = selector0,
                .current = shard_id % banks,
                .bank_order = overlay::BANK_INNER,
            },
        .inner_stride = shard_bytes,
        .inner_start = local0 + (shard_id / banks) * shard_bytes,
    };
    return true;
}

template <TransferDir Dir, typename Accessor>
inline bool try_transfer_shard_noc_addr(
    const Accessor& acc, uint32_t shard_id, uint32_t offset, uint8_t noc, uint64_t& out, PopInfo& info) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    bool claimed = false;
    Walker& w = acquire_walker(
        walk_key(acc),
        Dir,
        WalkKind::ShardBases,
        acc.get_bank_base_address(),
        acc.get_aligned_page_size(),
        claimed,
        info);
    const uint32_t slot = w.slot();
    w.last_use = ++walker_clock;
    if (!claimed && w.has_base && w.shard == shard_id) {
        out = w.last_addr + offset;
        return true;
    }
    info.seeked = claimed || shard_id < w.next || shard_id >= w.run_end;
    if (info.seeked) {
        WalkProgram prog;
        uint64_t hi_bits = 0;
        if (plan_shard_bases(acc, shard_id, noc, prog, hi_bits)) {
            program_walker(w, prog);
            w.hi_bits = hi_bits;
            w.run_end = acc.dspec().num_shards();
        } else {
            w.hi_bits = seek_single_bank(
                w,
                acc.get_shard_noc_addr(shard_id, 0, noc),
                static_cast<uint64_t>(acc.dspec().shard_volume()) * acc.get_aligned_page_size());
            w.run_end = shard_id + 1;
        }
        w.next = shard_id;
    }
    w.shard = shard_id;
    w.has_base = true;
    w.last_addr = pop_index(w, slot, shard_id, info);
    out = w.last_addr + offset;
    return true;
}

}  // namespace tt_addrgen
