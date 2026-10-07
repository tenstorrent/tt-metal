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

#include <cstddef>
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

// The bank shift (where the bank number goes in the address) lives in a MISC field shared by an address generator's two
// sides. Maps can give DRAM and worker windows different shifts (grendel_qsr1: 33 and 24), so the walks below only put
// two walks on one generator when their shifts agree (shift_fits).
namespace att_check {
constexpr const noc_att::Window& dram = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Dram);
constexpr const noc_att::Window& worker = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Worker);
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
// Walks behind tensor_accessor::transfer_noc_addr() (api/tensor/transfer_noc_addr.h)
// ============================================================================
//
// Design and costs: addrgen_walker.md (next to this header), section 6.
//
// The address generator produces the next address of a programmed pattern; the NoC APIs ask for the address of page p.
// A "walk" bridges the two for one tensor (one binding, one direction, one kind of index) while the tensor is accessed
// as a stream: the hardware serves the walk's pages in order, and anything else uses the software address.
//
// Sides: each address generator has a source and a destination side, so a DM core has four. A read walks on a source
// side and a write on a destination side, two sides per direction.
//
// More walks in one direction than sides (default): spill and reload. A walk that needs a side when both are taken
// spills the least recently used one -- its position is read back (3 register reads) and parked with its programming --
// and a parked walk that comes back is reloaded the same way (programming + position written back, no software seek).
// With two sides per direction, least recently used is "not the side used last", so one byte per direction tracks it.
// This is the run-time stand-in for spill and reload placed by the compiler (addrgen_compiler_interface.md), which
// knows which tensor comes back next; LRU can only guess, and round-robin over more tensors than sides reloads every
// transfer. TT_TA_ADDRGEN_NO_SPILL: first use is sticky instead -- a side's first walk keeps it and the extra walks use
// software.
//   side 0: addrgen_1 source  (reads)    side 2: addrgen_0 destination (writes)
//   side 1: addrgen_0 source  (reads)    side 3: addrgen_1 destination (writes)
// Sides 0 and 2 come first because they are the ones that can later push straight into the command buffers the NoC
// APIs read and write with (addrgen N feeds command buffer N; reads use command buffer 1, writes command buffer 0).
//
// State: one small SideState per side, thread-local at a fixed address, so each check is an independent load (no
// chain through records). The walk's identity is a compile-time key (binding id + kind), so finding the walk is one
// compare per side of the direction.
//
// Per request for index i on the walk's side (serve()):
//   i == next, inside the run          -> pop (the hit path)
//   next < i, inside the run, small gap -> skip forward in hardware, then pop; a gap that repeats becomes the stride
//   i == next at the end of the run     -> re-seek (sharded runs; still the same stream)
//   anything else (behind, large jump)  -> if the walk was streaming (its last request was a hit): re-seek now,
//                                          keeping the stride. Otherwise software for this
//                                          transfer, and the hardware is re-taken as soon as a request continues at the
//                                          walk's stride or repeats the last miss's gap
//
// The ATT window bits are part of every walk's programming (the outer loop starts at window.compare), so each pop is
// the complete NoC address.
//
// Recipes (plan_*: where page i lives, as a programming and how many indices it covers):
//   Interleaved: one programming walks the whole tensor. BANK_INNER cycles the bank every page and the inner loop
//     advances one page per bank wrap, as InterleavedAddrGen computes in software. Needs the device's interleaved
//     banks to be an ascending stride-1 ATT selector run with no per-bank offset (interleaved_walkable).
//   Sharded, cross-bank (plan_cross_bank): when the shards along the innermost split dimension sit on banks with
//     ascending stride-1 selectors at the same bank-local address, one BANK_MIDDLE programming walks a segment in one
//     shard, the same segment in the next shard's bank, ..., then steps the outer loop to the next row (or, for a
//     split outermost dimension such as HEIGHT / round-robin, to the next shard slot in each bank).
//   Sharded, fallback (any rank, any distribution, DRAM or L1): software resolves the page's address and the length of
//     the run of following page ids that are contiguous in that same bank (contiguous_run); the address generator
//     walks the run with a single-bank BankingConfig and a page-stride inner loop.
//   shard_pages() and ShardView have their own walk kinds (WalkKind) -- see their sections at the end.
//
// Must be included after api/tensor/tensor_accessor.h (it is, via transfer_noc_addr.h).

// Outer-loop end bound. The outer loop starts at the ATT window's compare bits, so its end must exceed every window's
// compare (grendel_qsr1 has one at bit 48, the inner sentinel).
inline constexpr uint64_t kOuterEndSentinel = uint64_t{1} << 62;
namespace att_check {
static_assert(
    (noc_att::is_no_window(dram) || dram.compare < kOuterEndSentinel) &&
        (noc_att::is_no_window(worker) || worker.compare < kOuterEndSentinel),
    "an ATT window's compare bits must stay below the outer loop's end, or the walk wraps and loses them");
}  // namespace att_check

// Metal 2.0 TensorAccessors (built from a binding token) have a hardware recipe: their binding id is a compile-time
// constant in their type, which is the walk's identity. Accessors without one use software.
template <typename T, typename = void>
struct bound_tensor_accessor : std::false_type {};
template <typename DSpecT>
struct bound_tensor_accessor<TensorAccessor<DSpecT>, void>
    : std::bool_constant<DSpecT::binding_id != tensor_accessor::NO_BINDING_ID> {};

template <typename Accessor>
inline constexpr bool has_hw_recipe = bound_tensor_accessor<Accessor>::value;

// What a walk steps through. One tensor can have a walk of each kind and direction; they're separate walks.
enum class WalkKind : uint32_t {
    Pages = 0,       // global page ids (TensorAccessor / PageView / wrapper / pages())
    ShardBases = 1,  // shard ids -> each shard's base address (ShardView)
    ShardPages = 2,  // pages of one shard in storage order (shard_pages())
};

// Walk identity: binding id and kind. Never 0 (0 = free side).
template <typename Accessor, WalkKind Kind>
inline constexpr uint32_t walk_key = 0x80000000u | (Accessor::DSpec::binding_id << 2) | static_cast<uint32_t>(Kind);

using tensor_accessor::TransferDir;

inline constexpr uint32_t kNumSides = 4;
constexpr uint32_t first_side(TransferDir dir) { return dir == TransferDir::Write ? 2u : 0u; }
template <uint32_t S>
inline constexpr overlay::AddrGen side_generator = (S == 0 || S == 3) ? overlay::ADDRGEN_1 : overlay::ADDRGEN_0;
template <uint32_t S>
inline constexpr overlay::Side side_of = S < 2 ? overlay::Side::Src : overlay::Side::Dest;
// The other side of the same address generator: 0 <-> 3 (addrgen_1), 1 <-> 2 (addrgen_0).
constexpr uint32_t sibling_side(uint32_t s) { return 3u - s; }

// A walk's bank shift: its memory's ATT window (an accessor's tensor is in DRAM or in L1, a compile-time property, and
// every recipe below resolves its addresses in that window).
template <typename Accessor>
inline constexpr uint32_t walk_shift = interleaved_window<Accessor::DSpec::is_dram>().endpoint_shift;

// A forward gap of up to this many indices is skipped in hardware (one discarding pop, about one cycle per address);
// a larger jump uses software and re-takes the hardware when the stream continues.
inline constexpr uint32_t kMaxSkip = 64;

// Short-run fallback: a seek costs hundreds of cycles of software (and more where the recipe takes several address
// computations to decide), so a walk whose seeks keep covering fewer than kMinRun indices is slower than software.
// After two such seeks in a row the walk uses software for good (stride 0 marks it). Row-recipe walks restart cheaply
// and never count. TT_TA_ADDRGEN_MIN_RUN overrides (0 turns it off).
#if defined(TT_TA_ADDRGEN_MIN_RUN)
inline constexpr uint32_t kMinRun = TT_TA_ADDRGEN_MIN_RUN;
#else
inline constexpr uint32_t kMinRun = 4;
#endif
inline constexpr uint8_t kShortSeeksToSoftware = 2;

// One walk: on a side (sides[]) or parked (parked[]). All zero = free: thread_local zero-initialized storage (.tbss)
// starts fresh for every kernel launch.
struct SideState {
    // ShardBases: the base popped for `shard` (ShardView transfers repeat it with offsets). Pages, row-recipe walks
    // (row_pages != 0): the outer-loop value at the current row's start, for a cheap restart at the next row.
    uint64_t last_addr;
    uint32_t owner;      // walk_key of the walk; 0 = free
    uint32_t next;       // index the hardware produces on the next pop
    uint32_t stride;     // indices the hardware advances per pop
    uint32_t run_end;    // first index the current programming does not cover
    uint32_t last;       // index of the previous request on this walk
    uint32_t last_gap;   // gap of the last request that wasn't a hit (a hit's gap is the stride; see serve())
    uint32_t miss_gap;   // gap of the last request served in software (a repeat re-takes the hardware)
    uint32_t shard;  // ShardPages: the shard being walked. ShardBases: the shard of last_addr. Pages, row walks: the
                     // row step
    // The walk's programming minus its start position, kept so a spilled walk can be reloaded: a spill reads back only
    // the position (overlay::save_position_addrgen). Narrowed to fit: bank-local strides and ends fit 32 bits and the
    // bank registers are 8 bits; a programming that doesn't fit is not restorable and is dropped instead of parked.
    // The outer loop's end is always kOuterEndSentinel.
    uint32_t inner_stride;
    uint32_t inner_end;  // 0 = kInnerEndSentinel
    uint32_t outer_stride;
    uint8_t bank_base;
    uint8_t bank_size;
    uint8_t bank_skip;
    uint8_t bank_shift;
    uint8_t bank_order;
    uint8_t has_base;    // ShardBases: last_addr is valid. Pages, row walks: the bank each row starts on
    uint8_t restorable;  // the programming fits the fields above
    uint8_t streaming;   // the last request was a hit: the next index of the current programming
    // Row restart (plan_row_round_robin): pages per row (0: not a row walk) and rows left in the band after the
    // current one; each row after the first is a 3-register position write instead of a seek.
    uint16_t row_pages;
    uint8_t rows_left;
    uint8_t short_seeks;  // consecutive seeks that covered fewer than kMinRun indices
};
static_assert(sizeof(SideState) == 64 && sizeof(SideState) % sizeof(uint64_t) == 0);
struct ParkedWalk {
    SideState walk;
    overlay::AddrgenPosition pos;
};
#if defined(TT_TA_ADDRGEN_NO_SPILL)
inline constexpr uint32_t kNumParked = 0;
#else
// One parked walk: three walks in a direction (two on sides, one parked). A walk spilled while the pool is full is
// forgotten and re-seeks when it comes back.
inline constexpr uint32_t kNumParked = 1;
#endif
static_assert(offsetof(SideState, owner) % 8 == 0 && offsetof(SideState, next) == offsetof(SideState, owner) + 4);
static_assert(offsetof(SideState, stride) % 8 == 0 && offsetof(SideState, run_end) == offsetof(SideState, stride) + 4);
inline thread_local SideState sides[kNumSides];

// owner/next and stride/run_end are adjacent, 8-byte aligned 32-bit pairs: read each pair with one load
// (low word = the first field).
inline __attribute__((always_inline)) uint64_t load_pair(const uint32_t& first) {
    using word = const uint64_t __attribute__((may_alias));
    return *reinterpret_cast<word*>(&first);
}
inline thread_local ParkedWalk parked[kNumParked > 0 ? kNumParked : 1];
inline thread_local uint8_t last_side[2];        // per direction: the side used last (the other one is the LRU)
inline thread_local uint8_t generator_ready[2];  // this kernel already reset generator g

// Whether a walk with bank shift `shift` can go on side s: the other side of its generator, which shares the shift
// field, is free or uses the same shift. (Programming s writes the shared field, so a mismatch would silently move the
// other walk's bank number to the wrong bits.)
inline bool shift_fits(uint32_t s, uint32_t shift) {
    const SideState& other = sides[sibling_side(s)];
    return other.owner == 0 || other.bank_shift == shift;
}

// A DM core's 8 KB of thread-local storage + stack already holds ~6.6 KB of DFB/CB interface state. At 640 bytes of
// walker state the sharded path overflowed the remaining stack and hung; ~380 bytes ran the whole suite.
inline constexpr uint32_t kWalkerTlsBytes =
    sizeof(sides) + sizeof(parked) + sizeof(last_side) + sizeof(generator_ready);
static_assert(kWalkerTlsBytes <= 400, "walk state shares the DM core's thread-local storage and stack; keep it small");

// One programming of a side. Loop semantics (checked by AddrgenLoopProbe): the inner loop counts from inner_start in
// steps of inner_stride and wraps to 0 at inner_end; each wrap carries into the next loop out (the bank loop under
// BANK_MIDDLE, the outer loop under BANK_INNER once the banks wrap). The outer loop keeps outer_start as its base and
// adds outer_stride per carry. Address = inner + outer + (bank << endpoint_shift); outer_start includes the ATT
// window's compare bits, so the address is the complete NoC address.
struct WalkProgram {
    overlay::BankingConfig banking;
    uint64_t inner_stride = 0;  // one page (page walks) or one shard (shard-base walks)
    uint64_t inner_start = 0;
    uint64_t inner_end = kInnerEndSentinel;  // default: never wraps
    uint64_t outer_start = 0;
    uint64_t outer_stride = 0;
};

// A seek's result: the programming that makes the next pop index i, and the first index it does not cover.
struct Seek {
    WalkProgram prog;
    uint32_t run_end = UINT32_MAX;
    // Row recipe: the run is one row; the next rows_left rows of the band are the same programming at
    // outer + k * row_step, starting on bank row_bank (see plan_row_round_robin).
    uint32_t row_pages = 0;
    uint32_t rows_left = 0;
    uint32_t row_step = 0;
    uint32_t row_bank = 0;
    uint64_t row_outer = 0;  // the outer-loop value at the current row's start
};

// A generator is reset once, the first time this kernel uses either of its sides, to clear whatever an earlier kernel
// left (face size and the like, which no walk programs). Never again: a reset clears both sides, and the other side may
// hold a live walk. Every register a walk depends on is written when it is programmed.
template <overlay::AddrGen G>
inline __attribute__((always_inline)) void ensure_generator_reset() {
    if (!generator_ready[G]) {
        overlay::reset_addrgen<G>();
        generator_ready[G] = 1;
    }
}

// Program side S and record the programming in its state (for a later reload). Every register is written each time, so
// nothing from a previous programming leaks in.
template <uint32_t S>
inline void program_side(const WalkProgram& prog) {
    constexpr overlay::AddrGen G = side_generator<S>;
    constexpr overlay::Side D = side_of<S>;
    SideState& s = sides[S];
    const overlay::BankingConfig& b = prog.banking;
    s.inner_stride = static_cast<uint32_t>(prog.inner_stride);
    s.inner_end = prog.inner_end == kInnerEndSentinel ? 0 : static_cast<uint32_t>(prog.inner_end);
    s.outer_stride = static_cast<uint32_t>(prog.outer_stride);
    s.bank_base = static_cast<uint8_t>(b.base);
    s.bank_size = static_cast<uint8_t>(b.size);
    s.bank_skip = static_cast<uint8_t>(b.skip);
    s.bank_shift = static_cast<uint8_t>(b.endpoint_id_shift);
    s.bank_order = static_cast<uint8_t>(b.bank_order);
    s.restorable = (prog.inner_stride >> 32) == 0 && (prog.outer_stride >> 32) == 0 &&
                   (prog.inner_end == kInnerEndSentinel || (prog.inner_end != 0 && (prog.inner_end >> 32) == 0)) &&
                   ((b.base | b.size | b.skip | b.endpoint_id_shift) >> 8) == 0;
    ensure_generator_reset<G>();
    overlay::setup_banking_addrgen<G, D>(prog.banking);
    overlay::setup_inner_loop_addrgen<G, D>(prog.inner_stride, prog.inner_end, prog.inner_start);
    overlay::setup_outer_loop_addrgen<G, D>(prog.outer_stride, kOuterEndSentinel, prog.outer_start);
}

// Returns the side's current address and advances it by `amount` addresses (1 = next).
template <uint32_t S>
inline __attribute__((always_inline)) uint64_t pop_side(uint32_t amount) {
    return overlay::pop_addrgen<side_generator<S>, side_of<S>>(amount);
}

// Push: the sides whose generator feeds the command buffer the NoC API issues that direction on can hand the address
// straight to it instead of returning it. Side 0 is generator 1's source side, which writes the read command buffer's
// (1) SRC_ADDR; side 2 is generator 0's destination side, which writes the write command buffer's (0) DEST_ADDR. Sides
// 1 and 3 feed the other direction's buffer, so they always pop.
template <uint32_t S>
inline constexpr bool is_push_side = S == 0 || S == 2;
static_assert(side_generator<0> == overlay::ADDRGEN_1 && side_of<0> == overlay::Side::Src);
static_assert(side_generator<2> == overlay::ADDRGEN_0 && side_of<2> == overlay::Side::Dest);

// Returned instead of an address when the walker pushed it into the command buffer: no NoC address has bit 63 set.
inline constexpr uint64_t kAddrInCmdBuf = ~0ull;

// Writes side S's current address into its command buffer and advances it by `amount` addresses. The push's skip
// count is in addition to its own advance of one (unlike pop's), hence amount - 1.
template <uint32_t S>
inline __attribute__((always_inline)) void push_side(uint32_t amount) {
    static_assert(is_push_side<S>);
    if constexpr (S == 0) {
        __builtin_riscv_ttrocc_addrgen_push_src_pop_x(overlay::ADDRGEN_1, amount - 1);
    } else {
        __builtin_riscv_ttrocc_addrgen_push_dest_pop_x(overlay::ADDRGEN_0, amount - 1);
    }
}

// Row restart: put side S at a new position of its current programming.
template <uint32_t S>
inline void set_side_position(uint32_t bank, uint64_t inner, uint64_t outer) {
    overlay::set_position_addrgen<side_generator<S>, side_of<S>>(
        overlay::AddrgenPosition{.inner_address = inner, .outer_address = outer, .bank_current = bank});
}

// Spill: read side S's position back (its programming is already in sides[S]).
template <uint32_t S>
inline void save_side(overlay::AddrgenPosition& pos) {
    overlay::save_position_addrgen<side_generator<S>, side_of<S>>(pos);
}

// Reload: write the programming in sides[S] and position `pos` into side S; the walk continues exactly where it
// stopped.
template <uint32_t S>
inline void restore_side(const overlay::AddrgenPosition& pos) {
    const SideState& s = sides[S];
    const overlay::AddrgenProgram prog{
        .banking =
            {
                .endpoint_id_shift = s.bank_shift,
                .size = s.bank_size,
                .skip = s.bank_skip,
                .base = s.bank_base,
                .bank_order = static_cast<overlay::bank_order_e>(s.bank_order),
            },
        .inner_stride = s.inner_stride,
        .inner_end = s.inner_end == 0 ? kInnerEndSentinel : s.inner_end,
        .outer_stride = s.outer_stride,
        .outer_end = kOuterEndSentinel,
    };
    ensure_generator_reset<side_generator<S>>();
    overlay::restore_addrgen<side_generator<S>, side_of<S>>(prog, pos);
}

// What one transfer cost, for the test instrumentation (TransferStats and the address trace in transfer_noc_addr.h).
// Empty in other builds, so nothing is written for it on the hit path.
#if defined(TT_TA_ADDRGEN_STATS) || defined(TT_TA_ADDRGEN_TRACE)
struct PopInfo {
    bool seeked = false;    // the walk was (re)programmed in software
    bool skipped = false;   // the hardware skipped forward over untransferred indices
    bool restored = false;  // a parked walk was reloaded into a side (and the side's walk spilled, if it had one)
    bool fallback = false;  // served in software: no side for the walk (NO_SPILL), or the request broke the stream
    bool pushed = false;    // the address went straight into the command buffer (kAddrInCmdBuf)
};
#define TT_TA_NOTE(info, field) ((info).field = true)
#else
struct PopInfo {};
#define TT_TA_NOTE(info, field) ((void)(info))
#endif

// The seek routines below are the slow path, and they are deliberately not inlined. Every tensor binding is its own
// type (Paul's BindingId), so each accessor gets its own copy of these; inlined, the copies' locals (plans, programs)
// would all sit in the calling kernel's single stack frame at once -- a three-tensor sharded kernel used ~750 bytes of
// frame that way, on a DM core whose stack has ~1 KB. As calls, only one seek's locals exist at a time.
#define TT_TA_SEEK_NOINLINE __attribute__((noinline))

// Interleaved: the next pop is page `page_id`; the programming covers every later page.
template <bool IsDram>
TT_TA_SEEK_NOINLINE inline Seek plan_interleaved(uint32_t bank_base, uint32_t page_size, uint32_t page_id) {
    constexpr uint32_t num_banks = IsDram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const noc_att::Window& window = interleaved_window<IsDram>();
    return Seek{
        .prog =
            {
                .banking =
                    {
                        .endpoint_id_shift = window.endpoint_shift,
                        .size = num_banks,
                        .skip = 1,
                        .base = interleaved_bank_selector<IsDram>(0),
                        .current = page_id % num_banks,  // relative to base (see configure_addrgen_src_interleaved)
                        .bank_order = overlay::BANK_INNER,
                    },
                .inner_stride = page_size,
                .inner_start = bank_base + static_cast<uint64_t>(page_id / num_banks) * page_size,
                .outer_start = window.compare,
            },
        .run_end = UINT32_MAX,
    };
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

// A walk that starts at full NoC address `addr` and steps `stride` bytes within that one bank.
TT_TA_SEEK_NOINLINE inline WalkProgram plan_single_bank(uint64_t addr, uint64_t stride) {
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr));
    return WalkProgram{
        .banking =
            {
                .endpoint_id_shift = window.endpoint_shift,
                .size = 1,
                .skip = 1,
                .base = window.selector(addr),
                .current = 0,
                .bank_order = overlay::BANK_INNER,
            },
        .inner_stride = stride,
        .inner_start = window.local_address(addr),
        .outer_start = window.compare,
    };
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

// Whether the shards of page_id's band can be walked by one BANK_MIDDLE programming: >= 2 whole shards along k, on
// banks with ascending stride-1 selectors in one ATT window, all at the same bank-local address for the same row.
// k == 0 (e.g. HEIGHT, round-robin): the band is every shard, and a bank wrap moves to the next slot in each bank.
template <typename Accessor>
inline bool plan_cross_bank(const Accessor& acc, uint32_t page_id, uint8_t noc, Seek& seek) {
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
    const uint64_t addr0 = ::tensor_accessor::detail::transfer_noc_addr(acc, band_origin, 0, noc);
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr0));
    const uint32_t selector0 = window.selector(addr0);
    const uint64_t local0 = window.local_address(addr0);
    for (uint32_t j = 1; j < band_banks; ++j) {
        const uint64_t addr = ::tensor_accessor::detail::transfer_noc_addr(acc, band_origin + j * segment, 0, noc);
        if (window.local_address(addr) != local0 || window.selector(addr) != selector0 + j || !window.matches(addr)) {
            return false;
        }
    }
    const uint32_t shard_j = pos.coord_k / pos.shard_k;
    uint64_t outer_start = local0;
    if (pos.k == 0) {
        // Round-robin slots: shard band_banks lands back on the first bank, one shard further in. Verify once.
        if (shards_k > band_banks) {
            const uint64_t wrap = ::tensor_accessor::detail::transfer_noc_addr(acc, band_banks * segment, 0, noc);
            if (window.selector(wrap) != selector0 ||
                window.local_address(wrap) != local0 + static_cast<uint64_t>(segment) * page_size) {
                return false;
            }
        }
        outer_start += static_cast<uint64_t>(shard_j / band_banks) * segment * page_size;
    }
    seek.prog.banking = overlay::BankingConfig{
        .endpoint_id_shift = window.endpoint_shift,
        .size = band_banks,
        .skip = 1,
        .base = selector0,
        .current = shard_j % band_banks,
        .bank_order = overlay::BANK_MIDDLE,
    };
    seek.prog.inner_stride = page_size;
    seek.prog.inner_start =
        static_cast<uint64_t>((pos.coord_k % pos.shard_k) * pos.inner_volume + pos.inner_flat) * page_size;
    seek.prog.inner_end = static_cast<uint64_t>(segment) * page_size;
    seek.prog.outer_start = window.compare + outer_start;
    seek.prog.outer_stride = static_cast<uint64_t>(segment) * page_size;
    const uint32_t row_pages = pos.extent_k * pos.inner_volume;
    seek.run_end = page_id + (pos.k == 0 ? acc.dspec().tensor_volume() - page_id
                                         : rows_left_in_band(acc, pos) * row_pages - (page_id - row_start));
    return true;
}

// One row of a band whose shards wrap around the banks (k >= 1, more shards along k than banks, round-robin: e.g. 3
// shards per row on 2 banks land on banks 0, 1, 0, the third one shard slot deeper). No single programming walks the
// band -- a row ends mid-cycle -- but one BANK_MIDDLE programming walks a row: a segment, the next bank, ..., and a
// bank wrap steps the outer loop to the next shard slot in each bank. Applies when every shard j of the row sits on
// bank (c0 + j) % B at slot (c0 + j) / B, B = the accessor's bank count, all banks one ascending stride-1 selector run.
template <typename Accessor>
inline bool plan_row_round_robin(const Accessor& acc, uint32_t page_id, uint8_t noc, Seek& seek) {
    const ShardedPosition pos = sharded_position(acc, page_id);
    if (pos.k < 1 || pos.extent_k % pos.shard_k != 0) {
        return false;
    }
    const uint32_t shards_k = pos.extent_k / pos.shard_k;
    const uint32_t num_banks = acc.dspec().num_banks();
    if (num_banks < 2 || shards_k <= num_banks) {
        return false;  // no wrap within the row (plan_cross_bank's case, or a single bank)
    }
    const uint32_t page_size = acc.get_aligned_page_size();
    const uint32_t segment = pos.shard_k * pos.inner_volume;                    // pages
    const uint64_t segment_bytes = static_cast<uint64_t>(segment) * page_size;  // one row of a shard
    // A bank holds its shards back to back, one whole (padded) shard per slot.
    const uint64_t slot_bytes = static_cast<uint64_t>(acc.dspec().shard_volume()) * page_size;
    const uint32_t row_start = page_id - pos.coord_k * pos.inner_volume - pos.inner_flat;
    const uint64_t addr0 = ::tensor_accessor::detail::transfer_noc_addr(acc, row_start, 0, noc);
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr0));
    const uint32_t selector0 = window.selector(addr0);
    const uint64_t local0 = window.local_address(addr0);
    // The first wrap (a shard whose selector isn't one more than its left neighbour's) gives the row's first bank:
    // shard w is bank 0, so the row starts at bank c0 = B - w.
    uint32_t w = 1;
    while (w < shards_k && window.selector(::tensor_accessor::detail::transfer_noc_addr(
                               acc, row_start + w * segment, 0, noc)) == selector0 + w) {
        ++w;
    }
    if (w > num_banks) {
        return false;
    }
    const uint32_t c0 = (num_banks - w) % num_banks;
    if (selector0 < c0) {
        return false;
    }
    const uint32_t base = selector0 - c0;
    for (uint32_t j = 1; j < shards_k; ++j) {
        const uint64_t addr = ::tensor_accessor::detail::transfer_noc_addr(acc, row_start + j * segment, 0, noc);
        const uint32_t b = c0 + j;
        if (!window.matches(addr) || window.selector(addr) != base + b % num_banks ||
            window.local_address(addr) != local0 + static_cast<uint64_t>(b / num_banks) * slot_bytes) {
            return false;
        }
    }
    const uint32_t shard_j = pos.coord_k / pos.shard_k;
    const uint32_t b = c0 + shard_j;
    seek.prog.banking = overlay::BankingConfig{
        .endpoint_id_shift = window.endpoint_shift,
        .size = num_banks,
        .skip = 1,
        .base = base,
        .current = b % num_banks,
        .bank_order = overlay::BANK_MIDDLE,
    };
    seek.prog.inner_stride = page_size;
    seek.prog.inner_start =
        static_cast<uint64_t>((pos.coord_k % pos.shard_k) * pos.inner_volume + pos.inner_flat) * page_size;
    seek.prog.inner_end = segment_bytes;
    seek.prog.outer_start = window.compare + local0 + static_cast<uint64_t>(b / num_banks) * slot_bytes;
    seek.prog.outer_stride = slot_bytes;
    seek.run_end = row_start + pos.extent_k * pos.inner_volume;  // the end of this row
    // The band's next rows are the same programming one shard row (segment) further into every slot.
    seek.row_pages = pos.extent_k * pos.inner_volume;
    seek.rows_left = rows_left_in_band(acc, pos) - 1;
    seek.row_step = static_cast<uint32_t>(segment_bytes);
    seek.row_bank = c0;
    seek.row_outer = window.compare + local0;
    return true;
}

// Sharded: prefer one BANK_MIDDLE programming across the shards of page_id's band (plan_cross_bank), or across one row
// when the band's shards wrap around the banks (plan_row_round_robin); otherwise resolve page_id in software and let
// the address generator walk its contiguous run in that one bank.
template <typename Accessor>
TT_TA_SEEK_NOINLINE inline Seek plan_sharded(const Accessor& acc, uint32_t page_id, uint8_t noc) {
    Seek seek;
    if (plan_cross_bank(acc, page_id, noc, seek) || plan_row_round_robin(acc, page_id, noc, seek)) {
        return seek;
    }
    seek.prog = plan_single_bank(
        ::tensor_accessor::detail::transfer_noc_addr(acc, page_id, 0, noc), acc.get_aligned_page_size());
    seek.run_end = page_id + contiguous_run(acc, page_id);
    return seek;
}

// Re-seek side S's walk so that index i is the next pop, advancing `stride` per pop, and pop it.
template <uint32_t S, typename Planner>
TT_TA_SEEK_NOINLINE inline uint64_t reseek(
    SideState& s, uint32_t i, uint32_t stride, const void* pc, uint32_t pa, uint8_t pn, PopInfo& info) {
    const Seek seek = Planner::plan(pc, pa, pn, i);
    program_side<S>(seek.prog);
    s.run_end = seek.run_end;
    if (seek.row_pages != 0 && seek.row_pages <= UINT16_MAX && (seek.row_step >> 31) == 0) {
        s.row_pages = static_cast<uint16_t>(seek.row_pages);
        s.rows_left = static_cast<uint8_t>(seek.rows_left < 255 ? seek.rows_left : 255);
        s.shard = seek.row_step;
        s.has_base = static_cast<uint8_t>(seek.row_bank);
        s.last_addr = seek.row_outer;
        s.short_seeks = 0;
    } else {
        s.row_pages = 0;
        if constexpr (Planner::kShortRunFallback && kMinRun != 0) {
            // Two seeks in a row that each cover fewer than kMinRun indices: this walk is cheaper in software.
            const bool short_run = seek.run_end - i < kMinRun;
            s.short_seeks = short_run ? static_cast<uint8_t>(s.short_seeks + 1) : 0;
        }
    }
    s.stride = stride;
    s.last_gap = stride;
    s.miss_gap = 0;
    s.last = i;
    s.next = i + stride;
    s.streaming = 0;  // a fresh programming: streaming again once a request continues it
    TT_TA_NOTE(info, seeked);
    return pop_side<S>(stride);
}

// Request index i from side S, whose walk this is: every case (see the policy at the top). True with the address when
// the hardware serves it; false when the request uses software. serve() handles the hit inline and calls this for the
// rest.
//
// The previous request: `last` is only stored off the hit path. While the walk is streaming (its last request was a
// hit), the previous request is next - stride.
template <uint32_t S, typename Planner>
TT_TA_SEEK_NOINLINE inline bool serve_slow(
    uint32_t i, const void* pc, uint32_t pa, uint8_t pn, uint64_t& out, PopInfo& info) {
    SideState& s = sides[S];
    if (s.short_seeks >= kShortSeeksToSoftware) {
        TT_TA_NOTE(info, fallback);  // short-run fallback: this walk uses software from now on
        return false;
    }
    const uint32_t last = s.streaming ? s.next - s.stride : s.last;
    if (i == s.next && i < s.run_end) {
        s.next = i + s.stride;
        s.streaming = 1;
        out = pop_side<S>(s.stride);
        return true;
    }
    if (i > s.next && i < s.run_end && i - s.next <= kMaxSkip) {
        // Ahead: skip forward in hardware. The step after this one is the request gap if the previous request had the
        // same gap (a steady stride, e.g. every Nth page), else 1 -- so a jump (a block edge, a new row) doesn't put
        // the hardware past the next sequential request. A skip doesn't count as streaming: random access skips by
        // luck.
        const uint32_t gap = i - last;
        const uint32_t prev_gap = s.streaming ? s.stride : s.last_gap;  // the previous request's gap
        const uint32_t step = gap == prev_gap ? gap : 1;
        (void)pop_side<S>(i - s.next);
        s.last_gap = gap;
        s.stride = step;
        s.last = i;
        s.next = i + step;
        s.streaming = 0;
        TT_TA_NOTE(info, skipped);
        out = pop_side<S>(step);
        return true;
    }
    if (i == s.next && i == s.run_end && s.row_pages != 0 && s.rows_left != 0 && s.stride == 1) {
        // Row walk at the end of its row, and the next row is in the same band: same programming, one shard row
        // further. Write the position only.
        --s.rows_left;
        s.last_addr += s.shard;
        set_side_position<S>(s.has_base, 0, s.last_addr);
        s.run_end = i + s.row_pages;
        s.next = i + 1;
        s.last = i;
        s.streaming = 0;
        TT_TA_NOTE(info, seeked);
        out = pop_side<S>(1);
        return true;
    }
    if (i == s.next) {
        out = reseek<S, Planner>(
            s, i, s.stride, pc, pa, pn, info);  // past the end of the run: the next run of the same stream
        return true;
    }
    // Behind, or a large jump. A walk that was streaming re-seeks right away, keeping its stride: the jump back to the
    // next block, column or pass of a regular pattern. A walk whose previous request also missed uses software for this
    // one -- random access -- and re-takes the hardware once the access continues at the walk's stride, or repeats the
    // gap of the previous software request.
    const uint32_t gap = i > last ? i - last : 0;
    const bool continues = gap != 0 && (gap == s.stride || gap == s.miss_gap);
    if (s.streaming || continues) {
        out = reseek<S, Planner>(s, i, continues ? gap : s.stride, pc, pa, pn, info);
        return true;
    }
    s.miss_gap = gap;
    s.last = i;
    TT_TA_NOTE(info, fallback);
    return false;
}

// The hit path: i is the walk's next index (`next`, already loaded with the owner) and inside the current programming.
// One paired load (stride, run end), one store (next), the pop; `streaming` is written only when it changes. Endless:
// the programming covers every later index (interleaved), so there is no run end to check. Anything else goes to
// serve_slow(). push: the caller can take the address in the command buffer, so a push side pushes it instead of
// popping it (out = kAddrInCmdBuf); only hits push, the slow path always returns the address.
template <uint32_t S, bool Endless, typename Planner>
inline __attribute__((always_inline)) bool serve(
    uint32_t i, uint32_t next, const void* pc, uint32_t pa, uint8_t pn, bool push, uint64_t& out, PopInfo& info) {
    if (__builtin_expect(i == next, 1)) {
        SideState& s = sides[S];
        const uint64_t stride_end = load_pair(s.stride);
        const uint32_t stride = static_cast<uint32_t>(stride_end);
        if (__builtin_expect(Endless || i < static_cast<uint32_t>(stride_end >> 32), 1)) {
            s.next = i + stride;
            if (__builtin_expect(!s.streaming, 0)) {
                s.streaming = 1;
            }
            if constexpr (is_push_side<S>) {
                if (push) {
                    push_side<S>(stride);
                    out = kAddrInCmdBuf;
                    TT_TA_NOTE(info, pushed);
                    return true;
                }
            }
            out = pop_side<S>(stride);
            return true;
        }
    }
    return serve_slow<S, Planner>(i, pc, pa, pn, out, info);
}

// Test instrumentation (TT_TA_ADDRGEN_STATS builds): cycles per step of a reload, see TransferStats.
#if defined(TT_TA_ADDRGEN_STATS)
#define TT_TA_RELOAD_MARK(t) \
    uint64_t t;              \
    asm volatile("rdcycle %0" : "=r"(t))
#define TT_TA_RELOAD_ACCUMULATE(t0, t1, t2, t3, t4)                                                        \
    do {                                                                                                   \
        ::tensor_accessor::detail::transfer_stats.reload_save_cycles += static_cast<uint32_t>(t1 - t0);    \
        ::tensor_accessor::detail::transfer_stats.reload_swap_cycles += static_cast<uint32_t>(t2 - t1);    \
        ::tensor_accessor::detail::transfer_stats.reload_restore_cycles += static_cast<uint32_t>(t3 - t2); \
        ::tensor_accessor::detail::transfer_stats.reload_serve_cycles += static_cast<uint32_t>(t4 - t3);   \
    } while (0)
#else
#define TT_TA_RELOAD_MARK(t) ((void)0)
#define TT_TA_RELOAD_ACCUMULATE(t0, t1, t2, t3, t4) ((void)0)
#endif

// Exchange two walk states in place, 8 bytes at a time (no temporary walk on the stack).
inline void swap_walks(SideState& a, SideState& b) {
    using word = uint64_t __attribute__((may_alias));
    word* wa = reinterpret_cast<word*>(&a);
    word* wb = reinterpret_cast<word*>(&b);
    for (uint32_t k = 0; k < sizeof(SideState) / sizeof(word); ++k) {
        const word t = wa[k];
        wa[k] = wb[k];
        wb[k] = t;
    }
}

template <uint32_t S, typename Planner>
TT_TA_SEEK_NOINLINE inline bool take_side(
    uint32_t key, uint32_t i, const void* pc, uint32_t pa, uint8_t pn, uint64_t& out, PopInfo& info) {
    SideState& s = sides[S];
    const bool spill = kNumParked > 0 && s.owner != 0 && s.restorable;
    for (uint32_t p = 0; p < kNumParked; ++p) {
        if (parked[p].walk.owner == key) {
            // Reload: the side's walk and the parked one trade places -- the side's position is read back, the parked
            // walk's programming and position written in, and the side's walk parked where the other one was.
            TT_TA_RELOAD_MARK(t0);
            overlay::AddrgenPosition spilled;
            if (spill) {
                save_side<S>(spilled);
            }
            TT_TA_RELOAD_MARK(t1);
            swap_walks(s, parked[p].walk);
            TT_TA_RELOAD_MARK(t2);
            restore_side<S>(parked[p].pos);
            TT_TA_RELOAD_MARK(t3);
            if (spill) {
                parked[p].pos = spilled;
            } else {
                parked[p].walk.owner = 0;  // the side's walk can't be restored (or there was none): forget it
            }
            TT_TA_NOTE(info, restored);
            const bool served = serve_slow<S, Planner>(i, pc, pa, pn, out, info);
            TT_TA_RELOAD_MARK(t4);
            TT_TA_RELOAD_ACCUMULATE(t0, t1, t2, t3, t4);
            return served;
        }
    }
    // Claim: park the side's walk in a free entry (or forget it when there is none), then seek to i.
    if (spill) {
        for (uint32_t p = 0; p < kNumParked; ++p) {
            if (parked[p].walk.owner == 0) {
                save_side<S>(parked[p].pos);
                parked[p].walk = s;
                break;
            }
        }
    }
    s = SideState{};
    s.owner = key;
    out = reseek<S, Planner>(s, i, 1, pc, pa, pn, info);
    return true;
}

// The walk isn't on a side of its direction: take a free side, else the least recently used one (spilling its walk;
// TT_TA_ADDRGEN_NO_SPILL: software instead), only where the other side of the generator fits its bank shift.
template <TransferDir Dir, typename Planner>
TT_TA_SEEK_NOINLINE inline bool walk_slow(
    uint32_t key,
    uint32_t shift,
    uint32_t i,
    const void* pc,
    uint32_t pa,
    uint8_t pn,
    uint64_t& out,
    PopInfo& info,
    uint32_t& side) {
    constexpr uint32_t A = first_side(Dir);
    constexpr uint32_t B = A + 1;
    constexpr uint32_t d = Dir == TransferDir::Write ? 1u : 0u;
    const bool fits_a = shift_fits(A, shift);
    const bool fits_b = shift_fits(B, shift);
    uint32_t target = kNumSides;
    if (sides[A].owner == 0 && fits_a) {
        target = A;
    } else if (sides[B].owner == 0 && fits_b) {
        target = B;
    } else if (kNumParked > 0) {
        // Spill: the least recently used side (the one not used last) if it fits, else the other one.
        const uint32_t lru = last_side[d] == A ? B : A;
        const uint32_t mru = lru == A ? B : A;
        const bool fits_lru = lru == A ? fits_a : fits_b;
        const bool fits_mru = mru == A ? fits_a : fits_b;
        target = fits_lru ? lru : (fits_mru ? mru : kNumSides);
    }
    if (target == kNumSides) {
        side = kNumSides;
        TT_TA_NOTE(info, fallback);
        return false;
    }
    side = target;
    last_side[d] = target;
    return target == A ? take_side<A, Planner>(key, i, pc, pa, pn, out, info)
                       : take_side<B, Planner>(key, i, pc, pa, pn, out, info);
}

// Record side S as the direction's most recently used (the LRU choice for spills); written only when it changes, and
// not at all without spills.
template <uint32_t d, uint32_t S>
inline __attribute__((always_inline)) void touch_side() {
    if constexpr (kNumParked > 0) {
        if (__builtin_expect(last_side[d] != S, 0)) {
            last_side[d] = S;
        }
    }
}

// Serve index i of the walk `key` (bank shift `shift`) in direction Dir. The hit path: one paired load (owner, next)
// per side of the direction, then serve(). Everything else -- taking a side, spilling, reloading -- is walk_slow().
// `Planner::plan(pc, pa, pn, i)` returns the Seek that makes i the next pop; it is only called off the hit path, so the
// hit path just forwards its plain arguments in registers. Endless: the walk's programming
// covers every later index (interleaved). Returns the side used (or kNumSides when the request uses software) through
// `side`. push: see serve(); the direction's first side is its push side.
template <TransferDir Dir, bool Endless, typename Planner>
inline __attribute__((always_inline)) bool walk(
    uint32_t key,
    uint32_t shift,
    uint32_t i,
    const void* pc,
    uint32_t pa,
    uint8_t pn,
    bool push,
    uint64_t& out,
    PopInfo& info,
    uint32_t& side) {
    constexpr uint32_t A = first_side(Dir);
    constexpr uint32_t B = A + 1;
    constexpr uint32_t d = Dir == TransferDir::Write ? 1u : 0u;
    static_assert(is_push_side<A> && !is_push_side<B>);
    const uint64_t a = load_pair(sides[A].owner);
    if (__builtin_expect(static_cast<uint32_t>(a) == key, 1)) {
        side = A;
        touch_side<d, A>();
        return serve<A, Endless, Planner>(i, static_cast<uint32_t>(a >> 32), pc, pa, pn, push, out, info);
    }
    const uint64_t b = load_pair(sides[B].owner);
    if (static_cast<uint32_t>(b) == key) {
        side = B;
        touch_side<d, B>();
        return serve<B, Endless, Planner>(i, static_cast<uint32_t>(b >> 32), pc, pa, pn, false, out, info);
    }
    return walk_slow<Dir, Planner>(key, shift, i, pc, pa, pn, out, info, side);
}

// The planners: stateless types whose plan() rebuilds the Seek from plain arguments -- the accessor (pc), one extra
// word (pa: the shard id for shard_pages()) and the NoC id (pn). The hit path only forwards these registers; plan()
// runs in the cold functions.
template <typename Accessor>
struct PagesPlanner {
    static constexpr bool kShortRunFallback = true;
    static Seek plan(const void* pc, uint32_t, uint8_t noc, uint32_t i) {
        const Accessor& acc = *static_cast<const Accessor*>(pc);
        if constexpr (Accessor::DSpec::is_interleaved) {
            return plan_interleaved<Accessor::DSpec::is_dram>(
                acc.get_bank_base_address(), acc.get_aligned_page_size(), i);
        } else {
            return plan_sharded(acc, i, noc);
        }
    }
};

template <typename Accessor>
struct ShardPagesPlanner {
    static constexpr bool kShortRunFallback = false;  // one seek per shard by design
    static Seek plan(const void* pc, uint32_t shard_id, uint8_t noc, uint32_t i) {
        const Accessor& acc = *static_cast<const Accessor*>(pc);
        const uint32_t page_size = acc.get_aligned_page_size();
        return Seek{
            .prog = plan_single_bank(
                ::tensor_accessor::detail::transfer_shard_noc_addr(acc, shard_id, i * page_size, noc), page_size),
            .run_end = static_cast<uint32_t>(acc.dspec().shard_volume()),
        };
    }
};

// Hardware transfer address of `page_id` (+ offset). Returns false (and leaves `out` untouched) when the request uses
// software: this device's tables don't fit the recipe, or the walk policy sends it there (PopInfo.fallback, in
// instrumented builds).
// MayPush: the caller issues on the direction's command buffer and accepts kAddrInCmdBuf (only for offset 0: a pushed
// address can't have the offset added).
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline bool try_transfer_noc_addr(
    const Accessor& acc, uint32_t page_id, uint32_t offset, uint8_t noc, uint64_t& out, PopInfo& info) {
    static_assert(has_hw_recipe<Accessor>);
    constexpr bool is_interleaved = Accessor::DSpec::is_interleaved;
    if constexpr (is_interleaved) {
        // False only on a device or ATT map whose interleaved banks aren't one ascending selector run (the host then
        // builds the kernel with TT_TA_ADDRGEN_INTERLEAVED_{DRAM,L1}_SW); with row-major L1 banks it holds for L1.
        if constexpr (!interleaved_walkable<Accessor::DSpec::is_dram>) {
            return false;
        }
    }
    uint32_t side;
    if (!walk<Dir, is_interleaved, PagesPlanner<Accessor>>(
            walk_key<Accessor, WalkKind::Pages>,
            walk_shift<Accessor>,
            page_id,
            &acc,
            0,
            noc,
            MayPush && offset == 0,
            out,
            info,
            side)) {
        return false;
    }
    out += offset;
    return true;
}

// ---- shard_pages(): pages of one shard, in storage order ----
//
// A shard's pages are consecutive in its bank (bank_page_offset = shard_in_bank * shard_volume + page_in_shard), so
// one single-bank walk covers the whole shard. A new shard starts a new run (re-seek). Padding pages the iterator
// skips are skipped in hardware.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline bool try_transfer_shard_page_noc_addr(
    const Accessor& acc,
    uint32_t shard_id,
    uint32_t page_in_shard,
    uint32_t offset,
    uint8_t noc,
    uint64_t& out,
    PopInfo& info) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    constexpr uint32_t key = walk_key<Accessor, WalkKind::ShardPages>;
    // Another shard: its page indices restart, so it's a new run. Make the request look like the run's start, on a side
    // or parked (a reloaded walk must not skip ahead inside the old shard's run).
    auto new_run = [&](SideState& w) {
        if (w.owner == key && w.shard != shard_id) {
            w.shard = shard_id;
            w.next = page_in_shard;
            w.run_end = 0;  // forces the re-seek path in serve()
        }
    };
    constexpr uint32_t A = first_side(Dir);
    new_run(sides[A]);
    new_run(sides[A + 1]);
    for (uint32_t p = 0; p < kNumParked; ++p) {
        new_run(parked[p].walk);
    }
    uint32_t side;
    if (!walk<Dir, false, ShardPagesPlanner<Accessor>>(
            key, walk_shift<Accessor>, page_in_shard, &acc, shard_id, noc, MayPush && offset == 0, out, info, side)) {
        return false;
    }
    sides[side].shard = shard_id;
    out += offset;
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
TT_TA_SEEK_NOINLINE inline Seek plan_shard_bases(const Accessor& acc, uint32_t shard_id, uint8_t noc) {
    const auto& ds = acc.dspec();
    const uint32_t num_shards = ds.num_shards();
    const uint32_t num_banks = ds.num_banks();
    const uint32_t banks = num_shards < num_banks ? num_shards : num_banks;
    const uint64_t shard_bytes = static_cast<uint64_t>(ds.shard_volume()) * acc.get_aligned_page_size();
    const Seek single{
        .prog =
            plan_single_bank(::tensor_accessor::detail::transfer_shard_noc_addr(acc, shard_id, 0, noc), shard_bytes),
        .run_end = shard_id + 1,
    };
    if (banks < 2) {
        return single;
    }
    const uint64_t addr0 = ::tensor_accessor::detail::transfer_shard_noc_addr(acc, 0, 0, noc);
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr0));
    const uint32_t selector0 = window.selector(addr0);
    const uint64_t local0 = window.local_address(addr0);
    for (uint32_t j = 1; j < banks; ++j) {
        const uint64_t addr = ::tensor_accessor::detail::transfer_shard_noc_addr(acc, j, 0, noc);
        if (window.local_address(addr) != local0 || window.selector(addr) != selector0 + j || !window.matches(addr)) {
            return single;
        }
    }
    if (num_shards > banks) {
        // Shard `banks` must wrap back to the first bank, one shard further in (round-robin, not shard-contiguous).
        const uint64_t wrap = ::tensor_accessor::detail::transfer_shard_noc_addr(acc, banks, 0, noc);
        if (window.selector(wrap) != selector0 || window.local_address(wrap) != local0 + shard_bytes) {
            return single;
        }
    }
    return Seek{
        .prog =
            {
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
                .outer_start = window.compare,
            },
        .run_end = num_shards,
    };
}

template <typename Accessor>
struct ShardBasesPlanner {
    static constexpr bool kShortRunFallback = false;  // a base serves every offset into its shard
    static Seek plan(const void* pc, uint32_t, uint8_t noc, uint32_t i) {
        return plan_shard_bases(*static_cast<const Accessor*>(pc), i, noc);
    }
};

template <TransferDir Dir, typename Accessor>
inline bool try_transfer_shard_noc_addr(
    const Accessor& acc, uint32_t shard_id, uint32_t offset, uint8_t noc, uint64_t& out, PopInfo& info) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    constexpr uint32_t key = walk_key<Accessor, WalkKind::ShardBases>;
    constexpr uint32_t A = first_side(Dir);
    for (uint32_t s = A; s < A + 2; ++s) {
        if (sides[s].owner == key && sides[s].has_base && sides[s].shard == shard_id) {
            out = sides[s].last_addr + offset;
            return true;
        }
    }
    uint32_t side;
    uint64_t base;
    if (!walk<Dir, false, ShardBasesPlanner<Accessor>>(
            key, walk_shift<Accessor>, shard_id, &acc, 0, noc, false, base, info, side)) {
        return false;
    }
    sides[side].shard = shard_id;
    sides[side].has_base = 1;
    sides[side].last_addr = base;
    out = base + offset;
    return true;
}

}  // namespace tt_addrgen
