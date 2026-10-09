// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tensor_accessor::generated_noc_addr() asks this header for the NoC address of a tensor used to issue a NoC
// transaction. Instead of computing it in software (TensorAccessor::get_noc_addr()), a sequence serves it. A sequence
// is an address-generator side programmed with the tensor's layout, which produces the next page's address on each pop.
// A request the hardware can't serve returns false, and the caller uses the software address.
//
// On a hit, a sequence on a push side writes the address straight into the command buffer the NoC API issues on,
// instead of returning it; the NoC API then issues without writing that address (api/tensor/noc_traits.h, PushIssue).
// Only the remote address is ever generated; the local address is written by the NoC API.
//
// Run-time work a compiler could do statically instead:
//   - Side assignment and spill/reload (serve_request(), serve_request_slow(), take_side()): each request finds its
//     sequence by comparing a compile-time key against the direction's two sides, and a third sequence spills the least
//     recently used one. With the kernel's tensors known, each sequence's side could be fixed at compile time and
//     spills placed where a loop switches tensors.
//   - Seeks (reseek(), plan_*): a loop whose page id is affine in its counter (page = a * i + b) could program the
//     sequence once before the loop, with a pop amount of a.
//   - The hit check (serve_on_side()): with both of the above, a transfer is a single push or pop.

#pragma once

#ifndef NOC_ATT_ENABLED
#error "addrgen_sequencer.h requires the ATT address backend (NOC_ATT_ENABLED)"
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "internal/tt-2xx/quasar/noc/att/att_config.h"
#include "internal/tt-2xx/quasar/noc_address_backend.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/addrgen_state.h"

namespace tt_addrgen {

// ============================================================================
// 1. ATT mapping
// ============================================================================
//
// An ATT NoC address is window compare bits | (selector << endpoint_shift) | bank-local offset. A sequence produces all
// three: the bank loop gives the selector, the inner loop the offset, and the outer loop starts at the window's
// compare bits, so every pop is the complete NoC address.

// Inner-loop end bound. X_END is an absolute address compared against the running inner address, so it must exceed
// any local address a sequence can reach (the caller bounds each sequence by page count, not by this).
inline constexpr uint64_t kInnerEndSentinel = uint64_t{1} << 48;

// Outer-loop end bound. The outer loop starts at the ATT window's compare bits, so its end must exceed every window's
// compare.
inline constexpr uint64_t kOuterEndSentinel = uint64_t{1} << 62;

template <bool IsDram>
constexpr const noc_att::Window& interleaved_window() {
    return noc_att::map_window(ACTIVE_ATT_MAP, IsDram ? noc_att::WindowClass::Dram : noc_att::WindowClass::Worker);
}

namespace att_check {
constexpr const noc_att::Window& dram = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Dram);
constexpr const noc_att::Window& worker = noc_att::map_window(ACTIVE_ATT_MAP, noc_att::WindowClass::Worker);
static_assert(
    (noc_att::is_no_window(dram) || dram.compare < kOuterEndSentinel) &&
        (noc_att::is_no_window(worker) || worker.compare < kOuterEndSentinel),
    "an ATT window's compare bits must stay below the outer loop's end, or the sequence wraps and loses them");
}  // namespace att_check

template <bool IsDram>
inline __attribute__((always_inline)) uint32_t interleaved_bank_selector(uint32_t bank) {
    return interleaved_window<IsDram>().selector(noc_address_backend::bank_address<IsDram>(bank, 0, noc_index));
}

// Only Metal 2.0 TensorAccessors (built from a binding token) have a hardware recipe. Their binding id is a
// compile-time constant in their type, which is the sequence's identity. Accessors without one use software.
template <typename T, typename = void>
struct bound_tensor_accessor : std::false_type {};
template <typename DSpecT>
struct bound_tensor_accessor<TensorAccessor<DSpecT>, void>
    : std::bool_constant<DSpecT::binding_id != tensor_accessor::NO_BINDING_ID> {};

template <typename Accessor>
inline constexpr bool has_hw_recipe = bound_tensor_accessor<Accessor>::value;

template <typename Accessor>
inline constexpr uint32_t sequence_shift = interleaved_window<Accessor::DSpec::is_dram>().endpoint_shift;

// ============================================================================
// 2. Sides
// ============================================================================
//
// A *side* is one of the four independent loop sets (bank, inner and outer loops) of a DM core's address generators:
// addrgen_0 and addrgen_1 each have a source side and a destination side. A side holds one tensor's sequence at a time.
//
// Which sides a request may use is fixed by the hardware wiring:
//   - The NoC APIs issue reads on command buffer 1 and writes on command buffer 0, and address generator N can push
//     only into command buffer N: a read's remote address goes to the command buffer's SRC_ADDR, a write's to
//     DEST_ADDR.
//   - So the side that can push a read's address is addrgen_1's source side, and the side that can push a write's is
//     addrgen_0's destination side. The other two sides can produce addresses too, but only by popping them back to the
//     RISC-V: their push would land in the wrong command buffer.
//   - Reads use the two source sides and writes the two destination sides, each direction trying its push side first.
//   - The two sides of one generator share its MISC register, so they must agree on the bank shift (shift_fits).

using tensor_accessor::TransferDir;

// Numbered so each direction's two sides are adjacent, push side first.
//   kReadPushSide   addrgen_1 source:       reads; pushes into command buffer 1's SRC_ADDR
//   kReadPopSide    addrgen_0 source:       reads; pops only
//   kWritePushSide  addrgen_0 destination:  writes; pushes into command buffer 0's DEST_ADDR
//   kWritePopSide   addrgen_1 destination:  writes; pops only
inline constexpr uint32_t kReadPushSide = 0;
inline constexpr uint32_t kReadPopSide = 1;
inline constexpr uint32_t kWritePushSide = 2;
inline constexpr uint32_t kWritePopSide = 3;
inline constexpr uint32_t kNumSides = 4;
constexpr uint32_t push_side_of(TransferDir dir) { return dir == TransferDir::Write ? kWritePushSide : kReadPushSide; }
constexpr uint32_t pop_side_of(TransferDir dir) { return dir == TransferDir::Write ? kWritePopSide : kReadPopSide; }
template <uint32_t S>
inline constexpr overlay::AddrGen side_generator =
    (S == kReadPushSide || S == kWritePopSide) ? overlay::ADDRGEN_1 : overlay::ADDRGEN_0;
template <uint32_t S>
inline constexpr overlay::Side side_of =
    (S == kReadPushSide || S == kReadPopSide) ? overlay::Side::Src : overlay::Side::Dest;
// The other side of the same address generator.
constexpr uint32_t sibling_side(uint32_t s) {
    return s == kReadPushSide   ? kWritePopSide
           : s == kWritePopSide ? kReadPushSide
           : s == kReadPopSide  ? kWritePushSide
                                : kReadPopSide;
}

template <uint32_t S>
inline constexpr bool is_push_side = S == kReadPushSide || S == kWritePushSide;
static_assert(side_generator<kReadPushSide> == overlay::ADDRGEN_1 && side_of<kReadPushSide> == overlay::Side::Src);
static_assert(side_generator<kWritePushSide> == overlay::ADDRGEN_0 && side_of<kWritePushSide> == overlay::Side::Dest);

// Returned instead of an address when the sequencer pushed it into the command buffer: no NoC address has bit 63 set.
inline constexpr uint64_t kAddrPushed = ~0ull;

// ============================================================================
// 3. Sequence state
// ============================================================================

// What a sequence steps through. One tensor can have a sequence of each kind and direction; they're separate sequences.
enum class SequenceKind : uint32_t {
    Pages = 0,       // global page ids (TensorAccessor / PageView / wrapper / pages())
    ShardBases = 1,  // shard ids -> each shard's base address (ShardView)
    ShardPages = 2,  // shard * shard_volume + page_in_shard (shard_pages())
};

// Sequence identity: binding id and kind. Never 0 (0 = free side).
template <typename Accessor, SequenceKind Kind>
inline constexpr uint32_t sequence_key = 0x80000000u | (Accessor::DSpec::binding_id << 2) | static_cast<uint32_t>(Kind);

// A forward gap of up to this many indices is skipped in hardware (one discarding pop, about one cycle per address);
// a larger jump uses software and re-takes the hardware when the sequence continues.
inline constexpr uint32_t kMaxSkip = 64;

// Short-run fallback: a seek costs hundreds of cycles of software (and more where the recipe takes several address
// computations to decide), so a sequence whose seeks keep covering fewer than kMinRun indices is slower than software.
// After kShortSeeksToSoftware such seeks in a row the sequence uses software for good. Row-recipe sequences restart
// cheaply and never count.
// This is a run-time guess at what the compiler could decide statically (which loops use the hardware path at all);
// once the compiler does that, the fallback is likely unnecessary and can be removed.
inline constexpr uint32_t kMinRun = 4;
inline constexpr uint8_t kShortSeeksToSoftware = 2;

// One sequence: on a side (sides[]) or parked (parked[]). All zero = free
struct SideState {
    union {
        uint64_t base_addr;  // ShardBases: the base popped for base_shard (ShardView transfers reuse it with offsets)
        uint64_t row_outer;  // row sequences (row_pages != 0): the outer-loop value at the current row's start
    };
    uint32_t owner;     // sequence_key of the sequence; 0 = free
    uint32_t next;      // index the hardware produces on the next pop
    uint32_t stride;    // indices the hardware advances per pop
    uint32_t run_end;   // first index the current programming does not cover
    uint32_t last;      // index of the previous request on this sequence
    uint32_t last_gap;  // gap of the last request that wasn't a hit (a hit's gap is the stride; see serve_on_side())
    uint32_t miss_gap;  // gap of the last request served in software (a repeat re-takes the hardware)
    union {
        uint32_t base_shard;  // ShardBases: the shard of base_addr
        uint32_t row_step;    // row sequences: bytes the outer loop moves from one row of the band to the next
    };
    // The sequence's programming minus its start position, kept so a spilled sequence can be reloaded: a spill reads
    // back only the position (overlay::save_position_addrgen). Narrowed to fit: bank-local strides and ends fit 32 bits
    // and the bank registers are 8 bits; a programming that doesn't fit is not restorable and is dropped instead of
    // parked. The outer loop's end is always kOuterEndSentinel.
    uint32_t inner_stride;
    uint32_t inner_end;  // 0 = kInnerEndSentinel
    uint32_t outer_stride;
    uint8_t bank_base;
    uint8_t bank_size;
    uint8_t bank_skip;
    uint8_t bank_shift;
    uint8_t bank_order;
    union {
        uint8_t has_base;  // ShardBases: base_addr is valid
        uint8_t row_bank;  // row sequences: the bank each row starts on
    };
    uint8_t restorable;  // the programming fits the fields above
    uint8_t last_hit;    // the last request was a hit: the next index of the current programming
    // Row restart (plan_row_round_robin): pages per row (0: not a row sequence) and rows left in the band after the
    // current one; each row after the first is a 3-register position write instead of a seek.
    uint16_t row_pages;
    uint8_t rows_left;
    uint8_t short_seeks;  // consecutive seeks that covered fewer than kMinRun indices
};
static_assert(sizeof(SideState) == 64 && sizeof(SideState) % sizeof(uint64_t) == 0);
static_assert(offsetof(SideState, owner) % 8 == 0 && offsetof(SideState, next) == offsetof(SideState, owner) + 4);
static_assert(offsetof(SideState, stride) % 8 == 0 && offsetof(SideState, run_end) == offsetof(SideState, stride) + 4);

struct ParkedSequence {
    SideState sequence;
    overlay::AddrgenPosition pos;
};
// Kept here as a toggle because spilling and reloading is expensive and needs to be measured by microbenchmarks.
#if defined(TT_TA_ADDRGEN_NO_SPILL)
inline constexpr uint32_t kNumParked = 0;
#else
// One parked sequence: three sequences in a direction (two on sides, one parked). A sequence spilled while the pool is
// full is forgotten and re-seeks when it comes back.
inline constexpr uint32_t kNumParked = 1;
#endif

inline thread_local SideState sides[kNumSides];
inline thread_local ParkedSequence parked[kNumParked > 0 ? kNumParked : 1];
inline thread_local uint8_t last_side[2];        // per direction: the side used last (the other one is the LRU)
inline thread_local uint8_t generator_ready[2];  // this kernel already reset generator g

inline constexpr uint32_t kSequencerTlsBytes =
    sizeof(sides) + sizeof(parked) + sizeof(last_side) + sizeof(generator_ready);
static_assert(
    kSequencerTlsBytes <= 400, "sequence state shares the DM core's thread-local storage and stack; keep it small");

// owner/next and stride/run_end are adjacent, 8-byte aligned 32-bit pairs: read each pair with one load
// (low word = the first field).
inline __attribute__((always_inline)) uint64_t load_pair(const uint32_t& first) {
    using word = const uint64_t __attribute__((may_alias));
    return *reinterpret_cast<word*>(&first);
}

// Whether a sequence with bank shift `shift` can go on side s: the other side of its generator, which shares the shift
// field, is free or uses the same shift. (Programming s writes the shared field, so a mismatch would silently move the
// other sequence's bank number to the wrong bits.)
inline bool shift_fits(uint32_t s, uint32_t shift) {
    const SideState& other = sides[sibling_side(s)];
    return other.owner == 0 || other.bank_shift == shift;
}

// ============================================================================
// 4. Side operations
// ============================================================================

// One programming of a side. Loop semantics: the inner loop counts from inner_start in
// steps of inner_stride and wraps to 0 at inner_end; each wrap carries into the next loop out (the bank loop under
// BANK_MIDDLE, the outer loop under BANK_INNER once the banks wrap). The outer loop keeps outer_start as its base and
// adds outer_stride per carry. Address = inner + outer + (bank << endpoint_shift); outer_start includes the ATT
// window's compare bits, so the address is the complete NoC address.
struct SideProgram {
    overlay::BankingConfig banking;
    uint64_t inner_stride = 0;  // one page (page sequences) or one shard (shard-base sequences)
    uint64_t inner_start = 0;
    uint64_t inner_end = kInnerEndSentinel;  // default: never wraps
    uint64_t outer_start = 0;
    uint64_t outer_stride = 0;
};

// A seek's result: the programming that makes the next pop index i, and the first index it does not cover.
struct Seek {
    SideProgram prog;
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
// left.
template <overlay::AddrGen G>
inline __attribute__((always_inline)) void ensure_generator_reset() {
    if (!generator_ready[G]) {
        overlay::reset_addrgen<G>();
        generator_ready[G] = 1;
    }
}

// Program side S and record the programming in its state (for a later reload). Every register is written each time.
template <uint32_t S>
inline void program_side(const SideProgram& prog) {
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

// Push from side S (kReadPushSide or kWritePushSide): the address generator writes its current address into the
// command buffer it is wired to (command buffer 1's SRC_ADDR for reads, command buffer 0's DEST_ADDR for writes) and
// advances by `amount` addresses. The push's skip count is in addition to its own advance of one (unlike pop's), hence
// amount - 1.
template <uint32_t S>
inline __attribute__((always_inline)) void push_side(uint32_t amount) {
    static_assert(is_push_side<S>);
    if constexpr (S == kReadPushSide) {
        __builtin_riscv_ttrocc_addrgen_push_src_pop_x(overlay::ADDRGEN_1, amount - 1);
    } else {
        __builtin_riscv_ttrocc_addrgen_push_dest_pop_x(overlay::ADDRGEN_0, amount - 1);
    }
}

// Produce side S's current address and advance it by `amount`: pushed into the command buffer (returns kAddrPushed)
// when the caller allows it and S is its direction's push side, else popped back to the RISC-V.
template <uint32_t S>
inline __attribute__((always_inline)) uint64_t advance_side(uint32_t amount, bool push) {
    if constexpr (is_push_side<S>) {
        if (push) {
            push_side<S>(amount);
            return kAddrPushed;
        }
    }
    return pop_side<S>(amount);
}

// Row restart: put side S at a new position of its current programming.
template <uint32_t S>
inline void set_side_position(uint32_t bank, uint64_t inner, uint64_t outer) {
    overlay::set_position_addrgen<side_generator<S>, side_of<S>>(
        overlay::AddrgenPosition{.inner_address = inner, .outer_address = outer, .bank_current = bank});
}

// Shard restart: point side S's single-bank sequence at another bank.
template <uint32_t S>
inline void set_side_bank(uint32_t bank_base) {
    overlay::set_bank_base_addrgen<side_generator<S>, side_of<S>>(bank_base);
}

// Spill: read side S's position back (its programming is already in sides[S]).
template <uint32_t S>
inline void save_side(overlay::AddrgenPosition& pos) {
    overlay::save_position_addrgen<side_generator<S>, side_of<S>>(pos);
}

// Reload: write the programming in sides[S] and position `pos` into side S; the sequence continues exactly where it
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

// ============================================================================
// 5. Recipes
// ============================================================================
//
// A recipe (plan_*) turns "the next pop is index i" into a side programming (SideProgram) and the first index that
// programming no longer covers (Seek::run_end):
//   Interleaved (plan_interleaved): one programming walks the whole tensor. BANK_INNER cycles the bank every page and
//     the inner loop advances one page per bank wrap, as InterleavedAddrGen computes in software. Assumes the device's
//     interleaved banks are an ascending stride-1 ATT selector run with no per-bank offset (bank i's selector = bank
//     0's + i), which the allocator and ATT map guarantee.
//   Sharded, cross-bank (plan_cross_bank): when the shards along the innermost split dimension sit on banks with
//     ascending stride-1 selectors at the same bank-local address, one BANK_MIDDLE programming walks a segment in one
//     shard, the same segment in the next shard's bank, ..., then steps the outer loop to the next row (or, for a
//     split outermost dimension such as HEIGHT / round-robin, to the next shard slot in each bank).
//   Sharded, one row (plan_row_round_robin): when a row's shards wrap around the banks, one BANK_MIDDLE programming
//     walks a row, and the band's later rows are the same programming one shard row further (a row restart).
//   Sharded, single bank (plan_sharded's fallback; any rank, any distribution, DRAM or L1): software resolves the
//     page's address and the length of the run of following page ids that are contiguous in that same bank
//     (contiguous_run); the address generator walks the run with a single-bank programming.
//   shard_pages() and ShardView: their own sequence kinds, with their entry points at the end of this file.
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
                        .current = page_id % num_banks,  // relative to base: bank = base + current
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

// A sequence that starts at full NoC address `addr` and steps `stride` bytes within that one bank.
TT_TA_SEEK_NOINLINE inline SideProgram plan_single_bank(uint64_t addr, uint64_t stride) {
    const noc_att::Window& window =
        noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr));
    return SideProgram{
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

// ============================================================================
// 6. Policy
// ============================================================================
//
// Per request for index i on the sequence's side (serve_on_side(), then serve_on_side_slow()):
//   i == next, inside the run            -> pop, or push on a push side (the hit path)
//   i is a shard's first page            -> shard_pages() sequences: move the sequence to that shard's bank (3 register
//                                           writes, no seek)
//   next < i, inside the run, small gap  -> skip forward in hardware, then pop; a gap that repeats becomes the stride
//   i == next at the end of a row        -> row sequences with rows left in the band: move to the next row (3 register
//                                           writes, no seek)
//   i == next at the end of the run      -> re-seek (the next run of the same sequence)
//   anything else (behind, large jump)   -> if the sequence's last request was a hit: re-seek now,
//                                           keeping the stride. Otherwise software for this transfer, and the hardware
//                                           is re-taken as soon as a request continues at the sequence's stride or
//                                           repeats the last miss's gap
// A sequence whose seeks keep covering fewer than kMinRun indices goes to software for good (short-run fallback).
//
// Side assignment (serve_request(), serve_request_slow(), take_side()): a request first compares its sequence's key
// against the two sides of its direction. A sequence on neither takes a free side, else the least recently used one:
// that side's sequence is spilled - its position read back (3 register reads) and parked with its programming. A
// parked sequence that comes back is reloaded the same way (programming + position written back, no seek). With two
// sides per direction, least recently used is "not the side used last", so one byte per direction tracks it.
// Round-robin over more tensors than sides reloads on every transfer. TT_TA_ADDRGEN_NO_SPILL: first use is sticky
// instead - a side's first sequence keeps it and the extra sequences use software.

// Re-seek side S's sequence so that index i is the next address, advancing `stride` per request, and produce it
// (pushed or popped, see advance_side).
template <uint32_t S, typename Recipe>
TT_TA_SEEK_NOINLINE inline uint64_t reseek(
    SideState& s, uint32_t i, uint32_t stride, const void* acc_ptr, uint32_t arg, uint8_t noc, bool push) {
    const Seek seek = Recipe::plan(acc_ptr, arg, noc, i);
    program_side<S>(seek.prog);
    s.run_end = seek.run_end;
    if (seek.row_pages != 0 && seek.row_pages <= UINT16_MAX && (seek.row_step >> 31) == 0) {
        s.row_pages = static_cast<uint16_t>(seek.row_pages);
        s.rows_left = static_cast<uint8_t>(seek.rows_left < 255 ? seek.rows_left : 255);
        s.row_step = seek.row_step;
        s.row_bank = static_cast<uint8_t>(seek.row_bank);
        s.row_outer = seek.row_outer;
        s.short_seeks = 0;
    } else {
        s.row_pages = 0;
        if constexpr (Recipe::kShortRunFallback) {
            // Two seeks in a row that each cover fewer than kMinRun indices: this sequence is cheaper in software.
            const bool short_run = seek.run_end - i < kMinRun;
            s.short_seeks = short_run ? static_cast<uint8_t>(s.short_seeks + 1) : 0;
        }
    }
    s.stride = stride;
    s.last_gap = stride;
    s.miss_gap = 0;
    s.last = i;
    s.next = i + stride;
    s.last_hit = 0;  // a fresh programming: a hit again once a request continues it
    return advance_side<S>(stride, push);
}

// Request index i from side S, whose sequence this is: every case (see the policy at the top). True with the address
// when the hardware serves it; false when the request uses software. serve_on_side() handles the hit inline and calls
// this for the rest.
//
// The previous request: `last` is only stored off the hit path. While the sequence's last request was a hit, the
// previous request is next - stride.
template <uint32_t S, typename Recipe>
TT_TA_SEEK_NOINLINE inline bool serve_on_side_slow(
    uint32_t i, const void* acc_ptr, uint32_t arg, uint8_t noc, bool push, uint64_t& out) {
    SideState& s = sides[S];
    if (s.short_seeks >= kShortSeeksToSoftware) {
        return false;  // short-run fallback: this sequence uses software from now on
    }
    const uint32_t last = s.last_hit ? s.next - s.stride : s.last;
    if (i == s.next && i < s.run_end) {
        s.next = i + s.stride;
        s.last_hit = 1;
        out = advance_side<S>(s.stride, push);
        return true;
    }
    if constexpr (Recipe::kShardRestart) {
        if (i % arg == 0) {
            // The first page of a shard (the next shard in order, a thread's next shard, or any other): same
            // single-bank programming, another bank and start. Software computes that one address; three register
            // writes move the sequence there instead of a seek.
            const uint64_t addr = Recipe::page_addr(acc_ptr, arg, noc, i);
            const noc_att::Window& window =
                noc_att::map_window(ACTIVE_ATT_MAP, noc_att::matching_window_class(ACTIVE_ATT_MAP, addr));
            const uint32_t bank = window.selector(addr);
            set_side_bank<S>(bank);
            set_side_position<S>(0, window.local_address(addr), window.compare);
            s.bank_base = static_cast<uint8_t>(bank);
            s.restorable = s.restorable && (bank >> 8) == 0;
            s.run_end = i + arg;
            s.stride = 1;
            s.last = i;
            s.next = i + 1;
            s.last_hit = 0;
            out = advance_side<S>(1, push);
            return true;
        }
    }
    if (i > s.next && i < s.run_end && i - s.next <= kMaxSkip) {
        // Ahead: skip forward in hardware. The step after this one is the request gap if the previous request had the
        // same gap (a steady stride, e.g. every Nth page), else 1 - so a jump (a block edge, a new row) doesn't put
        // the hardware past the next sequential request. A skip doesn't count as a hit: random access skips by
        // luck.
        const uint32_t gap = i - last;
        const uint32_t prev_gap = s.last_hit ? s.stride : s.last_gap;  // the previous request's gap
        const uint32_t step = gap == prev_gap ? gap : 1;
        (void)pop_side<S>(i - s.next);
        // The skip's pop result is unused: keep it from being in flight with the next one (AIHWE-6506).
        overlay::rocc_nop();
        s.last_gap = gap;
        s.stride = step;
        s.last = i;
        s.next = i + step;
        s.last_hit = 0;
        out = advance_side<S>(step, push);
        return true;
    }
    if (i == s.next && i == s.run_end && s.row_pages != 0 && s.rows_left != 0 && s.stride == 1) {
        // Row sequence at the end of its row, and the next row is in the same band: same programming, one shard row
        // further. Write the position only.
        --s.rows_left;
        s.row_outer += s.row_step;
        set_side_position<S>(s.row_bank, 0, s.row_outer);
        s.run_end = i + s.row_pages;
        s.next = i + 1;
        s.last = i;
        s.last_hit = 0;
        out = advance_side<S>(1, push);
        return true;
    }
    if (i == s.next) {
        out = reseek<S, Recipe>(
            s, i, s.stride, acc_ptr, arg, noc, push);  // past the end of the run: the next run of the same sequence
        return true;
    }
    // Behind, or a large jump. A sequence whose last request was a hit re-seeks right away, keeping its stride: the
    // jump back to the next block, column or pass of a regular pattern. A sequence whose previous request also missed
    // uses software for this one - random access - and re-takes the hardware once the access continues at the
    // sequence's stride, or repeats the gap of the previous software request.
    const uint32_t gap = i > last ? i - last : 0;
    const bool continues = gap != 0 && (gap == s.stride || gap == s.miss_gap);
    if (s.last_hit || continues) {
        out = reseek<S, Recipe>(s, i, continues ? gap : s.stride, acc_ptr, arg, noc, push);
        return true;
    }
    s.miss_gap = gap;
    s.last = i;
    return false;
}

// The hit path: i is the sequence's next index (`next`, already loaded with the owner) and inside the current
// programming. One paired load (stride, run end), one store (next), the pop; `last_hit` is written only when it
// changes. Endless: the programming covers every later index (interleaved), so there is no run end to check. Anything
// else goes to serve_on_side_slow(). push: the caller can take the address in the command buffer, so a push side pushes
// it instead of popping it (out = kAddrPushed), on the hit path and the slow path alike.
template <uint32_t S, bool Endless, typename Recipe>
inline __attribute__((always_inline)) bool serve_on_side(
    uint32_t i, uint32_t next, const void* acc_ptr, uint32_t arg, uint8_t noc, bool push, uint64_t& out) {
    if (__builtin_expect(i == next, 1)) {
        SideState& s = sides[S];
        const uint64_t stride_end = load_pair(s.stride);
        const uint32_t stride = static_cast<uint32_t>(stride_end);
        if (__builtin_expect(Endless || i < static_cast<uint32_t>(stride_end >> 32), 1)) {
            s.next = i + stride;
            if (__builtin_expect(!s.last_hit, 0)) {
                s.last_hit = 1;
            }
            out = advance_side<S>(stride, push);
            return true;
        }
    }
    return serve_on_side_slow<S, Recipe>(i, acc_ptr, arg, noc, push, out);
}

// Exchange two sequence states in place. Used in a three tensor round-robin case
inline void swap_sequences(SideState& a, SideState& b) {
    using word = uint64_t __attribute__((may_alias));
    word* wa = reinterpret_cast<word*>(&a);
    word* wb = reinterpret_cast<word*>(&b);
    for (uint32_t k = 0; k < sizeof(SideState) / sizeof(word); ++k) {
        const word t = wa[k];
        wa[k] = wb[k];
        wb[k] = t;
    }
}

template <uint32_t S, typename Recipe>
TT_TA_SEEK_NOINLINE inline bool take_side(
    uint32_t key, uint32_t i, const void* acc_ptr, uint32_t arg, uint8_t noc, bool push, uint64_t& out) {
    SideState& s = sides[S];
    const bool spill = kNumParked > 0 && s.owner != 0 && s.restorable;
    for (uint32_t p = 0; p < kNumParked; ++p) {
        if (parked[p].sequence.owner == key) {
            // Reload: the side's sequence and the parked one trade places - the side's position is read back, the
            // parked sequence's programming and position written in, and the side's sequence parked where the other one
            // was.
            overlay::AddrgenPosition spilled;
            if (spill) {
                save_side<S>(spilled);
            }
            swap_sequences(s, parked[p].sequence);
            restore_side<S>(parked[p].pos);
            if (spill) {
                parked[p].pos = spilled;
            } else {
                parked[p].sequence.owner = 0;  // the side's sequence can't be restored (or there was none):
            }
            const bool served = serve_on_side_slow<S, Recipe>(i, acc_ptr, arg, noc, push, out);
            return served;
        }
    }
    // Claim: park the side's sequence in a free entry (or forget it when there is none), then seek to i.
    if (spill) {
        for (uint32_t p = 0; p < kNumParked; ++p) {
            if (parked[p].sequence.owner == 0) {
                save_side<S>(parked[p].pos);
                parked[p].sequence = s;
                break;
            }
        }
    }
    s = SideState{};
    s.owner = key;
    out = reseek<S, Recipe>(s, i, 1, acc_ptr, arg, noc, push);
    return true;
}

// The sequence isn't on a side of its direction: take a free side, else the least recently used one,
// only where the other side of the generator fits its bank shift.
template <TransferDir Dir, typename Recipe>
TT_TA_SEEK_NOINLINE inline bool serve_request_slow(
    uint32_t key,
    uint32_t shift,
    uint32_t i,
    const void* acc_ptr,
    uint32_t arg,
    uint8_t noc,
    bool push,
    uint64_t& out,
    uint32_t& served_side) {
    constexpr uint32_t PushSide = push_side_of(Dir);
    constexpr uint32_t PopSide = pop_side_of(Dir);
    constexpr uint32_t d = Dir == TransferDir::Write ? 1u : 0u;
    const bool fits_push = shift_fits(PushSide, shift);
    const bool fits_pop = shift_fits(PopSide, shift);
    uint32_t target = kNumSides;
    if (sides[PushSide].owner == 0 && fits_push) {
        target = PushSide;
    } else if (sides[PopSide].owner == 0 && fits_pop) {
        target = PopSide;
    } else if (kNumParked > 0) {
        // Spill: the least recently used side (the one not used last) if it fits, else the other one.
        const uint32_t lru = last_side[d] == PushSide ? PopSide : PushSide;
        const uint32_t mru = lru == PushSide ? PopSide : PushSide;
        const bool fits_lru = lru == PushSide ? fits_push : fits_pop;
        const bool fits_mru = mru == PushSide ? fits_push : fits_pop;
        target = fits_lru ? lru : (fits_mru ? mru : kNumSides);
    }
    if (target == kNumSides) {
        served_side = kNumSides;
        return false;
    }
    served_side = target;
    last_side[d] = target;
    return target == PushSide ? take_side<PushSide, Recipe>(key, i, acc_ptr, arg, noc, push, out)
                              : take_side<PopSide, Recipe>(key, i, acc_ptr, arg, noc, push, out);
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

// Serve request i of the tensor sequence `key` (bank shift `shift`) in direction Dir: find which of the direction's two
// sides holds the sequence and serve i there (serve_on_side()). The hit path is one paired load (owner, next) per side
// checked, push side first. If neither side holds it, serve_request_slow() takes a side (spilling or reloading).
//   Endless:     the sequence's programming covers every later index (interleaved), so there is no run end to check.
//   Recipe:      how to (re)program a side for index i (PagesRecipe, ...); used only off the hit path, so the hit path
//                just forwards acc_ptr / arg / noc in registers.
//   push:        the caller can take the address in the command buffer; only the push side can do it.
//   out:         the address (or kAddrPushed) when this returns true; untouched when it returns false (software).
//   served_side: which side served the request (kNumSides when it used software). Only ShardView uses it: it caches
//                the popped shard base on that side, so later transfers into the same shard skip the request.
template <TransferDir Dir, bool Endless, typename Recipe>
inline __attribute__((always_inline)) bool serve_request(
    uint32_t key,
    uint32_t shift,
    uint32_t i,
    const void* acc_ptr,
    uint32_t arg,
    uint8_t noc,
    bool push,
    uint64_t& out,
    uint32_t& served_side) {
    constexpr uint32_t PushSide = push_side_of(Dir);
    constexpr uint32_t PopSide = pop_side_of(Dir);
    constexpr uint32_t d = Dir == TransferDir::Write ? 1u : 0u;
    static_assert(is_push_side<PushSide> && !is_push_side<PopSide>);
    const uint64_t push_owner_next = load_pair(sides[PushSide].owner);
    if (__builtin_expect(static_cast<uint32_t>(push_owner_next) == key, 1)) {
        served_side = PushSide;
        touch_side<d, PushSide>();
        return serve_on_side<PushSide, Endless, Recipe>(
            i, static_cast<uint32_t>(push_owner_next >> 32), acc_ptr, arg, noc, push, out);
    }
    const uint64_t pop_owner_next = load_pair(sides[PopSide].owner);
    if (static_cast<uint32_t>(pop_owner_next) == key) {
        served_side = PopSide;
        touch_side<d, PopSide>();
        return serve_on_side<PopSide, Endless, Recipe>(
            i, static_cast<uint32_t>(pop_owner_next >> 32), acc_ptr, arg, noc, false, out);
    }
    return serve_request_slow<Dir, Recipe>(key, shift, i, acc_ptr, arg, noc, push, out, served_side);
}

// ============================================================================
// 7. Entry points
// ============================================================================
//
//
// Recipe types, one per kind of index (passed to serve_request() as the Recipe template argument):
//   PagesRecipe       page ids (TensorAccessor, PageView, wrapper, pages()): plan_interleaved or plan_sharded
//   ShardPagesRecipe  shard * shard_volume + page_in_shard (shard_pages()): plan_single_bank for the rest of the shard
//   ShardBasesRecipe  shard ids (ShardView): plan_shard_bases
// Each is stateless: plan() rebuilds the Seek from plain arguments - the accessor (acc_ptr), one extra word (arg: the
// shard volume for shard_pages()) and the NoC id (noc). The hit path only forwards these registers; plan() runs only
// in the cold functions (reseek).
//   kShortRunFallback: count the sequence's short seeks (see kMinRun).
//   kShardRestart: index i is shard * arg + page_in_shard, and a shard's first page restarts the sequence there
//     (page_addr() gives that page's address).
template <typename Accessor>
struct PagesRecipe {
    static constexpr bool kShortRunFallback = true;
    static constexpr bool kShardRestart = false;
    static Seek plan(const void* acc_ptr, uint32_t, uint8_t noc, uint32_t i) {
        const Accessor& acc = *static_cast<const Accessor*>(acc_ptr);
        if constexpr (Accessor::DSpec::is_interleaved) {
            return plan_interleaved<Accessor::DSpec::is_dram>(
                acc.get_bank_base_address(), acc.get_aligned_page_size(), i);
        } else {
            return plan_sharded(acc, i, noc);
        }
    }
};

template <typename Accessor>
struct ShardPagesRecipe {
    // The sequence's index is shard * shard_volume + page_in_shard (arg = shard_volume): consecutive shards are
    // consecutive runs, and a request for a shard's first page restarts the sequence in that shard's bank (serve_slow).
    static constexpr bool kShortRunFallback = false;  // one run per shard by design
    static constexpr bool kShardRestart = true;
    static uint64_t page_addr(const void* acc_ptr, uint32_t shard_volume, uint8_t noc, uint32_t i) {
        const Accessor& acc = *static_cast<const Accessor*>(acc_ptr);
        return ::tensor_accessor::detail::transfer_shard_noc_addr(
            acc, i / shard_volume, (i % shard_volume) * acc.get_aligned_page_size(), noc);
    }
    static Seek plan(const void* acc_ptr, uint32_t shard_volume, uint8_t noc, uint32_t i) {
        const Accessor& acc = *static_cast<const Accessor*>(acc_ptr);
        return Seek{
            .prog = plan_single_bank(page_addr(acc_ptr, shard_volume, noc, i), acc.get_aligned_page_size()),
            .run_end = (i / shard_volume + 1) * shard_volume,  // the end of i's shard
        };
    }
};

// Hardware generated address of `page_id` (+ offset). Returns false (and leaves `out` untouched) when the request uses
// software. SW fallback when this ATT doesn't fit the recipe, or the sequence policy sends it there.
// MayPush: the caller issues on the direction's command buffer and accepts kAddrPushed (only for offset 0: a pushed
// address can't have the offset added).
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline __attribute__((always_inline)) bool try_generate_noc_addr(
    const Accessor& acc, uint32_t page_id, uint32_t offset, uint8_t noc, uint64_t& out) {
    static_assert(has_hw_recipe<Accessor>);
    constexpr bool is_interleaved = Accessor::DSpec::is_interleaved;
    uint32_t served_side;
    if (!serve_request<Dir, is_interleaved, PagesRecipe<Accessor>>(
            sequence_key<Accessor, SequenceKind::Pages>,
            sequence_shift<Accessor>,
            page_id,
            &acc,
            0,
            noc,
            /*push=*/MayPush && offset == 0,
            out,
            served_side)) {
        return false;
    }
    out += offset;
    return true;
}

// ---- shard_pages(): pages of one shard, in storage order ----
//
// A shard's pages are consecutive in its bank (bank_page_offset = shard_in_bank * shard_volume + page_in_shard), so
// one single-bank sequence covers a shard. The sequence's index is shard * shard_volume + page_in_shard: consecutive
// shards are consecutive runs, and reaching a shard's first page moves the sequence to that shard with three register
// writes (the shard restart in serve_slow) instead of a seek. Padding pages the iterator skips are skipped in hardware.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline __attribute__((always_inline)) bool try_generate_shard_page_noc_addr(
    const Accessor& acc, uint32_t shard_id, uint32_t page_in_shard, uint32_t offset, uint8_t noc, uint64_t& out) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    constexpr uint32_t key = sequence_key<Accessor, SequenceKind::ShardPages>;
    const uint32_t shard_volume = acc.dspec().shard_volume();
    uint32_t served_side;
    if (!serve_request<Dir, false, ShardPagesRecipe<Accessor>>(
            key,
            sequence_shift<Accessor>,
            shard_id * shard_volume + page_in_shard,
            &acc,
            shard_volume,
            noc,
            /*push=*/MayPush && offset == 0,
            out,
            served_side)) {
        return false;
    }
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
struct ShardBasesRecipe {
    static constexpr bool kShortRunFallback = false;  // a base serves every offset into its shard
    static constexpr bool kShardRestart = false;
    static Seek plan(const void* acc_ptr, uint32_t, uint8_t noc, uint32_t i) {
        return plan_shard_bases(*static_cast<const Accessor*>(acc_ptr), i, noc);
    }
};

template <TransferDir Dir, typename Accessor>
inline __attribute__((always_inline)) bool try_generate_shard_noc_addr(
    const Accessor& acc, uint32_t shard_id, uint32_t offset, uint8_t noc, uint64_t& out) {
    static_assert(has_hw_recipe<Accessor> && !Accessor::DSpec::is_interleaved);
    constexpr uint32_t key = sequence_key<Accessor, SequenceKind::ShardBases>;
    constexpr uint32_t dir_sides[] = {push_side_of(Dir), pop_side_of(Dir)};
    for (const uint32_t s : dir_sides) {
        if (sides[s].owner == key && sides[s].has_base && sides[s].base_shard == shard_id) {
            out = sides[s].base_addr + offset;
            return true;
        }
    }
    uint32_t served_side;
    uint64_t base;
    if (!serve_request<Dir, false, ShardBasesRecipe<Accessor>>(
            key, sequence_shift<Accessor>, shard_id, &acc, 0, noc, /*push=*/false, base, served_side)) {
        return false;
    }
    // Cache the base on the side that served it: later transfers into this shard reuse it (the loop above).
    sides[served_side].base_shard = shard_id;
    sides[served_side].has_base = 1;
    sides[served_side].base_addr = base;
    out = base + offset;
    return true;
}

}  // namespace tt_addrgen
