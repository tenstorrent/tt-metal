// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, unpack thread, unpack on Blackhole: L1 tiles to SrcA, SrcB or Dest. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/), except where noted. The unpackers' part of every Compute API op that reads L1
// (matmul, add, sub, mul, reduce, copy); the math thread's part is the op's own file. The shape of an op:
// llk_lib/thread.h.
//
//   unpack_params                   one struct; unpack_configure and unpack_execute take the same one
//   unpack_params_from(Plan)        fills it from the Resolver's Plan
//   unpack_configure<P>()           programs both unpackers completely, through the instruction stream: replay bodies,
//                                   address counters, a Dest stream's output stride, descriptors and formats, the
//                                   config context (see "configure" below)
//   unpack_execute<P>(a, b, steps, runs, tiles_per_run)
//                                   every step of the walk: for each run, point the unpackers at that run's tiles,
//                                   unpack the held operand, then every streamed tile; a Dest stream's hand-off
//                                   with math around the walk. The walk's three loop counts are arguments: each a
//                                   ct (P.walk's number) or, for a size known only at run time, an rt of at
//                                   most that number (resolver/work_counts.h)
//   unpack<On, Plan>(counts, in0[, in1])
//                                   the per-thread call the Compute API makes, one L1 address per input: configures
//                                   and executes, only when this build is for thread On (the unpack thread). counts
//                                   (WorkCounts) is the run-time side of the Plan's Work
//
// One unpack for every op. Unpack does not know which op it serves: the Resolver's Plan describes the work as data,
// and this file executes that description:
//
//   streams  one per unpacker in use. Unpacker 0 writes SrcA (configured through THCON SEC0), unpacker 1 writes
//            SrcB (SEC1). A stream says which formats to convert between, the tile geometry, and how its L1
//            address moves: per streamed tile, per run, per step
//   walk     the loop nest over those streams (see UnpackWalk in resolver/plan.h): steps x runs x tiles_per_run,
//            with an optional held stream unpacked once per run
//
// Three 1.0 unpack families are this as data: unpack_A is one stream, unpack_AB two streams with nothing held,
// unpack_AB_matmul two streams with the reused operand held. The fourth, tilize, would be a row-major stream
// (Layout::RowMajor): not ported, configure refuses it.
//
// Unpack to Dest. A stream whose Plan says Where::Dest goes through unpacker 0 to Dest instead of SrcA (32-bit
// formats with a 32-bit Dest; see unpack_tile_to_dest and unpack_dest_write_begin below). Its tiles land one after another
// from a Dest row the math thread names, so unpack and math hand off around the whole call (unpack_dest_write_begin /
// unpack_dest_write_end):
//   math    waits until its earlier Dest writes are done, then sends the absolute Dest row through the mailbox
//   unpack  takes the row, points both contexts at it, unpacks every tile (each UNPACR moves its Dest address on by
//           one tile through the channel 1 Z counter), then posts semaphore UNPACK_TO_DEST once the unpacker is idle
//   math    waits for UNPACK_TO_DEST and takes it; Dest now holds the tiles
// No Src bank changes hands on the way to Dest: those UNPACRs keep the bank (no data valid).
//
// Transpose. A stream with `transpose` set is transposed inside each 16x16 face by unpacker 0 (Haloize_mode). The
// face order is the consumer's to fix (copy's MOVA2Ds; matmul's SrcA address modifiers). Unpacker 0 cannot transpose
// on the way to Dest.
//
// Two config contexts: each unpacker keeps its L1 base address in a config register per context. The unpack thread
// writes the next run's addresses into one context while the unpackers still work from the other, and flips
// between them after every run. Semaphore UNPACK_SYNC counts the contexts handed to the unpackers and not yet
// finished; at most two may be in flight.
//
// configure assumes nothing about what ran before; removing writes an earlier op already made is left to the
// compiler (config folding), so configure issues everything through the instruction stream, in a form such a compiler
// can remove. There is no init, uninit or reconfig.
//
// Departures from the shape (llk_lib/thread.h), and why:
//   - configure's order is the reverse of the other ops': first what is not config and needs no wait (replay bodies,
//     address counters, GPRs), then the unpackers' own config words, and only then the wait for idle unpackers, around
//     the words they read while data moves (SrcASet, SrcAUnsigned, SrcBUnsigned). Not forced by the hardware; kept
//     because any other order changes what is issued. See "What the unpackers read, and when".
//   - configure does not set the config context: it writes the hardware's context offset from the software's
//     (platform::unpack_config_context), so the alternation of contexts carries on from the previous op. Forced by the
//     hardware as it is used here: the previous op's last run may still be working from its context when this op's
//     first run stores its base addresses ("Context" below).
//
// What the unpackers read, and when. configure writes the unpackers' words in stream order behind the previous op's
// UNPACRs, which may still be moving data (or standing still in the unpacker, waiting for a Src bank). That is right
// for a word an UNPACR takes when it leaves the Wait Gate, and wrong for one it reads while data moves:
//   read while data moves   SRCA_SET (SetOvrdWithAddr): tt-isa-documentation, UNPACR_Regular.md, "Instruction
//                           scheduling" ("at each of its writes ..., not only when it starts"); measured for Src
//                           writes too. SrcAUnsigned, SrcBUnsigned: the SFPI compiler's instruction table ("read
//                           when the data reaches the source or Dest registers"); not measured (no test unpacks an integer format).
//                           These three are written behind a wait for both unpackers, inside its scope, so a
//                           compiler that deletes the wait keeps it whenever one of them changes.
//   taken at the start      everything else configure writes: formats (InDataFormat, Out_data_format), Haloize_mode,
//                           Tile_x_dim / XDim, ZDim, Throttle_mode, Unpack_if_sel, the Src write base, the X end and
//                           the input and output address counters. Measured on the board, word by word: a write
//                           issued behind the UNPACRs, at every distance from "right behind the last one" to "after
//                           they finished", and with the last unpack standing still in the unpacker, leaves every
//                           tile exact, while the same write to SRCA_SET corrupts it
//                           (tests/python_tests/test_llk2_unpack_latch_probe.py). The same table of the compiler
//                           says so ("samples config at the buffer head"), and UNPACR_Regular.md has the UNPACR hold
//                           its thread for the two cycles in which it works out its input address.
// The same property carries the config contexts: a run's SEMGET and the context flip follow its UNPACRs with no wait,
// and the next run but one stores its base addresses into the same context as soon as that SEMGET has run.
// A Dest stream ends behind a wait for unpacker 0 (unpack_dest_write_end), which holds everything issued after it, so
// nothing of the next configure reaches an unpack to Dest in flight (UNPACR_Regular.md: also required before the next
// UNPACR to SrcA).
// A compiler that tracks Tensix state deletes the wait with its writes when their values are in place. By design.
//
// Ported from: llk_unpack_A.h, llk_unpack_AB.h, llk_unpack_AB_matmul.h and cunpack_common.h (configure_unpack_AB, the
// two config contexts and UNPACK_SYNC, the unpack-to-Dest handshake).
//
// Not HAL: LLK_ASSERT (llk_assert.h).

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_lib/formats.h"
#include "llk_lib/platform/format.h"
#include "llk_lib/platform/gpr_map.h"
#include "llk_lib/platform/l1.h"
#include "llk_lib/semaphores.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/work_counts.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#include "llk_lib/platform/issue.h"
#include "llk_lib/platform/limits.h"
#include "llk_lib/platform/mailbox.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- parameters -----------------------------------------------------------------------------------------------------

/// One unpacker's operand: formats, geometry, and how its L1 address moves. Address steps are in 16-byte units.
struct unpack_stream
{
    bool active;              ///< false: this unpacker is not used
    DataFormat l1_format;     ///< format in L1 (the unpacker's InDataFormat)
    DataFormat reg_format;    ///< format written to the register (OutDataFormat)
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tile_words; ///< one tile in L1, in L1 words
    Layout layout;            ///< how the operand sits in L1
    Broadcast broadcast;      ///< how a smaller operand is spread over the tile
    bool transpose;           ///< transpose inside each face on the way in (unpacker 0 to SrcA only)
    std::uint32_t per_tile;   ///< address step after each streamed tile, inside a run
    std::uint32_t per_run;    ///< address step from one run to the next
    std::uint32_t per_step;   ///< address step from one walk step to the next
    bool to_dest;             ///< write Dest instead of SrcA (unpacker 0 only, 32-bit formats)
};

/// Parameters of one unpack operation, for any op. configure and execute take the same struct.
struct unpack_params
{
    unpack_stream src_a; ///< unpacker 0 -> SrcA
    unpack_stream src_b; ///< unpacker 1 -> SrcB
    UnpackWalk walk;     ///< the loop nest over the two streams (resolver/plan.h)
};

/// One operand's UnpackPlan as an unpacker stream: address steps from tiles to L1 words.
constexpr unpack_stream unpack_stream_from(const UnpackPlan& u)
{
    const std::uint32_t tile_words = u.tile_bytes / l1_word_bytes;
    return {
        .active     = u.to != Where::None,
        .l1_format  = u.reads,
        .reg_format = u.writes,
        .face_r_dim = u.face_r_dim,
        .num_faces  = u.num_faces,
        .tile_words = tile_words,
        .layout     = u.layout,
        .broadcast  = u.broadcast,
        .transpose  = u.transpose_faces,
        .per_tile   = u.per_tile * tile_words,
        .per_run    = u.per_run * tile_words,
        .per_step   = u.per_step * tile_words,
        .to_dest    = u.to == Where::Dest,
    };
}

/// One of the two unpackers, under every name the HAL has for it.
struct unpacker
{
    std::uint32_t index;                 ///< 0 or 1: which half of a context's replay slots its per-tile body takes
    hal::unpack::Engine engine;          ///< the engine an UNPACR names
    cfg::Sec section;                    ///< its THCON section: its tile descriptor, config and base addresses
    hal::counters::adc::Client counters; ///< its address counters
    std::uint32_t per_tile_gpr;          ///< the GPR its replay body adds the per-tile address step from (platform/gpr_map.h)
};

/// Unpacker 0: writes SrcA, or Dest.
inline constexpr unpacker unpacker0 {
    .index        = 0,
    .engine       = hal::unpack::Engine::Unpacker0,
    .section      = cfg::Sec::S0,
    .counters     = hal::counters::adc::Client::Unpacker0,
    .per_tile_gpr = platform::gpr_map::unpack::unpacker0_tile_words};

/// Unpacker 1: writes SrcB.
inline constexpr unpacker unpacker1 {
    .index        = 1,
    .engine       = hal::unpack::Engine::Unpacker1,
    .section      = cfg::Sec::S1,
    .counters     = hal::counters::adc::Client::Unpacker1,
    .per_tile_gpr = platform::gpr_map::unpack::unpacker1_tile_words};

/// Whether an operand whose Plan sends it to `to` goes through unpacker u: unpacker 0 writes SrcA or Dest, unpacker 1
/// writes SrcB.
constexpr bool unpack_goes_through(const unpacker& u, Where to)
{
    return u.index == 0 ? (to == Where::SrcA || to == Where::Dest) : to == Where::SrcB;
}

inline constexpr std::uint32_t unpack_no_input = 2; ///< unpack_input_of: the unpacker takes neither input

/// Which input of a Plan unpacker u takes: 0 (in0), 1 (in1) or unpack_no_input. The one place that decides it, for the
/// streams (unpack_params_from) and for the addresses (the entry).
constexpr std::uint32_t unpack_input_of(const Plan& plan, const unpacker& u)
{
    return unpack_goes_through(u, plan.in0.to) ? 0 : unpack_goes_through(u, plan.in1.to) ? 1 : unpack_no_input;
}

/// The Plan of any op as unpack parameters: each operand goes to the unpacker its Plan names.
constexpr unpack_params unpack_params_from(const Plan& plan)
{
    const UnpackPlan none {};
    const auto operand_of = [&](const unpacker& u) -> const UnpackPlan&
    {
        const std::uint32_t input = unpack_input_of(plan, u);
        return input == 0 ? plan.in0 : input == 1 ? plan.in1 : none;
    };
    return {.src_a = unpack_stream_from(operand_of(unpacker0)), .src_b = unpack_stream_from(operand_of(unpacker1)), .walk = plan.walk};
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint32_t unpack_max_contexts_in_flight = 2;      ///< two config contexts
inline constexpr std::uint32_t unpack_gpr_half_max           = 0xffff; ///< scalar::set_low writes the low 16 bits of a GPR
inline constexpr std::uint32_t unpack_throttle_mode          = 2;      ///< Throttle_mode 2: the 512-bit L1 read interface
/// Rows UNPACR skips: it takes its output row from the output address, then subtracts 4 before writing Src or Dest
/// (tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md, functional model, "Row -= 4").
inline constexpr std::uint32_t unpack_skipped_rows = 4;
/// The output address (in datums) that lands on row 0 of Src or Dest, both contexts; the value configure_unpack_AB uses.
inline constexpr std::uint32_t unpack_src_write_base        = unpack_skipped_rows * dest_row_datums;
inline constexpr std::uint32_t unpack_dest_row_bytes_32bit  = dest_row_datums * sizeof(std::uint32_t); ///< one Dest row of 32-bit datums
/// The word that holds unpacker 0's AddDestAddrCntr (bit 8) and ForcedSharedExp (bits 0-7) is written whole (see
/// "Whole-word registers" under configure). A whole-word WRCFG is anchored on the field at bit 0 of its word, which
/// here is ForcedSharedExp. The value: AddDestAddrCntr 1 (the address counter is added to the write base),
/// ForcedSharedExp 0.
inline constexpr const cfg::Field& unpack_add_dest_addr_cntr_anchor = cfg::Unpacker[0].ForcedSharedExp;
inline constexpr std::uint32_t unpack_add_dest_addr_cntr_word       = 1u << cfg::Unpacker[0].AddDestAddrCntr.shamt(cfg::Sec::S0);
static_assert(
    unpack_add_dest_addr_cntr_anchor.shamt(cfg::Sec::S0) == 0 &&
        unpack_add_dest_addr_cntr_anchor.addr32(cfg::Sec::S0) == cfg::Unpacker[0].AddDestAddrCntr.addr32(cfg::Sec::S0),
    "the anchor is the field at bit 0 of AddDestAddrCntr's word");
/// GPR that carries AddDestAddrCntr's word to WRCFG; nothing else in LLK 2.0 writes it.
inline constexpr std::uint32_t unpack_add_dest_addr_cntr_gpr = platform::gpr_map::unpack::scratch1;

// Held and streamed operands.

/// SrcA's operand is the held one (unpacked once per run).
constexpr bool unpack_held_is_a(const unpack_params& params)
{
    return params.walk.held == Where::SrcA;
}

/// SrcB's operand is the held one (unpacked once per run).
constexpr bool unpack_held_is_b(const unpack_params& params)
{
    return params.walk.held == Where::SrcB;
}

/// Unpacker 0 is in use and streams (its operand is unpacked once per streamed tile).
constexpr bool unpack_streams_a(const unpack_params& params)
{
    return params.src_a.active && !unpack_held_is_a(params);
}

/// Unpacker 1 is in use and streams (its operand is unpacked once per streamed tile).
constexpr bool unpack_streams_b(const unpack_params& params)
{
    return params.src_b.active && !unpack_held_is_b(params);
}

// Tile geometry: how much of a tile one UNPACR covers.

/// Faces of a tile the unpacker reads. A row broadcast needs only the top face row, which holds the tile's row 0
/// (faces 0 and 1 of a 32x32 tile); a scalar broadcast only face 0, which holds datum [0, 0]; every other stream
/// reads all faces (a column broadcast reads column 0 from faces 0 and 2, so it takes the whole tile).
constexpr std::uint32_t unpack_faces(const unpack_stream& s)
{
    return s.broadcast == Broadcast::Row ? faces_per_face_row : s.broadcast == Broadcast::Scalar ? 1 : s.num_faces;
}

/// The last datum index of a tile along X, for the unpacker's X counter: every face read, every row, face_cols
/// datums per row. 0 for an unused unpacker.
constexpr std::uint32_t unpack_x_end(const unpack_stream& s)
{
    return s.active ? unpack_faces(s) * s.face_r_dim * face_cols - 1 : 0;
}

// Dest streams.

/// Bytes one tile takes in a 32-bit Dest. This is the channel 1 Z stride of a Dest stream: how far its output address
/// moves per Z step, so that each UNPACR (Z += 1) writes the next tile.
constexpr std::uint32_t unpack_dest_tile_stride(const unpack_stream& s)
{
    return s.num_faces * s.face_r_dim * unpack_dest_row_bytes_32bit;
}

/// The unpacker's output address for Dest row `row`. The unpacker counts its output in datums, from the same
/// unpack_src_write_base offset it uses for Src (unpack_skipped_rows), so a row is dest_row_datums further on.
constexpr std::uint32_t unpack_dest_write_address(std::uint32_t row)
{
    return unpack_src_write_base + row * dest_row_datums;
}

// Replay slots, and the config context they belong to.

/// Replay slots of one streamed operand's per-tile body: UNPACR, RDCFG, ADDDMAREG, STALLWAIT, WRCFG, NOP
/// (unpack_record_streamed_tile).
inline constexpr std::uint32_t unpack_stream_body_slots = 6;
/// Replay slots a context reserves: unpacker 0's stream body, then unpacker 1's.
inline constexpr std::uint32_t unpack_context_slots = 2 * unpack_stream_body_slots;
static_assert(
    unpack_max_contexts_in_flight * unpack_context_slots <= hal::replay::detail::BUFFER_SIZE, "the stream bodies of both contexts fit the replay buffer");

// The config context is a run-time value (it alternates per run, across ops), but everything that depends on it is
// issued as constants: one arm per context, each a fixed instruction word or replay range, never a computed one. A
// replay range is a template argument, and a compiler that tracks Tensix state can follow two constant arms, where a
// computed instruction word is one it can neither prove redundant nor move. So each such body is written once, as a
// template on the context, and the run-time context only selects the arm (unpack_set_context,
// unpack_set_base_addresses, unpack_play_streamed_tiles).

/// The field of an unpacker's THCON section that holds its L1 base address for config context `context`.
constexpr const cfg::Field& unpack_base_address_field(const std::uint32_t context)
{
    return context == 0 ? cfg::Thcon[cfg::Reg3].Base_address : cfg::Thcon[cfg::Reg3].Base_cntx1_address;
}

/// The replay slots of one operand's per-tile body in one context: the same for every op.
constexpr hal::replay::BufferRange unpack_stream_slots(const unpacker& u, const std::uint32_t context)
{
    return {context * unpack_context_slots + u.index * unpack_stream_body_slots, unpack_stream_body_slots};
}

/// The slots one streamed tile of P plays in `context`: every streamed operand's body, SrcA's first (adjacent slots).
constexpr hal::replay::BufferRange unpack_tile_slots(const unpack_params& params, const std::uint32_t context)
{
    const std::uint32_t streamed_operands = (unpack_streams_a(params) ? 1 : 0) + (unpack_streams_b(params) ? 1 : 0);
    return {unpack_stream_slots(unpack_streams_a(params) ? unpacker0 : unpacker1, context).start, streamed_operands * unpack_stream_body_slots};
}

// Stepping a stream with the address counter.
//
// unpack_record_streamed_tile (below) moves a streamed operand to its next tile by rewriting its base address: UNPACR, then
// RDCFG, ADDDMAREG, a STALLWAIT until the add has landed, WRCFG and a NOP, per tile and per stream, one stream after
// the other. The unpacker writes a tile in 16 cycles; that sequence takes the unpack thread about 14, of which 5 are
// the wait. With two streamed operands (add, sub, mul) the thread is the limit: 30 cycles per tile, measured, while
// the FPU needs 8 and the two unpackers could deliver a tile pair in 16.
//
// The unpacker can step by itself. It reads an uncompressed tile from
//     base + header + FirstDatum * datum bytes,   FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X
// with X, Y, Z, W its channel 0 address counters and XDim, YDim, ZDim the tile descriptor's (tt-isa-documentation,
// UNPACR_Regular.md). llk-lib programs XDim = one face's datums, YDim = 1, ZDim = faces per tile, and unpacks a whole
// tile with one UNPACR from X = Y = Z = W = 0. So Y + Z counts faces: an UNPACR that post-increments them by one
// tile's faces (each field holds at most 3: Z += 3, Y += 1 for four faces) leaves the next UNPACR at the next tile.
// The model masks the counters to 8 bits and does not compare them with the dims, so a run can step through 85 tiles
// (Z = 3 per tile); a longer run keeps the base-address rewrite.
//
// A streamed tile is then one instruction, both streams start back to back, and nothing is recorded: no replay, no
// GPR. The counters go back to 0 behind every run's last UNPACR, as tt-llk resets the Z counters behind its
// face-by-face UNPACRs: every run starts at its base address (the run's first tile), and the next op's
// configure finds the counters as it sets them.
//
// Only for a stream whose tiles follow each other in L1 with nothing between them (tile bytes = datums * datum bytes:
// the float and integer formats; a block float tile carries its exponents in front) and that moves by one tile, or
// not at all (a broadcast operand streamed next to the other one: no step, so nothing but the UNPACR). A walk steps
// by counter when every streamed operand can; otherwise it keeps the base-address rewrite.

inline constexpr std::uint32_t unpack_post_increment_max = 3; ///< UNPACR's Y and Z post-increments are two-bit fields
inline constexpr std::uint32_t unpack_counter_max        = 0xff; ///< the unpacker reads 8 bits of an input address counter
/// Tiles a run can step through: the Z counter moves unpack_post_increment_max per tile and must stay within its 8
/// bits.
inline constexpr std::uint32_t unpack_counter_step_max_tiles = unpack_counter_max / unpack_post_increment_max;

/// One streamed operand can step with the address counter: see above.
constexpr bool unpack_counter_steps(const unpack_stream& s)
{
    const bool contiguous = s.tile_words * l1_word_bytes == s.num_faces * s.face_r_dim * face_cols * platform::format::datum_bytes(s.l1_format);
    const bool one_tile   = s.per_tile == s.tile_words && contiguous && s.num_faces <= 2 * unpack_post_increment_max;
    return s.active && !s.to_dest && s.layout == Layout::Tilized && (s.per_tile == 0 || one_tile);
}

/// Every streamed operand of P steps with the address counter, and a run is short enough for the counters.
constexpr bool unpack_counter_step_walk(const unpack_params& params)
{
    return (unpack_streams_a(params) || unpack_streams_b(params)) && (!unpack_streams_a(params) || unpack_counter_steps(params.src_a)) &&
           (!unpack_streams_b(params) || unpack_counter_steps(params.src_b)) && params.walk.tiles_per_run <= unpack_counter_step_max_tiles;
}

/// The UNPACR post-increments that move a counter-stepped stream to its next tile: Z + Y = faces per tile.
constexpr hal::unpack::AddressCounterPostIncrements unpack_tile_post_increments(const unpack_stream& s)
{
    if (s.per_tile == 0)
    {
        return {};
    }
    const std::uint32_t z = s.num_faces < unpack_post_increment_max ? s.num_faces : unpack_post_increment_max;
    return {.channel0 = {.y = static_cast<std::uint8_t>(s.num_faces - z), .z = static_cast<std::uint8_t>(z)}};
}

/// A counter-stepped stream that moves (a broadcast one stays): its input counters must return to 0 between runs.
constexpr bool unpack_counter_moves(const unpack_params& params, const unpacker& u)
{
    return u.index == 0 ? (unpack_streams_a(params) && params.src_a.per_tile != 0) : (unpack_streams_b(params) && params.src_b.per_tile != 0);
}

// One tile: the UNPACRs configure records and execute issues.

/// Unpacks one tile of a stream to Dest: one UNPACR for the whole tile, then the channel 1 Z counter moves the
/// output on by one Dest tile (unpack_dest_tile_stride). No bank is handed to math: flipping SrcA after an UNPACR to
/// Dest is a hardware bug (ISA: SrcASrcB.md, Unpackers[0].WroteDst), and math does not read SrcA here.
template <unpacker Un, unpack_stream St>
inline __attribute__((always_inline)) void unpack_tile_to_dest()
{
    static_assert(Un.index == 0, "only unpacker 0 writes Dest");
    constexpr std::uint8_t next_dest_tile = 1; // channel 1 Z += 1: one Dest tile further on
    hal::unpack::run<hal::unpack::DataTransfer {
        .engine = Un.engine, .increments = {.channel1 = {.z = next_dest_tile}}, .handoff = hal::unpack::SourceHandoff::Keep}>();
}

/// Unpacks one tile of a stream: to Dest (unpack_tile_to_dest), or to Src with one UNPACR. The UNPACR is the same for
/// every broadcast; how much of the tile it covers is the X counter's end, which configure sets (unpack_x_end):
///   None  one UNPACR covers the whole tile (the X counter runs to unpack_x_end): faces 0..3 land in Src rows 0..63.
///   Row   one UNPACR covers the top face row only (unpack_x_end stops after faces 0 and 1): Src rows 0..15 hold face 0 and
///         rows 16..31 face 1, so tile row 0 is Src rows 0 and 16. Math broadcasts those rows down every face.
///         tt-llk (llk_unpack_AB.h, BroadcastType::ROW) unpacks faces 0, 1, 0, 1 one at a time, one per SrcA face,
///         because its math takes one face per Src bank; here math takes a whole tile per bank, so faces 0 and 1 once
///         are enough.
///   Col   as None: the whole tile. Math reads column 0 of faces 0 and 2 (tt-llk unpacks face 0, then face 2, one
///         per SrcA face row).
///   Scalar one UNPACR covers face 0 only (unpack_x_end stops after it): Src row 0 holds datum [0, 0].
/// Either way the bank is handed to math with the last datum.
template <unpacker Un, unpack_stream St>
inline __attribute__((always_inline)) void unpack_tile()
{
    static_assert(!(St.to_dest && St.broadcast != Broadcast::None), "unpack to Dest takes whole tiles: no broadcast");
    if constexpr (St.to_dest)
    {
        unpack_tile_to_dest<Un, St>();
    }
    else
    {
        // One UNPACR covers the tile (the X counter runs to unpack_x_end), then hands the bank to math. With Haloize_mode
        // on, each 16-row face is transposed as it is written.
        hal::unpack::run<hal::unpack::DataTransfer {.engine = Un.engine, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid}>();
    }
}

/// Unpacks one tile of a counter-stepped stream and moves its input counters to the next tile.
template <unpacker Un, unpack_stream St>
inline __attribute__((always_inline)) void unpack_tile_and_step()
{
    hal::unpack::run<hal::unpack::DataTransfer {
        .engine = Un.engine, .increments = unpack_tile_post_increments(St), .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid}>();
}

// ---- configure ------------------------------------------------------------------------------------------------------

//
// unpack_configure programs the unpackers completely and unconditionally, and issues every write as a Tensix
// instruction: the RISC-V queues the whole configure behind the previous op's unpacks and goes on. The three words the
// unpackers read while data moves wait, in the stream, until they are idle (platform::stall_scope). A
// state-tracking compiler deletes each write whose value is already in place, and the wait with them when no write
// is left that needs it. (Waiting on the RISC-V and storing the config through MMIO instead would empty the pipeline
// at every op boundary, and a compiler cannot remove a RISC-V loop or an MMIO store.) How it is laid out:
//
//   Order    unpack_configure is the list of its steps, with the reason for each one's place.
//   Replay   Each streamed operand's per-tile body has its own fixed replay slots per context (unpack_stream_slots), so an
//            operand's record is the same instructions in the same slots for every op that streams it. There is no
//            MOP: execute plays the context's slots once per streamed tile (unpack_play_streamed_tiles). MopCfg is written
//            by RISC-V stores behind mop_sync, which waits on the RISC-V for the MOP expander: one more pipeline
//            drain per configure, and a MOP whose body is one replay saves nothing over the replay.
//   Context  The config context is not reset to 0. With nothing waiting for the previous op's last run, that run
//            may still be using context 0 when this op's first run stores its base addresses (MMIO, at once):
//            execute only waits until at most one run is in flight, which is safe exactly because consecutive runs
//            alternate contexts. So the alternation carries on across ops, and configure makes the hardware's
//            context offset match the software's (unp_cfg_context), whatever a previous kernel left.
//   Whole-word registers   AddDestAddrCntr (no read-back on ttsim, so no read-modify-write) is written whole from a
//            GPR through WRCFG.
//   Dest     unpack_dest_write_begin writes the Dest row into both contexts through the instruction stream too: an MMIO
//            store would reach the register while the previous op's unpacks may still read it.

/// Replay body for unpacker Un's streamed operand in one context: unpack a tile, then base address (Base, the
/// context's) += its per-tile address step. The step is in Un's per_tile_gpr (set by configure).
template <unpacker Un, unpack_stream St, const cfg::Field& Base>
inline __attribute__((always_inline)) void unpack_record_streamed_tile()
{
    unpack_tile<Un, St>();
    cfg::read<cfg::Access::TensixCfgUnit, Base, Un.section>(hal::gpr<platform::gpr_map::unpack::scratch0>());
    hal::scalar::add(hal::gpr<platform::gpr_map::unpack::scratch0>(), hal::gpr<Un.per_tile_gpr>(), hal::gpr<platform::gpr_map::unpack::scratch0>());
    hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>(); // the add must land before WRCFG reads the GPR
    cfg::write<cfg::Access::TensixCfgUnit, Base, Un.section, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<platform::gpr_map::unpack::scratch0>());
}

/// Records unpacker Un's streamed operand's per-tile body in both contexts' slots.
template <unpacker Un, unpack_stream St>
inline __attribute__((always_inline)) void unpack_record_stream()
{
    hal::replay::record<unpack_stream_slots(Un, 0)>([] { unpack_record_streamed_tile<Un, St, unpack_base_address_field(0)>(); });
    hal::replay::record<unpack_stream_slots(Un, 1)>([] { unpack_record_streamed_tile<Un, St, unpack_base_address_field(1)>(); });
}

/// Configure, step 1: the per-tile replay body of each streamed operand, in both contexts' slots. Not config: a
/// REPLAY record is taken before the Wait Gate and executes nothing, so it needs no wait. A counter-stepped walk
/// records nothing: its streamed tiles are single UNPACRs.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_replay()
{
    if constexpr (unpack_streams_a(P) && !unpack_counter_step_walk(P))
    {
        unpack_record_stream<unpacker0, P.src_a>();
    }
    if constexpr (unpack_streams_b(P) && !unpack_counter_step_walk(P))
    {
        unpack_record_stream<unpacker1, P.src_b>();
    }
}

/// Configure, step 2: the unpackers' address counters for P's streams. Not config: the counters change only through
/// this thread's instructions, in order.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_counters()
{
    using namespace hal::counters::adc;
    hal::counters::set<AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<unpacker0.counters, unpacker1.counters>()>();
    hal::counters::set<AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<unpacker0.counters, unpacker1.counters>()>();
    // X runs over the whole tile, so one UNPACR unpacks it (channel 1 is the Src side).
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpack_x_end(P.src_a)}}.for_clients<unpacker0.counters>()>();
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpack_x_end(P.src_b)}}.for_clients<unpacker1.counters>()>();
}

/// Configure, step 3: the GPRs. The streamed operands' address steps, in the GPRs the replay bodies add from (a
/// counter-stepped walk reads none), and AddDestAddrCntr's word, for the WRCFG of step 4. Not config: the GPRs change
/// only through this thread's instructions.
/// The compile-time forms: the run-time set_low(gpr, value) builds its word at run time, which a state-tracking
/// compiler cannot name when it summarises the function for its callers, and that summary then says "may do
/// anything" (every value unknown after each call of a kernel helper).
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_gprs()
{
    if constexpr (!unpack_counter_step_walk(P))
    {
        hal::scalar::set_low<P.src_a.per_tile>(hal::gpr<unpacker0.per_tile_gpr>());
        hal::scalar::set_low<P.src_b.per_tile>(hal::gpr<unpacker1.per_tile_gpr>());
    }
    hal::scalar::set<unpack_add_dest_addr_cntr_word>(hal::gpr<unpack_add_dest_addr_cntr_gpr>());
}

/// Configure, step 4: AddDestAddrCntr's word, whole, from its GPR. The wait is for the scalar unit only: the GPR write
/// of step 3 must land before the WRCFG reads the GPR. The WRCFG needs no NOP behind it: execute issues a STALLWAIT
/// before the first UNPACR that reads the word.
inline __attribute__((always_inline)) void unpack_configure_add_dest_addr_cntr()
{
    constexpr auto write_add_dest_addr_cntr = []
    {
        cfg::write<cfg::Access::TensixCfgUnit, unpack_add_dest_addr_cntr_anchor, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<unpack_add_dest_addr_cntr_gpr>());
    };
    platform::stall_scope<StallTarget::Config, StallCondition::ScalarIdle>(write_add_dest_addr_cntr);
}

/// Unpacker Un's tile descriptor (what it reads) and config (what it writes), one write each, in its THCON section.
/// Every field of the descriptor words an uncompressed tile uses is named, and every field of the config words but
/// those of the contexts never selected.
template <unpacker Un, unpack_stream St, std::uint32_t XDim>
inline __attribute__((always_inline)) void unpack_write_descriptor_and_config()
{
    constexpr cfg::Sec S = Un.section;
    static_assert(St.layout == Layout::Tilized, "row-major (tilize) streams are not ported yet");
    static_assert(!St.to_dest || Un.index == 0, "only unpacker 0 writes Dest");
    static_assert(!(St.to_dest && St.transpose), "the unpacker cannot transpose on the way to Dest (ISA: undefined)");
    static_assert(!St.to_dest || datum_bits(St.reg_format) == 32, "only 32-bit formats are unpacked to Dest (unpack_dest_tile_stride assumes them)");
    constexpr auto& td = cfg::Thcon[cfg::Reg0].TileDescriptor;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<td.InDataFormat, S, format_id(St.l1_format)>(),
        cfg::set<td.Uncompressed, S, 1>(),     // tiles in L1 are not zero-compressed
        cfg::set<td.BlobsPerXyPlane, S, 0>(),  // no blobs: compression is off
        cfg::set<td.XDim, S, XDim>(),          // datums per face; unpacker 0 takes it per context instead (Tile_x_dim)
        cfg::set<td.YDim, S, 1>(),             // one face row of XDim datums per Y
        cfg::set<td.ZDim, S, St.num_faces>()); // faces per tile

    using r2 = cfg::ThconReg2Fields;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r2::Out_data_format, S, format_id(St.reg_format)>(),
        cfg::set<r2::Throttle_mode, S, unpack_throttle_mode>(),
        cfg::set<r2::Context_count, S, 0>(), // contexts are selected explicitly (unpack_set_context), not auto-incremented
        cfg::set<r2::Haloize_mode, S, St.transpose ? 1u : 0u>(),
        cfg::set<r2::Tileize_mode, S, 0>(),            // tilized input: no row-major tile making
        cfg::set<r2::Unpack_Src_Reg_Set_Upd, S, 0>(),  // an UNPACR that keeps the bank does not move the Src row on
        cfg::set<r2::Unpack_If_Sel, S, 0>(),           // unused: Unpack_if_sel_cntx* select Src or Dest per context
        cfg::set<r2::Upsample_rate, S, 0>(),           // no upsampling
        cfg::set<r2::Ovrd_data_format, S, 0>(),        // formats from the descriptor and Out_data_format, not per context
        cfg::set<r2::Upsample_and_interleave, S, 0>(), // no upsampling, so nothing to interleave
        // The per-context fields: the unpackers only ever work from contexts 0 and 1 (unpack_set_context,
        // Context_count 0), so those two are programmed and no UNPACR reads the fields of contexts 2..7.
        cfg::set<r2::Shift_amount_cntx0, S, 0>(), // no left shift into Src
        cfg::set<r2::Shift_amount_cntx1, S, 0>(),
        cfg::set<r2::Disable_zero_compress_cntx0, S, 1>(), // zero compression off
        cfg::set<r2::Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<r2::Unpack_if_sel_cntx0, S, St.to_dest ? 1u : 0u>(), // write Dest (1) or Src (0), contexts 0 and 1
        cfg::set<r2::Unpack_if_sel_cntx1, S, St.to_dest ? 1u : 0u>(),
        cfg::set<r2::Force_shared_exp, S, 0>()); // exponents come from L1, not a fixed shared one
}

/// Configure, step 5: the unpackers' own config words: a Dest stream's output stride, descriptors and configs, the
/// Src write base and unpacker 0's face size, for both contexts. An UNPACR takes all of them at its start, so they
/// follow the previous op's unpacks in stream order, with no wait.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_formats()
{
    // Output strides. An unpacker writes at its write base plus channel 1's Y, Z and W counters times these strides
    // (UNPACR_Regular.md: there is no X term). configure sets the three counters to 0 (unpack_configure_counters) and
    // only a Dest stream's UNPACRs move one of them (Z, on by one tile: unpack_tile_to_dest). So a Dest stream's Z
    // stride is written, and W's with it (the same word); every other stride multiplies a counter that stays 0 for
    // the whole op, whatever the register holds, and is not written: unpacker 1's, and unpacker 0's for a Src stream.
    using cfg::UnpackerReg;
    if constexpr (P.src_a.to_dest)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, unpack_dest_tile_stride(P.src_a)>(),
            cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    }

    // Descriptors and configs. Unpacker 0 takes its X dim from the per-context Tile_x_dim below, so 0 here.
    constexpr std::uint32_t src_a_face_datums = P.src_a.face_r_dim * face_cols;
    constexpr std::uint32_t src_b_face_datums = P.src_b.face_r_dim * face_cols;
    unpack_write_descriptor_and_config<unpacker0, P.src_a, 0>();
    unpack_write_descriptor_and_config<unpacker1, P.src_b, src_b_face_datums>();

    // Src write base for both contexts, and unpacker 0's face size per context.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, unpack_src_write_base>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, unpack_src_write_base>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0, src_a_face_datums>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0, src_a_face_datums>());
}

/// Selects config context Context for both unpackers.
template <std::uint32_t Context>
inline __attribute__((always_inline)) void unpack_select_context()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, Context>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, Context>());
}

/// Selects config context 0 or 1 for both unpackers. Configure's step 6 (the hardware's context offset = the
/// software's), and execute's flip after every run.
inline __attribute__((always_inline)) void unpack_set_context(std::uint32_t context)
{
    if (context == 0)
    {
        unpack_select_context<0>();
    }
    else
    {
        unpack_select_context<1>();
    }
}

/// Configure, part of step 7: SrcA bank select by address: off when unpacking to Dest (ISA: UNPACR to Dest with it
/// set is undefined). The unpacker reads this while it writes, not once per UNPACR (ISA, UNPACR to Dest: "at each
/// write").
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_src_a_set()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, P.src_a.to_dest ? 0u : 1u>();
}

/// Configure, part of step 7: the Src signedness (unsigned Src for UINT8), in the ALU word. The unpacker reads these
/// two bits while it writes, so unpack_configure_live_words issues them behind its wait for the unpackers, inside the
/// wait's scope. They are one-byte masked writes (RMWCIB), each atomic in the Configuration Unit, so they need no
/// mutex; a mutex acquire must stay out of the scope (platform::stall_scope).
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_signedness()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0, P.src_a.l1_format == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, P.src_b.l1_format == DataFormat::UInt8 ? 1u : 0u>());
}

/// Configure, step 7: the three words the unpackers read while data moves (SrcASet, SrcAUnsigned, SrcBUnsigned),
/// behind a wait for both unpackers and inside its scope, so a compiler that deletes the wait keeps it whenever one
/// of them changes ("What the unpackers read, and when").
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_live_words()
{
    platform::stall_scope<StallTarget::All, StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>(
        []
        {
            unpack_configure_src_a_set<P>();
            unpack_configure_signedness<P>();
        });
}

/// Configure, step 8: the words shared with other threads that an UNPACR takes at its start. ALU: no stochastic
/// rounding; FP8 E4M3 per unpacker (in the packers' THCON words). The read-modify-writes run under the RMW mutex.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure_shared_words()
{
    hal::sync::mutex::acquire<hal::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, unpacker0.section, P.src_a.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, unpacker1.section, P.src_b.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    hal::sync::mutex::release<hal::sync::Mutex::M0>();
}

/// Programs both unpackers for P through the instruction stream, completely. The body is the order of what is issued;
/// the order is this op's departure from the shape (file comment), and each step's comment says why it sits there.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure()
{
    static_assert(!P.src_a.active || (P.src_a.num_faces == faces_per_full_tile && P.src_a.face_r_dim == max_face_rows), "32x32 tiles only, for now");
    static_assert(!P.src_b.active || (P.src_b.num_faces == faces_per_full_tile && P.src_b.face_r_dim == max_face_rows), "32x32 tiles only, for now");
    static_assert(unpack_streams_a(P) || unpack_streams_b(P), "nothing streams: every op streams at least one operand");
    static_assert(P.src_a.per_tile <= unpack_gpr_half_max && P.src_b.per_tile <= unpack_gpr_half_max, "address step does not fit a GPR half");

    // Not config, no wait: taken in order behind the previous op's instructions.
    unpack_configure_replay<P>();   // 1. the streamed operands' per-tile bodies
    unpack_configure_counters<P>(); // 2. the address counters
    unpack_configure_gprs<P>();     // 3. the address steps the bodies add; AddDestAddrCntr's word
    // Config an UNPACR takes at its start: stream order behind the previous op's unpacks is enough.
    unpack_configure_add_dest_addr_cntr(); // 4. AddDestAddrCntr, behind a wait for the scalar unit only
    unpack_configure_formats<P>();         // 5. a Dest stream's stride, descriptors, formats, write base, face size
    unpack_set_context(platform::unpack_config_context()); // 6. the hardware's context offset = the software's
    // Config the unpackers read while data moves: behind a wait for both of them, inside its scope.
    unpack_configure_live_words<P>(); // 7. SrcASet, SrcAUnsigned, SrcBUnsigned
    // The shared words last and outside the scope, so the wait above does not hold the mutex acquire: the wait still
    // gates it in the hardware (it blocks every class until the unpackers are idle), so the mutex is never held
    // across the wait, and a compiler can drop the wait and the mutex pair independently.
    unpack_configure_shared_words<P>(); // 8. ALU rounding, Unp_LF8_4b_exp, under the RMW mutex
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// One run's streamed tiles from the replay: context Context's per-tile slots, once per tile.
template <unpack_params P, std::uint32_t Context, class Tiles>
inline __attribute__((always_inline)) void unpack_replay_streamed_tiles(const Tiles& tiles_per_run)
{
    for (std::uint32_t tile = 0; tile < wrapper::value(tiles_per_run); tile++)
    {
        hal::replay::run<unpack_tile_slots(P, Context)>();
    }
}

/// One run's streamed tiles after unpack_configure: by counter, or the current context's per-tile slots.
template <unpack_params P, class Tiles>
inline __attribute__((always_inline)) void unpack_play_streamed_tiles(const Tiles& tiles_per_run)
{
    if constexpr (unpack_counter_step_walk(P))
    {
        // One UNPACR per streamed operand and tile, the same in both contexts; SrcA's first, as the replay bodies.
        for (std::uint32_t tile = 0; tile < wrapper::value(tiles_per_run); tile++)
        {
            if constexpr (unpack_streams_a(P))
            {
                unpack_tile_and_step<unpacker0, P.src_a>();
            }
            if constexpr (unpack_streams_b(P))
            {
                unpack_tile_and_step<unpacker1, P.src_b>();
            }
        }
    }
    else if (platform::unpack_config_context() == 0)
    {
        unpack_replay_streamed_tiles<P, 0>(tiles_per_run);
    }
    else
    {
        unpack_replay_streamed_tiles<P, 1>(tiles_per_run);
    }
}

/// Behind a run of a counter-stepped walk: the moving streams' input Y and Z back to 0, so the next run (and the next
/// op) starts at its base address.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_rewind_stream_counters()
{
    using namespace hal::counters::adc;
    constexpr AddrCounter rewind {.channel0 = {.y = 0, .z = 0}};
    if constexpr (unpack_counter_moves(P, unpacker0) && unpack_counter_moves(P, unpacker1))
    {
        hal::counters::set<rewind.for_clients<unpacker0.counters, unpacker1.counters>()>();
    }
    else if constexpr (unpack_counter_moves(P, unpacker0))
    {
        hal::counters::set<rewind.for_clients<unpacker0.counters>()>();
    }
    else if constexpr (unpack_counter_moves(P, unpacker1))
    {
        hal::counters::set<rewind.for_clients<unpacker1.counters>()>();
    }
}

/// Writes context Context's base addresses (16-byte units) for the active streams.
template <unpack_params P, std::uint32_t Context>
inline __attribute__((always_inline)) void unpack_set_context_base_addresses(std::uint32_t src_a_base, std::uint32_t src_b_base)
{
    if constexpr (P.src_a.active)
    {
        cfg::write<cfg::Access::MMIO, unpack_base_address_field(Context), unpacker0.section>(src_a_base);
    }
    if constexpr (P.src_b.active)
    {
        cfg::write<cfg::Access::MMIO, unpack_base_address_field(Context), unpacker1.section>(src_b_base);
    }
}

/// Writes the current context's base addresses (16-byte units) for the active streams.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_set_base_addresses(std::uint32_t src_a_base, std::uint32_t src_b_base)
{
    if (platform::unpack_config_context() == 0)
    {
        unpack_set_context_base_addresses<P, 0>(src_a_base, src_b_base);
    }
    else
    {
        unpack_set_context_base_addresses<P, 1>(src_a_base, src_b_base);
    }
}

/// One step of the walk: `runs` runs of tiles_per_run streamed tiles. src_a_address and src_b_address are the byte
/// addresses of each operand's first tile.
template <unpack_params P, class Runs, class Tiles>
inline __attribute__((always_inline)) void unpack_step(
    std::uint32_t src_a_address, std::uint32_t src_b_address, std::uint32_t step, const Runs& runs, const Tiles& tiles_per_run)
{
    const std::uint32_t src_a_step_base = platform::l1::word_address(src_a_address) + step * P.src_a.per_step;
    const std::uint32_t src_b_step_base = platform::l1::word_address(src_b_address) + step * P.src_b.per_step;
    for (std::uint32_t run = 0; run < wrapper::value(runs); run++)
    {
        const std::uint32_t src_a_base = src_a_step_base + run * P.src_a.per_run;
        const std::uint32_t src_b_base = src_b_step_base + run * P.src_b.per_run;
        if constexpr (P.src_a.active)
        {
            LLK_ASSERT(platform::l1::is_valid_word_address(src_a_base), "unpacker 0's address (the SrcA or Dest operand) is outside L1");
        }
        if constexpr (P.src_b.active)
        {
            LLK_ASSERT(platform::l1::is_valid_word_address(src_b_base), "unpacker 1's address (the SrcB operand) is outside L1");
        }

        // Take a free context and point the unpackers at this run's tiles. The STALLWAIT keeps the unpackers from
        // starting before the RISC-V's config writes have landed.
        while (hal::sync::semaphore::read<unpack_sync>() >= unpack_max_contexts_in_flight)
        {
        }
        unpack_set_base_addresses<P>(src_a_base, src_b_base);
        hal::sync::semaphore::post<hal::sync::Access::MMIO, unpack_sync>();
        hal::sync::wait::stall<StallTarget::Unpack, StallCondition::RiscvAccessProcessed>();

        if constexpr (unpack_held_is_a(P))
        {
            unpack_tile<unpacker0, P.src_a>();
        }
        if constexpr (unpack_held_is_b(P))
        {
            unpack_tile<unpacker1, P.src_b>();
        }
        unpack_play_streamed_tiles<P>(tiles_per_run);
        if constexpr (unpack_counter_step_walk(P))
        {
            unpack_rewind_stream_counters<P>(); // behind the run's last UNPACR: the next run starts at its base address
        }

        // The SEMGET passes the Wait Gate behind the run's UNPACRs, each of which has taken its base address by
        // then: from here on the context may be written again (the UNPACRs may still be moving data).
        hal::sync::semaphore::get<unpack_sync>();
        platform::unpack_config_context() = 1 - platform::unpack_config_context();
        unpack_set_context(platform::unpack_config_context());
    }
}

// Unpack to Dest: the hand-off with math.

/// Before the walk of a Dest stream: wait until math has taken the previous call's tiles, take the Dest row math
/// sends through the mailbox (blocks until it arrives), and point both contexts' output at it: through the
/// instruction stream, behind a wait for unpacker 0, because the previous op's unpacks read this word.
inline __attribute__((always_inline)) void unpack_dest_write_begin()
{
    hal::sync::wait::semaphore<StallTarget::Unpack, hal::sync::SemaphoreCondition::WhileMaximum, unpack_to_dest_sync>();
    // The row is absolute (math adds its Dest section base).
    const std::uint32_t row     = platform::mailbox::read(platform::mailbox::Thread::Math);
    const std::uint32_t address = unpack_dest_write_address(row);
    platform::stall_scope<StallTarget::Config, StallCondition::Unpacker0Idle>(
        [address]
        {
            cfg::write<cfg::Access::TensixCfgUnit>(
                cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0>(address),
                cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0>(address));
        });
}

/// After the walk of a Dest stream: once unpacker 0 is idle, every tile is in Dest; tell math.
inline __attribute__((always_inline)) void unpack_dest_write_end()
{
    hal::sync::wait::stall<StallTarget::Sync, StallCondition::Unpacker0Idle>();
    hal::sync::semaphore::post<hal::sync::Access::Tensix, unpack_to_dest_sync>();
}

/// Every step of the walk; around it, for a Dest stream, the hand-off with math. src_a_address and src_b_address are
/// the byte addresses of each operand's first tile. steps, runs and tiles_per_run are the walk's loop counts: a
/// ct each (P.walk's number), or an rt of at most that number for a count known only at run time. Every
/// compile-time choice (how a streamed tile steps, the replay bodies) was made on P.walk's numbers, which a run-time
/// count does not exceed.
template <unpack_params P, class Steps, class Runs, class Tiles, ctrt<Steps, std::uint32_t> = 0, ctrt<Runs, std::uint32_t> = 0, ctrt<Tiles, std::uint32_t> = 0>
inline __attribute__((always_inline)) void unpack_execute(
    std::uint32_t src_a_address, std::uint32_t src_b_address, const Steps& steps, const Runs& runs, const Tiles& tiles_per_run)
{
    if constexpr (wrapper::is_rt_v<Steps> || wrapper::is_rt_v<Runs> || wrapper::is_rt_v<Tiles>)
    {
        LLK_ASSERT(wrapper::value(steps) <= P.walk.steps, "the run-time step count is above its bound");
        LLK_ASSERT(wrapper::value(runs) <= P.walk.runs, "the run-time run count is above its bound");
        LLK_ASSERT(wrapper::value(tiles_per_run) <= P.walk.tiles_per_run, "the run-time count of tiles per run is above its bound");
    }
    if constexpr (P.src_a.to_dest)
    {
        unpack_dest_write_begin();
    }
    for (std::uint32_t step = 0; step < wrapper::value(steps); step++)
    {
        unpack_step<P>(src_a_address, src_b_address, step, runs, tiles_per_run);
    }
    if constexpr (P.src_a.to_dest)
    {
        unpack_dest_write_end();
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The unpack thread's part of a Compute API call. in0_address and in1_address are the byte addresses of the Plan's
/// in0 and in1; an input the Plan does not unpack (copy's in1; a Dest operand) has none, and its argument is not read.
/// counts is the run-time side of the Plan's Work: the walk's loops run to the counts it holds (an empty object when
/// every count is compile time).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void unpack(const WorkCounts<Pl.work>& counts, std::uint32_t in0_address, std::uint32_t in1_address = 0)
{
    static_assert(On == Thread::Unpack, "the unpackers are driven from the unpack thread");
    constexpr unpack_params params = unpack_params_from(Pl);
    // Every TRISC compiles this call; only the unpack thread's build keeps its body (llk_lib/thread.h, runs_on).
    if constexpr (runs_on<On>)
    {
        // Each unpacker's address is its input's; an unpacker that takes no input never reads its address.
        const std::uint32_t src_a_address = unpack_input_of(Pl, unpacker0) == 1 ? in1_address : in0_address;
        const std::uint32_t src_b_address = unpack_input_of(Pl, unpacker1) == 1 ? in1_address : in0_address;
        unpack_configure<params>();
        unpack_execute<params>(
            src_a_address,
            src_b_address,
            plan_count<Pl.walk.steps_of, Pl.walk.steps>(counts),
            plan_count<Pl.walk.runs_of, Pl.walk.runs>(counts),
            plan_count<Pl.walk.tiles_of, Pl.walk.tiles_per_run>(counts));
    }
}

} // namespace llk2::llk_lib
