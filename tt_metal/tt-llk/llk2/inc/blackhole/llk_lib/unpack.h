// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, unpack thread, Blackhole. One implementation for every op, written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/).
//
// llk-lib is the layer under the Compute API: it splits a Compute API call into per-thread work and, on each
// thread, programs the hardware for it. Unpack does not know which op it serves. The Resolver's Plan describes the
// work as data, and this file executes that description:
//
//   streams  one per unpacker in use. Unpacker 0 writes SrcA (configured through THCON SEC0), unpacker 1 writes
//            SrcB (SEC1). A stream says which formats to convert between, the tile geometry, and how its L1
//            address moves: per streamed tile, per run, per step
//   walk     the loop nest over those streams (see UnpackWalk in resolver/plan.h): steps x runs x tiles_per_run,
//            with an optional held stream unpacked once per run
//
// What used to be four LLK families maps onto this as data: unpack_A is one stream, unpack_AB two streams with
// nothing held, unpack_AB_matmul two streams with the reused operand held, and tilize a row-major stream.
//
// The calls, in the order they run:
//   unpack_params_from(Plan)        Plan -> unpack_params (compile time)
//   unpack_configure<P>()           programs both unpackers completely: formats, strides, descriptors, the GPRs
//                                   holding the address steps, and a replay + MOP that unpacks every streamed
//                                   operand once and moves its address on
//   unpack_execute<P>(a, b, step)   one step of the walk: for each run, point the unpackers at that run's tiles,
//                                   unpack the held operand, and run the MOP tiles_per_run times
//   unpack<On, Plan>(in0[, in1])    the per-thread call the Compute API makes: plans on every thread, configures
//                                   and executes only when this build is for thread On (the unpack thread); one
//                                   address for a one-input op
//
// Unpack to Dest. A stream whose Plan says Where::Dest goes through unpacker 0 to Dest instead of SrcA (32-bit
// formats with a 32-bit Dest; see unpack_tile_to_dest and dest_write_begin below). Its tiles land one after another
// from a Dest row the math thread names, so unpack and math hand off around the whole call (dest_write_begin /
// dest_write_end):
//   math    waits until its earlier Dest writes are done, then sends the absolute Dest row through the mailbox
//   unpack  takes the row, points both contexts at it, unpacks every tile (each UNPACR moves its Dest address on by
//           one tile through the channel 1 Z counter), then posts semaphore UNPACK_TO_DEST once the unpacker is idle
//   math    waits for UNPACK_TO_DEST and takes it; Dest now holds the tiles
// No Src bank changes hands on the way to Dest: those UNPACRs keep the bank (no data valid).
//
// Transpose. A stream with `transpose` set is transposed inside each 16x16 face by unpacker 0 (Haloize_mode). The
// face order is the consumer's to fix (copy's MOVA2Ds; matmul transpose is not ported). Unpacker 0 cannot transpose
// on the way to Dest.
//
// Two config contexts: each unpacker keeps its L1 base address in a config register per context. The unpack thread
// writes the next run's addresses into one context while the unpackers still work from the other, and flips
// between them after every run. Semaphore UNPACK_SYNC counts the contexts handed to the unpackers and not yet
// finished; at most two may be in flight.
//
// configure assumes nothing about what ran before; removing writes an earlier op already made is left to the
// compiler (config folding). There is no init, uninit or reconfig.
//
// Not HAL yet: the format helpers (masked_data_format through reg_fmt, llk_lib/formats.h; datum_size_in_bytes and
// canonical_unpA_y_stride, from cunpack_common.h), the software state ckernel keeps (unp_cfg_context), the thread
// mailboxes (mailbox_read, ckernel.h) the unpack-to-Dest handoff reads the Dest row from, L1_ADDRESS and
// is_valid_L1_address (llk_memory_checks.h) with LLK_ASSERT, and the GPR indices p_gpr_unpack::TMP0, TILE_SIZE_A and
// TILE_SIZE_B (ckernel_gpr_map.h).

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_lib/formats.h"
#include "llk_lib/semaphores.h"
#include "llk_lib/thread.h"
#include "llk_memory_checks.h"
#include "resolver/plan.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- constants -----------------------------------------------------------------------------------------------------

inline constexpr std::uint32_t max_contexts_in_flight = 2;      ///< two config contexts
inline constexpr std::uint32_t mop_zmask_context0     = 0;      ///< MOP zmask that selects the plain (context 0) op on every iteration
inline constexpr std::uint32_t mop_zmask_context1     = 0xff;   ///< MOP zmask that selects the shadow (context 1) op on every iteration
inline constexpr std::uint32_t gpr_half_mask          = 0xffff; ///< scalar::set_low writes the low 16 bits of a GPR
inline constexpr std::uint32_t unpacker_throttle_mode = 2;      ///< Throttle_mode 2: the 512-bit L1 read interface, as configure_unpack_AB
/// Rows UNPACR skips: it takes its output row from the output address, then subtracts 4 before writing Src or Dest
/// (tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md, functional model, "Row -= 4").
inline constexpr std::uint32_t unpacr_skipped_rows = 4;
/// The output address (in datums) that lands on row 0 of Src or Dest, both contexts; the value configure_unpack_AB uses.
inline constexpr std::uint32_t src_write_base        = unpacr_skipped_rows * dest_row_datums;
inline constexpr std::uint32_t replay_ops_per_stream = 6; ///< UNPACR, RDCFG, ADDDMAREG, STALLWAIT, WRCFG, NOP (record_streamed_tile)
inline constexpr std::uint32_t dest_row_bytes_32bit  = dest_row_datums * sizeof(std::uint32_t); ///< one Dest row of 32-bit datums

// ---- parameters ----------------------------------------------------------------------------------------------------

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
    std::uint32_t tile_step;  ///< address step after each streamed tile, inside a run
    std::uint32_t run_step;   ///< address step from one run to the next
    std::uint32_t step_step;  ///< address step from one walk step to the next
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
constexpr unpack_stream stream_from(const UnpackPlan& u)
{
    return {
        u.to != Where::None,
        u.reads,
        u.writes,
        u.face_r_dim,
        u.num_faces,
        u.tile_bytes / l1_word_bytes,
        u.layout,
        u.broadcast,
        u.transpose_faces,
        u.tile_step * (u.tile_bytes / l1_word_bytes),
        u.run_step * (u.tile_bytes / l1_word_bytes),
        u.step_step * (u.tile_bytes / l1_word_bytes),
        u.to == Where::Dest,
    };
}

/// Unpacker 0 writes SrcA or Dest; unpacker 1 writes SrcB.
constexpr bool on_unpacker0(Where w)
{
    return w == Where::SrcA || w == Where::Dest;
}

/// The Plan of any op as unpack parameters: each operand goes to the unpacker its Plan names.
constexpr unpack_params unpack_params_from(const Plan& p)
{
    const UnpackPlan none {};
    const UnpackPlan& a = on_unpacker0(p.in0.to) ? p.in0 : on_unpacker0(p.in1.to) ? p.in1 : none;
    const UnpackPlan& b = p.in0.to == Where::SrcB ? p.in0 : p.in1.to == Where::SrcB ? p.in1 : none;
    return {stream_from(a), stream_from(b), p.walk};
}

/// SrcA's operand is the held one (unpacked once per run).
constexpr bool held_is_a(const unpack_params& p)
{
    return p.walk.held == Where::SrcA;
}

/// SrcB's operand is the held one (unpacked once per run).
constexpr bool held_is_b(const unpack_params& p)
{
    return p.walk.held == Where::SrcB;
}

/// Unpacker 0 is in use and streams (its operand is unpacked every MOP iteration).
constexpr bool streams_a(const unpack_params& p)
{
    return p.src_a.active && !held_is_a(p);
}

/// Unpacker 1 is in use and streams (its operand is unpacked every MOP iteration).
constexpr bool streams_b(const unpack_params& p)
{
    return p.src_b.active && !held_is_b(p);
}

/// Replay slots one context's MOP body takes: replay_ops_per_stream per streamed operand.
constexpr std::uint32_t replay_ops_per_context(const unpack_params& p)
{
    return replay_ops_per_stream * ((streams_a(p) ? 1 : 0) + (streams_b(p) ? 1 : 0));
}

/// Faces of a tile the unpacker reads. A row broadcast needs only the top face row, which holds the tile's row 0
/// (faces 0 and 1 of a 32x32 tile); every other stream reads all faces.
constexpr std::uint32_t unpacked_faces(const unpack_stream& s)
{
    return s.broadcast == Broadcast::Row ? faces_per_face_row : s.num_faces;
}

/// The last datum index of a tile along X, for the unpacker's X counter: every face read, every row, face_cols
/// datums per row. 0 for an unused unpacker.
constexpr std::uint32_t x_end(const unpack_stream& s)
{
    return s.active ? unpacked_faces(s) * s.face_r_dim * face_cols - 1 : 0;
}

/// Bytes one face takes in Src: how far the unpacker's Src (ch1) address moves per Z step.
constexpr std::uint32_t src_face_stride(const unpack_stream& s)
{
    return face_cols * max_face_rows * ckernel::datum_size_in_bytes(reg_fmt(s.reg_format));
}

/// Bytes one tile takes in a 32-bit Dest: how far a Dest stream's output (ch1) address moves per Z step, so the
/// next UNPACR writes the next tile.
constexpr std::uint32_t dest_tile_stride(const unpack_stream& s)
{
    return s.num_faces * s.face_r_dim * dest_row_bytes_32bit;
}

/// The channel 1 Z stride of a stream: a tile in Dest (Dest streams step a tile per UNPACR), a face in Src.
constexpr std::uint32_t output_z_stride(const unpack_stream& s)
{
    return s.to_dest ? dest_tile_stride(s) : src_face_stride(s);
}

/// The unpacker's output address for Dest row `row`. The unpacker counts its output in datums, from the same
/// src_write_base offset it uses for Src (unpacr_skipped_rows), so a row is dest_row_datums further on.
constexpr std::uint32_t dest_write_address(std::uint32_t row)
{
    return src_write_base + row * dest_row_datums;
}

// ---- configure -----------------------------------------------------------------------------------------------------

/// Selects config context 0 or 1 for both unpackers.
inline __attribute__((always_inline)) void set_unpack_context(std::uint32_t context)
{
    if (context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}

/// One unpacker's tile descriptor (what it reads) and config (what it writes), one write each. S is the
/// unpacker's THCON section: S0 for unpacker 0, S1 for unpacker 1. Every field of the two config words is named,
/// so the write replaces them whole.
template <cfg::Sec S, unpack_stream St, std::uint32_t XDim>
inline __attribute__((always_inline)) void write_descriptor_and_config()
{
    static_assert(St.layout == Layout::Tilized, "row-major (tilize) streams are not ported yet");
    static_assert(!St.to_dest || S == cfg::Sec::S0, "only unpacker 0 writes Dest");
    static_assert(!(St.to_dest && St.transpose), "the unpacker cannot transpose on the way to Dest (ISA: undefined)");
    static_assert(!St.to_dest || datum_bits(St.reg_format) == 32, "only 32-bit formats are unpacked to Dest (dest_tile_stride assumes them)");
    constexpr auto& td = cfg::Thcon[cfg::Reg0].TileDescriptor;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<td.InDataFormat, S, reg_fmt(St.l1_format)>(),
        cfg::set<td.Uncompressed, S, 1>(),     // tiles in L1 are not zero-compressed
        cfg::set<td.BlobsPerXyPlane, S, 0>(),  // no blobs: compression is off
        cfg::set<td.XDim, S, XDim>(),          // datums per face; unpacker 0 takes it per context instead (Tile_x_dim)
        cfg::set<td.YDim, S, 1>(),             // one face row of XDim datums per Y
        cfg::set<td.ZDim, S, St.num_faces>()); // faces per tile

    using r2 = cfg::ThconReg2Fields;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<r2::Out_data_format, S, reg_fmt(St.reg_format)>(),
        cfg::set<r2::Throttle_mode, S, unpacker_throttle_mode>(),
        cfg::set<r2::Context_count, S, 0>(), // contexts are selected explicitly (set_unpack_context), not auto-incremented
        cfg::set<r2::Haloize_mode, S, St.transpose ? 1u : 0u>(),
        cfg::set<r2::Tileize_mode, S, 0>(),            // tilized input: no row-major tile making
        cfg::set<r2::Unpack_Src_Reg_Set_Upd, S, 0>(),  // an UNPACR that keeps the bank does not move the Src row on
        cfg::set<r2::Unpack_If_Sel, S, 0>(),           // unused: Unpack_if_sel_cntx* select Src or Dest per context
        cfg::set<r2::Upsample_rate, S, 0>(),           // no upsampling
        cfg::set<r2::Ovrd_data_format, S, 0>(),        // formats from the descriptor and Out_data_format, not per context
        cfg::set<r2::Upsample_and_interleave, S, 0>(), // no upsampling, so nothing to interleave
        cfg::set<r2::Shift_amount_cntx0, S, 0>(),      // no left shift into Src, any context (cntx0..3)
        cfg::set<r2::Shift_amount_cntx1, S, 0>(),
        cfg::set<r2::Shift_amount_cntx2, S, 0>(),
        cfg::set<r2::Shift_amount_cntx3, S, 0>(),
        cfg::set<r2::Disable_zero_compress_cntx0, S, 1>(), // zero compression off in every context (cntx0..7)
        cfg::set<r2::Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<r2::Unpack_if_sel_cntx0, S, St.to_dest ? 1u : 0u>(), // write Dest (1) or Src (0), contexts 0 and 1
        cfg::set<r2::Unpack_if_sel_cntx1, S, St.to_dest ? 1u : 0u>(),
        cfg::set<r2::Force_shared_exp, S, 0>()); // exponents come from L1, not a fixed shared one
}

/// Unpacks one tile of a stream to Dest: one UNPACR for the whole tile, then the channel 1 Z counter moves the
/// output on by one Dest tile (output_z_stride). No bank is handed to math: flipping SrcA after an UNPACR to Dest
/// is a hardware bug (ISA: SrcASrcB.md, Unpackers[0].WroteDst), and math does not read SrcA here.
template <hal::unpack::Engine E, unpack_stream St>
inline __attribute__((always_inline)) void unpack_tile_to_dest()
{
    static_assert(E == hal::unpack::Engine::Unpacker0, "only unpacker 0 writes Dest");
    constexpr std::uint8_t next_dest_tile = 1; // channel 1 Z += 1: one Dest tile further on
    hal::unpack::run<hal::unpack::DataTransfer {.engine = E, .increments = {.channel1 = {.z = next_dest_tile}}, .handoff = hal::unpack::SourceHandoff::Keep}>();
}

/// Unpacks one tile of a stream. Broadcast and face-by-face variants add their UNPACR sequences here.
///   None  one UNPACR covers the whole tile (the X counter runs to x_end): faces 0..3 land in Src rows 0..63.
///   Row   one UNPACR covers the top face row only (x_end stops after faces 0 and 1): Src rows 0..15 hold face 0 and
///         rows 16..31 face 1, so tile row 0 is Src rows 0 and 16. Math broadcasts those rows down every face.
///         tt-llk (llk_unpack_AB.h, BroadcastType::ROW) unpacks faces 0, 1, 0, 1 one at a time, one per SrcA face,
///         because its math takes one face per Src bank; here math takes a whole tile per bank, so faces 0 and 1 once
///         are enough.
/// Either way the bank is handed to math with the last datum.
template <hal::unpack::Engine E, unpack_stream St>
inline __attribute__((always_inline)) void unpack_tile()
{
    static_assert(St.broadcast == Broadcast::None || St.broadcast == Broadcast::Row, "column and scalar broadcast streams are not ported yet");
    static_assert(!(St.to_dest && St.broadcast != Broadcast::None), "unpack to Dest takes whole tiles: no broadcast");
    if constexpr (St.to_dest)
    {
        unpack_tile_to_dest<E, St>();
    }
    else
    {
        // One UNPACR covers the tile (the X counter runs to x_end), then hands the bank to math. With Haloize_mode
        // on, each 16-row face is transposed as it is written.
        hal::unpack::run<hal::unpack::DataTransfer {.engine = E, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid}>();
    }
}

/// Replay body for one streamed operand in one context: unpack a tile, then base address += its tile step.
/// StepGpr holds the step (set by configure).
template <hal::unpack::Engine E, unpack_stream St, cfg::Sec S, const cfg::Field& Base, std::uint32_t StepGpr>
inline __attribute__((always_inline)) void record_streamed_tile()
{
    using ckernel::p_gpr_unpack;
    unpack_tile<E, St>();
    cfg::read<cfg::Access::TensixCfgUnit, Base, S>(hal::gpr<p_gpr_unpack::TMP0>());
    hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<StepGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
    hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>(); // the add must land before WRCFG reads the GPR
    cfg::write<cfg::Access::TensixCfgUnit, Base, S, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<p_gpr_unpack::TMP0>());
}

/// One context's MOP body: every streamed operand, SrcA's first. Base is that context's base-address field.
template <unpack_params P, const cfg::Field& Base>
inline __attribute__((always_inline)) void record_context_body()
{
    using ckernel::p_gpr_unpack;
    if constexpr (streams_a(P))
    {
        record_streamed_tile<hal::unpack::Engine::Unpacker0, P.src_a, cfg::Sec::S0, Base, p_gpr_unpack::TILE_SIZE_A>();
    }
    if constexpr (streams_b(P))
    {
        record_streamed_tile<hal::unpack::Engine::Unpacker1, P.src_b, cfg::Sec::S1, Base, p_gpr_unpack::TILE_SIZE_B>();
    }
}

/// Programs both unpackers for P: counters, strides, formats, descriptors, contexts, then the MOP.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure()
{
    using namespace hal::counters::adc;
    using ckernel::p_gpr_unpack;
    static_assert(!P.src_a.active || (P.src_a.num_faces == faces_per_full_tile && P.src_a.face_r_dim == max_face_rows), "32x32 tiles only, for now");
    static_assert(!P.src_b.active || (P.src_b.num_faces == faces_per_full_tile && P.src_b.face_r_dim == max_face_rows), "32x32 tiles only, for now");
    static_assert(replay_ops_per_context(P) > 0, "nothing streams: every op streams at least one operand");

    // Wait until math has taken every context, then reset the unpackers' address counters.
    while (hal::sync::semaphore::read<unpack_sync>() > 0)
    {
    }
    hal::counters::set<AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();
    hal::counters::set<AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();
    // X runs over the whole tile, so one UNPACR unpacks it (channel 1 is the Src side).
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end(P.src_a)}}.for_clients<Client::Unpacker0>()>();
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end(P.src_b)}}.for_clients<Client::Unpacker1>()>();

    // Output strides: one face per Z step (one tile for a Dest stream; W unused); SrcA's Y stride for its register format.
    using cfg::UnpackerReg;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, output_z_stride(P.src_a)>(),
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, src_face_stride(P.src_b)>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    constexpr std::uint32_t src_a_y_stride = ckernel::unpacker::canonical_unpA_y_stride(reg_fmt(P.src_a.reg_format));
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Ystride, cfg::Sec::S0, src_a_y_stride>();

    // ALU: unsigned Src for UINT8, no stochastic rounding; FP8 E4M3 per unpacker. The word is shared with other
    // threads, so the read-modify-write runs under the RMW mutex.
    hal::sync::mutex::acquire<hal::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0, P.src_a.l1_format == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, P.src_b.l1_format == DataFormat::UInt8 ? 1u : 0u>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0, P.src_a.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1, P.src_b.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    hal::sync::mutex::release<hal::sync::Mutex::M0>();

    // Descriptors and configs. Unpacker 0 takes its X dim from the per-context Tile_x_dim below, so 0 here.
    constexpr std::uint32_t src_a_face_datums = P.src_a.face_r_dim * face_cols;
    constexpr std::uint32_t src_b_face_datums = P.src_b.face_r_dim * face_cols;
    write_descriptor_and_config<cfg::Sec::S0, P.src_a, 0>();
    write_descriptor_and_config<cfg::Sec::S1, P.src_b, src_b_face_datums>();

    // Src write base for both contexts, and unpacker 0's face size per context.
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, src_write_base>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, src_write_base>());
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0, src_a_face_datums>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0, src_a_face_datums>());

    // SrcA bank select by address (off when unpacking to Dest: ISA, UNPACR to Dest with it set is undefined), and
    // unpacker 0's Src address counter added to the write base.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, P.src_a.to_dest ? 0u : 1u>();
    // Written as the whole word (enable bit set, ForcedSharedExp 0): a field-wise write would read the register
    // back first, which ttsim does not support for this register.
    constexpr std::uint32_t add_dest_addr_cntr_word = 1u << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {add_dest_addr_cntr_word});

    unp_cfg_context = 0;
    set_unpack_context(0);

    // The streamed operands' address steps, in the GPRs the replay adds from.
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), P.src_a.tile_step & gpr_half_mask);
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), P.src_b.tile_step & gpr_half_mask);
    static_assert(P.src_a.tile_step <= gpr_half_mask && P.src_b.tile_step <= gpr_half_mask, "address step does not fit a GPR half");

    // The MOP body, once per context: replay slots [0, n) for context 0 and [n, 2n) for context 1.
    constexpr std::uint32_t n = replay_ops_per_context(P);
    constexpr hal::replay::BufferRange context0 {0, n}, context1 {n, n}, both {0, max_contexts_in_flight * n};
    hal::replay::record<both>(
        []
        {
            record_context_body<P, cfg::Thcon[cfg::Reg3].Base_address>();
            record_context_body<P, cfg::Thcon[cfg::Reg3].Base_cntx1_address>();
        });
    // MOP template 0: each iteration runs the current context's body (the zmask picks context 1's shadow op).
    hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template0> {
        .start_op = hal::replay::get_operation<context0>(), .start_op_shadow = hal::replay::get_operation<context1>()}>();
}

// ---- execute -------------------------------------------------------------------------------------------------------

/// Writes the current context's base addresses (16-byte units) for the active streams.
template <unpack_params P>
inline __attribute__((always_inline)) void set_base_addresses(std::uint32_t src_a_base, std::uint32_t src_b_base)
{
    if (unp_cfg_context == 0)
    {
        if constexpr (P.src_a.active)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(src_a_base);
        }
        if constexpr (P.src_b.active)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(src_b_base);
        }
    }
    else
    {
        if constexpr (P.src_a.active)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(src_a_base);
        }
        if constexpr (P.src_b.active)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S1>(src_b_base);
        }
    }
}

/// One step of the walk. src_a_address and src_b_address are the byte addresses of each operand's first tile.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_execute(std::uint32_t src_a_address, std::uint32_t src_b_address, std::uint32_t step)
{
    const std::uint32_t src_a_step_base = L1_ADDRESS(src_a_address) + step * P.src_a.step_step;
    const std::uint32_t src_b_step_base = L1_ADDRESS(src_b_address) + step * P.src_b.step_step;
    for (std::uint32_t run = 0; run < P.walk.runs; run++)
    {
        const std::uint32_t src_a_base = src_a_step_base + run * P.src_a.run_step;
        const std::uint32_t src_b_base = src_b_step_base + run * P.src_b.run_step;
        LLK_ASSERT(
            (!P.src_a.active || is_valid_L1_address(src_a_base)) && (!P.src_b.active || is_valid_L1_address(src_b_base)), "unpack address is outside L1");

        // Take a free context and point the unpackers at this run's tiles. The STALLWAIT keeps the unpackers from
        // starting before the RISC-V's config writes have landed.
        while (hal::sync::semaphore::read<unpack_sync>() >= max_contexts_in_flight)
        {
        }
        set_base_addresses<P>(src_a_base, src_b_base);
        hal::sync::semaphore::post<hal::sync::Access::MMIO, unpack_sync>();
        hal::sync::wait::stall<StallTarget::Unpack, StallCondition::RiscvAccessProcessed>();

        if constexpr (held_is_a(P))
        {
            unpack_tile<hal::unpack::Engine::Unpacker0, P.src_a>();
        }
        if constexpr (held_is_b(P))
        {
            unpack_tile<hal::unpack::Engine::Unpacker1, P.src_b>();
        }
        hal::mop::Runner<hal::mop::MopTemplate::Template0>::run(P.walk.tiles_per_run, unp_cfg_context == 0 ? mop_zmask_context0 : mop_zmask_context1);

        // The Tensix-side SEMGET runs after the unpacks above, so the context is freed only once they finish.
        hal::sync::semaphore::get<unpack_sync>();
        unp_cfg_context = 1 - unp_cfg_context;
        set_unpack_context(unp_cfg_context);
    }
}

// ---- unpack to Dest: the handoff with math ---------------------------------------------------------------------

/// Before the walk of a Dest stream: wait until math has taken the previous call's tiles, take the Dest row math
/// sends through the mailbox (blocks until it arrives), and point both contexts' output at it.
inline __attribute__((always_inline)) void dest_write_begin()
{
    hal::sync::wait::semaphore<StallTarget::Unpack, hal::sync::SemaphoreCondition::WhileMaximum, unpack_to_dest_sync>();
    // Not HAL: the thread mailboxes. The row is absolute (math adds its Dest section base).
    const std::uint32_t row     = ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
    const std::uint32_t address = dest_write_address(row);
    // Both contexts in one whole-word write: no read-back, which ttsim does not support for this register.
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0>(address), cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0>(address));
}

/// After the walk of a Dest stream: once unpacker 0 is idle, every tile is in Dest; tell math.
inline __attribute__((always_inline)) void dest_write_end()
{
    hal::sync::wait::stall<StallTarget::Sync, StallCondition::Unpacker0Idle>();
    hal::sync::semaphore::post<hal::sync::Access::Tensix, unpack_to_dest_sync>();
}

/// The unpack thread's part of a two-input Compute API call. in0_address and in1_address are the byte addresses of
/// the call's input tensors.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void unpack(std::uint32_t in0_address, std::uint32_t in1_address)
{
    static_assert(On == Thread::Unpack, "the unpackers are driven from the unpack thread");
    constexpr unpack_params p = unpack_params_from(Pl);
    // Every TRISC compiles this call (the static_assert checks the call names the right thread); only the unpack
    // thread's build keeps its body. On math and pack this branch is discarded, so their builds carry no unpack code.
    if constexpr (runs_on<On>)
    {
        const std::uint32_t src_a_address = on_unpacker0(Pl.in0.to) ? in0_address : in1_address;
        const std::uint32_t src_b_address = Pl.in0.to == Where::SrcB ? in0_address : in1_address;
        unpack_configure<p>();
        if constexpr (p.src_a.to_dest)
        {
            dest_write_begin();
        }
        for (std::uint32_t step = 0; step < p.walk.steps; step++)
        {
            unpack_execute<p>(src_a_address, src_b_address, step);
        }
        if constexpr (p.src_a.to_dest)
        {
            dest_write_end();
        }
    }
}

/// The unpack thread's part of a one-input Compute API call (copy): unpacker 1 idles, so there is no second address.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void unpack(std::uint32_t in0_address)
{
    static_assert(Pl.in1.to == Where::None, "this op's Plan has a second input: pass its address");
    unpack<On, Pl>(in0_address, in0_address /* read by no unpacker: unpacker 1 idles */);
}

} // namespace llk2::llk_lib
