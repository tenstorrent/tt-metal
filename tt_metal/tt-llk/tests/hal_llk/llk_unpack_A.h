// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_A.h. Same instruction sequence and access kinds as production,
// expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace llk_unpack_a_detail
{
template <EltwiseBinaryReuseDestType binary_reuse_dest>
constexpr std::uint32_t dest_reuse_dummy_unpack()
{
    static_assert(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB);
    constexpr std::uint32_t source = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? SrcA : SrcB;
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
    // (UNP_ZEROSRC, SET_DVALID, WAIT_LIKE_UNPACR) has no HAL descriptor, so the production encoding is kept.
    return TT_OP_UNPACR_NOP(source, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

} // namespace llk_unpack_a_detail

namespace hal_port::unpack_A
{
using hal::unpack::Engine;
using hal::unpack::SourceHandoff;
namespace adc = hal::counters::adc;

// UNPACR with thread-override context 0 and the given channel Z increments:
// TT_OP_UNPACR(Src, (ch1_z << 4) | ch0_z, 0, 0, 0, 1, SetDvalid, RAREFYB_DISABLE, 0, 0, 0, 0, 1).
constexpr std::uint32_t unpack(const Engine engine, const std::uint8_t ch0_z, const std::uint8_t ch1_z, const SourceHandoff handoff)
{
    return hal::unpack::get_operation(hal::unpack::DataTransfer {
        .engine     = engine,
        .increments = {.channel0 = {.z = ch0_z}, .channel1 = {.z = ch1_z}},
        .handoff    = handoff,
    });
}

inline constexpr std::uint32_t unpack_srca                            = unpack(Engine::Unpacker0, 1, 0, SourceHandoff::FlipAndSetDataValid);
inline constexpr std::uint32_t unpack_srca_to_dest                    = unpack(Engine::Unpacker0, 1, 1, SourceHandoff::Keep); // ch0/ch1 z_inc
inline constexpr std::uint32_t unpack_srca_to_dest_column             = unpack(Engine::Unpacker0, 2, 2, SourceHandoff::Keep); // ch0/ch1 z_inc
inline constexpr std::uint32_t unpack_srca_to_dest_transpose_of_faces = unpack(Engine::Unpacker0, 2, 1, SourceHandoff::Keep); // ch1_z+=1, ch0_z+=2
inline constexpr std::uint32_t unpack_srcb                            = unpack(Engine::Unpacker1, 1, 0, SourceHandoff::FlipAndSetDataValid);
inline constexpr std::uint32_t unpack_srcb_inc_z_0                    = unpack(Engine::Unpacker1, 0, 0, SourceHandoff::FlipAndSetDataValid);
// Transpose-of-faces SrcA unpacks: ch0_z += 2 (down a column of faces) or += 1 (2-face tile).
inline constexpr hal::unpack::DataTransfer unpack_srca_z2 {
    .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 2}}, .handoff = SourceHandoff::FlipAndSetDataValid};
inline constexpr hal::unpack::DataTransfer unpack_srca_z1 {
    .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 1}}, .handoff = SourceHandoff::FlipAndSetDataValid};

// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
// (UNP_ZEROSRC, SET_DVALID, WAIT_LIKE_UNPACR) has no HAL descriptor, so the production encoding is kept.
inline constexpr std::uint32_t unpack_srca_set_dvalid =
    TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
inline constexpr std::uint32_t unpack_srcb_set_dvalid =
    TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
// HAL-GAP: the clear-source UNPACR_NOP flavor (CLR_SRC_0 / CLR_SRC, WAIT_LIKE_UNPACR) has no HAL descriptor.
inline constexpr std::uint32_t clear_srca = TT_OP_UNPACR_NOP(
    SrcA,
    0 /*Stream_Id*/,
    0 /*Msg_Clr_Cnt*/,
    0 /*Set_Dvalid*/,
    0 /*Clr_to1_fmt_Ctrl*/,
    p_unpacr_nop::WAIT_LIKE_UNPACR,
    0 /*Bank_Clr_Ctrl*/,
    p_unpacr_nop::CLR_SRC_0,
    p_unpacr_nop::CLR_SRC);

// Single-counter sets: production's SETADCZW, emitted by the HAL planner as SETADC (one instruction either way).
// TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001): set srcA ch0_z = 1
inline constexpr std::uint32_t srca_set_z_1 = adc::AddrCounter {.channel0 = {.z = 1}}.for_clients<adc::Client::Unpacker0>().operation();
// TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001): set srcB ch0_z = 2
inline constexpr std::uint32_t srcb_set_z_2 = adc::AddrCounter {.channel0 = {.z = 2}}.for_clients<adc::Client::Unpacker1>().operation();
// TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001): set srcB ch0_z = 0
inline constexpr std::uint32_t srcb_clear_z = adc::AddrCounter {.channel0 = {.z = 0}}.for_clients<adc::Client::Unpacker1>().operation();
// TT_OP_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 1, 0b0101): srcA ch0_z = 1, ch1_z = 2
inline constexpr std::uint32_t srca_set_z_1_dest_z_2 =
    adc::AddrCounter {.channel0 = {.z = 1}, .channel1 = {.z = 2}}.for_clients<adc::Client::Unpacker0>().operation();

} // namespace hal_port::unpack_A

/**
 * @brief Whether @ref _llk_unpack_A_ programs the operand's L1 address into unpacker A (SEC0).
 *
 * Otherwise the address goes to unpacker B (SEC1): a broadcast or acc_to_dest operand's address goes
 * to unpacker B unless DEST is reused as SrcB or the operand is unpacked to DEST. For example,
 * acc_to_dest with DEST_TO_SRCA unpacks the operand to SrcB and moves DEST to SrcA.
 * This only selects the address register; the MOP decides which sources are actually unpacked.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @return true if the operand's L1 base address goes to unpacker A (SEC0), false if it goes to unpacker B (SEC1).
 */
template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
constexpr bool _llk_unpack_A_address_on_unpacker_A_()
{
    return ((BType == BroadcastType::NONE) && (!acc_to_dest)) || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB || unpack_to_dest;
}

/**
 * @brief Whether the unpack A configuration asserts check unpacker B (SEC1) rather than unpacker A (SEC0).
 *
 * Unpacker B is checked only when it both holds the operand's address (see
 * @ref _llk_unpack_A_address_on_unpacker_A_) and unpacks it from L1: broadcasts, and acc_to_dest with
 * DEST_TO_SRCA. acc_to_dest without DEST reuse also routes the address to unpacker B, but the
 * transpose-of-faces MOP that transpose_init configures with it unpacks only SrcA (SrcB is zeroed),
 * and its paired per-tile unpack uses acc_to_dest = false, so unpacker A is checked there.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @return true if the asserts should check unpacker B (SEC1), false for unpacker A (SEC0).
 */
template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
constexpr bool _llk_unpack_A_asserts_check_unpacker_B_()
{
    return !_llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>() &&
           ((BType != BroadcastType::NONE) || (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA));
}

/**
 * @brief Program the unpacker MOP for a single-operand (A) unpack.
 *
 * Selects the UNPACR instruction sequence based on broadcast type, dest-reuse mode and
 * whether data is unpacked straight to the dest register, covering transpose-of-faces and
 * 32-bit-to-dest paths.
 *
 * @note On the plain path, a face shorter than 16 rows zeroes the unpacker's own SrcA bank before each face,
 * because datacopy moves all 16 rows.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Whether faces are reordered (transposed) during the unpack.
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_mop_config_(
    const bool transpose_of_faces, const ckernel::TensorShape tensor_shape, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        !((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");
    static_assert(
        !(((acc_to_dest) || (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)) && (unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_A_mop_config_", tensor_shape);
    const std::uint8_t num_faces = tensor_shape.total_num_faces();

    using namespace hal::mop;
    using namespace hal_port::unpack_A;

    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        if (transpose_of_faces && num_faces == 4)
        {
            const std::uint32_t outerloop = 2;
            const std::uint32_t innerloop = 2;
            program(loop(outerloop, loop(innerloop, unpack_srca_to_dest_transpose_of_faces), srca_set_z_1_dest_z_2));
        }
        else if (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            constexpr std::uint32_t outerloop = BType == BroadcastType::ROW ? 2 : 1;
            constexpr std::uint32_t innerloop = 1;
            program(loop(outerloop, loop(innerloop, unpack_srca_to_dest)));
        }
        else if (BType == BroadcastType::COL)
        {
            constexpr std::uint32_t outerloop = 2;
            constexpr std::uint32_t innerloop = 1;
            program(loop(outerloop, loop(innerloop, unpack_srca_to_dest_column)));
        }
        else
        {
            const std::uint32_t outerloop     = num_faces;
            constexpr std::uint32_t innerloop = 1;
            program(loop(outerloop, loop(innerloop, unpack_srca_to_dest)));
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        if constexpr (acc_to_dest)
        {
            // The dummy SrcA publication is the same word whether or not dest is reused as SrcA.
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces, add support for dest to srcB
            program(loop(outerloop, unpack_srcb, loop(innerloop, unpack_srca_set_dvalid, unpack_srca_set_dvalid), srcb_set_z_2));
        }
        else
        {
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 1; // TODO: add support for num_faces
            program(loop(outerloop, unpack_srca_set_dvalid, loop(innerloop, unpack_srcb, srcb_set_z_2), unpack_srcb));
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        const std::uint32_t outerloop = tensor_shape.num_faces_r_dim;
        const std::uint32_t innerloop = tensor_shape.num_faces_c_dim;
        if constexpr (acc_to_dest)
        {
            // Publish one dummy SrcA DVALID per tensor face; the word is the same whether or not
            // dest is reused as SrcA.
            program(loop(outerloop, loop(innerloop, unpack_srcb, unpack_srca_set_dvalid), srcb_clear_z));
        }
        else
        {
            program(loop(outerloop, loop(innerloop, unpack_srcb), srcb_clear_z));
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = 1;
        program(loop(outerloop, unpack_srca_set_dvalid, loop(innerloop, unpack_srcb_inc_z_0)));
    }
    else
    {
        if (transpose_of_faces)
        {
            constexpr hal::replay::BufferRange replay_range {0 /*start*/, 2 /*count*/};
            hal::replay::record<replay_range>(
                [num_faces]
                {
                    TTI_INSN(unpack_srcb_set_dvalid);
                    if (num_faces > 2)
                    {
                        hal::unpack::run<unpack_srca_z2>(); // inc srcA ch0_z+=2
                    }
                    else
                    {
                        hal::unpack::run<unpack_srca_z1>(); // inc srcA ch0_z+=1
                    }
                });

            const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
            const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;
            // Unpack faces 0/2 && 1/3 to srcA, or 0/1 for 2 face tile. The end op is a NOP unless the tile has a
            // second column of faces, as production's set_end_op is conditional.
            program(loop(outerloop, loop(innerloop, hal::replay::get_operation<replay_range>()), num_faces > 2 ? srca_set_z_1 : TT_OP_NOP));
        }
        else
        {
            if constexpr (acc_to_dest)
            {
                // Use unpacker-bank readiness for destination-reuse dummy source publication so
                // unpack can prepare the next bank while math consumes the current bank.
                static constexpr std::uint32_t unpack_srca_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                        : unpack_srca;

                static constexpr std::uint32_t unpack_srcb_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCB>()
                        : unpack_srcb;

                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                program(loop(outerloop, loop(innerloop, unpack_srca_reuse, unpack_srcb_reuse)));
            }
            else
            {
                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                // Datacopy moves all 16 rows of a face. With a partial face, zero the unpacker's own SrcA bank before
                // each face, so the rows past the face hold zeros and not the previous op's data. One template either
                // way (production edits its template in place), so the function stays small enough to inline.
                const bool partial_face = tensor_shape.face_r_dim < FACE_R_DIM;
                program(loop(
                    outerloop,
                    partial_face ? clear_srca : unpack_srca,
                    partial_face ? loop(innerloop, unpack_srca, unpack_srcb_set_dvalid) : loop(innerloop, unpack_srcb_set_dvalid)));
            }
        }
    }
}

// _llk_unpack_A_init_ without the sync-mode request, for inits that reuse its state setup but keep their own
// two-context execute (_llk_unpack_fast_untilize_init_).
template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
inline void _llk_unpack_A_init_state_(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const ckernel::TensorShape tensor_shape,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_A_init_", tensor_shape);
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    LLK_ASSERT(BType != BroadcastType::COL || num_faces == 4, "Unary Broadcast Column requires num_faces == 4 (32x32 only)");
    LLK_ASSERT(transpose_of_faces == 0 || face_r_dim == 16, "Partial faces are not supported for transpose datacopy, face_r_dim must be 16 rows");
    LLK_ASSERT(transpose_of_faces == 0 || num_faces == 4 || num_faces == 1, "Transpose requires num_faces == 4 or 1 (32x32 and 16x16 only)");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");
    // The unpack-to-dest ROW MOP copies row 0 of faces 0 and 1 into dest faces 0 and 1. On a 32x16 tile
    // face 1 is the bottom face, so its half of the output takes the wrong row. Same MOP as Wormhole, where
    // the output is wrong; ttsim cannot run this path.
    LLK_ASSERT(
        !(BType == BroadcastType::ROW && should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format) &&
          is_tensor_shape_2x1_face_grid(tensor_shape)),
        "Row broadcast with unpack to dest does not support 32x16 tiles");

    namespace cfg = hal::cfg;

    // Set transpose register to prevent state pollution
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face_16x16_transpose);

    // x-start/x-end is per-unpacker state, so program it on exactly the unpacker(s) the MOP issues a
    // real (non-ZEROSRC) UNPACR against; a zeroed source does not read L1, so its X counter is unused.
    // The unpack-to-dest (SrcA) path is only taken when the input is actually 32-bit; otherwise the MOP
    // falls through to the normal/broadcast path, so gate on the shared should_unpack_to_dest() predicate
    // the MOP uses, so the two cannot diverge.
    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        // SrcA -> dest. ROW and SCALAR broadcast only unpack a single row; everything else a full face.
        if constexpr (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            hal_port::config_unpacker_x_end<p_setadc::UNP_A>(1);
        }
        else
        {
            hal_port::config_unpacker_x_end<p_setadc::UNP_A>(face_r_dim);
        }
    }
    else
    {
        //   plain datacopy            -> SrcA           (unpacker A)
        //   acc_to_dest, no reuse      -> SrcA and SrcB  (both)
        //   acc_to_dest DEST_TO_SRCA   -> SrcB only      (SrcA comes from DEST, e.g. hardswish x*hardsigmoid(x))
        //   acc_to_dest DEST_TO_SRCB   -> SrcA only      (SrcB comes from DEST)
        //   broadcast                  -> SrcB
        constexpr bool reads_srca       = (BType == BroadcastType::NONE) && !(acc_to_dest && binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA);
        constexpr bool reads_srcb       = (BType != BroadcastType::NONE) || (acc_to_dest && binary_reuse_dest != EltwiseBinaryReuseDestType::DEST_TO_SRCB);
        constexpr std::uint32_t UNP_SEL = (reads_srca && reads_srcb) ? p_setadc::UNP_AB : (reads_srca ? p_setadc::UNP_A : p_setadc::UNP_B);
        hal_port::config_unpacker_x_end<UNP_SEL>(face_r_dim);
    }

    _llk_unpack_A_mop_config_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, tensor_shape, unpack_src_format, unpack_dst_format);
}

// Push the sync mode _llk_unpack_A_ needs; every init it pairs with (_llk_unpack_A_init_, _llk_unpack_A_rmsnorm_init_)
// calls this first and hal_port::unpack_sync_mode_end on the result last, so the wait overlaps the init's setup.
template <bool unpack_to_dest>
inline bool _llk_unpack_A_sync_mode_begin_()
{
    if constexpr (!unpack_to_dest)
    {
        // _llk_unpack_A_ uses config context 0 only, ordered by Auto TTSync (AutoTTSync.md), as _llk_unpack_AB_init_.
        // Free when already on, so a re-init inside a loop costs no drain; _llk_unpack_A_uninit_ turns it off again.
        hal_port::reset_config_context();
        return hal_port::unpack_sync_mode_begin<hal_port::UnpackSyncMode::AutoTTSync>();
    }
    else
    {
        // The unpack-to-dest instantiation keeps the two-context handshake.
        return hal_port::unpack_sync_mode_begin<hal_port::UnpackSyncMode::Handshake>();
    }
}

/**
 * @brief Initialize the unpacker for a single-operand (A) unpack.
 *
 * Configures the within-face transpose register and per-unpacker datum count, then programs
 * the MOP for the requested broadcast/dest-reuse/unpack-to-dest mode.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Nonzero to reorder (transpose) faces during the unpack.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose (haloize mode).
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note On the plain path, a face shorter than 16 rows zeroes the unpacker's own SrcA bank before each face
 *       (see @ref _llk_unpack_A_mop_config_).
 * @note ROW broadcast with 32-bit unpack to dest does not support a 32x16 tile (2x1 face grid).
 * @note Without unpack_to_dest this turns Auto TTSync on (config context 0 only); call @ref _llk_unpack_A_uninit_
 *       as the matching teardown to turn it off again.
 * @ref _llk_unpack_A_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ is the matching init on the math thread (datacopy/transpose consumer).
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_init_(
    const std::uint32_t transpose_of_faces          = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const ckernel::TensorShape tensor_shape         = ckernel::DEFAULT_TENSOR_SHAPE,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    const bool sync_mode_pushed = _llk_unpack_A_sync_mode_begin_<unpack_to_dest>();
    _llk_unpack_A_init_state_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces, within_face_16x16_transpose, tensor_shape, unpack_src_format, unpack_dst_format);
    hal_port::unpack_sync_mode_end(sync_mode_pushed);
}

/**
 * @brief Teardown after single-operand (A) unpacking: turn the unpack thread's Auto TTSync off again.
 *
 * The unpacker x-start/x-end (datum-count) state is transient and reprogrammed by each operation's
 * init (see tt-llk#1036). Auto TTSync, enabled by @ref _llk_unpack_A_init_ without unpack_to_dest, is sticky per thread.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @note Call @ref _llk_unpack_A_init_ with matching template args before this function.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_A_uninit_()
{
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
}

/**
 * @brief Unpack a single tile (operand A) from L1 into the SrcA/SrcB or dest register.
 *
 * Programs the operand base address and runs the configured MOP. Without unpack_to_dest, and with Auto TTSync on
 * (@ref _llk_unpack_A_init_), it uses config context 0 only, ordered by Auto TTSync; otherwise (unpack_to_dest, or after
 * @ref _llk_unpack_A_sdpa_init_, which leaves the mode alone) the two-context semaphore handshake. When unpacking 32-bit datums to dest,
 * also manages the dest write address and completion handshake.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param address: L1 address of the source tile.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @note Call @ref _llk_unpack_A_init_ with matching template args before this function, and
 *       @ref _llk_unpack_A_uninit_ after it as the matching teardown.
 * @note With unpack_to_dest, a 32-bit input and BType != NONE, this also hands the math thread one SrcB bank
 *       per tile; only @ref _llk_math_eltwise_unary_datacopy_ with the same BType releases it.
 * @ref _llk_math_eltwise_unary_datacopy_ on the math thread consumes the tile unpacked here.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_(const std::uint32_t address, const std::uint32_t unpack_src_format = 0, const std::uint32_t unpack_dst_format = 0)
{
    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    // _llk_unpack_A_init_ (and the rmsnorm init) turn Auto TTSync on; _llk_unpack_A_sdpa_init_ leaves the mode as it is, so
    // its kernels keep the two-context handshake below and never pay for the mode switch.
    if (!unpack_to_dest && hal_port::unpack_sync_mode_state == hal_port::UnpackSyncMode::AutoTTSync) [[likely]]
    {
        volatile std::uint32_t tt_reg_ptr *cfg = hcfg::detail::state_cfg_bank();
        // Single config context with Auto TTSync: this base store stalls until the previous tile's MOP expansion has
        // passed the Wait Gate, where its UNPACRs have taken their base addresses (MemoryOrdering.md), and the first
        // UNPACR of this MOP waits until the store has landed (AutoTTSync.md). This replaces the busy-context semaphore,
        // the STALLWAIT on RISC CFG writes and the context flip. Whole-word store, as production.
        if constexpr (_llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>())
        {
            constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
            static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32, "unpacker A context-0 base word");
            cfg[base_cntx0] = address;
        }
        else
        {
            constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[1].Cntx[0].Base.addr32(hcfg::Sec::S0);
            static_assert(base_cntx0 == THCON_SEC1_REG3_Base_address_ADDR32, "unpacker B context-0 base word");
            cfg[base_cntx0] = address;
        }
        // Clear z/w start counters right before the MOP: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
        adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        return;
    }

    // Clear z/w start counters: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Program srcA and srcB base addresses
    // Production's get_cfg_pointer (RISC pointer select only), so the per-tile branch layout stays as production's.
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    hal_port::wait_for_next_context(2);

    // Set upk0/1 L1 read addr: whole-word MMIO stores, as production.
    if constexpr (_llk_unpack_A_address_on_unpacker_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>())
    {
        constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
        constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
        static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A base words");
        const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? base_cntx0 : base_cntx1;
        cfg[upk0_reg]                = address;
    }
    else
    {
        constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[1].Cntx[0].Base.addr32(hcfg::Sec::S0);
        constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[1].Cntx[1].Base.addr32(hcfg::Sec::S0);
        static_assert(base_cntx0 == THCON_SEC1_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC1_REG3_Base_cntx1_address_ADDR32, "unpacker B base words");
        const std::uint32_t upk1_reg = (unp_cfg_context == 0) ? base_cntx0 : base_cntx1;
        cfg[upk1_reg]                = address;
    }

    // Trisc::SEMPOST for context acquire
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    if constexpr (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            hal_port::set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            hal_port::wait_for_dest_available();
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // Run MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    if constexpr (unpack_to_dest && (BType != BroadcastType::NONE))
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            // The math side broadcasts the tile in DEST through SrcB (MOVD2B/MOVB2D), so hand it a SrcB bank.
            // SETDVALID from the math thread is unsupported on Blackhole.
            TTI_INSN(hal_port::unpack_A::unpack_srcb_set_dvalid); // HAL-GAP: zero-source + set-data-valid UNPACR_NOP
        }
    }

    if (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            hal_port::unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
        }
    }

    // T6::SEMGET for context release, after unpack_to_dest_tile_done's config writes so that
    // configure_unpack_AB's wait_for_idle also waits for them
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}
