// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h. Same Tensix instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "hal/sync.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

namespace hal_port
{
namespace transpose_dest
{
namespace move = hal::move;
using move::RowAddressing;

// Replay buffer ranges (production: lltt::replay / lltt::replay_insn with these start/length pairs).
inline constexpr hal::replay::BufferRange REPLAY_32B {ckernel::math::replay_buf_offset, 16};           // 32-bit sequence
inline constexpr hal::replay::BufferRange REPLAY_32B_LO {ckernel::math::replay_buf_offset, 8};         // MOVD2B lo x4, TRNSPSRCB, MOVB2A x3
inline constexpr hal::replay::BufferRange REPLAY_32B_HI {ckernel::math::replay_buf_offset + 8, 8};     // MOVD2B hi x4, TRNSPSRCB, MOVB2D x3
inline constexpr hal::replay::BufferRange REPLAY_32B_HI_D2B {ckernel::math::replay_buf_offset + 8, 5}; // MOVD2B hi x4, TRNSPSRCB
inline constexpr hal::replay::BufferRange REPLAY_32B_B2A {ckernel::math::replay_buf_offset + 5, 3};    // MOVB2A x3

// TTI_MOVD2B(Lo32 ? DEST_32B_LOW : DEST_NORM, SrcBRow, AddrMod, MOV_4_ROWS, DestRow)
template <bool Lo32, std::uint32_t SrcBRow, std::uint8_t AddrMod, std::uint32_t DestRow>
inline constexpr move::Transfer<move::Dst, move::SrcB> MOVD2B {
    .source_row = DestRow, .destination_row = SrcBRow, .address_mode = AddrMod, .number_of_rows = move::rows::Four, .source_32bit_low = Lo32};

template <bool Lo32, std::uint32_t SrcBRow, std::uint8_t AddrMod, std::uint32_t DestRow>
inline __attribute__((always_inline)) void movd2b()
{
    move::run<MOVD2B<Lo32, SrcBRow, AddrMod, DestRow>>();
}

// TTI_MOVB2D(DEST_NORM, SrcBRow, AddrMod, MOV_4_ROWS, DestRow)
template <std::uint32_t SrcBRow, std::uint8_t AddrMod, std::uint32_t DestRow, RowAddressing Addressing = RowAddressing::Absolute>
inline constexpr move::Transfer<move::SrcB, move::Dst> MOVB2D {
    .source_row = SrcBRow, .destination_row = DestRow, .destination_addressing = Addressing, .address_mode = AddrMod, .number_of_rows = move::rows::Four};

template <std::uint32_t SrcBRow, std::uint8_t AddrMod, std::uint32_t DestRow, RowAddressing Addressing = RowAddressing::Absolute>
inline __attribute__((always_inline)) void movb2d()
{
    move::run<MOVB2D<SrcBRow, AddrMod, DestRow, Addressing>>();
}

// TTI_MOVA2D(Lo32 ? DEST_32B_LOW : DEST_NORM, SrcARow, AddrMod, MOV_8_ROWS, DestRow)
template <bool Lo32, std::uint32_t SrcARow, std::uint8_t AddrMod, std::uint32_t DestRow, RowAddressing Addressing = RowAddressing::Absolute>
inline __attribute__((always_inline)) void mova2d()
{
    move::run<move::Transfer<move::SrcA, move::Dst> {
        .source_row             = SrcARow,
        .destination_row        = DestRow,
        .destination_addressing = Addressing,
        .address_mode           = AddrMod,
        .number_of_rows         = move::rows::Eight,
        .destination_32bit_low  = Lo32}>();
}

// TTI_TRNSPSRCB
inline __attribute__((always_inline)) void transpose_srcb()
{
    hal::src<hal::SrcB>.transpose();
}

} // namespace transpose_dest
} // namespace hal_port

// local function declarations
template <bool is_32bit>
inline void transpose_dest_configure_addrmod();
template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop();

/**
 * @brief Transpose a full 32-bit tile in place at dest index 0 (relative to the current bank) via format switching.
 *
 * Because the FPU transposes 16-bit datums, each 32-bit datum is transposed as two halves: the high 16 bits are
 * transposed and written first, then the cached low 16 bits, so the low bits don't clobber the high bits.
 * transpose_of_faces additionally transposes the face arrangement, not just elements within each face.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @note Toggles DISABLE_IMPLIED_SRCA_FMT and the SrcA format/FP32/zero-flag config bits, restoring them on exit.
 */
template <bool transpose_of_faces>
inline void transpose_dest_32b()
{
    using namespace hal_port::transpose_dest;
    using hal_port::set_disable_implied_srca_fmt;
    using hal_port::set_fp32_enabled;
    using hal_port::set_srca_format;
    using Mop = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    // Disable implied SrcA format inference so manual SrcA format switches take effect.
    // On Blackhole the ALU format is normally inferred; we must disable this for
    // the SrcA format switching approach to control MOVD2B/MOVB2D behavior.
    set_disable_implied_srca_fmt<1>();

    // Zero-flag PRESERVE for the hi16/lo16 MOV sequence is established by the caller
    // (_llk_math_transpose_dest_), covering both the 32-bit and 16-bit paths.

    // Transpose all the low 16 bit elements of all faces and put them in SrcA.
    // Eventually Dest=0, SrcA=0.
    set_srca_format<DataFormat::Float16_b>();
    Mop::run();

    if constexpr (!transpose_of_faces)
    {
        // Transpose Face 0 hi bits and put back in Dest register.
        set_srca_format<DataFormat::Tf32>();
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_0, 12>(); // Dest=16, SrcA=0

        // Transpose Face 1 hi bits and put back in Dest register.
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_0, 12>(); // Dest=32, SrcA=0

        // Transpose Face 2 hi bits and put back in Dest register.
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_0, 12>(); // Dest=48, SrcA=0

        // Transpose Face 3 hi bits and put back in Dest register.
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_3, 12>(); // Dest=0, SrcA=0

        // Since all high bits have been written. We can set the appropriate formats and
        // write all the transposed low bits to Dest register.
        set_fp32_enabled<0>();
        set_srca_format<DataFormat::Float32>();

        mova2d<true, 0, ADDR_MOD_1, 0>();
        mova2d<true, 8, ADDR_MOD_1, 8>();
        mova2d<true, 16, ADDR_MOD_1, 16>();
        mova2d<true, 24, ADDR_MOD_1, 24>();
        mova2d<true, 32, ADDR_MOD_1, 32>();
        mova2d<true, 40, ADDR_MOD_1, 40>();
        mova2d<true, 48, ADDR_MOD_1, 48>();
        mova2d<true, 56, ADDR_MOD_1, 56>();
    }
    else
    {
        // Transpose Face 0 hi bits and put back in Dest register.
        set_srca_format<DataFormat::Tf32>();
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_1, 12>(); // Dest=0, SrcA=0

        set_fp32_enabled<0>();
        set_srca_format<DataFormat::Float32>();

        // Write transposed Face 0 low bits to Dest, to make room for Face 1 high bits
        mova2d<true, 0, ADDR_MOD_1, 0>();
        mova2d<true, 8, ADDR_MOD_0, 8>(); // Dest=16, SrcA=0

        set_fp32_enabled<1>();

        // Transpose Face 1 hi bits and move to SrcA
        set_srca_format<DataFormat::Tf32>();
        hal::replay::run<REPLAY_32B_HI_D2B>();
        hal::replay::run<REPLAY_32B_B2A>();
        TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // Dest=32, SrcA=0  HAL-GAP: no HAL API for MOVB2A

        // Transpose Face 2 hi bits and move to Face 1's area in Dest
        hal::replay::run<REPLAY_32B_HI_D2B>();
        movb2d<16, ADDR_MOD_1, move::relative_row<-16>(), RowAddressing::CounterRelative>();
        movb2d<20, ADDR_MOD_1, move::relative_row<-12>(), RowAddressing::CounterRelative>();
        movb2d<24, ADDR_MOD_1, move::relative_row<-8>(), RowAddressing::CounterRelative>();
        movb2d<28, ADDR_MOD_1, move::relative_row<-4>(), RowAddressing::CounterRelative>(); // Dest=32, SrcA=0

        // Write transposed Face 1 hi bits and move to Face 2's area in Dest
        mova2d<false, 0, ADDR_MOD_1, 0>();
        mova2d<false, 8, ADDR_MOD_0, 8>(); // Dest=48, SrcA=0

        // Transpose Face 3 hi bits and put back in Dest register
        hal::replay::run<REPLAY_32B_HI>();
        movb2d<28, ADDR_MOD_3, 12>();

        // Since all high bits have been written. We can set the appropriate formats and
        // write all the low bits to Dest register.
        set_fp32_enabled<0>();
        set_srca_format<DataFormat::Float32>();

        // Write transposed Face 1 low bits to Face 2's area in Dest
        mova2d<true, 16, ADDR_MOD_1, 32>();
        mova2d<true, 24, ADDR_MOD_1, 40>();

        // Write transposed Face 2 low bits to Face 1's area in Dest
        mova2d<true, 32, ADDR_MOD_1, 16>();
        mova2d<true, 40, ADDR_MOD_1, 24>();

        // Write transposed Face 3 low bits to Face 3's area in Dest
        mova2d<true, 48, ADDR_MOD_1, 48>();
        mova2d<true, 56, ADDR_MOD_1, 56>();
    }

    // Restore config state
    set_disable_implied_srca_fmt<0>();
    set_fp32_enabled<1>();
}

// Notes on these template parameters:
// 1. <transpose_of_faces=false, is_32bit=false>: not supported.
// 2. <transpose_of_faces=false, is_32bit=true>: 4x 16x16 face transpose; can be combined with _llk_unpack_A_ with transpose_of_faces=true.
// 3. <transpose_of_faces=true, is_32bit=false>: the default case (full 32x32 tile transpose, non-32-bit).
// 4. <transpose_of_faces=true, is_32bit=true>: full 32x32 tile transpose for 32-bit.
//
// We may want to revisit these template parameters, and perhaps the
// transpose_dest API generally as it's not currently widely used:
// https://github.com/tenstorrent/tt-llk/issues/290
/**
 * @brief Transpose a tile in place within the destination register.
 *
 * For 32-bit datums uses the format-switching path (low/high 16-bit halves transposed separately to avoid
 * clobbering); otherwise runs the preconfigured transpose MOP. transpose_of_faces controls whether the face
 * arrangement is transposed in addition to the elements within each face.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (uses the format-switching transpose path).
 * @param dst_index: Tile index into the destination register.
 * @note Call @ref _llk_math_transpose_dest_init_ with matching template args before this
 *       function, and @ref _llk_math_transpose_dest_uninit_ after it to restore modified state.
 * @note On the unpack thread, the tile must already be in dest (via @ref _llk_unpack_A_ datacopy);
 *       @ref _llk_unpack_set_srcb_dummy_valid_ marks SrcB valid so the MOVB2D/MOVD2B sequence can run.
 * @note <transpose_of_faces=false, is_32bit=false> is not supported.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_(const std::uint32_t dst_index)
{
    namespace sync = hal::sync;
    namespace rwc  = hal::counters::rwc;
    using Mop0     = hal::mop::Runner<hal::mop::MopTemplate::Template0>;

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    hal_port::reset_counters_abd_f();

    // Wait condition SRCA_VLD is required as MOVB2A doesn't automatically wait
    // for SrcA[MatrixUnit.SrcABank].AllowedClient == SrcClient::MatrixUnit.
    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    sync::wait::stall<
        sync::StallTarget::Math,
        sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcAValid | sync::StallCondition::SrcBValid>();

    // Both paths run a MOVD2B/MOVB2D/MOVA2D transpose sequence that flushes any datum with a zero low
    // byte (e.g. bf16 0x4400 = 768.0) when the Src zero-substitution flag is at the operand DEFAULT,
    // corrupting ordinary finite values (same hazard as reduce_row_perform_transpose). transpose_dest
    // owns the PRESERVE policy here — asserted in execute (not init) so it survives any
    // llk_math_hw_configure that ran after the init; skip-if-set keeps it cheap.
    hal_port::_configure_preserve_zero_flag_state_();

    if constexpr (is_32bit)
    {
        transpose_dest_32b<transpose_of_faces>();
    }
    else
    {
        // ckernel_unpack_template::run(2, 2): MOP_CFG(zmask >> 16) then MOP(0, count - 1, zmask & 0xFFFF)
        Mop0::configure_mask_high<2>();
        Mop0::run<2, 2, hal::mop::MaskHigh::Preset>();
    }

    // TTI_SETRWC(CLR_AB, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();
}

/**
 * @brief Program the address-mod slots for a destination-register transpose.
 *
 * @tparam is_32bit: True for 32-bit datums (uses SrcA-stepping mods for the format-switching path).
 */
template <bool is_32bit>
inline void transpose_dest_configure_addrmod()
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    if constexpr (is_32bit)
    {
        set_math_addr_mod<Sec::S0, MathAddrMod {.dest_incr = 16}>();
        set_math_addr_mod<Sec::S1, MathAddrMod {}>();
        set_math_addr_mod<Sec::S2, MathAddrMod {.srca_incr = 16, .dest_incr = 16}>();
        set_math_addr_mod<Sec::S3, MathAddrMod {.srca_clr = 1, .dest_clr = 1}>();
    }
    else
    {
        set_math_addr_mod<Sec::S0, MathAddrMod {.dest_incr = 16}>();
        set_math_addr_mod<Sec::S1, MathAddrMod {}>();
        set_math_addr_mod<Sec::S2, MathAddrMod {.dest_incr = 0x3ff & -16}>();
        set_math_addr_mod<Sec::S3, MathAddrMod {.dest_incr = 32}>();
    }
}

/**
 * @brief Build the transpose MOP, recording the per-face MOVD2B/TRNSPSRCB/MOVB2D-MOVB2A sequence into the replay buffer.
 *
 * The 32-bit path records a 16-instruction replay buffer and drives it with a ckernel_template; the non-32-bit path
 * records a 15-instruction replay buffer and assembles a ckernel_unpack_template that runs the seven per-face steps.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (records the format-switching transpose sequence).
 */
template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop()
{
    using namespace hal_port::transpose_dest;
    namespace mop    = hal::mop;
    namespace replay = hal::replay;

    if constexpr (is_32bit)
    {
        replay::record<REPLAY_32B>(
            []
            {
                movd2b<true, 16, ADDR_MOD_1, 0>();
                movd2b<true, 20, ADDR_MOD_1, 4>();
                movd2b<true, 24, ADDR_MOD_1, 8>();
                movd2b<true, 28, ADDR_MOD_1, 12>();

                transpose_srcb();

                // HAL-GAP: no HAL API for MOVB2A (the three below and the MOP ops)
                TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
                TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
                TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);

                movd2b<false, 16, ADDR_MOD_1, 0>();
                movd2b<false, 20, ADDR_MOD_1, 4>();
                movd2b<false, 24, ADDR_MOD_1, 8>();
                movd2b<false, 28, ADDR_MOD_1, 12>();

                transpose_srcb();

                movb2d<16, ADDR_MOD_1, 0>();
                movb2d<20, ADDR_MOD_1, 4>();
                movb2d<24, ADDR_MOD_1, 8>();
            });

        constexpr std::uint32_t loop_op_0          = replay::get_operation<REPLAY_32B_LO>();
        constexpr std::uint32_t loop_op_1          = TT_OP_MOVB2A(12, ADDR_MOD_2, p_movb2a::MOV_4_ROWS, 28); // HAL-GAP: MOVB2A
        constexpr std::uint32_t loop_op_last_outer = TT_OP_MOVB2A(12, ADDR_MOD_3, p_movb2a::MOV_4_ROWS, 28); // HAL-GAP: MOVB2A

        // ckernel_template(4, 1, loop_op_0, loop_op_1) + set_last_outer_loop_instr(loop_op_last_outer)
        mop::program(mop::loop(4, mop::loop(1, loop_op_0, loop_op_1)).patch(mop::last_outer, loop_op_last_outer));
    }
    else
    {
        replay::record<replay::BufferRange {16, 15}>(
            []
            {
                // ABCD  HAL-GAP: no HAL API for MOVB2A
                TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
                TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
                TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
                TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // dst += 16

                // EFGHIJKLM
                movd2b<false, 16, ADDR_MOD_1, 0>();
                movd2b<false, 20, ADDR_MOD_1, 4>();
                movd2b<false, 24, ADDR_MOD_1, 8>();
                movd2b<false, 28, ADDR_MOD_1, 12>();
                transpose_srcb();
                movb2d<16, ADDR_MOD_1, 0>();
                movb2d<20, ADDR_MOD_1, 4>();
                movb2d<24, ADDR_MOD_1, 8>();
                movb2d<28, ADDR_MOD_0, 12>(); // dst += 16

                // NO: counter-relative negative offsets (10 bits)
                mova2d<false, 0, ADDR_MOD_2, move::relative_row<0 - 32>(), RowAddressing::CounterRelative>(); // dst -= 16
                mova2d<false, 8, ADDR_MOD_2, move::relative_row<8 - 16>(), RowAddressing::CounterRelative>(); // dst -= 16
            });

        // The next 7 instructions expand as follows:
        // Face 0: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM)
        // Face 1: 4x MOVD2B, TRNSPSRCB, 4x MOVB2A, dst += 16 (EFGHI, ABCD..)
        // Face 2: 4x MOVD2B (dst -= 16), TRNSPSRCB, 4x MOVB2D, dst += 32 (..EFG, P, IJKL, Q)
        // Face 3: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM..)
        // Face 1: 2x MOVA2D (2x dst -= 16) (..NO)

        constexpr std::uint32_t EFGHIJKLM   = replay::get_operation<replay::BufferRange {20, 9}>();
        constexpr std::uint32_t EFGHI       = replay::get_operation<replay::BufferRange {20, 5}>();
        constexpr std::uint32_t ABCDEFG     = replay::get_operation<replay::BufferRange {16, 7}>();
        constexpr std::uint32_t P           = MOVD2B<false, 28, ADDR_MOD_2, 12>.get_operation(); // dst -= 16
        constexpr std::uint32_t IJKL        = replay::get_operation<replay::BufferRange {24, 4}>();
        constexpr std::uint32_t Q           = MOVB2D<28, ADDR_MOD_3, 12>.get_operation(); // dst += 32
        constexpr std::uint32_t EFGHIJKLMNO = replay::get_operation<replay::BufferRange {20, 11}>();

        // The following MOP config simply runs the above 7 instructions in order (when executed with zmask 0b10):
        // ckernel_unpack_template(true, true, EFGHIJKLM, EFGHI, ABCDEFG, P, /* skip A */ Q, /* B */ IJKL, /* skip B */ EFGHIJKLMNO)
        mop::program(mop::foreach_mask_bit(mop::if_clear(EFGHIJKLM, EFGHI, ABCDEFG, P, IJKL), mop::otherwise(Q, EFGHIJKLMNO)));
    }
}

/**
 * @brief Configure the math thread (address mods and MOP) for a destination-register transpose.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (configures the format-switching transpose path).
 * @note @ref _llk_math_transpose_dest_ runs the configured transpose with matching template args.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_init_()
{
    transpose_dest_configure_addrmod<is_32bit>();
    transpose_dest_configure_mop<transpose_of_faces, is_32bit>();

    namespace cfg = hal::cfg;
    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
}

/**
 * @brief Uninitialize/cleanup after a destination-register transpose, restoring modified state to defaults.
 *
 * @note Reverses @ref _llk_math_transpose_dest_init_; currently a no-op since all state is transient.
 */
inline void _llk_math_transpose_dest_uninit_()
{
    // No state to restore - all states are transient or default
}
