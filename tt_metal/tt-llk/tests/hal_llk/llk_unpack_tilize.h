// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_tilize.h. Same instruction sequence and access kinds as production,
// expressed through the HAL.

#pragma once

#include <cstdint>

#include "../../common/tensor_shape.h"
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
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port::unpack_tilize
{
using hal::unpack::DataTransfer;
using hal::unpack::Engine;
using hal::unpack::SourceHandoff;

// TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
inline constexpr DataTransfer unpack_srca_desc {
    .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 1}}, .handoff = SourceHandoff::FlipAndSetDataValid};
// TT_OP_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 0 /*no Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
inline constexpr DataTransfer unpack_srca_z11 {
    .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 1}, .channel1 = {.z = 1}}, .handoff = SourceHandoff::Keep};
// Same with Set Dvalid: flips the SrcA bank, handing the tile to math.
inline constexpr DataTransfer unpack_srca_z11_dvalid {
    .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 1}, .channel1 = {.z = 1}}, .handoff = SourceHandoff::FlipAndSetDataValid};
// TT_OP_UNPACR(SrcA, 0b01000000 /*CH1_Y+=1*/, 0, 0, 0, 1, 0, RAREFYB_DISABLE, 0, 0, 0, 0, 1): one 1x16 row to SrcA
inline constexpr DataTransfer unpack_srca_row {.engine = Engine::Unpacker0, .increments = {.channel1 = {.y = 1}}, .handoff = SourceHandoff::Keep};
// TT_OP_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1): last SrcA row of a face
inline constexpr DataTransfer unpack_srca_last_row {.engine = Engine::Unpacker0, .handoff = SourceHandoff::FlipAndSetDataValid};

inline constexpr std::uint32_t unpack_srca         = hal::unpack::get_operation<unpack_srca_desc>();
inline constexpr std::uint32_t unpack_srca_to_dest = hal::unpack::get_operation<unpack_srca_z11>();

// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
// (UNP_ZEROSRC, SET_DVALID, WAIT_LIKE_UNPACR) has no HAL descriptor, so the production encoding is kept.
inline constexpr std::uint32_t unpack_srcb_set_dvalid =
    TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

// HAL-GAP: CFGSHIFTMASK (Base_address += SCRATCH_SEC0_val) has no HAL API; production encodings are kept.
inline constexpr std::uint32_t base_cntx0_add_scratch = TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
inline constexpr std::uint32_t base_cntx1_add_scratch = TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);

// Write the unpack config word 0 (THCON_SEC0_REG2 word 0) through GPR TMP0:
// TT_SETDMAREG x2, TTI_STALLWAIT(STALL_CFG, THCON), TTI_WRCFG(TMP0, WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32).
inline void write_unpack_config_word0(const std::uint32_t word)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S0) == THCON_SEC0_REG2_Out_data_format_ADDR32, "unpack config word 0");
    hal::scalar::set(hal::gpr<p_gpr_unpack::TMP0>(), word);
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::TMP0>());
}

// Write SCRATCH_SEC0_val (the CFGSHIFTMASK increment) through GPR TMP0, as write_unpack_config_word0.
inline void write_scratch_sec0(const std::uint32_t value)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    static_assert(cfg::Scratch::val.addr32(cfg::Sec::S0) == SCRATCH_SEC0_val_ADDR32, "SCRATCH_SEC0_val word");
    hal::scalar::set(hal::gpr<p_gpr_unpack::TMP0>(), value);
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Scratch::val, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::TMP0>());
}

} // namespace hal_port::unpack_tilize

/**
 * @brief Program the unpacker MOP for the non-8-bit tilize path (Blackhole whole-tile workaround).
 *
 * Selects between unpacking to SrcA (with a SrcB dvalid NOP) or straight to dest. 8-bit formats
 * use the inline 2-context path in @ref _llk_unpack_tilize_ instead (no MOP).
 *
 * @param unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 */
inline void _llk_unpack_tilize_mop_config_(const bool unpack_to_dest = false)
{
    using namespace hal::mop;
    using namespace hal_port::unpack_tilize;
    // SrcB SET_DVALID + UNP_ZEROSRC: math's ELWADD branch (FP32 dest-accumulate) reads SrcB,
    // so it must be both dvalid and zeroed. MOVA2D branch ignores it (harmless handshake).
    // Unpack to dest: no start op (NOP), as production leaves it unset.
    program(
        loop(1 /*outerloop*/, unpack_to_dest ? TT_OP_NOP : unpack_srca, loop(1 /*innerloop*/, unpack_to_dest ? unpack_srca_to_dest : unpack_srcb_set_dvalid)));
}

/**
 * @brief Initialize the unpacker for a tilize operation.
 *
 * Disables face transpose, configures the unpacker into tileize mode (throttle, shift amount,
 * per-tile X/Z dims) for the given block column dimension, decides whether 32-bit datums must be
 * unpacked to dest, and:
 *   - 8-bit formats: programs the 1x16 face x_dim per context and (for num_faces==4) the
 *     SCRATCH_SEC0_val preload used by the inline CFGSHIFTMASK. No MOP is programmed —
 *     the UNPACR sequence is issued inline by @ref _llk_unpack_tilize_.
 *   - non-8-bit formats: programs the whole-tile x_dim/z_dim descriptor state and the
 *     MOP template via @ref _llk_unpack_tilize_mop_config_ (Blackhole workaround).
 *
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param ct_dim: Number of column tiles in the block, used to size the column dimension.
 * @param face_r_dim: Rows per face, valid values = <2, 4, 8, 16>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces). Not supported on the 8-bit path (asserted false).
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @note Call @ref _llk_unpack_tilize_uninit_ to restore the modified tile-descriptor state.
 * @ref _llk_unpack_tilize_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ (A2D, PackMode::Tilize) is the matching init on the math thread.
 */
inline void _llk_unpack_tilize_init_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be 1, 2, 4, 8, or 16 for tilize");
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");
    namespace cfg          = hal::cfg;
    namespace adc          = hal::counters::adc;
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    constexpr auto R5      = cfg::Thcon[cfg::Reg5];
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(TileDsc.ZDim.addr32(cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, "Z-dim word");
    static_assert(TileDsc.XDim.addr32(cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32, "X-dim word");
    static_assert(R5.Tile_x_dim_cntx0.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, "Tile_x_dim word");

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 0>();

    // Program Ch1.Z stride from unpack_dst_format's datum size, required by the
    // UNPACR AddrMode 0b00010001 used on the 8-bit tilize path.
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Zstride, cfg::Sec::S0>(canonical_unpA_z_stride(unpack_dst_format));

    // Clear Ch1.X and Ch1.Y start-counts once at init. Tilize's UNPACRs never
    // increment Ch1.X or Ch1.Y (AddrMode 0b00010001 sets only Ch0.Z++/Ch1.Z++),
    // so the counts stay at 0 through the whole tilize_block. With the counts at
    // 0, Ch1.X/Y strides multiply out of the SrcA write address (SrcA_row =
    // Ch1.Z*Zstride + Ch1.Y*Ystride + Ch1.X*Xstride + Base) and need not be
    // programmed here.
    // TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.x = 0, .y = 0}}.for_clients<adc::Client::Unpacker0>()>();

    // Z_dim = num_faces, the per-operand baseline. The non-8-bit branch below overwrites it with 1,
    // and that is the write the teardown reverts.
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.ZDim, cfg::Sec::S0>(num_faces);

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

    // Set face dim: TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = face_r_dim * FACE_C_DIM - 1}}.for_clients<adc::Client::Unpacker0>());

    const bool is_8bit_format        = IS_8BIT_FORMAT(unpack_src_format);
    const std::uint32_t shift_amount = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = shift_amount;

    hal_port::unpack_tilize::write_unpack_config_word0(config.val[0]);
    if (is_8bit_format)
    {
        LLK_ASSERT(!narrow_tile, "8-bit tilize narrow_tile not supported");
        LLK_ASSERT(!unpack_to_dest, "8-bit tilize unpack_to_dest not supported");

        // x_dim per context = FACE_C_DIM (1x16 face read). FACE_DIM_1x16 holds
        // FACE_C_DIM packed in both halfwords, so a 32b WRCFG programs both cntx0 and cntx1.
        cfg::write<cfg::Access::TensixCfgUnit, R5.Tile_x_dim_cntx0, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<p_gpr_unpack::FACE_DIM_1x16>());

        // SCRATCH_SEC0_val is the increment CFGSHIFTMASK adds to Base_address between
        // top and bottom face pairs. Only used when num_faces==4 (top pair to bottom pair).
        // For num_faces==2 (top pair only) no CFGSHIFTMASK fires, so the preload is skipped.
        //   bot_face_offset = shift_amount * face_r_dim       (in 16B words, jump from top to bottom faces)
        //   -2 compensates for Z=2 L1 offset (Z * XDim * DatumSize = 2 * 16 * 1 = 32 bytes = 2 16B-words)
        if (num_faces == 4)
        {
            const std::uint32_t base_address_increment = shift_amount * face_r_dim - 2;
            hal_port::unpack_tilize::write_scratch_sec0(base_address_increment);
        }
    }
    else
    {
        // Whole-tile unpacking (non-8-bit BH workaround): x_dim covers the entire tile, z_dim=1
        const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
        constexpr std::uint32_t Tile_z_dim = 1;
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<R5.Tile_x_dim_cntx0, cfg::Sec::S0>(Tile_x_dim), cfg::set<R5.Tile_x_dim_cntx1, cfg::Sec::S0>(Tile_x_dim));
        // Set x-dim to cover entire tile (face_r_dim * num_faces * FACE_C_DIM)
        cfg::write<cfg::Access::TensixCfgUnit, TileDsc.XDim, cfg::Sec::S0>(Tile_x_dim);
        // Set z-dim to 1 as X dim is set to cover the entire tile, so no need to iterate over faces.
        cfg::write<cfg::Access::TensixCfgUnit, TileDsc.ZDim, cfg::Sec::S0, Tile_z_dim>();

        // Set x-end for Unpackers to (face_r_dim * num_faces * FACE_C_DIM - 1): TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0)
        adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = Tile_x_dim - 1}}.for_clients<adc::Client::Unpacker0>());

        _llk_unpack_tilize_mop_config_(unpack_to_dest);
    }
}

/**
 * @brief Unpack and tilize a tile from L1 into SrcA or the dest register.
 *
 * Computes the L1 base address for the selected column tile and unpacks it into SrcA.
 *   - 8-bit formats: issues an inline 2-context UNPACR sequence (2 UNPACRs for num_faces=2, or
 *     4 UNPACRs + CFGSHIFTMASK for num_faces=4) plus a SrcB SET_DVALID + UNP_ZEROSRC handshake.
 *     Emits 1 DVALID per tile. No MOP; `face_r_dim` is not consumed by the sequence itself
 *     (already programmed via init's SCRATCH_SEC0_val preload) but drives the init-time preload.
 *   - non-8-bit formats: runs the whole-tile MOP programmed at init, and — when unpacking
 *     32-bit datums to dest — manages the dest write address and completion handshake.
 *
 * @param base_address: L1 base address of the source tile buffer.
 * @param tile_index: Column tile index selecting which tile to unpack.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param face_r_dim: Rows per face.
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces). Not supported on the 8-bit path.
 * @note Call @ref _llk_unpack_tilize_init_ before this function, and
 *       @ref _llk_unpack_tilize_uninit_ after it to restore modified state.
 */
inline void _llk_unpack_tilize_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    std::uint32_t unpack_src_format                 = 0,
    std::uint32_t unpack_dst_format                 = 0,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool narrow_tile                          = false)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");
    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    using namespace hal_port::unpack_tilize;
    constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A base words");
    // TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111): Z/W counters on UNP_A for both channels
    constexpr adc::AddrCounter clear_zw = adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0>();

    // Production's get_cfg_pointer (RISC pointer select only): its branch layout keeps the per-tile path as production's;
    // hcfg::detail::state_cfg_bank() flips the branch and cost up to 4% on short tilize blocks.
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);

    // 8-bit path: 2-context inline UNPACR sequence. AddrMode=0b00010001 advances BOTH
    // Ch0.Z (L1 input read) and Ch1.Z (SrcA output write) so face N lands in SrcA rows
    // 16*N..16*N+15 via the Zstride configured by configure_unpack_AB. This relies on
    // SRCA_SET_SetOvrdWithAddr = 1 (letting OutAddr drive the SrcA row directly, without
    // SrcRow being added). Contract: the bit is set to 1 by configure_unpack_AB at unpack
    // init and is the resting state. It is transiently cleared to 0 only by
    // set_dst_write_addr with a paired restore in unpack_to_dest_tile_done. Because this
    // path asserts !unpack_to_dest at init, we can never be inside that transient window
    // on entry -- the bit is guaranteed to be 1. NOTE: future FP8 unpack_to_dest support
    // would enter with the bit cleared and must either restore it explicitly or use a
    // different addressing scheme.
    // The active context selects between REG3_Base_address (cntx0) and REG3_Base_cntx1
    // _address for both the base-address cfg write and the CFGSHIFTMASK target.
    //   num_faces==4: 4 UNPACRs (face 3 with SetDvalid) + CFGSHIFTMASK between top and
    //                 bottom face pairs (Ch1.Z keeps advancing across the CFGSHIFTMASK
    //                 since it touches REG3 only).
    //   num_faces==2: 2 UNPACRs (top pair only, face 1 with SetDvalid). No CFGSHIFTMASK.
    if (IS_8BIT_FORMAT(unpack_src_format))
    {
        const std::uint32_t address = base_address + top_face_offset_address;
        LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

        // Clear Z/W counters on UNP_A for both channels (Ch0 for L1 read, Ch1 for SrcA write)
        adc::set<clear_zw>();

        // Wait for a free context (up to 2 tiles in flight)
        hal_port::wait_for_next_context(2);

        // Write top-face base address to the active context's REG3
        if (0 == unp_cfg_context)
        {
            cfg[base_cntx0] = address;
        }
        else
        {
            cfg[base_cntx1] = address;
        }
        sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

        // Stall unpacker until pending CFG writes from Trisc have completed
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

        if (num_faces == 4)
        {
            // Face 0: Ch0.Z=0, Ch1.Z=0 -> SrcA rows 0..15
            hal::unpack::run<unpack_srca_z11>();
            // Face 1: Ch0.Z=1, Ch1.Z=1 -> SrcA rows 16..31
            hal::unpack::run<unpack_srca_z11>();

            // Shift the active context's Base_address by SCRATCH_SEC0_val (= bot_face_offset - 2
            // in 16B words) so face 2/3 read correctly at Z=2/3 on the L1 side. CFGSHIFTMASK's
            // target is an instruction immediate, so we pick the variant at compile time per context.
            if (0 == unp_cfg_context)
            {
                TTI_INSN(base_cntx0_add_scratch); // HAL-GAP: CFGSHIFTMASK
            }
            else
            {
                TTI_INSN(base_cntx1_add_scratch); // HAL-GAP: CFGSHIFTMASK
            }
            hal::nop::thread();

            // Face 2: Ch0.Z=2, Ch1.Z=2 -> SrcA rows 32..47
            hal::unpack::run<unpack_srca_z11>();
            // Face 3: Ch0.Z=3, Ch1.Z=3 -> SrcA rows 48..63. SetDvalid flips the SrcA bank,
            // handing the full 64-row tile to math.
            hal::unpack::run<unpack_srca_z11_dvalid>();
        }
        else // num_faces == 2: top face pair only, no CFGSHIFTMASK
        {
            // Face 0: Ch0.Z=0, Ch1.Z=0 -> SrcA rows 0..15
            hal::unpack::run<unpack_srca_z11>();
            // Face 1: Ch0.Z=1, Ch1.Z=1 -> SrcA rows 16..31. SetDvalid flips the SrcA bank.
            hal::unpack::run<unpack_srca_z11_dvalid>();
        }

        // SrcB DVALID + UNP_ZEROSRC. Required so math's ELWADD branch (dest_acc=Yes,
        // is_int_fpu_en, or dst==UInt8) sees a zero SrcB; for the MOVA2D branch it's
        // a no-op handshake but kept uniform across formats.
        TTI_INSN(unpack_srcb_set_dvalid); // HAL-GAP: zero-source + set-data-valid UNPACR_NOP

        // T6::SEMGET for context release
        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

        // Flip to the other context for the next tile
        hal_port::switch_config_context_branched(unp_cfg_context);
    }
    else
    {
        // Non-8-bit path: whole-tile unpack via BH workaround configuration (x_dim covers full tile).
        std::uint32_t address = base_address + top_face_offset_address;
        LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

        // Clear z/w start counters
        adc::set<clear_zw>();

        // Wait for free context
        hal_port::wait_for_next_context(2);

        // Get tile address
        if (0 == unp_cfg_context)
        {
            cfg[base_cntx0] = address;
        }
        else
        {
            cfg[base_cntx1] = address;
        }

        // Trisc::SEMPOST for context acquire
        sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

        if (unpack_to_dest)
        {
            // Unpack to dest
            hal_port::set_dst_write_addr(unp_cfg_context, unpack_src_format);
            hal_port::wait_for_dest_available();
        }

        // Stall unpacker until pending CFG writes from Trisc have completed
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

        // Run MOP
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

        if (unpack_to_dest)
        {
            // Pair with set_dst_write_addr above (both keyed on unpack_src_format), so the
            // canonical Z-stride restore matches the value programmed on entry.
            hal_port::unpack_to_dest_tile_done(unp_cfg_context, unpack_src_format);
        }

        // T6::SEMGET for context release, after unpack_to_dest_tile_done's config writes so that
        // configure_unpack_AB's wait_for_idle also waits for them
        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

        // Switch unpacker config context
        hal_port::switch_config_context_branched(unp_cfg_context);
    }
}

/**
 * @brief Restore unpacker state after a tilize operation.
 *
 * Reverts the config @ref _llk_unpack_tilize_init_ wrote: tile-descriptor Z dim (dropped to 1 by
 * the non-8-bit init path) and X dim (written by that path only, never by the 8-bit one), unpack
 * config word 0 (clearing tilize and haloize mode) and Tile_x_dim_cntx0. x-start/x-end and the MOP
 * are reprogrammed by the next operation's init, so they are not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @param tensor_shape: Tile geometry; total_num_faces() restores the descriptor Z dimension
 *                      (valid values = <1, 2, 4>) and face_r_dim restores the canonical Tile_x_dim.
 * @note Call @ref _llk_unpack_tilize_init_ before this function.
 */
inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    namespace cfg                  = hal::cfg;
    namespace sync                 = hal::sync;
    constexpr auto TileDsc         = cfg::Thcon[cfg::Reg0].TileDescriptor;
    constexpr auto R5              = cfg::Thcon[cfg::Reg5];
    const std::uint32_t num_faces  = tensor_shape.total_num_faces();
    const std::uint32_t face_r_dim = tensor_shape.face_r_dim;

    sync::wait::stall<sync::StallTarget::Thcon, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // X-dim is 0 because Tile_x_dim_cntx0 below is what the unpacker consumes for srcA.
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.ZDim, cfg::Sec::S0>(num_faces);
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.XDim, cfg::Sec::S0, CANONICAL_UNPA_TILE_X_DIM>();

    // Zeroed config clears tileize_mode and haloize_mode along with the rest of word 0.
    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    // Load unpack config[0]
    hal_port::unpack_tilize::write_unpack_config_word0(config.val[0]);
    // canonical_unpA_tile_x_dim_cntx(face_r_dim): the face size in both context halves of the word
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R5.Tile_x_dim_cntx0, cfg::Sec::S0>(face_dim), cfg::set<R5.Tile_x_dim_cntx1, cfg::Sec::S0>(face_dim));
    hal::nop::thread();
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

/**
 * @brief Program the unpacker MOP/replay buffer for tilize-A-with-unpack-B.
 *
 * Builds a replay buffer that unpacks one 1x16 row of SrcA at a time and advances the SrcA L1
 * base address (per config context) by the programmed column stride.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 */
// TODO: add support for all the template parameters
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr std::uint32_t replay_buf_run_len  = 6;
    constexpr std::uint32_t replay_buf_half_len = replay_buf_run_len >> 1;

    using namespace hal_port::unpack_tilize;
    static_assert(replay_buf_run_len == 6 && replay_buf_half_len == 3, "replay ranges below assume a 6-instruction buffer");
    constexpr hal::replay::BufferRange replay_all {0 /*start*/, replay_buf_run_len};
    constexpr hal::replay::BufferRange replay_cntx0 {0 /*start*/, replay_buf_half_len};
    constexpr hal::replay::BufferRange replay_cntx1 {replay_buf_half_len /*start*/, replay_buf_half_len};

    // Lambda function to set up replay buffer
    hal::replay::record<replay_all>(
        []
        {
            // Unpacks 1x16 row of datums to SrcA
            hal::unpack::run<unpack_srca_row>();

            // THCON_SEC0_REG3_Base_address_ADDR32 =  THCON_SEC0_REG3_Base_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_INSN(base_cntx0_add_scratch); // HAL-GAP: CFGSHIFTMASK
            hal::nop::thread();

            // Unpacks 1x16 row of datums to SrcA
            hal::unpack::run<unpack_srca_row>();

            // THCON_SEC0_REG3_Base_cntx1_address_ADDR32 =  THCON_SEC0_REG3_Base_cntx1_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_INSN(base_cntx1_add_scratch); // HAL-GAP: CFGSHIFTMASK
            hal::nop::thread();
        });

    // Template 0, no SrcB, no halo: A0 runs when context is 0 (mask bit clear), SkipA0 when context is 1 (mask bit set).
    hal::mop::program(hal::mop::foreach_mask_bit(
        hal::mop::if_clear(hal::replay::get_operation<replay_cntx0>()), hal::mop::otherwise(hal::replay::get_operation<replay_cntx1>())));
}

/**
 * @brief Initialize the unpacker to tilize operand A while unpacking operand B.
 *
 * Programs the column stride used to advance SrcA's L1 address (via the CFGSHIFTMASK scratch
 * register), sets per-unpacker datum counts (one row for SrcA, full face for SrcB) and SrcA's Y
 * stride, disables face transpose, and programs the tilize-A-B MOP.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param unpack_src_format: Source data format of operand A in L1.
 * @param unpack_dst_format: Destination data format operand A is converted to.
 * @param ct_dim: Number of column tiles in the block, used to size the column stride.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @note Call @ref _llk_unpack_tilizeA_B_uninit_ to revert the config it writes; it re-asserts the canonical SrcA Y stride.
 * @ref _llk_unpack_tilizeA_B_ is the matching execute call.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces       = 4,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // Sets the block_c_dim for unpack to use to increment the L1 address
    const std::uint32_t c_dim_size = SCALE_DATUM_SIZE(unpack_src_format, ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM)) >> 4;

    namespace cfg = hal::cfg;

    // This sets the scratch register that CFGSHIFTMASK instruction uses to increment the L1 address
    hal_port::unpack_tilize::write_scratch_sec0(c_dim_size);
    hal::nop::thread();

    // Unpack 1 row of 1x16 at a time for SrcA
    hal_port::config_unpacker_x_end<p_setadc::UNP_A>(1);
    hal_port::config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);

    // Set the SrcA Y stride to one 1x16 row of datums with canonical_unpA_y_stride, the same helper
    // configure_unpack_AB, the srcA reconfig and _llk_unpack_tilizeA_B_uninit_ use. A Tf32 SrcA (Float32 L1
    // operand under 32-bit dest) has 4-byte datums; SCALE_DATUM_SIZE has no Tf32 arm and would give a quarter-row stride.
    const std::uint32_t unpA_ch1_y_stride = canonical_unpA_y_stride(unpack_dst_format);
    constexpr auto Unp0Ch1                = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(Unp0Ch1.Ystride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, "Unp0 ch1 Y-stride word");
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(unpA_ch1_y_stride);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 0>();

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(num_faces);
}

/**
 * @brief Tilize operand A and unpack operand B, face by face, into SrcA and SrcB.
 *
 * Loops over the faces, computing each face's SrcA L1 address, optionally clearing SrcA to
 * neginf/zero, unpacking the SrcB face, then unpacking the face's rows into SrcA (row by row via
 * the MOP) and setting data-valid, synchronizing through the unpack semaphore each iteration.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param face_r_dim: Rows per face.
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param address_b: L1 address of operand B's face data.
 * @param tile_index_a: Column tile index into operand A.
 * @param block_ct_dim: Number of column tiles in the block, used to compute face strides.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ with matching template args before this function, and
 *       @ref _llk_unpack_tilizeA_B_uninit_ after it to restore modified state.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_(
    std::uint32_t unpA_src_format,
    std::uint32_t face_r_dim,
    std::uint32_t base_address_a,
    std::uint32_t address_b,
    std::uint32_t tile_index_a,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t offset_address_a = SCALE_DATUM_SIZE(unpA_src_format, tile_index_a) << 1;
    const std::uint32_t address_a        = base_address_a + offset_address_a;

    const std::uint32_t block_c_dim = block_ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM) * face_r_dim;
    const bool run_r_dim_loop       = (face_r_dim > 1);

    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    using namespace hal_port::unpack_tilize;
    // TTI_UNPACR(SrcB, reload_srcB ? 0b0 : 0b1, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    constexpr hal::unpack::DataTransfer unpack_srcb {
        .engine = Engine::Unpacker1, .increments = {.channel0 = {.z = reload_srcB ? 0 : 1}}, .handoff = SourceHandoff::FlipAndSetDataValid};

    volatile std::uint32_t tt_reg_ptr* cfg = hal::cfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Clear z/w start counters for SrcA/B: TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    for (std::uint32_t n = 0; n < num_faces; n++)
    {
        /*
        Face 0: address = base_address
        Face 1: address = base_address + 1x16 row of datums
        Face 2: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim (address for the bottom 2 faces of tiles)
        Face 3: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim + 1x16 row of datums
        */
        std::uint32_t address_face_a = (n % 2 == 0) ? address_a : (address_a + (SCALE_DATUM_SIZE(unpA_src_format, FACE_C_DIM) >> 4));
        address_face_a += (n >= 2) ? ((SCALE_DATUM_SIZE(unpA_src_format, block_c_dim)) >> 4) : 0;

        // Wait for free context
        hal_port::wait_for_next_context(2);

        // HAL-GAP: the clear-source UNPACR_NOP flavors (UNP_CLRSRC with NEGINF / ZERO, WAIT_LIKE_UNPACR) have no
        // HAL descriptor, so the production encodings are kept.
        if constexpr (neginf_srcA)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC);
        }
        else if constexpr (zero_srcA_reduce)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
        }

        // Validate and configure addresses
        _llk_unpack_configure_addresses_(address_face_a, address_b, cfg);

        // Trisc::SEMPOST for context acquire
        sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

        // Stall unpacker until pending CFG writes from Trisc have completed
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

        // Reset Y counters for SrcA: TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010)
        adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0>()>();
        // Unpack SrcB 16x16 face & Set Data Valid

        // If reload_srcB, only first face needs to be loaded, otherwise CH0_Z+=1
        hal::unpack::run<unpack_srcb>();

        // Unpacks face_r_dim-1 rows of 1x16 datums to SrcA
        if (run_r_dim_loop)
        {
            hal::mop::Runner<hal::mop::MopTemplate::Template0>::run(face_r_dim - 1, unp_cfg_context == 0 ? 0 : 0xffff);
        }

        // Unpack last SrcA row of a 16x16 face and SetDvalid
        hal::unpack::run<unpack_srca_last_row>();

        // T6::SEMGET for context release
        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

        // Switch unpacker config context
        hal_port::switch_config_context(unp_cfg_context);
    }
}

/**
 * @brief Restore unpacker state after a tilize-A-with-unpack-B operation.
 *
 * Re-asserts the canonical SrcA Ch1 Y stride @ref _llk_unpack_tilizeA_B_init_ also wrote, and zeroes the Ch0/Ch1 Y
 * counters on both unpackers, which belong to the execute path rather than to init. The unpack
 * config word-0 write is wider than init's: init only clears haloize_mode, while this is a
 * full-word write that also forces out_data_format and throttle_mode and zeroes context_count,
 * tileize_mode, unpack_src_reg_set_update and shift_amount. x-start/x-end and the MOP are
 * reprogrammed by the next operation's init, so they are not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ before this function.
 */
inline void _llk_unpack_tilizeA_B_uninit_(const std::uint32_t unpack_dst_format)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    sync::wait::stall<sync::StallTarget::Thcon, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // _llk_unpack_tilizeA_B uses y-stride and updates y counter: TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010)
    adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    // Load unpack config[0]
    hal_port::unpack_tilize::write_unpack_config_word0(config.val[0]);
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(Unp0Ch1.Ystride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, "Unp0 ch1 Y-stride word");
    static_assert(Unp0Ch1.Ystride.shamt(cfg::Sec::S0) == UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, "Y-stride shift");
    static_assert(Unp0Ch1.Ystride.mask(cfg::Sec::S0) == UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK, "Y-stride mask");
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(canonical_unpA_y_stride(unpack_dst_format));
    hal::nop::thread();
}
