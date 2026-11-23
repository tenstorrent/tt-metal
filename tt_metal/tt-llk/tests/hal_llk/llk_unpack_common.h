// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_common.h (+ the unpack helpers of common/inc/cunpack_common.h
// reached by eltwise_binary_test). Same instruction sequence and access kinds as production, expressed through the HAL.

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h" // constants, types and pure helpers only (p_gpr_unpack, masked_data_format, canonical_*, ...)
#include "hal/cfg.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{

namespace adc = hal::counters::adc;

// Semaphore / mutex indices come from production's ckernel::semaphore / ckernel::mutex.
inline constexpr hal::sync::Semaphore UNPACK_SYNC_SEM = static_cast<hal::sync::Semaphore>(ckernel::semaphore::UNPACK_SYNC);
inline constexpr hal::sync::Mutex REG_RMW_MUTEX       = static_cast<hal::sync::Mutex>(ckernel::mutex::REG_RMW);

// Place value in field F of section S, for composing a complete CFG word (equivalent of production's bitfield unions).
template <const hal::cfg::Field &F, hal::cfg::Sec S = hal::cfg::Sec::S0>
constexpr std::uint32_t field_bits(const std::uint32_t value)
{
    return (value << F.shamt(S)) & F.mask(S);
}

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void unpacker_addr_counter_init()
{
    // TTI_SETADCXY(UNP_A | UNP_B, 0, 0, 0, 0, 0b1011): channel 0 X and Y, channel 1 Y
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    // TTI_SETADCZW(UNP_A | UNP_B, 0, 0, 0, 0, 0b1111): Z and W of both channels
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
}

// Wait for threshold of busy contexts to fall below total available contexts
inline void wait_for_next_context(const std::uint32_t num_contexts)
{
    while (hal::sync::semaphore::read<UNPACK_SYNC_SEM>() >= num_contexts)
    {
    }
}

inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    namespace cfg = hal::cfg;
    // Switch config context. Both offsets take the new context id (0 or 1), so one runtime SETC16 of
    // ctx * (bit of offset 0 | bit of offset 1) replaces the per-tile branch between two immediate encodings.
    constexpr std::uint32_t both_offsets_one = field_bits<cfg::UnpackMiscCfg::CfgContextOffset_0>(1) | field_bits<cfg::UnpackMiscCfg::CfgContextOffset_1>(1);
    const std::uint32_t next_context         = 1 - unp_cfg_context;
    LLK_ASSERT(next_context <= 1, "unpack config context must be 0 or 1");
    unp_cfg_context = next_context;
    cfg::write_thread_word<cfg::Access::TensixCfgUnit, cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0>(next_context * both_offsets_one);
}

inline void reset_config_context()
{
    namespace cfg = hal::cfg;
    // Switch config context
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

// Auto TTSync setting used by the unpack thread: track backend config (split per unpacker) against Tensix instructions.
inline void enable_unpack_auto_ttsync()
{
    namespace cfg = hal::cfg;
    // SETC16.md: after reset, Auto TTSync only tracks the right Config bank once a SETC16 of CFG_STATE_ID has executed.
    // The software tracker ckernel::cfg_state_id is 0 here (no state flip in this kernel), so write 0.
    LLK_ASSERT(ckernel::cfg_state_id == 0, "Auto TTSync setup assumes config state 0");
    cfg::write<cfg::Access::TensixCfgUnit, cfg::CfgStateId::StateID, cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::TensixTriscSync::TrackGlobalCfg, cfg::Sec::S0, 1>(),
        cfg::set<cfg::TensixTriscSync::EnSubdividedCfgForUnpacr, cfg::Sec::S0, 1>(),
        cfg::set<cfg::TensixTriscSync::TrackGPR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTDMARegs, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTensixInstructions, cfg::Sec::S0, 1>());
    // AutoTTSync.md: a change to the TENSIX_TRISC_SYNC fields must take effect before relying on it.
    // HAL-GAP: no HAL wrapper for the Manual TTSync idle wait; production's tensix_sync() is used.
    ckernel::tensix_sync();
}

// Turn Auto TTSync off again for the unpack thread. The setting is sticky per thread, so code relying on the
// two-context handshake (or on Auto TTSync being off for performance) must restore it.
inline void disable_unpack_auto_ttsync()
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::TensixTriscSync::TrackGlobalCfg, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::EnSubdividedCfgForUnpacr, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackGPR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTDMARegs, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTensixInstructions, cfg::Sec::S0, 0>());
    ckernel::tensix_sync();
}

// Sync on unpacker idle via waiting busy contexts counter 0
inline void wait_for_idle()
{
    while (hal::sync::semaphore::read<UNPACK_SYNC_SEM>() > 0)
    {
    }
}

template <bool is_fp32_dest_acc_en, bool row_pool = false, bool fpu_srnd_en = false, bool pack_srnd_en = false>
inline void configure_unpack_AB(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const bool transpose_xy_srca_en     = false,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpA_src_format), static_cast<DataFormat>(unpA_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpB_src_format), static_cast<DataFormat>(unpB_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    hal_port::wait_for_idle();

    // Reset address counters
    hal_port::unpacker_addr_counter_init();

    const std::uint32_t unpA_src_format_masked = masked_data_format(unpA_src_format);
    const std::uint32_t unpB_src_format_masked = masked_data_format(unpB_src_format);
    const std::uint32_t unpA_dst_format_masked = masked_data_format(unpA_dst_format);
    const std::uint32_t unpB_dst_format_masked = masked_data_format(unpB_dst_format);

    std::uint32_t unpA_ch1_x_stride = datum_size_in_bytes(unpA_dst_format_masked);
    std::uint32_t unpB_ch1_x_stride = datum_size_in_bytes(unpB_dst_format_masked);
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    // Z and W(not used) stride for dest address (ch1): whole-word MMIO stores
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    constexpr auto Unp1Ch1 = cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1];
    cfg::write<cfg::Access::MMIO, Unp0Ch1.Zstride, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<Unp0Ch1.Wstride>(0) | field_bits<Unp0Ch1.Zstride>(unpA_ch1_z_stride)});
    cfg::write<cfg::Access::MMIO, Unp1Ch1.Zstride, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<Unp1Ch1.Wstride>(0) | field_bits<Unp1Ch1.Zstride>(unpB_ch1_z_stride)});

    // Establish the canonical Y-stride baseline for srcA (ch1) here, so per-op inits and
    // uninits that touch the Y stride (e.g. tilizeA_B) set and re-assert this baseline
    // rather than snapshotting the previous register value into a GPR.
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(canonical_unpA_y_stride(unpA_dst_format_masked));

    // Math ALU_FORMAT_REG
    sync::mutex::acquire<REG_RMW_MUTEX>();

    // FP32 accumulation and SFPU to read dest as FP32
    // NOTE: This assumes these config fields are adjacent and in same register!!
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(unpA_src_format == to_underlying(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(unpB_src_format == to_underlying(DataFormat::UInt8) ? 1 : 0));

    // Set FP8 E4M3 mode, bit is accessible by unpacker/packer
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0>(
        ((unpA_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1>(
        ((unpB_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);

    sync::mutex::release<REG_RMW_MUTEX>();

    // Set tile descriptor (TILE_DESC_SIZE words per unpacker)
    static_assert(TILE_DESC_SIZE == 2, "tile descriptor is composed of 2 words");
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    // Input tile is uncompressed; x_dim is not used for unpA as the value is overridden by per context x_dim set below. Used for unpB
    cfg::write<cfg::Access::MMIO, TileDsc.Raw, cfg::Sec::S0, TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {
        field_bits<TileDsc.InDataFormat, cfg::Sec::S0>(unpA_src_format_masked) | field_bits<TileDsc.Uncompressed, cfg::Sec::S0>(1) |
            field_bits<TileDsc.XDim, cfg::Sec::S0>(0),
        field_bits<TileDsc.YDim, cfg::Sec::S0>(1) | field_bits<TileDsc.ZDim, cfg::Sec::S0>(unpA_num_faces)});
    cfg::write<cfg::Access::MMIO, TileDsc.Raw, cfg::Sec::S1, TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {
        field_bits<TileDsc.InDataFormat, cfg::Sec::S1>(row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked) |
            field_bits<TileDsc.Uncompressed, cfg::Sec::S1>(1) | field_bits<TileDsc.XDim, cfg::Sec::S1>(unpB_face_r_dim * FACE_C_DIM),
        field_bits<TileDsc.YDim, cfg::Sec::S1>(1) | field_bits<TileDsc.ZDim, cfg::Sec::S1>(unpB_num_faces)});

    // Set unpacker config (CONFIG_SIZE words per unpacker)
    static_assert(CONFIG_SIZE == 2, "unpacker config is composed of 2 words");
    constexpr auto R2 = cfg::Thcon[cfg::Reg2];
    // word 1: uncompress_cntx0_3 = uncompress_cntx4_7 = 0xf, i.e. zero-compression disabled in all 8 contexts
    const std::uint32_t uncompress_all =
        field_bits<R2.Disable_zero_compress_cntx0, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx1, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx2, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx3, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx4, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx5, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx6, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx7, cfg::Sec::S0>(1);
    cfg::write<cfg::Access::MMIO, R2.Out_data_format, cfg::Sec::S0, CONFIG_SIZE>(std::array<std::uint32_t, CONFIG_SIZE> {
        field_bits<R2.Out_data_format, cfg::Sec::S0>(unpA_dst_format_masked) | field_bits<R2.Throttle_mode, cfg::Sec::S0>(2) |
            field_bits<R2.Haloize_mode, cfg::Sec::S0>(transpose_xy_srca_en ? 1 : 0),
        uncompress_all});

    cfg::write<cfg::Access::MMIO, R2.Out_data_format, cfg::Sec::S1, CONFIG_SIZE>(std::array<std::uint32_t, CONFIG_SIZE> {
        field_bits<R2.Out_data_format, cfg::Sec::S1>(row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked) |
            field_bits<R2.Throttle_mode, cfg::Sec::S1>(2) | field_bits<R2.Haloize_mode, cfg::Sec::S1>(0),
        uncompress_all});

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    constexpr auto R5                      = cfg::Thcon[cfg::Reg5];
    const std::uint32_t Dest_cntx0_address = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg::write<cfg::Access::MMIO, R5.Dest_cntx0_address, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<R5.Dest_cntx0_address>(Dest_cntx0_address) | field_bits<R5.Dest_cntx1_address>(Dest_cntx1_address)});

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim = unpA_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::MMIO, R5.Tile_x_dim_cntx0, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<R5.Tile_x_dim_cntx0>(face_dim) | field_bits<R5.Tile_x_dim_cntx1>(face_dim)});

    // HAL-GAP: RISC MMIO stores to the GPR register file (regfile[...] = v) and the load that fences them
    // (sync_regfile_write) have no HAL API; kept as in production.
    constexpr std::uint32_t face_dim_16x16        = FACE_R_DIM * FACE_C_DIM;
    ckernel::regfile[p_gpr_unpack::FACE_DIM_8x16] = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_4x16] = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_2x16] = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_1x16] = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    ckernel::sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    // TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4)
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<cfg::Unpacker[0].AddDestAddrCntr>(1)});

    // Clear context ID
    hal_port::reset_config_context();
    // Auto TTSync is enabled by _llk_unpack_AB_init_, the only user of the single-context path, not here: other
    // unpack ops configured through this function keep the production two-context handshake, which hangs under
    // Auto TTSync (test_sdpa_custom_mm_mask_extent_restore).
}

template <std::uint32_t UNP_SEL, std::uint32_t face_r_dim>
inline void set_unpacker_x_end()
{
    // TTI_SETADCXX(UNP_SEL, face_r_dim * FACE_C_DIM - 1, 0x0): X start on channel 0, X end on channel 1
    constexpr adc::AddrCounter x_end {.channel0 = {.x = 0}, .channel1 = {.x = face_r_dim * FACE_C_DIM - 1}};
    if constexpr (UNP_SEL == p_setadc::UNP_A)
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker0>()>();
    }
    else if constexpr (UNP_SEL == p_setadc::UNP_B)
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker1>()>();
    }
    else
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    }
}

template <std::uint32_t UNP_SEL = p_setadc::UNP_AB>
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B || UNP_SEL == p_setadc::UNP_AB, "UNP_SEL must be UNP_A, UNP_B, or UNP_AB");
    LLK_ASSERT(
        face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == FACE_R_DIM, "face_r_dim must be 1, 2, 4, 8, or FACE_R_DIM");

    switch (face_r_dim)
    {
        case 1:
            set_unpacker_x_end<UNP_SEL, 1>();
            break;
        case 2:
            set_unpacker_x_end<UNP_SEL, 2>();
            break;
        case 4:
            set_unpacker_x_end<UNP_SEL, 4>();
            break;
        case 8:
            set_unpacker_x_end<UNP_SEL, 8>();
            break;
        default:
            set_unpacker_x_end<UNP_SEL, FACE_R_DIM>();
            break;
    }
}

} // namespace hal_port

/**
 * @brief Configure the unpacker hardware for both operands A and B.
 *
 * Programs the per-operand source/destination data formats, face dimensions and face counts via
 * configure_unpack_AB, and stores the per-operand tile sizes into the unpack GPRs.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param unpB_src_format: Source data format of operand B in L1.
 * @param unpA_dst_format: Destination data format operand A is converted to.
 * @param unpB_dst_format: Destination data format operand B is converted to.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_tile_size: Tile size of operand A stored to the tile-size GPR.
 * @param unpB_tile_size: Tile size of operand B stored to the tile-size GPR.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    const std::uint32_t unpA_tile_size = 0,
    const std::uint32_t unpB_tile_size = 0)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    hal_port::configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), LOWER_HALFWORD(unpA_tile_size));
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), LOWER_HALFWORD(unpB_tile_size));
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 *
 * Sets the ALU rounding-mode register bits enabling stochastic rounding on the units selected by
 * the template mode.
 *
 * @tparam stoch_rnd_mode: Which units use stochastic rounding, values = <None/Fpu/Pack/All>
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg               = hal::cfg;
    constexpr bool fpu_srnd_en  = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Pack);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>());
}

/**
 * @brief Validates L1 addresses and configures unpack base addresses in the configuration registers
 *
 * This helper function validates that both address_a and address_b are within the valid L1 memory region,
 * then configures the appropriate THCON base address registers based on the unpack configuration context.
 *
 * @param address_a: Address for unpacker A (THCON_SEC0).
 * @param address_b: Address for unpacker B (THCON_SEC1).
 * @param cfg: Pointer to configuration registers.
 * @param context: Unpack config context (0 or 1) whose base registers are written; defaults to unp_cfg_context.
 */
inline void _llk_unpack_configure_addresses_(
    const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg, const std::uint32_t context = unp_cfg_context)
{
    namespace hcfg = hal::cfg;
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    // Whole-word MMIO stores into the state-CFG bank `cfg` (the bank get_cfg_pointer() selected), as production.
    // The context-1 base word follows the context-0 word, so the context id indexes the store instead of branching.
    constexpr auto BaseA = hcfg::Unpacker[0].Cntx[0].Base;
    constexpr auto BaseB = hcfg::Unpacker[1].Cntx[0].Base;
    static_assert(BaseA.shamt(hcfg::Sec::S0) == 0 && BaseA.width == 32 && BaseB.shamt(hcfg::Sec::S0) == 0 && BaseB.width == 32, "base is a whole word");
    static_assert(hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0) == BaseA.addr32(hcfg::Sec::S0) + 1, "context-1 base A follows context 0");
    static_assert(hcfg::Unpacker[1].Cntx[1].Base.addr32(hcfg::Sec::S0) == BaseB.addr32(hcfg::Sec::S0) + 1, "context-1 base B follows context 0");
    LLK_ASSERT(context <= 1, "unpack config context must be 0 or 1");
    // Offset the bank as an integer: GCC can then still prove the stores miss ckernel::cfg_state_id and keep the bank
    // select out of the caller's tile loop, which an indexed pointer store (cfg[base + context]) prevents.
    volatile std::uint32_t tt_reg_ptr *cfg_context =
        reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(reinterpret_cast<std::uintptr_t>(cfg) + context * sizeof(std::uint32_t));
    cfg_context[BaseA.addr32(hcfg::Sec::S0)] = address_a;
    cfg_context[BaseB.addr32(hcfg::Sec::S0)] = address_b;
}
