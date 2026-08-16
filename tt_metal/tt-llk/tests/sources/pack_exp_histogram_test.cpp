// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Packer exponent-histogram probe (Blackhole).
//
// WormholeB0/TensixTile/TensixCoprocessor/Packers/ExponentHistogram.md documents a
// per-packer `uint8_t ExponentHistogram[32]` that is incremented for every datum the
// packer fetches from Dst, plus a running max exponent. It is gated on
// ThreadConfig[*].ENABLE_ACC_STATS_Enable, cleared by CLREXPHIST, and read back with
// SETDMAREG modes 6/7 (histogram halves) and 9 (max exponent).
//
// BlackholeA0 has NO packer documentation, so everything here is a measurement. The
// register surface does exist on BH:
//   * ENABLE_ACC_STATS_Enable is ThreadConfig (SETC16) index 45 on BH (46 on WH).
//   * CLREXPHIST is opcode 0x21 in the BH ISA yaml.
//   * BH assembly.yaml documents SETDMAREG Payload_SigSel[6:3] modes 6/7 as
//     "FPU stats 0/1 (payload bits 8:7 select packer)" and 9 as "FPU stats max exp".
//
// This kernel packs one tile of a KNOWN exponent distribution and dumps all four
// packers' 32-byte histograms, the max exponent, and PackerTileSize into a scratch
// tile of buffer_Res.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   HIST_EN        - SETC16 ENABLE_ACC_STATS_Enable = 1 on the pack thread
//   HIST_EN_UNPACK - same, from the unpack thread (the WH model ORs all three threads)
//   HIST_EN_MATH   - same, from the math thread
//   CLR_MODE       - 0 = never clear, 1 = CLREXPHIST from pack thread before the packs,
//                    2 = CLREXPHIST from math thread before dest_section_done
//   NUM_PACKS      - how many times the same Dst tile is packed (saturation / accumulation)
//   CLR_BETWEEN    - CLREXPHIST from the pack thread between consecutive packs
//   DIAG_TILE_INDEX- buffer_Res tile index used for the dump
//   DOWNSAMPLE_MASK- THCON_SEC0_REG1_Downsample_mask (always written; set_packer_config
//                    skips THCON_SEC0_REG1 word 3, so a stale mask survives ELF reload)

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (HIST_EN_UNPACK)
    {
        TTI_SETC16(ENABLE_ACC_STATS_Enable_ADDR32, 1);
    }

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (HIST_EN_MATH)
    {
        TTI_SETC16(ENABLE_ACC_STATS_Enable_ADDR32, 1);
    }

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, params.num_faces);

    // CLREXPHIST is a MATH-resource instruction (BH assembly.yaml: ex_resource MATH).
    // Issuing it from the math thread, before the packer is released, guarantees it is
    // ordered ahead of every PACR of this tile.
    if constexpr (CLR_MODE == 2)
    {
        TTI_CLREXPHIST;
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

constexpr std::uint32_t TDMA_PACKED_SIZE_T2 = RISCV_TDMA_REG_PACKED_SIZE + 0x080;

// SETDMAREG in SetSignals mode writes GPRs. Result size 2 (128 bit) forces the
// destination GPR index to a multiple of four (the functional model masks it with
// 0x3c), so the scratch block is GPR 28..31 -- p_gpr_pack's TMP0/TMP1/TMP_LO/TMP_HI.
constexpr std::uint32_t HIST_GPR = 28;
constexpr std::uint32_t POISON16 = 0x3EAD; // Payload_SigSel is 14 bits wide

// Payload_SigSel = (WhichPackers << 7) | (InputSource << 3) | InputHalfReg
constexpr std::uint32_t sigsel(std::uint32_t which, std::uint32_t src, std::uint32_t half)
{
    return (which << 7) | (src << 3) | half;
}

// Poison the scratch GPRs so a SETDMAREG that quietly does nothing is visible in the
// dump as 0x3EAD3EAD rather than as a plausible all-zero histogram.
inline void poison_gprs()
{
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 0);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 1);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 2);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 3);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 4);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 5);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 6);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_16BIT, POISON16, p_setdmareg::MODE_IMMEDIATE, 2 * HIST_GPR + 7);
}

inline void drain_gprs(volatile std::uint32_t* out)
{
    tensix_sync();
    out[0] = regfile[HIST_GPR + 0];
    out[1] = regfile[HIST_GPR + 1];
    out[2] = regfile[HIST_GPR + 2];
    out[3] = regfile[HIST_GPR + 3];
}

// One 128-bit read of half a packer's histogram (mode 6 = bytes 0..15, mode 7 = 16..31).
template <std::uint32_t WHICH, std::uint32_t SRC>
inline void read_hist_half(volatile std::uint32_t* out)
{
    poison_gprs();
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_128BIT, sigsel(WHICH, SRC, 0), p_setdmareg::MODE_SIGNAL, 2 * HIST_GPR);
    drain_gprs(out);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (HIST_EN)
    {
        TTI_SETC16(ENABLE_ACC_STATS_Enable_ADDR32, 1);
    }
    else
    {
        // Explicit 0: ThreadConfig survives an ELF reload, so a previous variant that
        // enabled the histogram would otherwise contaminate the "off" arm.
        TTI_SETC16(ENABLE_ACC_STATS_Enable_ADDR32, 0);
    }

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
        formats.pack_src,
        formats.pack_dst,
        16 * 16 * 4 /* tile_size */,
        FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces,
        false /* partial_face */,
        false /* narrow_tile */,
        params.RELU_CONFIG /* relu_config */);
    _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
        // Programmed unconditionally: set_packer_config never writes THCON_SEC0_REG1
        // word 3, so a Downsample_mask left behind by an earlier kernel survives an ELF
        // reload and silently decimates this pack.
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(DOWNSAMPLE_MASK);
    }

    _llk_packer_wait_for_math_done_();

    if constexpr (CLR_MODE == 1)
    {
        TTI_CLREXPHIST;
    }

    for (std::uint32_t i = 0; i < NUM_PACKS; ++i)
    {
        if constexpr (CLR_BETWEEN)
        {
            if (i != 0)
            {
                TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
                TTI_CLREXPHIST;
            }
        }
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    // Drain the packer, then let the RISC catch up with the Tensix instruction stream so
    // the reads below observe post-pack state.
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    tensix_sync();

    volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[DIAG_TILE_INDEX]);

    diag[0] = 0xC0DEBA5E; // ran-to-here sentinel

    read_hist_half<0, 6>(&diag[1]);
    read_hist_half<0, 7>(&diag[5]);
    read_hist_half<1, 6>(&diag[9]);
    read_hist_half<1, 7>(&diag[13]);
    read_hist_half<2, 6>(&diag[17]);
    read_hist_half<2, 7>(&diag[21]);
    read_hist_half<3, 6>(&diag[25]);
    read_hist_half<3, 7>(&diag[29]);

    // Mode 9: packer 0's running max exponent. Values[0] only, so a 32-bit result.
    poison_gprs();
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_32BIT, sigsel(0, 9, 0), p_setdmareg::MODE_SIGNAL, 2 * HIST_GPR);
    tensix_sync();
    diag[33] = regfile[HIST_GPR + 0];

    // Poison-only control: proves the poison writes themselves land, so an all-0x3EAD3EAD
    // histogram row means "SETDMAREG signal mode produced nothing", not "GPRs unwritable".
    poison_gprs();
    drain_gprs(&diag[34]);

    diag[38] = reg_read(TDMA_PACKED_SIZE_T2);
    diag[39] = HIST_EN ? 1u : 0u;
    diag[40] = CLR_MODE;
    diag[41] = NUM_PACKS;
    diag[42] = params.num_faces;
    diag[43] = params.buffer_Res[0];
    diag[44] = 0xC0DEE0D1;
}
#endif
