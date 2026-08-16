// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Packer zero-compression probe (Blackhole).
//
// Packs a single tile whose contents are a known sparse pattern, with the packer's
// zero-compression path either disabled (baseline) or enabled, and then dumps the
// packer sideband metadata registers into a scratch tile of buffer_Res so the host
// can read them back alongside the raw packed bytes.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   COMPRESS_EN             - clear THCON_SEC0_REG1_Disable_zero_compress before packing
//   ROW_START_SECTION_SIZE  - THCON_SEC0_REG1_Row_start_section_size, in 16B units
//   DIAG_TILE_INDEX         - buffer_Res tile index used for the metadata dump
//   DOWNSAMPLE_MASK         - THCON_SEC0_REG1_Downsample_mask (0 == disabled; always written)
//   ENABLE_OUT_FIFO         - THCON_SEC0_REG1_Enable_out_fifo, to read the metadata FIFO
//   CONCAT_ROWS             - issue the pack PACRs by hand with Concat set, so the whole
//                             tile is one compression row instead of one row per PACR

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
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, params.num_faces);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

// Packer sideband metadata, per TDMA-RISC memory map. The +0x040 / +0x080 strides select
// which Tensix thread's "last tile" is reported; the pack thread is T2.
constexpr std::uint32_t TDMA_PACKED_SIZE_T0     = RISCV_TDMA_REG_PACKED_SIZE;         // PackerTileSize(0, 0)
constexpr std::uint32_t TDMA_PACKED_SIZE_T1     = RISCV_TDMA_REG_PACKED_SIZE + 0x040; // PackerTileSize(0, 1)
constexpr std::uint32_t TDMA_PACKED_SIZE_T2     = RISCV_TDMA_REG_PACKED_SIZE + 0x080; // PackerTileSize(0, 2)
constexpr std::uint32_t TDMA_ALL_ZERO_FLAGS_P0  = 0xFFB11020;                         // Packers[0].AllZeroFlags
constexpr std::uint32_t TDMA_ACC_PACKED_SIZE_T2 = RISCV_TDMA_REG_ACC_PACKED_SIZE + 0x080;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
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

    // The LLK unconditionally programs Disable_zero_compress = 1 ("uncompress") in
    // set_packer_config. Flip it back, and reserve room for the row-start-index array,
    // before the first PACR of the tile.
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
        if constexpr (ROW_START_SECTION_SIZE != 0)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_RMW>(ROW_START_SECTION_SIZE);
        }
        if constexpr (COMPRESS_EN)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Disable_zero_compress_RMW>(0);
        }
        // Programmed unconditionally, including the 0 (disabled) case: set_packer_config never
        // writes THCON_SEC0_REG1 word 3, so a Downsample_mask left behind by an earlier kernel
        // survives an ELF reload and silently decimates this pack.
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(DOWNSAMPLE_MASK);
        if constexpr (ENABLE_OUT_FIFO)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Enable_out_fifo_RMW>(1);
        }
    }

    _llk_packer_wait_for_math_done_();
    if constexpr (CONCAT_ROWS)
    {
        // Same PACR sequence the Default pack MOP would issue (4 outer x 4 inner, with
        // ADDR_MOD_0 / ADDR_MOD_2 / ADDR_MOD_1-and-Last), but with Concat set on every
        // PACR except the last, so the whole tile is a single compression row.
        set_dst_write_addr<false>(0);
        program_packer_destination(L1_ADDRESS(params.buffer_Res[0]));
// Concat is 1 on every PACR but the last, so all 16 PACRs form one compression row.
#define PZC_PACR(AM, CONCAT, LAST)      \
    TTI_PACR(                           \
        p_pacr::CFG_CTXT_0,             \
        p_pacr::NO_ROW_PAD_ZERO,        \
        p_pacr::DST_ACCESS_NORMAL_MODE, \
        AM,                             \
        p_pacr::ADDR_CNT_CTXT_0,        \
        p_pacr::P_ZERO_OUTPUT_DISABLED, \
        p_pacr::ALL_INTF_ACTIVE,        \
        0,                              \
        CONCAT,                         \
        0,                              \
        0,                              \
        LAST)
#define PZC_FACE(LASTMOD, CONCAT_LAST, LAST) \
    PZC_PACR(ADDR_MOD_0, 1, 0);              \
    PZC_PACR(ADDR_MOD_0, 1, 0);              \
    PZC_PACR(ADDR_MOD_0, 1, 0);              \
    PZC_PACR(LASTMOD, CONCAT_LAST, LAST)

        PZC_FACE(ADDR_MOD_2, 1, 0);
        PZC_FACE(ADDR_MOD_2, 1, 0);
        PZC_FACE(ADDR_MOD_2, 1, 0);
        PZC_FACE(ADDR_MOD_1, 0, 1);
#undef PZC_FACE
#undef PZC_PACR
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
    else
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    // Drain the packer, then let the RISC catch up with the Tensix instruction stream so
    // the MMIO reads below observe post-pack state.
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    tensix_sync();

    volatile std::uint32_t* diag           = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[DIAG_TILE_INDEX]);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    diag[0]  = 0xC0DEBA5E; // ran-to-here sentinel
    diag[1]  = reg_read(TDMA_PACKED_SIZE_T0);
    diag[2]  = reg_read(TDMA_PACKED_SIZE_T1);
    diag[3]  = reg_read(TDMA_PACKED_SIZE_T2);
    diag[4]  = reg_read(TDMA_ALL_ZERO_FLAGS_P0);
    diag[5]  = reg_read(TDMA_ACC_PACKED_SIZE_T2);
    diag[6]  = reg_read(RISCV_TDMA_REG_FIFO_PACKED_TILE_STATUS);
    diag[7]  = cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0]; // row_ptr | exp section sizes
    diag[8]  = cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 1]; // L1_Dest_addr
    diag[9]  = cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2]; // uncompress | formats | ...
    diag[10] = cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3]; // downsample mask | ...
    diag[11] = params.buffer_Res[0];                                   // packed-tile base, byte address
    if constexpr (ENABLE_OUT_FIFO)
    {
        // Peek waits on an empty FIFO, so only touch it when the status word says
        // packer 0's FIFO is non-empty (bit 0 set == empty).
        if ((diag[6] & 0x1) == 0)
        {
            diag[13] = reg_read(RISCV_TDMA_REG_FIFO_PACKED_TILE_SIZE(0));     // Peek().TileSize
            diag[14] = reg_read(RISCV_TDMA_REG_FIFO_PACKED_TILE_ZEROMASK(0)); // Pop().AllZeroFlags
        }
        else
        {
            diag[13] = 0xDEADF1F0u;
            diag[14] = 0xDEADF1F0u;
        }
    }
    diag[15] = params.num_faces;
    diag[12] = 0xC0DEE0D1;
}
#endif
