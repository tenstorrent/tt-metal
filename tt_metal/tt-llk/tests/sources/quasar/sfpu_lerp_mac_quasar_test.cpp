// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Drives the ternary SFPU kernels lerp and mac (SFPU_TERNARY_OPERATION) the way the Compute API does.
// buffer_A holds TILE_CNT / 3 operand triples, one Dest section each: staged into Dest tiles 0, 1 and 2,
// result over tile 0, packed into the section's tile of buffer_Res.
//
// The op's init runs per section, immediately before the kernel, as mac_tile_init does: mac_init's recording
// starts at replay slot 0, which the bank-0 FPU recordings (eltwise binary, matmul, reduce, transpose_dest)
// and sfpi's own replay also use.
//
// Two staging paths, selected by `unpack_to_dest`:
//   * unpack_to_dest=true  — UNPACK writes Dest; SFPU reads/writes Dest; PACK reads Dest.
//   * unpack_to_dest=false — UNPACK -> SrcA -> FPU datacopy -> Dest; SFPU; PACK.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Operand tiles per Dest section; mac's recorded body fixes them, lerp takes them as arguments.
constexpr std::uint32_t TERNARY_OPERAND_TILES = 3;

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    LLK_ASSERT(params.TILE_CNT % TERNARY_OPERAND_TILES == 0, "buffer_A holds whole operand triples");
    const std::uint32_t num_sections = params.TILE_CNT / TERNARY_OPERAND_TILES;

    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
    constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // If Dst is 32b and MATH uses FPU datacopy (MOVA2D → ELWADD fallback), we need both SrcA and SrcB formats configured.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    }

    if constexpr (unpack_to_dest)
    {
        // The to-dest placers carry no synchronization: the dest-dvalid handshake brackets them, and the DEST
        // bank offset reset to 0 keeps every section at DEST tile 0.
        _llk_unpack_dest_init_();
        _llk_unpack_unary_operand_to_dest_init_(bfd_unpack);
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
    }
    else
    {
        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, TERNARY_OPERAND_TILES);
        // SrcA unpack is not a dest-dvalid client; do not inherit an UNP_DEST wait mask from a prior kernel.
        set_up_zero_dest_dvalid_handshake_for_unpack();
    }

    for (std::uint32_t section = 0; section < num_sections; ++section)
    {
        const std::uint32_t l1_tile_idx = section * TERNARY_OPERAND_TILES;
        if constexpr (unpack_to_dest)
        {
            _llk_unpack_unary_operand_to_dest_block_(l1_tile_idx, 0 /*dst_tile_idx*/, TERNARY_OPERAND_TILES);
            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        }
        else
        {
            _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(l1_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"
#include "sfpu_operations_quasar.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    LLK_ASSERT(params.TILE_CNT % TERNARY_OPERAND_TILES == 0, "buffer_A holds whole operand triples");
    LLK_ASSERT(params.DST_INDEX == 0, "mac's recorded body reads tiles 0, 1, 2 and writes tile 0");
    const std::uint32_t num_sections = params.TILE_CNT / TERNARY_OPERAND_TILES;

    const DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);

    // The dvalid chain must match UNPACK exactly. On the FPU path T1 is both the datacopy producer and the
    // SFPU producer.
    if constexpr (unpack_to_dest)
    {
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
    }
    else
    {
        set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
    }

    if constexpr (!unpack_to_dest)
    {
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(params.num_faces * params.TEST_FACE_R_DIM, 1 /*num_matrices*/);
    }

    for (std::uint32_t section = 0; section < num_sections; ++section)
    {
        if constexpr (!unpack_to_dest)
        {
            for (std::uint32_t i = 0; i < TERNARY_OPERAND_TILES; ++i)
            {
                _llk_math_eltwise_unary_datacopy_(i);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
        }

        // Init (see the header comment), then the kernel over all four faces: operands in tiles 0, 1, 2,
        // result over tile 0.
        test_utils::init_ternary_sfpu_operation_quasar<SFPU_TERNARY_OPERATION, is_fp32_dest_acc_en>();
        test_utils::call_ternary_sfpu_operation_quasar<SFPU_TERNARY_OPERATION, dest_sync, is_fp32_dest_acc_en>(
            0 /*src0_tile*/, 1 /*src1_tile*/, 2 /*src2_tile*/, 0 /*dst_tile*/);

        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }

    // Drain every execution unit driven by T1 before PACK takes over.
    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_sections = params.TILE_CNT / TERNARY_OPERAND_TILES;

    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);

    // Declare the same dvalid client chain that UNPACK and MATH use.
    if constexpr (unpack_to_dest)
    {
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    }
    else
    {
        set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    }

    for (std::uint32_t section = 0; section < num_sections; ++section)
    {
        _llk_pack_(params.DST_INDEX, section /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
        // Drain this bank's dest-waited pack before issuing the next.
        ckernel::wait_pack_idle();
    }
}

#endif
