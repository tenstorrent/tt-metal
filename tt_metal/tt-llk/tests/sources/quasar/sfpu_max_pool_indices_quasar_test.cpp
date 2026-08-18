// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-08-07_max_pool_indices_quasar_c0c71599
//
// Column-wise max reduction with argmax (max_pool_with_indices) on Quasar.
//
// The unified binary SFPU harness cannot express this op: it stages one input
// format for both operands and its golden is element-wise. This op is a
// reduction over a (values, indices) Dest tile PAIR, staged with two different
// formats, and it writes both tiles. The staging therefore follows the topk
// pair (sources/quasar/sfpu_topk_quasar_test.cpp), the only Quasar test that
// already moves a value tile and an integer index tile through the same
// pipeline.
//
// Buffer layout:
//   buffer_A[0] -> Dest[0] current values
//   buffer_A[1] -> Dest[1] prior accumulated values
//   buffer_A[2] -> Dest[2] current indices
//   buffer_A[3] -> Dest[3] prior accumulated indices
//   Dest[0]     -> buffer_Res[0] reduced values, row 0 of every column
//   Dest[2]     -> buffer_Res[1] winning indices, row 0 of every column
//
// The 16-bit path follows topk: T0 unpack (L1 -> SrcA, per-stage format) -> T1
// FPU A2D datacopy (SrcA -> Dest) -> T1 SFPU -> T2 pack. The 32-bit path stages
// all four tiles as raw Int32 words with UNP_DEST; the SFPU then loads value
// words as FP32 and index words as INT32. Direct staging avoids the FPU
// datacopy flushing small index bit patterns as FP32 denormals.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // MAX_POOL_CONFIG, is_fp32_dest_acc_en, dest_sync

// Container the index tags ride in from L1 to Dest and back.
//
// 16-bit Dest uses Int16 integer transport, matching the kernel's
// IDX_MODE = sfpmem::UINT16.
//
// 32-bit Dest uses Int32 raw-word transport, matching sfpmem::INT32. The value
// tiles share that raw staging format and are reinterpreted by sfpmem::FP32.
constexpr DataFormat MAX_POOL_INDEX_FORMAT = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::Int16;

constexpr std::uint32_t MAX_POOL_VALUES_TILE        = 0;
constexpr std::uint32_t MAX_POOL_VALUES_ACCUM_TILE  = 1;
constexpr std::uint32_t MAX_POOL_INDICES_TILE       = 2;
constexpr std::uint32_t MAX_POOL_INDICES_ACCUM_TILE = 3;
constexpr std::uint32_t MAX_POOL_STAGED_TILES       = 4;

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;

    if constexpr (is_fp32_dest_acc_en)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<false /*implied math*/, true /*fp32 dest*/, false /*int32 dest*/>();
    }
    else
    {
        // Dest dvalid chain: FPU (datacopy) -> SFPU (reduction) -> PACK.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc    = bd_val;
    td_val.buf_desc_id = buf_desc_id;

    if constexpr (is_fp32_dest_acc_en)
    {
        // All four 32-bit tiles are raw words. UNP_DEST preserves those words,
        // while the SFPU's explicit load modes choose FP32 or INT32 semantics.
        td_val.buf_desc.f.format = static_cast<std::uint8_t>(to_underlying(DataFormat::Int32));
        td_val.reg_data_format   = static_cast<std::uint8_t>(to_underlying(DataFormat::Int32));
        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(td_val);
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*transpose*/, true /*32b Dest*/, EltwiseBinaryReuseDestType::NONE, true /*unpack_to_dest*/>(
            buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, MAX_POOL_STAGED_TILES);
        _llk_unpack_unary_operand_<p_unpacr::UNP_DEST, EltwiseBinaryReuseDestType::NONE, true /*unpack_to_dest*/, dest_sync>(
            0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
    else
    {
        // Current and prior-accumulator value tiles: the swept L1 format.
        td_val.buf_desc.f.format = static_cast<std::uint8_t>(formats.unpack_A_src);
        td_val.reg_data_format   = static_cast<std::uint8_t>(formats.unpack_A_dst);
        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, false /*32b Dest*/>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(MAX_POOL_VALUES_TILE, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(MAX_POOL_VALUES_ACCUM_TILE, ckernel::DEFAULT_TENSOR_SHAPE);

        // Current and prior-accumulator index tags: integer payloads carried bit-exactly.
        td_val.buf_desc.f.format = static_cast<std::uint8_t>(to_underlying(MAX_POOL_INDEX_FORMAT));
        td_val.reg_data_format   = static_cast<std::uint8_t>(to_underlying(MAX_POOL_INDEX_FORMAT));
        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, false /*32b Dest*/>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(MAX_POOL_INDICES_TILE, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(MAX_POOL_INDICES_ACCUM_TILE, ckernel::DEFAULT_TENSOR_SHAPE);
    }
}

#endif // LLK_TRISC_UNPACK

// ============================================================================
// MATH TRISC
// ============================================================================

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu/ckernel_sfpu_max_pool_indices.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (is_fp32_dest_acc_en)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        // Math owns both the FPU (datacopy) and SFPU clients of the dvalid chain.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    const DataFormat value_math_format = static_cast<DataFormat>(formats.math);

    SFPU_BINARY_INIT_FN(max_pool_with_indices, sfpu::init_max_pool_with_indices, (false /*APPROX*/, MAX_POOL_LAYOUT));

    if constexpr (!is_fp32_dest_acc_en)
    {
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, false /*32b Dest*/>(num_rows, 1);

        // Value tiles -> Dest[0:2].
        _configure_alu_formats_<false /*EN_IMPLIED_MATH_FORMAT*/, false /*32b Dest*/>(
            value_math_format, value_math_format, false /*en_int32_dest_format*/, DataFormat::Invalid);
        _llk_math_eltwise_unary_datacopy_(MAX_POOL_VALUES_TILE);
        _llk_math_eltwise_unary_datacopy_(MAX_POOL_VALUES_ACCUM_TILE);

        // Index-tag tiles -> Dest[2:4], staged in MAX_POOL_INDEX_FORMAT.
        _configure_alu_formats_<false /*EN_IMPLIED_MATH_FORMAT*/, false /*32b Dest*/>(
            MAX_POOL_INDEX_FORMAT, MAX_POOL_INDEX_FORMAT, false /*en_int32_dest_format*/, DataFormat::Invalid);
        _llk_math_eltwise_unary_datacopy_(MAX_POOL_INDICES_TILE);
        _llk_math_eltwise_unary_datacopy_(MAX_POOL_INDICES_ACCUM_TILE);

        // All four tiles are staged: release the FPU dvalid to the SFPU client.
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    // VectorMode::None: both layouts cover all columns in one call.
    SFPU_BINARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        calculate_max_pool_with_indices,
        (false /*APPROX*/, is_fp32_dest_acc_en, MAX_POOL_NUM_ROWS, SFPU_ITERATIONS, MAX_POOL_LAYOUT, MAX_POOL_ACCUMULATE),
        MAX_POOL_VALUES_TILE,
        MAX_POOL_INDICES_TILE,
        MAX_POOL_VALUES_TILE, // unused out slot
        VectorMode::None,
        MAX_POOL_CHUNK);

    wait_sfpu_idle();
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    if constexpr (!is_fp32_dest_acc_en)
    {
        wait_fpu_idle();
    }
    wait_mop_idle();
}

#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 8;

    if constexpr (is_fp32_dest_acc_en)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc         = buffer_descriptor_u {0};
    tdma_desc.buf_desc.f.x_dim = params.TEST_FACE_C_DIM;
    tdma_desc.buf_desc.f.y_dim = params.TEST_FACE_R_DIM;
    tdma_desc.buf_desc.f.z_dim = params.num_faces;
    tdma_desc.buf_desc_id      = buf_desc_id;

    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);

    // Reduced values tile.
    tdma_desc.buf_desc.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
    tdma_desc.buf_desc.f.l1_addr_16B = L1_ADDRESS(params.buffer_Res[0]);
    tdma_desc.reg_data_format        = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
    _llk_pack_(MAX_POOL_VALUES_TILE, 0 /*tile index*/, ckernel::DEFAULT_TENSOR_SHAPE);

    // Winning indices tile, integer transport on both sides.
    tdma_desc.buf_desc.f.format      = static_cast<std::uint8_t>(to_underlying(MAX_POOL_INDEX_FORMAT));
    tdma_desc.buf_desc.f.l1_addr_16B = L1_ADDRESS(params.buffer_Res[1]);
    tdma_desc.reg_data_format        = static_cast<std::uint8_t>(to_underlying(MAX_POOL_INDEX_FORMAT));
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
    _llk_pack_(MAX_POOL_INDICES_TILE, 0 /*tile index*/, ckernel::DEFAULT_TENSOR_SHAPE);

    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
