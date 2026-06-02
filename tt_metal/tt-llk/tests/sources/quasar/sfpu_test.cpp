// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = params.TILE_CNT;

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    if constexpr (unpack_to_dest)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles_per_unpack);

    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/);

    if constexpr (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"

// Op header is selected per-variant by the Python driver (SFPU_DEFINES).
// build.h (pulled in by params.h) defines SFPU_INCLUDE_HEADER, so this must
// come after params.h.
#include SFPU_INCLUDE_HEADER

#ifndef SFPU_INIT
#define SFPU_INIT
#endif
#ifndef SFPU_ADDITIONAL_ARGS
#define SFPU_ADDITIONAL_ARGS
#endif
// Binary operand placement: gather operands from DEST tiles 0 and 1, write the
// result to tile 0. This is the kernel's default for every binary op; only ops
// that index by element offset instead of tile index (the int ops) override it.
#ifndef SFPU_BINARY_OPERANDS
#define SFPU_BINARY_OPERANDS 0u, 1u, 0u
#endif
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    if constexpr (!unpack_to_dest)
    {
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    // SFPU processes SFP_ROWS dest rows per issue, so every arity loops this many times.
    const std::uint32_t num_sfpu_iterations = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;

    /**
     * Per-op defines supplied by the build (SFPU_DEFINES in the Python driver):
     *   SFPU_<ARITY>_OP      - arity selector: SFPU_UNARY_OP / SFPU_BINARY_OP / SFPU_TERNARY_OP
     *   SFPU_INCLUDE_HEADER  - op header (included above, after params.h)
     *   SFPU_OP_CALL         - ckernel::sfpu::<function> to invoke; may be a lambda defined in
     *                          SFPU_INIT so that ops with non-standard signatures (fill, max/min,
     *                          add_int32, swiglu) fit the existing three dispatch patterns
     *   SFPU_TYPE            - SfpuType enum value (ternary init only)
     *   SFPU_INIT            - optional init statement(s); may also define an SFPU_OP_CALL wrapper
     *   SFPU_ADDITIONAL_ARGS - extra trailing args forwarded to the unary/ternary op call
     *   SFPU_BINARY_OPERANDS - binary operand placement (default: tile-index 0u,1u,0u)
     *
     * The three arities are asymmetric in Quasar LLK: unary/binary share a plain
     * _llk_math_eltwise_sfpu_init_() and forward (iterations[, offsets]) to the op;
     * only the ternary init is templated on SfpuType and its dispatch takes four
     * tile indices plus a VECTOR_MODE face selector.
     */

#if defined(SFPU_UNARY_OP)
    _llk_math_eltwise_sfpu_init_();
    SFPU_INIT
    // Apply the op in place to each datacopied tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(SFPU_OP_CALL, params.DST_INDEX + i, num_sfpu_iterations SFPU_ADDITIONAL_ARGS);
    }
#elif defined(SFPU_BINARY_OP)
    _llk_math_eltwise_sfpu_init_();
    SFPU_INIT
    // tile_stride is the element stride between DEST tiles, for offset-style ops
    // that override SFPU_BINARY_OPERANDS (e.g. "0, tile_stride, 0").
    const int tile_stride = static_cast<int>(params.num_faces * params.TEST_FACE_R_DIM);
    (void)tile_stride;
    _llk_math_eltwise_binary_sfpu_params_<false>(SFPU_OP_CALL, 0u, num_sfpu_iterations, SFPU_BINARY_OPERANDS);
#elif defined(SFPU_TERNARY_OP)
    // The ternary op takes its iteration count as a template arg (calculate_where's
    // ITERATIONS), not a runtime one, so num_sfpu_iterations is unused on this path.
    (void)num_sfpu_iterations;
    _llk_math_eltwise_ternary_sfpu_init_<SFPU_TYPE>();
    SFPU_INIT
    // in0/in1/in2 at DEST tiles 0/1/2, result to tile 0; VECTOR_MODE selects faces.
    _llk_math_eltwise_ternary_sfpu_params_(SFPU_OP_CALL, 0u, 1u, 2u, 0u, VECTOR_MODE SFPU_ADDITIONAL_ARGS);
#else
    LLK_ASSERT(false, "Missing SFPU arity define (SFPU_UNARY_OP / SFPU_BINARY_OP / SFPU_TERNARY_OP)");
#endif

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
}

#endif

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
    const std::uint32_t buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = 1;

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);

    // Packs only the result tile (DEST[DST_INDEX]); where produces one output tile
    // regardless of how many input tiles were loaded.
    _llk_pack_(params.DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
