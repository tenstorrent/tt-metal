// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Driver for the column-vector SFPU bodies in metal's
   experimental/llk_sfpu/ckernel_sfpu_sdpa.h.
 *
 * Those bodies have no LLK API of their own. Each consumer declares its own wrapper; ttnn's SDPA
 * does it in transformer/sdpa/device/kernels/compute/compute_common.hpp, one SFPU_UNARY_CALL per
 * body at VectorMode::C. This test declares the same wrapper, so what is under test is the
 * contract a consumer depends on: body, dispatch mode and init together.
 *
 * Footprint. Every body runs ITERATIONS_HALF_FACE = 4 iterations at a dst_reg stride of 2. Within
 * a 16x16 face, one sfpi dst_reg unit is a 32-lane vector covering 16 rows of columns {j, j+8},
 * for unit j in 0..7. The stride-2 walk lands on units 0/2/4/6, and VectorMode::C repeats it on
 * faces 0 and 2, so the bodies write columns {0,2,4,6,8,10,12,14} of all 32 rows and leave the
 * rest of the tile alone. Only column 0 carries meaning for the caller, since these operands are
 * row-reduce outputs broadcast down a column.
 *
 * That footprint was measured on Blackhole. The dst_reg unit layout is not documented in the
 * header, and the two readings that suggest themselves (4 rows x 8 columns, or 16 rows x 2
 * adjacent columns) both predict a contiguous column range, which is not what the hardware does.
 * It is unconfirmed on Wormhole: if the footprint assertion fails there, check the column mask in
 * the Python golden before suspecting the kernel.
 *
 * OP_CORRECTION drives the same skeleton over five DEST tiles instead of one.
 * calculate_fused_max_sub_exp_add_tile reads and writes sfpi dst_reg offsets 0/32/64/96/128, and
 * a 32x32 tile is 32 dst_reg units, so those are DEST tiles idst+0 through idst+4. Regions, in
 * the order ttnn's correction_block (compute_common.hpp:705) assigns dst_reg_0..dst_reg_4:
 *
 *   tile 0  in: prev_max     out: exp(scale * (prev_max   - cur_max))
 *   tile 1  in: worker_max   out: exp(scale * (worker_max - cur_max))
 *   tile 2  in: ignored      out: cur_max = max(prev_max, worker_max)
 *   tile 3  in: prev_sum     out: exp_worker*worker_sum + exp_prev*prev_sum
 *   tile 4  in: worker_sum   out: exp_worker * worker_sum
 *
 * Four of the five are modified in place, so this test packs all five back out, tile 4 included
 * even though ttnn discards it. The body also stores tile 2 and reloads it from DEST within the
 * same iteration to form the two differences.
 */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Which body to drive. Mirrors ColumnVectorOp in helpers/llk_params.py.
constexpr int OP_RECIP_LEGACY = 0; // calculate_recip_first_column<true>,  _reciprocal_compat_
constexpr int OP_RECIP_ITER   = 1; // calculate_recip_first_column<false>, sfpu_reciprocal_iter
// The two exp variants are named after the kernel each one runs, not after the
// SDPA_EXP_APPROX_MODE argument that selects it: true picks _ckernel_sfpu_exp_accurate_, false
// picks calculate_exponential_polynomial at degree 2 on bf16 dest and degree 4 on fp32.
constexpr int OP_EXP_ACCURATE = 2; // calculate_exponential_first_column<true,  EXP_SCALE_BF16>
constexpr int OP_EXP_POLY     = 3; // calculate_exponential_first_column<false, EXP_SCALE_BF16>
constexpr int OP_SOFTPLUS     = 4; // calculate_softplus_first_column
constexpr int OP_CORRECTION   = 5; // calculate_fused_max_sub_exp_add_tile

static_assert(COLVEC_OP >= OP_RECIP_LEGACY && COLVEC_OP <= OP_CORRECTION, "unhandled COLVEC_OP");

constexpr bool COLVEC_OP_IS_EXP = (COLVEC_OP == OP_EXP_ACCURATE || COLVEC_OP == OP_EXP_POLY);

// Derived from the op rather than passed in, so the tile count cannot disagree with the body it
// is sized for. Only the correction body works on more than one tile.
constexpr std::uint32_t NUM_DST_TILES = (COLVEC_OP == OP_CORRECTION) ? 5 : 1;

// The dispatch always targets the base tile. The correction body reaches its other four regions
// by fixed dst_reg offsets from there.
constexpr std::uint32_t COLVEC_DST_INDEX = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// ckernel_sfpu_sdpa.h reads DST_ACCUM_MODE and APPROX as preprocessor macros at include time
// rather than as template parameters, so both are supplied here from the generated build.h.
// sfpu_operations.h uses the same pattern around ckernel_sfpu_binop_with_unary.h. One consequence
// is that every dest-acc and approx combination is a separate build, which is how the harness
// sweeps them in any case. They stay defined for the rest of this section because the recip init
// below is parameterised on the same two values.
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#define APPROX         APPROX_MODE

// ckernel_sfpu_sdpa.h declares sdpa_insert_sfpnop ALWI, but ALWI comes from the Compute API
// (api/compute/common_globals.h), which cannot be included here because it pulls in metal's
// generated chlkc_list.h. This is the definition that header uses.
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"

using namespace ckernel;

// Per-body SFPU init, mirroring what the ttnn SDPA compute kernel runs ahead of each call.
//
// The two reciprocal paths need different inits. legacy_compat=true routes to
// _reciprocal_compat_, which materialises its constants inline, leaving recip_init responsible
// only for the general SFPU state; that is recip_tile_init() with its default legacy_compat, as
// used by compute_common.hpp's recip_block_inplace. legacy_compat=false routes to
// sfpu_reciprocal_iter, whose Newton step reads vConstFloatPrgm0 = 2.0f on Blackhole and the
// reciprocal polynomial coefficients in vConstFloatPrgm0/1/2 on Wormhole. Those come from
// sfpu_reciprocal_init, not recip_init: recip_init's !legacy_compat branch primes the
// _calculate_reciprocal_fast_* kernels, which this body never calls. compute_common.hpp:869 calls
// sfpu_reciprocal_init directly for the same reason.
//
// The exp bodies get exp_init parameterised on the same argument that selects the body, as ttnn
// does at compute_common.hpp:676. The correction body has no approx branch and always runs
// _ckernel_sfpu_exp_accurate_, so it takes the accurate path's init. exp_init's own scale
// argument is left at its default throughout: it feeds only the Schraudolph constants of the
// APPROX && CLAMP_NEGATIVE path, which none of these bodies call, and each carries its own scale
// as a bf16 pattern rather than exp_init's fp32 one.
//
// Softplus gets what softplus_tile_init() expands to, spelled out rather than routed through
// llk_math_eltwise_unary_sfpu_init<SfpuType::softplus>(): that header redeclares
// sfpu::softplus_init(), which ckernel_sfpu_softplus.h has already defined by this point, and the
// harness compiles with -Wredundant-decls -Werror.
inline void column_vector_op_init()
{
    if constexpr (COLVEC_OP == OP_RECIP_LEGACY)
    {
        sfpu::recip_init<APPROX_MODE, is_fp32_dest_acc_en, true /* legacy_compat */>();
    }
    else if constexpr (COLVEC_OP == OP_RECIP_ITER)
    {
        // General SFPU state (config register and the invariant ADDR_MOD_7), then the reciprocal
        // constant the iter path consumes: the exp_tile_init<false>() plus
        // sfpu_reciprocal_init<false>() pair that precedes recip_tile_first_column<false>.
        _llk_math_eltwise_unary_sfpu_init_once_();
        sfpu::sfpu_reciprocal_init<APPROX_MODE>();
    }
    else if constexpr (COLVEC_OP_IS_EXP || COLVEC_OP == OP_CORRECTION)
    {
        sfpu::exp_init<
            COLVEC_OP != OP_EXP_POLY /* APPROXIMATION_MODE: the accurate path for every op but the polynomial one */,
            0x3F800000 /* scale, unused by these bodies */,
            true /* CLAMP_NEGATIVE */,
            is_fp32_dest_acc_en>();
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_init_once_();
        sfpu::softplus_init();
    }
}

// The wrapper under test: each body dispatched the way its consumer dispatches it.
inline void column_vector_op(const std::uint32_t dst_index)
{
    if constexpr (COLVEC_OP == OP_RECIP_LEGACY)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_recip_first_column<true>, dst_index, VectorMode::C);
    }
    else if constexpr (COLVEC_OP == OP_RECIP_ITER)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_recip_first_column<false>, dst_index, VectorMode::C);
    }
    else if constexpr (COLVEC_OP == OP_EXP_ACCURATE)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_exponential_first_column<true, EXP_SCALE_BF16>, dst_index, VectorMode::C);
    }
    else if constexpr (COLVEC_OP == OP_EXP_POLY)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_exponential_first_column<false, EXP_SCALE_BF16>, dst_index, VectorMode::C);
    }
    else if constexpr (COLVEC_OP == OP_SOFTPLUS)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_softplus_first_column, dst_index, VectorMode::C, SOFTPLUS_BETA_BITS, SOFTPLUS_BETA_RECIPROCAL_BITS, SOFTPLUS_THRESHOLD_BITS);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_fused_max_sub_exp_add_tile, dst_index, VectorMode::C, static_cast<int>(EXP_SCALE_BF16));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    column_vector_op_init();

    // The correction body's five tiles do not fit a SyncHalf region under fp32 dest accumulation,
    // where the capacity is four. The Python side omits that pair.
    LLK_ASSERT(
        (NUM_DST_TILES <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "this body needs more DEST tiles than the dest_sync / dest_acc pair can hold");

    _llk_math_wait_for_dest_available_<dest_sync>();

    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    column_vector_op(COLVEC_DST_INDEX);

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();

    // Every tile is packed out. For the correction body that includes the region ttnn leaves in
    // DEST, since four of its five regions are modified in place.
    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
