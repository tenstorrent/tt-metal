// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_top32 presorted-1024 vehicle (lane GK, 2026-08-24) — the
// blaze-dstop32 SKIP_BLOCKED_VEHICLE follow-up (lane FD/EX).  Stages the
// transpose-tile ingest the blaze op run_top32_llk_presorted_1024_opt
// performs (each 1024-element chunk of 32 presorted descending runs is one
// 32x32 tile, ingested TRANSPOSED into Dst so the sorted runs become
// columns), then runs the pre-sorted top-32 pipeline:
//
//   chunk 0: ingest values -> Dst tile 0, indices -> Dst tile 2; prep(0)
//   chunk c: ingest values -> Dst tile 1, indices -> Dst tile 3;
//            prep(1) with the alternate direction; combine(0)
//   final(0): the row's top-32 [value | index] lands in even column 0 of
//             faces F0/F1 of Dst tiles 0 / 2 (the op packs exactly those
//             cells; this vehicle packs the full tiles and the checker
//             reads the contractual cells).
//
// The ingest transpose rides the unpacker (transpose_of_faces=1 +
// within_face_16x16_transpose=1, the topk_test.cpp iteration-0 pattern) —
// for the 16-bit path this is the same tile-transpose state the metal
// transpose_tile ingest produces.  Values are Float16_b, indices UInt16
// (the fp32_dest_acc_en=false path of the original kernel, the path the
// lane-EX lift lifted).
//
// DS_TOP32_IMPL selects the math-thread SFPU phase bodies:
//   0 = vendored byte-exact blaze original (blaze_vendored kernel_includes
//       ckernel_sfpu_deepseek_top32_rm.h)
//   1 = lane-EX builtin-bridge semantic lift (blaze_vendored
//       semantic/deepseek_top32_rm.hpp)
//   2 = 32-bit arm (is_fp32_dest_acc_en=true — the blaze sampling pipeline's
//       production flag): values bf16 via SrcA-transpose datacopy; indices
//       Int32 via unpack_to_dest (the unpacker cannot within-face-transpose
//       32-bit data) + the HAND in-tree math transpose LLK
//       (_llk_math_transpose_dest_, MOP/replay) for the index tiles; phases =
//       vendored original with is_fp32_dest_acc_en=true.
//   3 = same as 2 with the index-tile within-face transpose on the TYPED X6
//       surface (sfpi::face_transpose_dst_32b_batch, lane FV) — the
//       transpose-ingest algebra X6 was built for.  SrcB-format contract
//       edge: the math format is Float16_b (8b-exponent class) and nothing
//       in the pipeline rewrites ALU SrcB, so the X6 composition is exact;
//       index payloads with zero low bytes (e.g. 0x100) exercise the cfg
//       block's zero-flag arm naturally.
//       Bank-grant protocol (the FS bank-valid wedge class): the unpack
//       thread sends ONE _llk_unpack_set_srcb_dummy_valid_ per chunk after
//       that chunk's feeds; the math thread consumes exactly one grant per
//       chunk (the hand LLK's own per-tile CLR_AB for impl 2 /
//       face_transpose_release_banks per epoch for impl 3) — balanced on
//       every path, no early exits.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

#ifndef DS_TOP32_IMPL
#define DS_TOP32_IMPL 0
#endif

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t TILES_PER_CHUNK = 1; // 1024 elements = one 32x32 tile
constexpr bool APPROX                   = false;

// buffer_A layout per row: [values c0..c{N-1}][indices c0..c{N-1}].
inline constexpr std::uint32_t value_tile_l1_index(std::uint32_t r, std::uint32_t c)
{
    return r * (2 * DS_TOP32_NUM_CHUNKS) + c;
}

inline constexpr std::uint32_t index_tile_l1_index(std::uint32_t r, std::uint32_t c)
{
    return r * (2 * DS_TOP32_NUM_CHUNKS) + DS_TOP32_NUM_CHUNKS + c;
}

enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

// ============================================================================
// UNPACK TRISC — per (row, chunk, stage): one tile, transposed both levels.
// ============================================================================

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

#if DS_TOP32_IMPL >= 2

// 32-bit arm: values bf16 through SrcA (unpacker transposes both levels);
// indices Int32 unpack_to_dest (face reorder only — the within-face
// transpose happens on the math thread); ONE SrcB dummy-valid grant per
// chunk for the math-thread index transpose.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t idx_format = ckernel::to_underlying(DataFormat::Int32);

    for (std::uint32_t r = 0; r < DS_TOP32_NUM_ROWS; ++r)
    {
        for (std::uint32_t c = 0; c < DS_TOP32_NUM_CHUNKS; ++c)
        {
            const bool first_hardware_configuration = (r == 0 && c == 0);
            // Values: bf16 -> SrcA, transpose both levels.
            if (first_hardware_configuration)
            {
                _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                    formats.unpack_A_src,
                    formats.unpack_A_src,
                    formats.unpack_A_dst,
                    formats.unpack_A_dst,
                    FACE_R_DIM,
                    FACE_R_DIM,
                    4 /* num_faces */,
                    4 /* num_faces */);
            }
            else
            {
                _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                    formats.unpack_A_src, formats.unpack_A_dst, 16 * 16 * 4 /* tile_size */);
            }
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
                1 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
                L1_ADDRESS(params.buffer_A[r * DS_TOP32_NUM_CHUNKS + c]), formats.unpack_A_src, formats.unpack_A_dst);

            // Indices: Int32 unpack_to_dest with face reorder.
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                idx_format, idx_format, 16 * 16 * 4 /* tile_size */);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
                1 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, idx_format, idx_format);
            // Index tiles ride buffer_B: Int32 is a different L1 tile width
            // than the bf16 value tiles (the harness packs one format per
            // stimulus buffer).
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
                L1_ADDRESS(params.buffer_B[r * DS_TOP32_NUM_CHUNKS + c]), idx_format, idx_format);

            // ONE bank grant per chunk for the math-thread index transpose
            // (hand LLK per-tile pairing == X6 per-epoch, one tile per chunk).
            _llk_unpack_set_srcb_dummy_valid_();
        }
    }
}

#else // DS_TOP32_IMPL < 2

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t unpack_src_data_types[2] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[2] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    for (std::uint32_t r = 0; r < DS_TOP32_NUM_ROWS; ++r)
    {
        for (std::uint32_t c = 0; c < DS_TOP32_NUM_CHUNKS; ++c)
        {
            for (Stage stage : {Stage::Values, Stage::Indices})
            {
                const int stage_index                 = static_cast<int>(stage);
                const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
                const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

                const bool first_hardware_configuration = (r == 0 && c == 0 && stage == Stage::Values);
                if (first_hardware_configuration)
                {
                    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                        unpack_src_format,
                        unpack_src_format,
                        unpack_dst_format,
                        unpack_dst_format,
                        FACE_R_DIM,
                        FACE_R_DIM,
                        4 /* num_faces */,
                        4 /* num_faces */);
                }
                else
                {
                    // Reconfigure between the bf16 value stage and the u16 index
                    // stage (the topk_test.cpp two-stage pattern).
                    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                        unpack_src_format, unpack_dst_format, 16 * 16 * 4 /* tile_size */);
                }

                // Transpose-tile ingest: face reorder + within-face 16x16.
                _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    1 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, unpack_src_format, unpack_dst_format);

                const std::uint32_t tile_index = (stage == Stage::Values) ? value_tile_l1_index(r, c) : index_tile_l1_index(r, c);
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[tile_index]), unpack_src_format, unpack_dst_format);
            }
        }
    }
}

#endif // DS_TOP32_IMPL
#endif // LLK_TRISC_UNPACK

// ============================================================================
// MATH TRISC — datacopy ingest + the pre-sorted top-32 pipeline.
// ============================================================================

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace ckernel;

// The vendored original carries an unused constexpr (`increasing` in the
// final phase) that this tree's -Werror rejects; suppress around the include
// so the vendored file stays byte-exact (VENDORED.md discipline).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#if DS_TOP32_IMPL == 0 || DS_TOP32_IMPL >= 2
// Vendored byte-exact blaze original (impl 2/3 run its fp32-dest-acc arm).
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"
#elif DS_TOP32_IMPL == 1
// Lane-EX builtin-bridge semantic lift (pulls the original for the shared
// protocol helpers; phase bodies are the semantic:: lifts).
#include "blaze/kernels/sfpu/semantic/deepseek_top32_rm.hpp"
#else
#error "Unknown DS_TOP32_IMPL selector"
#endif
#pragma GCC diagnostic pop

inline void ds_top32_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
#if DS_TOP32_IMPL == 1
    ckernel::sfpu::semantic::_semantic_top32_rm_init_();
#else
    ckernel::sfpu::_top32_rm_init_();
#endif
}

// Debug bisection knob (lane GK): bit 0 = route prep to the ORIGINAL even
// under impl 1; bit 1 = route combine; bit 2 = route final.  0 for normal.
#ifndef DS_TOP32_LIFT_BISECT
#define DS_TOP32_LIFT_BISECT 0
#endif

template <bool top_min>
inline void ds_top32_prep(std::uint32_t dst_index)
{
#if DS_TOP32_IMPL != 1 || (DS_TOP32_LIFT_BISECT & 1)
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_prep_<APPROX, is_fp32_dest_acc_en, top_min>, dst_index, VectorMode::RC_custom, dst_index);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::semantic::_semantic_bitonic_top32_of_1024_rm_pre_sorted_prep_<APPROX, is_fp32_dest_acc_en, top_min>,
        dst_index,
        VectorMode::RC_custom,
        dst_index);
#endif
}

inline void ds_top32_combine(std::uint32_t dst_index)
{
#if DS_TOP32_IMPL != 1 || (DS_TOP32_LIFT_BISECT & 2)
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_combine_<APPROX, is_fp32_dest_acc_en>, dst_index, VectorMode::RC_custom, dst_index);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::semantic::_semantic_bitonic_top32_of_1024_rm_pre_sorted_combine_<APPROX, is_fp32_dest_acc_en>,
        dst_index,
        VectorMode::RC_custom,
        dst_index);
#endif
}

inline void ds_top32_final(std::uint32_t dst_index)
{
#if DS_TOP32_IMPL != 1 || (DS_TOP32_LIFT_BISECT & 4)
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_final_<APPROX, is_fp32_dest_acc_en>, dst_index, VectorMode::RC_custom, dst_index);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::semantic::_semantic_bitonic_top32_of_1024_rm_pre_sorted_final_<APPROX, is_fp32_dest_acc_en>,
        dst_index,
        VectorMode::RC_custom,
        dst_index);
#endif
}

// Ingest one chunk: values -> Dst tile value_dst, indices -> value_dst + 2
// (the kernel's dst_indices_offset = 128 = 2 tiles).
inline void ds_top32_ingest_chunk(std::uint32_t value_dst, const std::uint32_t (&math_data_types)[2], bool first_hardware_configuration)
{
    for (Stage stage : {Stage::Values, Stage::Indices})
    {
        const int stage_index           = static_cast<int>(stage);
        const std::uint32_t math_format = math_data_types[stage_index];

        if (first_hardware_configuration && stage == Stage::Values)
        {
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        }
        else
        {
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(math_format);
        }

        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(
            /*num_rows_per_matrix=*/4, /*math_format=*/math_format);

        const std::uint32_t dst_tile = (stage == Stage::Values) ? value_dst : (value_dst + 2);
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst_tile, math_format, math_format);
    }
}

#if DS_TOP32_IMPL >= 2

// 32-bit ingest: value tile bf16 SrcA-datacopy -> Dst tile value_dst (fp32),
// index tile Int32 unpack_to_dest -> Dst tile value_dst + 2, then the
// within-face transpose of the index tile (hand LLK for impl 2, X6 typed
// surface for impl 3), consuming the chunk's one bank grant.
inline void ds_top32_ingest_chunk_32b(std::uint32_t value_dst, std::uint32_t math_format, bool first_hardware_configuration)
{
    const std::uint32_t idx_format = ckernel::to_underlying(DataFormat::Int32);

    if (first_hardware_configuration)
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    }
    else
    {
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(math_format);
    }
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        /*num_rows_per_matrix=*/4, /*math_format=*/math_format);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
        value_dst, math_format, math_format);

    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(idx_format);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        /*num_rows_per_matrix=*/4, /*math_format=*/idx_format);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, true /* unpack_to_dest */>(
        value_dst + 2, idx_format, idx_format);

    const std::uint32_t index_dst = value_dst + 2;
#if DS_TOP32_IMPL == 2
    // Hand in-tree math transpose LLK (MOP/replay), within-face only —
    // faces were reordered by the unpacker.  Its own per-tile grant
    // consume/CLR_AB pairing (transpose_dest_test.cpp protocol).
    _llk_math_transpose_dest_init_<false /* transpose_of_faces */, true /* is_32bit */>();
    _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false /* transpose_of_faces */, true /* is_32bit */>(index_dst);
#else
    // X6 typed surface (lane FV vehicle form): one cfg epoch per index tile,
    // tile base through the LLK's own addressing seam, one grant consumed,
    // banks released at epoch end (drain on the only exit path).
    sfpi::face_transpose_cfg_enter();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(index_dst);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::face_transpose_dst_32b_batch<4, 0, /*OuterCfg=*/false>();
    sfpi::face_transpose_cfg_leave();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    sfpi::face_transpose_release_banks();
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(is_fp32_dest_acc_en, "deepseek top32 32-bit arms need fp32 dest accumulation");

    const std::uint32_t math_format = formats.math;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    for (std::uint32_t r = 0; r < DS_TOP32_NUM_ROWS; ++r)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();

        ds_top32_ingest_chunk_32b(0, math_format, r == 0);
        ds_top32_init();
        ds_top32_prep<false /* top_min */>(0);

        for (std::uint32_t c = 1; c < DS_TOP32_NUM_CHUNKS; ++c)
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            ds_top32_ingest_chunk_32b(1, math_format, false);
            ds_top32_prep<true /* top_min */>(1);
            ds_top32_combine(0);
        }

        ds_top32_final(0);

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#else // DS_TOP32_IMPL < 2

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(!is_fp32_dest_acc_en, "deepseek top32 vehicle: 16-bit Dst path (the lane-EX lift's contract)");

    const std::uint32_t math_data_types[2] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    for (std::uint32_t r = 0; r < DS_TOP32_NUM_ROWS; ++r)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();

        // Chunk 0 -> value tile 0 / index tile 2, then prep(0).
        ds_top32_ingest_chunk(0, math_data_types, r == 0);
        ds_top32_init();
        ds_top32_prep<false /* top_min */>(0);

        for (std::uint32_t c = 1; c < DS_TOP32_NUM_CHUNKS; ++c)
        {
            // The chunk's FPU ingest follows SFPU phase work inside the same
            // dest section: drain the SFPU before the datacopy touches Dst.
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            ds_top32_ingest_chunk(1, math_data_types, false);
            ds_top32_prep<true /* top_min */>(1);
            ds_top32_combine(0);
        }

        ds_top32_final(0);

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // DS_TOP32_IMPL
#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC — value tile 0 and index tile 2 per row (full tiles; the
// checker reads the contractual even-col-0 F0/F1 cells).
// ============================================================================

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#if DS_TOP32_IMPL >= 2
    // 32-bit arms: value tile = fp32 Dst datums, index tile = Int32.
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    (void)formats;
#endif
    const std::uint32_t pack_src_data_types[2] = {ckernel::to_underlying(DataFormat::Float32), ckernel::to_underlying(DataFormat::Int32)};
    const std::uint32_t pack_dst_data_types[2] = {ckernel::to_underlying(DataFormat::Float32), ckernel::to_underlying(DataFormat::Int32)};
#else
    const std::uint32_t pack_src_data_types[2] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[2] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};
#endif

    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t r = 0; r < DS_TOP32_NUM_ROWS; ++r)
    {
        _llk_packer_wait_for_math_done_();

        for (Stage stage : {Stage::Values, Stage::Indices})
        {
            const int stage_index               = static_cast<int>(stage);
            const std::uint32_t pack_src_format = pack_src_data_types[stage_index];
            const std::uint32_t pack_dst_format = pack_dst_data_types[stage_index];

            const bool first_hardware_configuration = (r == 0 && stage == Stage::Values);
            if (first_hardware_configuration)
            {
                _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(pack_src_format, pack_dst_format, 16 * 16 * 4 /* tile_size */);
            }
            else
            {
                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false /* is_tile_dim_reconfig_en */>(
                    pack_src_format,
                    pack_dst_format,
                    16 * 16 * 4 /* tile_size */,
                    FACE_R_DIM,
                    TILE_C_DIM,
                    4 /* num_faces */,
                    false /* partial_face */,
                    false /* narrow_tile */,
                    1 /* num_tiles */);
            }
            _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_dst_format);

            const std::uint32_t dst_tile = (stage == Stage::Values) ? 0 : 2;
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(dst_tile, L1_ADDRESS(params.buffer_Res[2 * r + stage_index]));
        }

        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}
#endif // LLK_TRISC_PACK
