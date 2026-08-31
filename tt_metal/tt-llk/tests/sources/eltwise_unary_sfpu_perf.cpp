// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;

    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;

    const auto& buffer_A = params.buffer_A;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        // acc_to_dest must be false to allow unpack_to_dest (the static assert in
        // llk_unpack_A forbids both together) — matches the functional kernel.
        _llk_unpack_A_init_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // In case of math isolate, we don't want any software synchronization from unpack to math.
            // So we just set/clear valid bits here - which is unavoidable hardware synchronization.
            // When unpack_to_dest is used, we assume the data is immediately ready in destination register.
            // Otherwise, we assume the data is immediately ready in source A/B registers.
            if (!unpack_to_dest)
            {
                // Set valid for source A always.
                // Set valid for source B only if dest_acc is enabled.
                // Works only when unpacking to dest is not used.
                _perf_unpack_loop_set_valid<
                    /* src A */ true,
                    /* src B */ is_fp32_dest_acc_en>(
                    /* iterations*/ num_faces * TILE_CNT * LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    // Accuracy-merge: read input from the runtime operand address
                    // (StimuliConfig layout) instead of the fixed PERF_INPUT_A, so
                    // the harness's stimuli line up with what the kernel consumes.
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest (see init) */, reuse_dest_type, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

// Fresh semantic bodies: keep after sfpu_operations.h in their own include
// block (their templates consume its transitive typed helpers at
// definition-time lookup; clang-format sorts blocks independently).
#include "fresh_cpp/log1p.h"
#include "fresh_cpp_operations.h"
// Storm-contract canonical per-op semantic bodies (new bodies never land in
// the legacy aggregator above).
#include "fresh_cpp/rdiv.h"
#include "fresh_cpp/relu.h"
#include "fresh_cpp/roundingops.h"
#include "fresh_cpp/rpow.h"
#include "fresh_cpp/selu.h"
#include "fresh_cpp/sign.h"
// Storm-contract semantic bodies (one op per header, fresh_cpp/README.md).
#include "fresh_cpp/abs.h"
#include "fresh_cpp/absint32.h"
#include "fresh_cpp/add1.h"
#include "fresh_cpp/bitwisenot.h"
#include "fresh_cpp/castfp32tofp16a.h"
#include "fresh_cpp/celu.h"
#include "fresh_cpp/comp.h"
// Byte-untouched legacy tt-llk LUT sigmoid (6-segment SFPLUTFP32 hand kernel,
// laneED sem-only audit): impl-3 hand arm of test_perf_lut_variant_fresh_cpp.
#include "sfpu/ckernel_sfpu_sigmoid.h"
// Canonical per-op semantic bodies (storm contract: fresh_cpp/README.md).
#include "fresh_cpp/acosh_fitted.h"
#include "fresh_cpp/celu_fitted.h"
#include "fresh_cpp/digamma_fitted.h"
#include "fresh_cpp/elu_fitted.h"
#include "fresh_cpp/exp_fitted.h"
#include "fresh_cpp/expm1_fitted.h"
#include "fresh_cpp/gelu_255_licensed.h"
#include "fresh_cpp/gelu_appx_licensed.h"
#include "fresh_cpp/gelu_fitted.h"
#include "fresh_cpp/i0_fitted.h"
#include "fresh_cpp/i1_fitted.h"
#include "fresh_cpp/lgamma_fitted.h"
#include "fresh_cpp/log1p_fitted.h"
#include "fresh_cpp/log_fitted.h"
#include "fresh_cpp/mish_fitted.h"
#include "fresh_cpp/polygamma_fitted.h"
#include "fresh_cpp/rsqrt_fitted.h"
#include "fresh_cpp/selu_fitted.h"
#include "fresh_cpp/sigmoid_fitted.h"
#include "fresh_cpp/sigmoid_lut_licensed.h"
#include "fresh_cpp/softplus.h"
#include "fresh_cpp/softshrink.h"
#include "fresh_cpp/softsign.h"
#include "fresh_cpp/sqrt.h"
#include "fresh_cpp/square.h"
#include "fresh_cpp/tanh.h"
#include "fresh_cpp/tanh_fitted.h"
#include "fresh_cpp/tanhderivative-lut.h"
#include "fresh_cpp/tanhderivative_fitted.h"
#include "fresh_cpp/tanhlut_licensed.h"
#include "fresh_cpp/tanhshrink.h"
#include "fresh_cpp/threshold.h"
#include "fresh_cpp/threshold_fitted.h"
#include "fresh_cpp/trigonometry.h"
#include "fresh_cpp/unarycomp.h"
#include "fresh_cpp/unarypower.h"
#include "fresh_cpp/unaryshift.h"
#include "fresh_cpp/xielu.h"

#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

// Fixed dispatch scalar for the fresh unary max/min selector, mirroring the
// production dispatch in sfpu_operations.h (0u = 0.0f) so both perf legs
// compute the identical operation.
constexpr std::uint32_t FRESH_UNARY_MAX_MIN_FLOAT_VALUE = 0u; // 0.0f

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        // CLAMP_NEGATIVE must match the accuracy harness (which passes it): for
        // approx exp it selects the clamped approx-exp branch in sfpu_operations.
        // Omitting it defaulted to false -> a different approx path -> mismatch.
        test_utils::
            call_unary_sfpu_operation_init<SFPU_UNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT, CLAMP_NEGATIVE>();
#if FRESH_CPP_IMPL == 3
        // impl 3 = explicitly selected byte-untouched hand LUT variant (laneED
        // sem-only audit): the legacy tt-llk 6-segment SFPLUTFP32 sigmoid needs
        // its LReg coefficient loads (no production dispatch reaches it).
        if constexpr (SFPU_UNARY_OPERATION == SfpuType::sigmoid)
        {
            llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>(ckernel::sfpu::_init_sigmoid_<APPROX_MODE>);
        }
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    // For unpack isolate scenario, math should only perform necessary synchronization and nothing else.
                    if constexpr (unpack_to_dest)
                    {
                        // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                        // "consumed" and can be overwritten with new data.
                        // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                        // this method will perform synchronization only and no actual data copy.
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            i % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                        _perf_math_loop_clear_valid<
                            /* src A */ true,
                            /* src B */ true>(
                            /* iterations*/ num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                            // "consumed" and can be overwritten with new data.
                            // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                            // this method will perform synchronization only and no actual data copy.
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                            _perf_math_loop_clear_valid<
                                /* src A */ true,
                                /* src B */ true>(
                                /* iterations*/ num_faces);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        // When data is not unpacked to dest, math needs to copy data from srcA to dest before starting SFPU operation.
                        // Otherwise, data is immediately ready in destination register.
                        if constexpr (!unpack_to_dest)
                        {
                            LLK_ASSERT(
                                (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }

                        if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exponential)
                        {
                            // Guard only variants that actually select this branch: a
                            // discarded `if constexpr` branch is still checked for
                            // non-dependent expressions, so an unconditional
                            // static_assert here rejects every APPROX_MODE / fp32-dest
                            // variant of this kernel (signbit, reciprocal, ...).
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "semantic exp selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_exp_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_sigmoid_appx_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
                        {
                            // Second semantic form: 3-range magnitude dispatch tree
                            // (the LUT-eligible shape); same contract as impl 1.
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_appx_tree_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::signbit)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_signbit_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_max || SFPU_UNARY_OPERATION == SfpuType::unary_min))
                        {
                            constexpr bool is_max = SFPU_UNARY_OPERATION == SfpuType::unary_max;
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_max_min_fresh_cpp,
                                (is_max, ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                FRESH_UNARY_MAX_MIN_FLOAT_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::ceil)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_ceil_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::equal_zero)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_eqz_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: remaining float zero-comparisons
                        // (production = the all-raw-TTI calculate_comp hand kernel).
                        else if constexpr (
                            FRESH_CPP_IMPL == 1 &&
                            (SFPU_UNARY_OPERATION == SfpuType::not_equal_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_zero ||
                             SFPU_UNARY_OPERATION == SfpuType::greater_than_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_equal_zero ||
                             SFPU_UNARY_OPERATION == SfpuType::greater_than_equal_zero))
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_fresh_cpp, (SFPU_UNARY_OPERATION, ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: semantic arm for the GeluAppx contract.
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: impl 3 = byte-untouched legacy tt-llk
                        // 6-segment SFPLUTFP32 sigmoid hand kernel.
                        else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sigmoid_, (APPROX_MODE, ITERATIONS), block_tile, VectorMode::None, ITERATIONS);
                        }
                        // Lane GI LICENSED semantic arms (owner ratification
                        // 2026-08-24, review_records/OWNER-RATIFICATION-arm-
                        // preference-lut-license.md item 2): impl 4 = the
                        // accuracy-licensed sem body (fresh_cpp/*_licensed.h).
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_appx_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_255_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_lut_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 4 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "licensed tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_lut_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_abs_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs_int32)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_abs_int32_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::bitwise_not)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_bitwise_not_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::add1)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_add1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cast_fp32_to_fp16a)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_cast_fp32_to_fp16a_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::celu)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_celu_fresh_cpp,
                                (is_fp32_dest_acc_en, ITERATIONS, ckernel::sfpu::FRESH_CELU_ALPHA_BITS, ckernel::sfpu::FRESH_CELU_ALPHA_RECIP_BITS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::clamp)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_clamp_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_CLAMP_LO,
                                ckernel::sfpu::FRESH_CLAMP_HI);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardtanh)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_hardtanh_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_CLAMP_LO,
                                ckernel::sfpu::FRESH_CLAMP_HI);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            // The fresh tanh states the bf16 production contract; guard
                            // only variants that actually select this branch.
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Lane CM fitted-kernel placeholders (tt-polynomial-fitter
                        // frontier selections; provenance in fresh_cpp/*_fitted.h).
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted sigmoid selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted gelu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh_derivative || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted tanh-derivative selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_derivative_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Lane CR fitted-kernel placeholders, wave 2 (tt-polynomial-fitter
                        // frontier selections; provenance in fresh_cpp/*_fitted.h).
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::digamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::digamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted digamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_digamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::lgamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::lgamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted lgamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::polygamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::polygamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted polygamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_polygamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i0)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i0 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted i0 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i0_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted i1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i1_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::mish)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::mish || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted mish selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mish_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted log selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log1p)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted log1p selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log1p_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::exponential)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted exponential selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_exponential_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::rsqrt)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::rsqrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted rsqrt selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsqrt_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::celu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::celu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted celu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_celu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::elu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted elu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_elu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::selu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted selu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_selu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::threshold)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::threshold || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted threshold selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_threshold_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::expm1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted expm1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::acosh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::acosh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted acosh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_acosh_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative_lut)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_derivative_lut_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::silu)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_silu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fmod)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_fmod_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR_RECIP);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::remainder)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_remainder_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR_RECIP);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh expm1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::sqrt || SFPU_UNARY_OPERATION == SfpuType::rsqrt))
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || (SFPU_UNARY_OPERATION != SfpuType::sqrt && SFPU_UNARY_OPERATION != SfpuType::rsqrt) ||
                                    (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh sqrt/rsqrt selectors support only non-approx, bf16 dest");
                            constexpr bool is_reciprocal = SFPU_UNARY_OPERATION == SfpuType::rsqrt;
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sqrt_rsqrt_fresh_cpp, (is_reciprocal, ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::power)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::power || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh unary power selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_power_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_POWER_EXPONENT);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::xielu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::xielu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh xielu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_xielu_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_XIELU_ALPHA,
                                ckernel::sfpu::FRESH_XIELU_ALPHA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh sigmoid selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cbrt)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::cbrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh cbrt selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_cbrt_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softplus)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::softplus || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh softplus selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_softplus_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_SOFTPLUS_BETA,
                                ckernel::sfpu::FRESH_SOFTPLUS_BETA_RECIP,
                                ckernel::sfpu::FRESH_SOFTPLUS_THRESHOLD);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardsigmoid)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_hardsigmoid_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh gelu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1_cw)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_cw_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::i1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh i1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::floor)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_floor_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::trunc)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_trunc_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::frac)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_frac_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rdiv)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rdiv || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh rdiv selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_rdiv_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_RDIV_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rpow)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rpow || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh rpow selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rpow_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::selu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh selu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_selu_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_SELU_SCALE,
                                ckernel::sfpu::FRESH_SELU_ALPHA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sign)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sign_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::relu_max)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_relu_max_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_RELU_MAX_THRESHOLD);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log1p)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh log1p selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log1p_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fill)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_fill_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FILL_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::heaviside)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_heaviside_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_HEAVISIDE_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_hardshrink_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_HARDSHRINK_LAMBDA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardmish)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_hardmish_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::elu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh elu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_elu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exp2)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exp2 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh exp2 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_exp2_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erf)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erf_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfc)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erfc_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfinv)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erfinv_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::digamma)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_digamma_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm S5 (fresh_cpp/ canonical per-op bodies).
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_softshrink_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                0x3f000000u /* lambda = 0.5f */);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softsign)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_softsign_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::square)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_square_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanhshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_tanhshrink_fresh_cpp,
                                (is_fp32_dest_acc_en, ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::threshold)
                        {
                            // Scalars mirror the production dispatch in sfpu_operations.h (threshold 5.0f, value 10.0f).
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_threshold_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None, 5.0f, 10.0f);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_ge || SFPU_UNARY_OPERATION == SfpuType::unary_le))
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_comp_fresh_cpp,
                                (SFPU_UNARY_OPERATION == SfpuType::unary_ge, ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                0x3f000000u /* value = 0.5f, the production dispatch scalar */);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::left_shift)
                        {
                            // Shift amount mirrors the production dispatch constant (SHIFT_AMOUNT = 3u).
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_shift_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None, 3u);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::acosh)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_acosh_fresh_cpp, (is_fp32_dest_acc_en, ITERATIONS), block_tile, VectorMode::None);
                        }
                        else
                        {
                            test_utils::call_unary_sfpu_operation<
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                SFPU_UNARY_OPERATION,
                                APPROX_MODE,
                                is_fp32_dest_acc_en,
                                ITERATIONS,
                                FAST_MODE,
                                STABLE_SORT,
                                CLAMP_NEGATIVE>(block_tile, formats.math);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    // Copy from srcA to dest
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");

                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);

                        // Start SFPU operation
                        if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exponential)
                        {
                            // Guard only variants that actually select this branch: a
                            // discarded `if constexpr` branch is still checked for
                            // non-dependent expressions, so an unconditional
                            // static_assert here rejects every APPROX_MODE / fp32-dest
                            // variant of this kernel (signbit, reciprocal, ...).
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "semantic exp selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_exp_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_sigmoid_appx_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
                        {
                            // Second semantic form: 3-range magnitude dispatch tree
                            // (the LUT-eligible shape); same contract as impl 1.
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_appx_tree_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::signbit)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_signbit_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_max || SFPU_UNARY_OPERATION == SfpuType::unary_min))
                        {
                            constexpr bool is_max = SFPU_UNARY_OPERATION == SfpuType::unary_max;
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_max_min_fresh_cpp,
                                (is_max, ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                FRESH_UNARY_MAX_MIN_FLOAT_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::ceil)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_ceil_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::equal_zero)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_eqz_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: remaining float zero-comparisons
                        // (production = the all-raw-TTI calculate_comp hand kernel).
                        else if constexpr (
                            FRESH_CPP_IMPL == 1 &&
                            (SFPU_UNARY_OPERATION == SfpuType::not_equal_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_zero ||
                             SFPU_UNARY_OPERATION == SfpuType::greater_than_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_equal_zero ||
                             SFPU_UNARY_OPERATION == SfpuType::greater_than_equal_zero))
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_fresh_cpp, (SFPU_UNARY_OPERATION, ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: semantic arm for the GeluAppx contract.
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // laneED sem-only audit: impl 3 = byte-untouched legacy tt-llk
                        // 6-segment SFPLUTFP32 sigmoid hand kernel.
                        else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sigmoid_, (APPROX_MODE, ITERATIONS), block_tile, VectorMode::None, ITERATIONS);
                        }
                        // Lane GI LICENSED semantic arms (owner ratification
                        // 2026-08-24, review_records/OWNER-RATIFICATION-arm-
                        // preference-lut-license.md item 2): impl 4 = the
                        // accuracy-licensed sem body (fresh_cpp/*_licensed.h).
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_appx_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_255_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_lut_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 4 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "licensed tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_lut_licensed_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_abs_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs_int32)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_abs_int32_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::bitwise_not)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_bitwise_not_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::add1)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_add1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cast_fp32_to_fp16a)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_cast_fp32_to_fp16a_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::celu)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_celu_fresh_cpp,
                                (is_fp32_dest_acc_en, ITERATIONS, ckernel::sfpu::FRESH_CELU_ALPHA_BITS, ckernel::sfpu::FRESH_CELU_ALPHA_RECIP_BITS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::clamp)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_clamp_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_CLAMP_LO,
                                ckernel::sfpu::FRESH_CLAMP_HI);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardtanh)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_hardtanh_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_CLAMP_LO,
                                ckernel::sfpu::FRESH_CLAMP_HI);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            // The fresh tanh states the bf16 production contract; guard
                            // only variants that actually select this branch.
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Lane CM fitted-kernel placeholders (tt-polynomial-fitter
                        // frontier selections; provenance in fresh_cpp/*_fitted.h).
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted tanh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted sigmoid selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted gelu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh_derivative || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted tanh-derivative selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_derivative_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Lane CR fitted-kernel placeholders, wave 2 (tt-polynomial-fitter
                        // frontier selections; provenance in fresh_cpp/*_fitted.h).
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::digamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::digamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted digamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_digamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::lgamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::lgamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted lgamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::polygamma)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::polygamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted polygamma selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_polygamma_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i0)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i0 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted i0 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i0_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted i1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i1_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::mish)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::mish || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted mish selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mish_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted log selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log1p)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted log1p selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log1p_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::exponential)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted exponential selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_exponential_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::rsqrt)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::rsqrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted rsqrt selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsqrt_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::celu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::celu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted celu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_celu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::elu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted elu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_elu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::selu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted selu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_selu_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::threshold)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::threshold || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted threshold selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_threshold_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::expm1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted expm1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::acosh)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::acosh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fitted acosh selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_acosh_fitted_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative_lut)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tanh_derivative_lut_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::silu)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_silu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fmod)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_fmod_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR_RECIP);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::remainder)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_remainder_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR,
                                ckernel::sfpu::FRESH_FMOD_DIVISOR_RECIP);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh expm1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::sqrt || SFPU_UNARY_OPERATION == SfpuType::rsqrt))
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || (SFPU_UNARY_OPERATION != SfpuType::sqrt && SFPU_UNARY_OPERATION != SfpuType::rsqrt) ||
                                    (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh sqrt/rsqrt selectors support only non-approx, bf16 dest");
                            constexpr bool is_reciprocal = SFPU_UNARY_OPERATION == SfpuType::rsqrt;
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sqrt_rsqrt_fresh_cpp, (is_reciprocal, ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::power)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::power || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh unary power selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_power_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_POWER_EXPONENT);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::xielu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::xielu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh xielu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_xielu_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_XIELU_ALPHA,
                                ckernel::sfpu::FRESH_XIELU_ALPHA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh sigmoid selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sigmoid_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cbrt)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::cbrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh cbrt selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_cbrt_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softplus)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::softplus || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh softplus selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_softplus_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_SOFTPLUS_BETA,
                                ckernel::sfpu::FRESH_SOFTPLUS_BETA_RECIP,
                                ckernel::sfpu::FRESH_SOFTPLUS_THRESHOLD);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardsigmoid)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_hardsigmoid_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh gelu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1_cw)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_expm1_cw_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::i1)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh i1 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_i1_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::floor)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_floor_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::trunc)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_trunc_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::frac)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_frac_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rdiv)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rdiv || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh rdiv selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_rdiv_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_RDIV_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rpow)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rpow || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh rpow selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rpow_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::selu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh selu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_selu_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_SELU_SCALE,
                                ckernel::sfpu::FRESH_SELU_ALPHA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sign)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sign_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::relu_max)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_relu_max_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_RELU_MAX_THRESHOLD);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log1p)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh log1p selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_log1p_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fill)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_fill_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_FILL_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::heaviside)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_heaviside_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_HEAVISIDE_VALUE);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_hardshrink_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                ckernel::sfpu::FRESH_HARDSHRINK_LAMBDA);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardmish)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_hardmish_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::elu)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh elu selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_elu_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exp2)
                        {
                            static_assert(
                                FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exp2 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                                "fresh exp2 selector supports only non-approx, bf16 dest");
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_exp2_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erf)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erf_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfc)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erfc_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfinv)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_erfinv_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::digamma)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_digamma_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        // Storm S5 (fresh_cpp/ canonical per-op bodies).
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_softshrink_fresh_cpp,
                                (ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                0x3f000000u /* lambda = 0.5f */);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softsign)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_softsign_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::square)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_square_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanhshrink)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_tanhshrink_fresh_cpp,
                                (is_fp32_dest_acc_en, ITERATIONS),
                                block_tile,
                                VectorMode::None);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::threshold)
                        {
                            // Scalars mirror the production dispatch in sfpu_operations.h (threshold 5.0f, value 10.0f).
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_threshold_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None, 5.0f, 10.0f);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_ge || SFPU_UNARY_OPERATION == SfpuType::unary_le))
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_unary_comp_fresh_cpp,
                                (SFPU_UNARY_OPERATION == SfpuType::unary_ge, ITERATIONS),
                                block_tile,
                                VectorMode::None,
                                0x3f000000u /* value = 0.5f, the production dispatch scalar */);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::left_shift)
                        {
                            // Shift amount mirrors the production dispatch constant (SHIFT_AMOUNT = 3u).
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_shift_fresh_cpp, (ITERATIONS), block_tile, VectorMode::None, 3u);
                        }
                        else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::acosh)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_acosh_fresh_cpp, (is_fp32_dest_acc_en, ITERATIONS), block_tile, VectorMode::None);
                        }
                        else
                        {
                            test_utils::call_unary_sfpu_operation<
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                SFPU_UNARY_OPERATION,
                                APPROX_MODE,
                                is_fp32_dest_acc_en,
                                ITERATIONS,
                                FAST_MODE,
                                STABLE_SORT,
                                CLAMP_NEGATIVE>(block_tile, formats.math);
                        }
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const auto& buffer_Res          = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")

        // Configure packer hardware
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        // Initialize destination for packing
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
