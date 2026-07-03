// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"
#include "api/numeric/bfloat16.h"
#include "../metadata/metadata.hpp"
#ifdef TRISC_MATH
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sampling.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#endif
#endif

constexpr uint32_t FACE_ELEMS = 256;
constexpr uint16_t BF16_ONE = 0x3F80;
constexpr uint32_t ELEMS_PER_FACE_ROW = 16;

static inline uint32_t float_to_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline float bits_to_float(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

// Convert bf16 bit-pattern to float32 exactly
static inline float bf16_to_float(uint16_t bf) {
    uint32_t u32 = static_cast<uint32_t>(bf) << 16;
    return bits_to_float(u32);
}

// Convert float32 to bf16 bit-pattern using round-to-nearest
static inline uint16_t float_to_bf16_rne(float x) {
    uint32_t u = float_to_bits(x);

    // Preserve NaNs as NaNs. Make sure result mantissa is nonzero.
    const uint32_t exp_mask = 0x7F800000u;
    const uint32_t frac_mask = 0x007FFFFFu;
    if ((u & exp_mask) == exp_mask && (u & frac_mask) != 0) {
        uint16_t upper = static_cast<uint16_t>(u >> 16);
        // Ensure NaN payload remains NaN after truncation
        if ((upper & 0x007Fu) == 0) {
            upper |= 0x0001u;
        }
        return upper;
    }

    // Round-to-nearest-even when truncating low 16 bits
    // bias = 0x7FFF + lsb_of_upper
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;

    return static_cast<uint16_t>(u >> 16);
}

// Pack a single bf16 bit-pattern into a uint32 with two copies (low and high
// 16 bits both set to the same bf16 value).  This matches the packed scalar
// layout expected by `generate_bcast_unary_scalar`, and is the runtime
// equivalent of `float_to_bfloat16_packed` in utils.py.
static inline uint32_t bf16_pack_to_uint32(uint16_t bf16_val) {
    return (static_cast<uint32_t>(bf16_val) << 16) | static_cast<uint32_t>(bf16_val);
}

// Convenience: convert fp32 -> bf16 (RNE) and pack two copies into a uint32.
static inline uint32_t float_to_bf16_packed(float x) { return bf16_pack_to_uint32(float_to_bf16_rne(x)); }

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "api/socket_api.h"
#include "../micro_ops/host_io/kernels/pcie_noc_utils.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../kernel_includes/tt_metal/dm_utils.hpp"

FORCE_INLINE void generate_row0_bcast(const uint32_t cb_id, uint16_t bf16_val) {
    cb_reserve_back(cb_id, 1);
    auto* tile_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    const uint32_t packed = bf16_pack_to_uint32(bf16_val);
    // Face 0 row 0: 16 bf16 lanes = 8 u32 words.
    for (uint32_t i = 0; i < 8; ++i) {
        tile_u32[i] = packed;
    }
    // Face 1 row 0: same, offset by one face (256 bf16 = 128 u32 words).
    constexpr uint32_t FACE_U32 = 128;
    for (uint32_t i = 0; i < 8; ++i) {
        tile_u32[FACE_U32 + i] = packed;
    }
    cb_push_back(cb_id, 1);
}

#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/cumsum.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rand.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/rmsnorm.h"

#if defined(TRISC_UNPACK)
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_top32_rm_api.h"
#endif
#if defined(TRISC_MATH)
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_top32_rm_api.h"
#include "../kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_deepseek_top32_rm.h"
template <bool legacy_compat = true>
ALWI void sampling_recip_tile_scalar(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sampling_recip_scalar, (legacy_compat), idst, VectorMode::None);
}

ALWI void sampling_clamp_max_tile_scalar(uint32_t idst, uint32_t param) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sampling_clamp_max_scalar, idst, VectorMode::None, param);
}

ALWI void sampling_le_binary_tile_first_column(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sampling_binary_comp_first_column,
        (SfpuType::le),
        idst0,
        idst1,
        odst,
        VectorMode::C);
}

ALWI void sampling_lt_binary_tile_first_column(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sampling_binary_comp_first_column,
        (SfpuType::lt),
        idst0,
        idst1,
        odst,
        VectorMode::C);
}

ALWI void sampling_ge_binary_tile_first_column(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sampling_binary_comp_first_column,
        (SfpuType::ge),
        idst0,
        idst1,
        odst,
        VectorMode::C);
}

ALWI void sampling_mul_unary_tile_first_column(uint32_t idst, uint32_t param) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sampling_mul_unary_scalar_first_column, idst, VectorMode::C, param);
}

ALWI void sampling_add_binary_tile_first_column(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sampling_add_binary_first_column, idst0, idst1, odst, VectorMode::C);
}
#endif

// Sampling-local explicit-fidelity forks of the compute API helpers. Keep
// these local so the core API can continue defaulting to the kernel-wide
// MATH_FIDELITY macro, while sampling can force HiFi4 in only the softmax
// normalization path.
template <PoolType reduce_type, ReduceDim reduce_dim, MathFidelity math_fidelity>
ALWI void sampling_reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    // BH REDUCE_ROW SUM/AVG uses MVMUL with swapped operands (scaler→SrcA, data→SrcB)
    // Reconfig formats to match: SrcA=scaler format, SrcB=data
    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && (reduce_type != PoolType::MAX);
    if constexpr (swap_operands) {
        state_configure(icb_scaler, icb, ocb, call_line);
        reconfig_data_format(icb_scaler, icb);
    } else {
        state_configure(icb, icb_scaler, ocb, call_line);
    }
#else
    state_configure(icb, icb_scaler, ocb, call_line);
#endif
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, math_fidelity>(icb, icb_scaler)));
#else
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, math_fidelity>(icb)));
#endif
    PACK((llk_pack_reduce_mask_config<reduce_dim, ckernel::PackMode::Default>(ocb)));
}

template <PoolType reduce_type, ReduceDim reduce_dim, MathFidelity math_fidelity>
ALWI void sampling_reduce_tile(
    uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, math_fidelity>(icb, icb_scaler, idst)));
    UNPACK((llk_unpack_AB_reduce<reduce_type, reduce_dim>(icb, icb_scaler, itile, itile_scaler)));
#else
    MATH((llk_math_reduce(idst)));
    UNPACK((llk_unpack_AB_reduce(icb, icb_scaler, itile, itile_scaler)));
#endif
}

void generate_rand_tile(const uint32_t cb_id);

template <
    uint32_t in_cb,
    uint32_t out_cb,
    uint32_t exp_cb,
    uint32_t probs_cb,
    uint32_t probs_out_cb,
    uint32_t scaler_cb,
    uint32_t p_cb,
    uint32_t rand_cb,
    uint32_t rand_bcast_cb,
    uint32_t mask_cb,
    uint32_t num_tiles,
    bool enable_metadata = false,
    uint32_t metadata_output_l1_addr = 0,
    uint32_t inv_temp_bf16_ct = 0>
void trisc_fused_softmax_top_p_sampling_block() {
    DeviceZoneScopedN("SP-TOPP-TRISC");

    generate_rand_tile(rand_cb);
    cb_wait_front(in_cb, num_tiles);
    cb_wait_front(scaler_cb, 1);

    uint16_t temp_bf16;
    if constexpr (enable_metadata) {
        auto* metadata_ptr =
            reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_output_l1_addr);
        const float temperature = std::max(static_cast<float>(metadata_ptr->temperature), 0.01f);
        temp_bf16 = float_to_bf16_rne(1.0f / temperature);
    } else {
        temp_bf16 = static_cast<uint16_t>(inv_temp_bf16_ct);
    }
    // Step 1: Compute DST[0, 0, 0] = max(x_i, dim=0), x_i = in_cb
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-1");
        reconfig_data_format(in_cb, scaler_cb);
        sampling_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, MathFidelity::HiFi4>(in_cb, scaler_cb, exp_cb);

        tile_regs_acquire();
        sampling_reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW, MathFidelity::HiFi4>(in_cb, scaler_cb, 0, 0, 0);
        reduce_uninit();
    }
    // Step 2: Compute DST[0, 0, 0] = x_i - max(x_i, dim=0), x_i = in_cb,
    // max(x_i, dim=0) comes from DST in Step 1.
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-2");
        rmsnorm_bcast_scalar_reuse_tiles_init<
            EltwiseBinaryType::ELWSUB,
            /*num_tiles=*/1,
            MathFidelity::LoFi,
            /*unpack_full_transpose=*/true>(in_cb);
        rmsnorm_bcast_scalar_reuse_tiles<
            EltwiseBinaryType::ELWSUB,
            /*num_tiles=*/1,
            MathFidelity::LoFi,
            /*clear_dest=*/true>(in_cb, /*in_tile=*/0, /*src=*/0, /*dst=*/0);
    }
    // Step 3: Compute DST[0, 0, 0] = exp(x_i - max(x_i, dim=0)), x_i = in_cb, x_i - max(x_i, dim=0) comes from DST in
    // Step 2 NOTE: temp_bf16 is the temperature factor for the exp operation; sourced above (metadata-direct or
    // compile-time fallback).
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-3");
        exp_tile_init<false>();
        exp_tile<false, true>(0, VectorMode::C, temp_bf16);
    }
    tile_regs_commit();
    // Step 4: Pack the result into the exp_cb
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-4");
        cb_reserve_back(exp_cb, 1);
        tile_regs_wait();
        pack_reconfig_data_format(exp_cb);
        pack_tile(0, exp_cb);
        cb_push_back(exp_cb, 1);
        tile_regs_release();
        cb_pop_front(in_cb, num_tiles);
        cb_wait_front(exp_cb, 1);
    }
    {
        // Step 5: Compute DST[0, 0, 0] = sum(exp(x_i - max(x_i, dim=0))), exp(x_i - max(x_i, dim=0)) = exp_cb
        // Since the result is a column strip, we need to reduce along the column dimension
        DeviceZoneScopedN("SP-TOPP-TRISC-5");
        reconfig_data_format(exp_cb, scaler_cb);
        sampling_reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL, MathFidelity::HiFi4>(exp_cb, scaler_cb, probs_cb);
        tile_regs_acquire();
        // MATH wait for DST register to be available
        sampling_reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL, MathFidelity::HiFi4>(exp_cb, scaler_cb, 0, 0, 0);
        reduce_uninit();
        // Step 6: Compute DST[0, 0, 0] = 1/sum(exp(x_i - max(x_i, dim=0))), sum(exp(x_i - max(x_i, dim=0))) comes from
        // DST in Step 5
        recip_tile_init();
        MATH((sampling_recip_tile_scalar(0)));
        // Step 7: Compute DST[0] = exp(x_i - max(x_i, dim=0)) * 1/sum(exp(x_i - max(x_i, dim=0))).
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-6");
        rmsnorm_bcast_scalar_reuse_tiles_init<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4>(exp_cb);
        rmsnorm_bcast_scalar_reuse_tiles<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4,
            /*clear_dest=*/true>(exp_cb, /*in_tile=*/0, /*src=*/0, /*dst=*/0);
    }
    tile_regs_commit();
    cb_pop_front(exp_cb, 1);
    tile_regs_wait();
    // Step 8: Pack T(probs) into probs_cb. BRISC pops this slot at the
    // tail of SP-CPROBS; TRISC re-reads it (cb_wait_front + copy_tile)
    // in Block B below as the cumsum input.
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-7");
        cb_reserve_back(probs_cb, 1);
        pack_reconfig_data_format(probs_cb);
        pack_tile(0, probs_cb);
        cb_push_back(probs_cb, 1);
        tile_regs_release();
        reconfig_data_format_srca<false, true>(exp_cb, probs_cb);
        cb_wait_front(probs_cb, 1);
        cb_wait_front(p_cb, 1);
        tile_regs_acquire();
        // Step 9: DST[0] = T(probs), re-loaded from probs_cb (bf16).
        copy_tile_to_dst_init_short(probs_cb);
        copy_tile(probs_cb, 0, 0);
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-8");
        // Step 10: Compute DST[0] = cumsum(probs, dim=0). cumsum_tile is a
        // unary SFPU op that scans top-to-bottom per column; column 0 holds
        // the running CDF over the original row-0 scores.
        cumsum_tile_init();
        cumsum_tile(0, /*first=*/true);
    }
    // Step 11: DST[1] = p (column-0 broadcast staged by BRISC).
    copy_tile_to_dst_init_short(p_cb);
    copy_tile(p_cb, 0, 1);
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-9");
        // Step 12: DST[2] = (cumsum < p) ? 1.0 : 0.0
        lt_binary_tile_init();
        MATH((sampling_lt_binary_tile_first_column(0, 1, 2)));
        // Step 13: DST[2] *= BIG. Below-cutoff lanes blow up to ~BIG so the
        // upcoming Pass 4 MIN-reduce skips them; above-cutoff lanes stay at 0.
        constexpr uint32_t BIG_VAL_FP32_U32 = 0x42C80000u;  // 100.0f
        binop_with_scalar_tile_init();
        MATH((sampling_mul_unary_tile_first_column(2, BIG_VAL_FP32_U32)));
        // Step 14: DST[3] = cumsum + (mask * BIG) = filtered cumsum.
        add_binary_tile_init();
        MATH((sampling_add_binary_tile_first_column(0, 2, 3)));
        tile_regs_commit();
        tile_regs_wait();
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-10");
        // Step 15: Pack T(cumsum) into out_cb (used by Pass 4's rescale MUL).
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        // Step 16: Pack T(filtered cumsum) into exp_cb (input to Pass 4 MIN).
        cb_reserve_back(exp_cb, 1);
        pack_reconfig_data_format(exp_cb);
        pack_tile(3, exp_cb);
        cb_push_back(exp_cb, 1);
        tile_regs_release();
        cb_pop_front(p_cb, 1);
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-11");
        cb_wait_front(exp_cb, 1);
        cb_wait_front(rand_bcast_cb, 1);
        cb_wait_front(out_cb, 1);
        reconfig_data_format(exp_cb, scaler_cb);
        tile_regs_acquire();
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-12");
        copy_tile_to_dst_init_short(exp_cb);
        copy_tile(exp_cb, 0, 0);
        sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
        sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
    }
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-13");
        // Step 17.5: Clamp cum_kept <= 1.0. The MIN-reduce sentinel trick
        constexpr uint32_t SP_ONE_FP32 = 0x3F800000u;  // 1.0f
        MATH((sampling_clamp_max_tile_scalar(0, SP_ONE_FP32)));
        // Step 18: Compute DST[0] = 1/cum_kept
        recip_tile_init();
        MATH((sampling_recip_tile_scalar(0)));
    }
    // Step 18.5: Compute DST[3] = probs * 1/cum_kept = rescaled (renormalized) PMF.

    {
        DeviceZoneScopedN("SP-TOPP-TRISC-13b");
        rmsnorm_bcast_scalar_reuse_tiles_init<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4>(probs_cb);
        rmsnorm_bcast_scalar_reuse_tiles<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4,
            /*clear_dest=*/true>(probs_cb, /*in_tile=*/0, /*src=*/0, /*dst=*/3);
    }
    cb_pop_front(probs_cb, 1);
    // Step 18.75: Recompute 1/cum_kept into DST[0] for Step 19 to consume.
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-13c");
        copy_tile_to_dst_init_short(exp_cb);
        copy_tile(exp_cb, 0, 0);
        sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
        sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
        // Same clamp as Step 17.5: see comment above.
        constexpr uint32_t SP_ONE_FP32 = 0x3F800000u;  // 1.0f
        MATH((sampling_clamp_max_tile_scalar(0, SP_ONE_FP32)));
        recip_tile_init();
        MATH((sampling_recip_tile_scalar(0)));
    }
    // Step 19: Compute DST[2] = cumsum * 1/cum_kept (rescaled CDF over the kept set).
    // SrcA reads out_cb (Step 15's T(cumsum)), SrcB scalar comes from
    // DST[0] (Step 18.75's recomputed 1/cum_kept).
    // NOTE: rescaled_cumsum_i = cumsum(softmax_out_i, dim=0) * 1/cum_kept
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-14");
        rmsnorm_bcast_scalar_reuse_tiles_init<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4>(out_cb);
        rmsnorm_bcast_scalar_reuse_tiles<
            EltwiseBinaryType::ELWMUL,
            /*num_tiles=*/1,
            MathFidelity::HiFi4,
            /*clear_dest=*/false>(out_cb, /*in_tile=*/0, /*src=*/0, /*dst=*/2);
    }
    // Step 20: DST[1] = rand (column-0 broadcast staged by BRISC into rand_bcast_cb).
    copy_tile_to_dst_init_short(rand_bcast_cb);
    copy_tile(rand_bcast_cb, 0, 1);
    // Step 21: DST[2] = (rescaled_cumsum >= rand) ? 1.0 : 0.0.
    {
        DeviceZoneScopedN("SP-TOPP-TRISC-15");
        ge_binary_tile_init();
        MATH((sampling_ge_binary_tile_first_column(2, 1, 2)));
        tile_regs_commit();
    }
    tile_regs_wait();
    if constexpr (mask_cb == scaler_cb) {
        cb_pop_front(scaler_cb, 1);
    }
    // Step 22: Pack the mask into mask_cb (dedicated TRISC -> BRISC channel).
    if constexpr (mask_cb == scaler_cb) {
        cb_pop_front(scaler_cb, 1);
    }
    cb_reserve_back(mask_cb, 1);
    pack_reconfig_data_format(mask_cb);
    pack_tile(2, mask_cb);
    cb_push_back(mask_cb, 1);
    // Step 21.5: Hand the rescaled PMF (DST[3] from Step 18.5) off to BRISC.
    cb_reserve_back(probs_out_cb, 1);
    pack_reconfig_data_format(probs_out_cb);
    pack_tile(3, probs_out_cb);
    cb_push_back(probs_out_cb, 1);

    tile_regs_release();
    cb_pop_front(exp_cb, 1);
    cb_pop_front(rand_bcast_cb, 1);
    if constexpr (mask_cb == scaler_cb) {
        cb_pop_front(mask_cb, 1);
    } else {
        cb_pop_front(scaler_cb, 1);
    }
}

void generate_rand_tile(const uint32_t cb_id) {
    uint32_t rand_scale = 0;
    const float one_f = 1.0f;
    std::memcpy(&rand_scale, &one_f, sizeof(uint32_t));
    uint32_t rand_from = 0;
    cb_reserve_back(cb_id, 1);
    tile_regs_acquire();
    rand_tile(0, rand_from, rand_scale);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_id);
    pack_tile(0, cb_id, 0);
    tile_regs_release();
    cb_push_back(cb_id, 1);
}

template <
    uint32_t in_scores_cb,
    uint32_t in_indices_cb,
    uint32_t out_scores_cb,
    uint32_t out_indices_cb,
    bool presorted = false>
void run_top32_llk(uint32_t row_elements, uint32_t num_input_tiles, uint32_t phase_number) {
    constexpr uint32_t value_offset_tiles = 0;
    constexpr uint32_t index_offset_tiles = 2;
    constexpr uint32_t decreasing = 0;
    constexpr uint32_t increasing = 1;

    cb_wait_front(in_scores_cb, num_input_tiles);
    cb_wait_front(in_indices_cb, num_input_tiles);
    cb_reserve_back(out_scores_cb, 1);
    cb_reserve_back(out_indices_cb, 1);

    tile_regs_acquire();

    uint32_t num_faces = 4;
    PACK((void)num_faces);
    reconfig_data_format_srca(in_scores_cb);
    UNPACK((llk_unpack_A_top32_rm_init(in_scores_cb)));
    UNPACK((llk_unpack_A_top32_rm(in_scores_cb, 0, num_faces)));
    MATH((llk_math_top32_rm_init(in_scores_cb)));
    MATH((llk_math_top32_rm(in_scores_cb, value_offset_tiles, num_faces)));

    reconfig_data_format_srca(in_indices_cb);
    UNPACK((llk_unpack_A_top32_rm_init(in_indices_cb)));
    UNPACK((llk_unpack_A_top32_rm(in_indices_cb, 0, num_faces)));
    MATH((llk_math_top32_rm_init(in_indices_cb)));
    MATH((llk_math_top32_rm(in_indices_cb, index_offset_tiles, num_faces)));

    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::_top32_rm_init_)));
    if constexpr (presorted) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles,
            VectorMode::RC_custom,
            decreasing,
            false /*skip_second*/));
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_phases_steps_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles,
            VectorMode::RC_custom,
            decreasing));
    }
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _bitonic_top32_merge_,
        (false, DST_ACCUM_MODE, false /*idir*/),
        value_offset_tiles,
        VectorMode::RC_custom,
        false /*across_tiles*/));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _bitonic_top32_rebuild_,
        (false, DST_ACCUM_MODE),
        value_offset_tiles,
        VectorMode::RC_custom,
        decreasing,
        true /*skip_second*/));

    for (uint32_t i = 64; i < row_elements; i += 64) {
        if (i + 64 > row_elements) {
            num_faces = 2;
        } else {
            num_faces = 4;
        }

        PACK((void)num_faces);
        reconfig_data_format_srca(in_scores_cb);
        UNPACK((llk_unpack_A_top32_rm_init(in_scores_cb)));
        UNPACK((llk_unpack_A_top32_rm(in_scores_cb, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(in_scores_cb)));
        MATH((llk_math_top32_rm(in_scores_cb, value_offset_tiles + 1, num_faces)));

        reconfig_data_format_srca(in_indices_cb);
        UNPACK((llk_unpack_A_top32_rm_init(in_indices_cb)));
        UNPACK((llk_unpack_A_top32_rm(in_indices_cb, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(in_indices_cb)));
        MATH((llk_math_top32_rm(in_indices_cb, index_offset_tiles + 1, num_faces)));

        if constexpr (presorted) {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _bitonic_top32_rebuild_,
                (false, DST_ACCUM_MODE),
                value_offset_tiles + 1,
                VectorMode::RC_custom,
                decreasing,
                false /*skip_second*/));
        } else {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _bitonic_top32_phases_steps_,
                (false, DST_ACCUM_MODE),
                value_offset_tiles + 1,
                VectorMode::RC_custom,
                decreasing));
        }
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_merge_,
            (false, DST_ACCUM_MODE, false /*idir*/),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            false /*across_tiles*/));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            increasing,
            true /*skip_second*/));

        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_merge_,
            (false, DST_ACCUM_MODE, false /*idir*/),
            value_offset_tiles,
            VectorMode::RC_custom,
            true /*across_tiles*/));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles,
            VectorMode::RC_custom,
            decreasing,
            true /*skip_second*/));
    }
    tile_regs_commit();
    tile_regs_wait();

    // The custom TTI_SETADCXX(PAC, 1 - 1, 0x0) below packs 1 element per row across
    // 16 rows (32 elements total) for the topk output, which requires
    // pack_reads_per_xy_plane = FACE_R_DIM = 16 so the tile position generator counts
    // 16 rows before resetting. Save FACE_R_DIM here and restore 1 after pack_tile.
    ckernel::pack_reconfig_data_format(out_scores_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)));
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_tile(value_offset_tiles, out_scores_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)));

    ckernel::pack_reconfig_data_format(out_indices_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)));
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_tile(index_offset_tiles, out_indices_cb);
    PACK(TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0));
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)));

    tile_regs_release();

    cb_pop_front(in_scores_cb, num_input_tiles);
    cb_pop_front(in_indices_cb, num_input_tiles);
    // Phase 1's llk_unpack_A_top32_rm_init modifies four unpacker registers for
    // row-major mode. transpose_init restores Haloize_mode and X counter,
    // but Tile_x_dim and Y_stride must be restored explicitly here.
    UNPACK(TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));
    UNPACK(TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(
        FACE_R_DIM * FACE_C_DIM | (FACE_R_DIM * FACE_C_DIM << 16))));
    UNPACK((cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(FACE_C_DIM)));

    cb_push_back(out_scores_cb, 1);
    cb_push_back(out_indices_cb, 1);
}

template <uint32_t in_scores_cb, uint32_t in_indices_cb, uint32_t out_scores_cb, uint32_t out_indices_cb>
void run_top32_llk_presorted_1024_opt(uint32_t row_elements, uint32_t num_input_tiles, uint32_t phase_number) {
    constexpr uint32_t value_offset_tiles = 0;
    constexpr uint32_t index_offset_tiles = 2;
    constexpr uint32_t decreasing = 0;
    constexpr uint32_t increasing = 1;
    constexpr uint32_t chunk_size = 1024;

    // deepseek_compute_kernel_hw_startup<true>(
    //     in_scores_cb,
    //     in_scores_cb,
    //     out_scores_cb);

    cb_wait_front(in_scores_cb, num_input_tiles);
    cb_wait_front(in_indices_cb, num_input_tiles);
    cb_reserve_back(out_scores_cb, 1);
    cb_reserve_back(out_indices_cb, 1);

    tile_regs_acquire();

    const uint32_t num_chunks = row_elements / chunk_size;

    // Step 1: load first 1024 values/indices chunk with transpose.
    reconfig_data_format_srca(in_scores_cb);
    transpose_init(in_scores_cb);
    transpose_tile(in_scores_cb, 0, value_offset_tiles);

    reconfig_data_format_srca(in_indices_cb);
    transpose_init(in_indices_cb);
    transpose_tile(in_indices_cb, 0, index_offset_tiles);

    // Step 2: prepare first chunk for pre-sorted combine pipeline.
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::_top32_rm_init_)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _bitonic_top32_of_1024_rm_pre_sorted_prep_,
        (false, DST_ACCUM_MODE, decreasing),
        value_offset_tiles,
        VectorMode::RC_custom,
        value_offset_tiles));

    // Steps 3-5: ingest remaining full 1024 chunks and combine.
    for (uint32_t i = 1; i < num_chunks; ++i) {
        reconfig_data_format_srca(in_scores_cb);
        transpose_init(in_scores_cb);
        transpose_tile(in_scores_cb, i, value_offset_tiles + 1);

        reconfig_data_format_srca(in_indices_cb);
        transpose_init(in_indices_cb);
        transpose_tile(in_indices_cb, i, index_offset_tiles + 1);

        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_of_1024_rm_pre_sorted_prep_,
            (false, DST_ACCUM_MODE, increasing),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            value_offset_tiles + 1));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_of_1024_rm_pre_sorted_combine_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles,
            VectorMode::RC_custom,
            value_offset_tiles));
    }

    // Step 6: collapse per-face top-32 to a single top-32.
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _bitonic_top32_of_1024_rm_pre_sorted_final_,
        (false, DST_ACCUM_MODE),
        value_offset_tiles,
        VectorMode::RC_custom,
        value_offset_tiles));

    // Steps 7-9: handle trailing (<1024) values in 64-element chunks.
    uint32_t num_faces = 4;
    PACK((void)num_faces);
    for (uint32_t i = num_chunks * chunk_size; i < row_elements; i += 64) {
        num_faces = (i + 64 > row_elements) ? 2 : 4;

        reconfig_data_format_srca(in_scores_cb);
        UNPACK((llk_unpack_A_top32_rm_init(in_scores_cb)));
        UNPACK((llk_unpack_A_top32_rm(in_scores_cb, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(in_scores_cb)));
        MATH((llk_math_top32_rm(in_scores_cb, value_offset_tiles + 1, num_faces)));

        reconfig_data_format_srca(in_indices_cb);
        UNPACK((llk_unpack_A_top32_rm_init(in_indices_cb)));
        UNPACK((llk_unpack_A_top32_rm(in_indices_cb, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(in_indices_cb)));
        MATH((llk_math_top32_rm(in_indices_cb, index_offset_tiles + 1, num_faces)));

        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            decreasing,
            false /*skip_second*/));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_merge_,
            (false, DST_ACCUM_MODE, false /*idir*/),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            false /*across_tiles*/));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles + 1,
            VectorMode::RC_custom,
            increasing,
            true /*skip_second*/));

        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_merge_,
            (false, DST_ACCUM_MODE, false /*idir*/),
            value_offset_tiles,
            VectorMode::RC_custom,
            true /*across_tiles*/));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_top32_rebuild_,
            (false, DST_ACCUM_MODE),
            value_offset_tiles,
            VectorMode::RC_custom,
            decreasing,
            true /*skip_second*/));
    }
    tile_regs_commit();
    tile_regs_wait();

    // Step 10: pack final top-32 scores/indices.
    // The custom TTI_SETADCXX(PAC, 1 - 1, 0x0) below packs 1 element per row across
    // 16 rows (32 elements total) for the topk output, which requires
    // pack_reads_per_xy_plane = FACE_R_DIM = 16 so the tile position generator counts
    // 16 rows before resetting. Save FACE_R_DIM here and restore 1 after pack_tile.
    ckernel::pack_reconfig_data_format(out_scores_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)));
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_tile(value_offset_tiles, out_scores_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)));

    ckernel::pack_reconfig_data_format(out_indices_cb);
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)));
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_tile(index_offset_tiles, out_indices_cb);

    PACK(TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0));
    PACK((cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)));

    // Reset unpacker state for downstream operations (softmax)
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0)));
    UNPACK(TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));
    UNPACK(TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));

    tile_regs_release();

    cb_pop_front(in_scores_cb, num_input_tiles);
    cb_pop_front(in_indices_cb, num_input_tiles);
    cb_push_back(out_scores_cb, 1);
    cb_push_back(out_indices_cb, 1);
}

#endif  // COMPILE_FOR_TRISC

constexpr uint32_t MIN_TOPK_ALIGNMENT = 32;

namespace deepseek_b1_ops {

struct TopKSampling {
    template <
        uint32_t NumValues,
        uint32_t TopK,
        uint32_t WinnerPageBytes,
        uint32_t NumSenders,
        uint32_t ExpectedRemoteIncs,
        uint32_t ReceiverSemaphoreAddr,
        uint32_t LocalReadySemaphoreAddr,
        uint32_t MeshMode,
        uint32_t Stage1Sender,
        uint32_t Stage1Receiver,
        uint32_t Stage2Sender,
        uint32_t Stage2Receiver,
        uint32_t Stage1SlotBaseOffset,
        uint32_t Stage1NumSlots,
        uint32_t Stage1ExpectedRemoteIncs,
        uint32_t Stage1LocalSlotOffset,
        uint32_t Stage2SlotBaseOffset,
        uint32_t Stage2NumSlots,
        uint32_t Stage2ExpectedRemoteIncs,
        uint32_t Stage2LocalSlotOffset,
        uint32_t MeshLocalSendSlotOffset,
        uint32_t SenderIdx,
        uint32_t SocketMode = 0,
        uint32_t SocketCBId = 0xFFFFFFFF,
        uint32_t SocketPageSizeBytes = 0,
        uint32_t ScoresCBId = 0xFFFFFFFF,
        uint32_t ScoresNumPages = 0,
        uint32_t WinnerCBId = 0xFFFFFFFF,
        uint32_t SoftmaxInCBId = 0xFFFFFFFF,
        uint32_t SoftmaxOutCBId = 0xFFFFFFFF,
        uint32_t SoftmaxExpCBId = 0xFFFFFFFF,
        uint32_t ScalerCBId = 0xFFFFFFFF,
        uint32_t TempCBId = 0xFFFFFFFF,
        uint32_t InvTempBF16 = 1,
        uint32_t TopKInScoresCBId = 0xFFFFFFFF,
        uint32_t TopKInIndicesCBId = 0xFFFFFFFF,
        uint32_t TopKOutScoresCBId = 0xFFFFFFFF,
        uint32_t TopKOutIndicesCBId = 0xFFFFFFFF,
        uint32_t Phase2ScoresByteOffset = 0,
        uint32_t Phase2IndicesByteOffset = 0,
        uint32_t MeshStageScoresCBId = 0xFFFFFFFF,
        uint32_t MeshStageIndicesCBId = 0xFFFFFFFF,
        uint32_t ScoresScratchStage2Offset = 0,
        uint32_t IndicesScratchStage2Offset = 0,
        uint32_t ScoresScratchAddr = 0,
        uint32_t IndicesScratchAddr = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t num_values = NumValues;
        static constexpr uint32_t topk_k = TopK;
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t num_senders = NumSenders;
        static constexpr uint32_t expected_remote_incs = ExpectedRemoteIncs;
        static constexpr uint32_t receiver_semaphore_addr = ReceiverSemaphoreAddr;
        static constexpr uint32_t local_ready_semaphore_addr = LocalReadySemaphoreAddr;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage1_sender = Stage1Sender == 1;
        static constexpr bool stage1_receiver = Stage1Receiver == 1;
        static constexpr bool stage2_sender = Stage2Sender == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
        static constexpr uint32_t stage1_slot_base_offset = Stage1SlotBaseOffset;
        static constexpr uint32_t stage1_num_slots = Stage1NumSlots;
        static constexpr uint32_t stage1_expected_remote_incs = Stage1ExpectedRemoteIncs;
        static constexpr uint32_t stage1_local_slot_idx = Stage1LocalSlotOffset;
        static constexpr uint32_t stage2_slot_base_offset = Stage2SlotBaseOffset;
        static constexpr uint32_t stage2_num_slots = Stage2NumSlots;
        static constexpr uint32_t stage2_expected_remote_incs = Stage2ExpectedRemoteIncs;
        static constexpr uint32_t stage2_local_slot_idx = Stage2LocalSlotOffset;
        static constexpr uint32_t mesh_local_send_slot_offset = MeshLocalSendSlotOffset;
        static constexpr uint32_t sender_idx = SenderIdx;
        static constexpr uint32_t socket_mode = SocketMode;
        static constexpr uint32_t socket_cb_id = SocketCBId;
        static constexpr uint32_t socket_page_size_bytes = SocketPageSizeBytes;
        static constexpr uint32_t scores_cb_id = ScoresCBId;
        static constexpr uint32_t scores_num_pages = ScoresNumPages;
        static constexpr uint32_t winner_cb_id = WinnerCBId;
        static constexpr uint32_t softmax_in_cb = SoftmaxInCBId;
        static constexpr uint32_t softmax_out_cb = SoftmaxOutCBId;
        static constexpr uint32_t softmax_exp_cb = SoftmaxExpCBId;
        static constexpr uint32_t scaler_cb = ScalerCBId;
        static constexpr uint32_t temp_cb = TempCBId;
        static constexpr uint32_t inv_temp_bf16 = InvTempBF16;
        static constexpr uint32_t topk_in_scores_cb = TopKInScoresCBId;
        static constexpr uint32_t topk_in_indices_cb = TopKInIndicesCBId;
        static constexpr uint32_t topk_out_scores_cb = TopKOutScoresCBId;
        static constexpr uint32_t topk_out_indices_cb = TopKOutIndicesCBId;
        static constexpr uint32_t phase1_num_input_tiles = (NumValues + 1023) / 1024;
        static constexpr uint32_t phase2_scores_byte_offset = Phase2ScoresByteOffset;
        static constexpr uint32_t phase2_indices_byte_offset = Phase2IndicesByteOffset;
        static constexpr uint32_t phase2_num_input_tiles = TopK <= MIN_TOPK_ALIGNMENT
                                                               ? (NumSenders * MIN_TOPK_ALIGNMENT + 1023) / 1024
                                                               : (NumSenders * TopK + 1023) / 1024;
        static constexpr uint32_t topk_effective_k = TopK <= MIN_TOPK_ALIGNMENT ? MIN_TOPK_ALIGNMENT : TopK;
        static constexpr uint32_t topk_scores_slot_bytes = (topk_effective_k * sizeof(uint16_t) + 31u) & ~31u;
        static constexpr uint32_t topk_indices_slot_bytes = (topk_effective_k * sizeof(uint32_t) + 31u) & ~31u;
        static constexpr uint32_t mesh_stage_scores_cb = MeshStageScoresCBId;
        static constexpr uint32_t mesh_stage_indices_cb = MeshStageIndicesCBId;
        static constexpr uint32_t scores_scratch_stage2_offset = ScoresScratchStage2Offset;
        static constexpr uint32_t indices_scratch_stage2_offset = IndicesScratchStage2Offset;
        static constexpr uint32_t scores_scratch_addr = ScoresScratchAddr;
        static constexpr uint32_t indices_scratch_addr = IndicesScratchAddr;
    };

    template <
        uint32_t WinnerPageBytes,
        uint32_t LocalReadySemaphoreAddr,
        uint32_t SocketMode = 0,
        uint32_t SocketCBId = 0,
        uint32_t SocketPageSizeBytes = 0,
        uint32_t TopK = 32,
        uint32_t SoftmaxOutCBId = 0xFFFFFFFF,
        uint32_t RandCBId = 0xFFFFFFFF,
        uint32_t WinnerCBId = 0xFFFFFFFF,
        uint32_t PBF16 = 0,
        uint32_t TopKScoresSlotBytes = 0,
        uint32_t MeshMode = 0,
        uint32_t Stage2Receiver = 0,
        uint32_t OutputAddr = 0,
        uint32_t RandOutputAddr = 0,
        uint32_t InvTempBF16 = 0,
        uint32_t SoftmaxInCBId = 0xFFFFFFF,
        uint32_t TempCBId = 0xFFFFFFFF,
        uint32_t DeferSocketOutput = 0,
        uint32_t EnableMetadata = 0,
        uint32_t CopyProbabilities = 0,
        uint32_t MetadataOutputL1Addr = 0,
        uint32_t CopyProbabilitiesToQ = 0,
        uint32_t PBcastCBId = 0xFFFFFFFF,
        uint32_t RandBcastCBId = 0xFFFFFFFF,
        uint32_t ProbsOutCBId = 0xFFFFFFFF,
        uint32_t MaskCBId = 0xFFFFFFFF,
        uint32_t MaskAliasesScaler = 0>
    struct WriterCTArgs {
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t local_ready_semaphore_addr = LocalReadySemaphoreAddr;
        static constexpr uint32_t socket_mode = SocketMode;
        static constexpr uint32_t socket_cb_id = SocketCBId;
        static constexpr uint32_t socket_page_size_bytes = SocketPageSizeBytes;
        static constexpr uint32_t topk_k = TopK;
        static constexpr uint32_t softmax_out_cb = SoftmaxOutCBId;
        static constexpr uint32_t rand_cb = RandCBId;
        static constexpr uint32_t winner_cb_id = WinnerCBId;
        static constexpr float p = __builtin_bit_cast(float, PBF16);
        static constexpr uint32_t topk_scores_slot_bytes = TopKScoresSlotBytes;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
        static constexpr uint32_t output_addr = OutputAddr;
        static constexpr uint32_t rand_output_addr = RandOutputAddr;
        static constexpr uint32_t softmax_in_cb = SoftmaxInCBId;
        static constexpr bool defer_socket_output = DeferSocketOutput == 1;
        static constexpr bool enable_metadata = EnableMetadata == 1;
        static constexpr bool copy_probabilities = CopyProbabilities == 1;
        static constexpr bool copy_probabilities_to_q = CopyProbabilitiesToQ == 1;
        static constexpr uint32_t metadata_output_l1_addr = MetadataOutputL1Addr;
        static constexpr uint32_t p_bcast_cb = PBcastCBId;
        static constexpr uint32_t rand_bcast_cb = RandBcastCBId;
        static constexpr uint32_t probs_out_cb = ProbsOutCBId;
        static constexpr uint32_t mask_cb = MaskCBId;
        static constexpr bool mask_aliases_scaler = MaskAliasesScaler == 1;
        static_assert(
            !CopyProbabilities || EnableMetadata,
            "EnableMetadata must be true when CopyProbabilities is true as we copy into the metadata buffer");
        static_assert(
            !EnableMetadata || MetadataOutputL1Addr != 0,
            "MetadataOutputL1Addr must be set when EnableMetadata is true");
    };

    template <
        uint32_t SoftmaxInCBId,
        uint32_t SoftmaxOutCBId,
        uint32_t SoftmaxExpCBId,
        uint32_t SoftmaxSubCBId,
        uint32_t MaxCBId,
        uint32_t SumCBId,
        uint32_t ScalerCBId,
        uint32_t ProbsOutCBId,
        uint32_t RandCBId = 0xFFFFFFFF,
        uint32_t Seed = 520,
        uint32_t TopK = 32,
        uint32_t MeshMode = 0,
        uint32_t Stage1Receiver = 0,
        uint32_t Stage2Receiver = 0,
        uint32_t NumValues = 0,
        uint32_t NumSenders = 0,
        uint32_t TopKInScoresCBId = 0xFFFFFFFF,
        uint32_t TopKInIndicesCBId = 0xFFFFFFFF,
        uint32_t TopKOutScoresCBId = 0xFFFFFFFF,
        uint32_t TopKOutIndicesCBId = 0xFFFFFFFF,
        uint32_t MeshStageScoresCBId = 0xFFFFFFFF,
        uint32_t MeshStageIndicesCBId = 0xFFFFFFFF,
        uint32_t Stage1RowElements = 0,
        uint32_t Stage1NumInputTiles = 0,
        uint32_t Stage2RowElements = 0,
        uint32_t Stage2NumInputTiles = 0,
        uint32_t MaskCBId = 0xFFFFFFFF,
        uint32_t MaskAliasesScaler = 0,
        uint32_t EnableMetadata = 0,
        uint32_t MetadataOutputL1Addr = 0,
        uint32_t InvTempBF16 = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t softmax_in_cb = SoftmaxInCBId;
        static constexpr uint32_t softmax_out_cb = SoftmaxOutCBId;
        static constexpr uint32_t softmax_exp_cb = SoftmaxExpCBId;
        static constexpr uint32_t softmax_sub_cb = SoftmaxSubCBId;
        static constexpr uint32_t max_cb = MaxCBId;
        static constexpr uint32_t sum_cb = SumCBId;
        static constexpr uint32_t p_bcast_cb = SoftmaxSubCBId;
        static constexpr uint32_t rand_bcast_cb = SumCBId;
        static constexpr uint32_t scaler_cb = ScalerCBId;
        static constexpr uint32_t probs_out_cb = ProbsOutCBId;
        static constexpr uint32_t rand_cb = RandCBId;
        static constexpr uint32_t seed = Seed;
        static constexpr uint32_t topk_k = TopK;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage1_receiver = Stage1Receiver == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
        static constexpr uint32_t num_values = NumValues;
        static constexpr uint32_t num_senders = NumSenders;
        static constexpr uint32_t phase2_row_elements =
            NumSenders * (TopK <= MIN_TOPK_ALIGNMENT ? MIN_TOPK_ALIGNMENT : TopK);
        static constexpr uint32_t phase2_num_input_tiles = (phase2_row_elements + 1023) / 1024;
        static constexpr uint32_t topk_in_scores_cb = TopKInScoresCBId;
        static constexpr uint32_t topk_in_indices_cb = TopKInIndicesCBId;
        static constexpr uint32_t topk_out_scores_cb = TopKOutScoresCBId;
        static constexpr uint32_t topk_out_indices_cb = TopKOutIndicesCBId;
        static constexpr uint32_t phase1_num_input_tiles = (NumValues + 1023) / 1024;
        static constexpr uint32_t mesh_stage_scores_cb = MeshStageScoresCBId;
        static constexpr uint32_t mesh_stage_indices_cb = MeshStageIndicesCBId;
        static constexpr uint32_t stage1_row_elements = Stage1RowElements;
        static constexpr uint32_t stage1_num_input_tiles = Stage1NumInputTiles;
        static constexpr uint32_t stage2_row_elements = Stage2RowElements;
        static constexpr uint32_t stage2_num_input_tiles = Stage2NumInputTiles;
        static constexpr uint32_t mask_cb = MaskCBId;
        static constexpr bool mask_aliases_scaler = MaskAliasesScaler == 1;
        static constexpr bool enable_metadata = EnableMetadata == 1;
        static constexpr uint32_t metadata_output_l1_addr = MetadataOutputL1Addr;
        static constexpr uint32_t inv_temp_bf16 = InvTempBF16;
    };

    struct ReaderArgs {
        uint32_t scores_addr;
        uint32_t indices_addr;
        uint32_t output_addr;
        uint32_t final_noc_x;
        uint32_t final_noc_y;
        uint32_t global_sem_addr;
        uint32_t global_stage2_sem_addr;
    };

    struct WriterArgs {
        uint32_t final_noc_x;
        uint32_t final_noc_y;
        uint32_t socket_config_addr = 0;
        // Optional persistent-mode next-iteration signal routing (BRISC path).
        uint32_t persistent_enable = 0;
        uint32_t persistent_dst_noc_x = 0;
        uint32_t persistent_dst_noc_y = 0;
        uint32_t persistent_dst_mesh_id = 0;
        uint32_t persistent_dst_chip_id = 0;
        uint32_t persistent_dst_sem_addr = 0;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

#if defined(COMPILE_FOR_BRISC)
    struct BriscMeshSendMetadata {
        uint32_t dst_mesh_id;
        uint32_t dst_chip_id;
        uint32_t dst_scores_addr;
        uint32_t dst_indices_addr;
        uint32_t dst_sem_addr;
    };

#endif

    template <typename CTArgs, bool IsActiveCore, bool IsFinalCore, bool IsMeshSenderCore>
    class Op {
    public:
        size_t persistent_fabric_arg_idx = 0;
        void operator()(const RTArgs& args) { impl(args); }

        void set_seed(uint32_t seed = 0xFFFFFFFF) {
#if defined(COMPILE_FOR_TRISC)
            if (seed != 0xFFFFFFFF) {
                rand_tile_init(seed);
            }
#endif
        }

#if defined(COMPILE_FOR_BRISC)
        FORCE_INLINE void send_persistent_next_iter_inc_via_fabric_brisc(const WriterArgs& args, size_t& arg_idx) {
            if (args.persistent_enable == 0) {
                return;
            }
            constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
            auto route_id = PacketHeaderPool::allocate_header_n(1);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;
            set_unicast_route(
                packet_header,
                static_cast<uint16_t>(args.persistent_dst_chip_id),
                static_cast<uint16_t>(args.persistent_dst_mesh_id),
                1);
            packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                get_noc_addr(args.persistent_dst_noc_x, args.persistent_dst_noc_y, args.persistent_dst_sem_addr), 1});

            auto fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            fabric_sender.open();
            fabric_sender.wait_for_empty_write_slot();
            fabric_sender.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
            fabric_sender.close();
            noc_async_full_barrier();
        }

        FORCE_INLINE void send_d2h_token_from_cb_brisc(const WriterArgs& args) {
            const uint32_t socket_config_addr = args.socket_config_addr;
            SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
            set_sender_socket_page_size(sender_socket, CTArgs::socket_page_size_bytes);
            const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
            const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

            socket_reserve_pages(sender_socket, 1);
            cb_wait_front(CTArgs::socket_cb_id, 1);
            const uint32_t read_addr = get_read_ptr(CTArgs::socket_cb_id);

            noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                read_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                sizeof(uint32_t));
            noc_async_writes_flushed();

            cb_pop_front(CTArgs::socket_cb_id, 1);
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            update_socket_config(sender_socket);
            noc_async_write_barrier();
        }

        FORCE_INLINE void send_d2d_token_from_cb_brisc(const WriterArgs& args) {
            const uint32_t socket_config_addr = args.socket_config_addr;
            SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
            set_sender_socket_page_size(sender_socket, CTArgs::socket_page_size_bytes);

            socket_reserve_pages(sender_socket, 1);
            cb_wait_front(CTArgs::socket_cb_id, 1);
            const uint32_t read_addr = get_read_ptr(CTArgs::socket_cb_id);
            for (uint32_t i = 0; i < sender_socket.num_downstreams; i++) {
                sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
                noc_async_write(
                    read_addr,
                    get_noc_addr(
                        downstream_enc.d2d.downstream_noc_x,
                        downstream_enc.d2d.downstream_noc_y,
                        sender_socket.write_ptr + sender_socket.downstream_fifo_addr),
                    CTArgs::socket_page_size_bytes);
            }
            noc_async_write_barrier();

            cb_pop_front(CTArgs::socket_cb_id, 1);
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            update_socket_config(sender_socket);
        }

        FORCE_INLINE void send_deferred_socket_output_brisc(const WriterArgs& args) {
            if constexpr (IsFinalCore && CTArgs::defer_socket_output && CTArgs::socket_mode == 1) {
                send_d2h_token_from_cb_brisc(args);
            } else if constexpr (IsFinalCore && CTArgs::defer_socket_output && CTArgs::socket_mode == 2) {
                send_d2d_token_from_cb_brisc(args);
            }
        }
#endif

    private:
#if defined(COMPILE_FOR_NCRISC)
        FORCE_INLINE bool is_better_candidate(
            uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
            return bfloat16_greater(candidate_score, best_score) ||
                   ((candidate_score == best_score) && (candidate_index < best_index));
        }

        // Sorted insertion: maintain a descending-sorted array of K (score, index) pairs
        // in two separate output arrays.  After iterating over all num_values elements,
        // out_scores[0..K-1] and out_indices[0..K-1] contain the top-K in descending order.
        FORCE_INLINE void phase1_reduce_local_topk(
            volatile tt_l1_ptr uint16_t* scores_ptr,
            volatile tt_l1_ptr uint32_t* indices_ptr,
            volatile tt_l1_ptr uint16_t* out_scores,
            volatile tt_l1_ptr uint32_t* out_indices) {
            constexpr uint32_t K = CTArgs::topk_k;

            for (uint32_t j = 0; j < K; ++j) {
                out_scores[j] = NEG_INF_BFLOAT16;
                out_indices[j] = 0xFFFFFFFF;
            }

            for (uint32_t i = 0; i < CTArgs::num_values; ++i) {
                const uint16_t score = scores_ptr[i];
                const uint32_t index = indices_ptr[i];

                if (!is_better_candidate(score, index, out_scores[K - 1], out_indices[K - 1])) {
                    continue;
                }

                uint32_t pos = K - 1;
                while (pos > 0 && is_better_candidate(score, index, out_scores[pos - 1], out_indices[pos - 1])) {
                    --pos;
                }

                for (uint32_t j = K - 1; j > pos; --j) {
                    out_scores[j] = out_scores[j - 1];
                    out_indices[j] = out_indices[j - 1];
                }

                out_scores[pos] = score;
                out_indices[pos] = index;
            }
        }

        FORCE_INLINE void phase1_send_topk_to_final(
            uint32_t local_scores_addr,
            uint32_t local_indices_addr,
            uint32_t dst_scores_l1_addr,
            uint32_t dst_indices_l1_addr,
            uint32_t final_noc_x,
            uint32_t final_noc_y) {
            const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
            const uint64_t dst_sem_noc_addr = final_noc_base | static_cast<uint64_t>(CTArgs::receiver_semaphore_addr);
            noc_async_write_one_packet<true, true>(
                local_scores_addr, final_noc_base | dst_scores_l1_addr, CTArgs::topk_scores_slot_bytes);
            noc_async_write_one_packet<true, true>(
                local_indices_addr, final_noc_base | dst_indices_l1_addr, CTArgs::topk_indices_slot_bytes);
            noc_async_posted_writes_flushed();
            noc_semaphore_inc(dst_sem_noc_addr, 1);
            noc_async_atomic_barrier();
        }

        FORCE_INLINE void wait_and_reset_semaphore(volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t expected_count) {
            noc_semaphore_wait(sem_ptr, expected_count);
            noc_semaphore_set(sem_ptr, 0);
        }

        // N-way merge of per-core sorted-descending top-K arrays into a single
        // global top-K (also descending).  Each core contributed a sorted array;
        // we maintain one head pointer per core and greedily pick the best head
        // K times.  O(K * N) comparisons -- for K=32, N~100 that is ~3200.
        FORCE_INLINE void phase2_merge_global_topk(
            uint32_t scores_base,
            uint32_t indices_base,
            volatile tt_l1_ptr uint16_t* global_scores,
            volatile tt_l1_ptr uint32_t* global_indices) {
            constexpr uint32_t K = CTArgs::topk_k;
            constexpr uint32_t N = CTArgs::num_senders;

            uint8_t heads[N];
            for (uint32_t c = 0; c < N; ++c) {
                heads[c] = 0;
            }

            for (uint32_t out = 0; out < K; ++out) {
                uint16_t best_score = NEG_INF_BFLOAT16;
                uint32_t best_index = 0xFFFFFFFF;
                uint32_t best_core = 0;

                for (uint32_t c = 0; c < N; ++c) {
                    if (heads[c] >= K) {
                        continue;
                    }
                    auto s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        scores_base + c * CTArgs::topk_scores_slot_bytes);
                    auto idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        indices_base + c * CTArgs::topk_indices_slot_bytes);
                    if (is_better_candidate(s[heads[c]], idx[heads[c]], best_score, best_index)) {
                        best_score = s[heads[c]];
                        best_index = idx[heads[c]];
                        best_core = c;
                    }
                }

                global_scores[out] = best_score;
                global_indices[out] = best_index;
                heads[best_core]++;
            }
        }

        // Write top-K scores and indices to contiguous scratch regions.
        // Layout: [all scores contiguous] [all indices contiguous]
        FORCE_INLINE void write_topk_slot(
            uint32_t scores_slot_addr, uint32_t indices_slot_addr, uint32_t scores_base, uint32_t indices_base) {
            constexpr uint32_t K = CTArgs::topk_k;
            // auto dst_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_slot_addr);
            // auto dst_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_slot_addr);
            // for (uint32_t i = 0; i < K; ++i) {
            //     dst_scores[i] = scores[i];
            // }
            // for (uint32_t i = 0; i < K; ++i) {
            //     dst_indices[i] = indices[i];
            // }
            noc_async_read(get_noc_addr(scores_base), scores_slot_addr, CTArgs::topk_scores_slot_bytes);
            noc_async_read(get_noc_addr(indices_base), indices_slot_addr, CTArgs::topk_indices_slot_bytes);
            noc_async_read_barrier();
        }

        // N-way merge of sorted top-K arrays received in mesh stage scratch slots.
        // Scratch layout per stage (contiguous):
        //   [scores_0 | scores_1 | ... | scores_N-1]
        //   [indices_0 | indices_1 | ... | indices_N-1]
        FORCE_INLINE void phase3_merge_mesh_stage_topk_slots(
            uint32_t stage_scores_base,
            uint32_t stage_indices_base,
            uint32_t stage_num_slots,
            volatile tt_l1_ptr uint16_t* out_scores,
            volatile tt_l1_ptr uint32_t* out_indices) {
            constexpr uint32_t K = CTArgs::topk_k;

            uint8_t heads[16];  // max mesh dimension
            for (uint32_t s = 0; s < stage_num_slots; ++s) {
                heads[s] = 0;
            }

            for (uint32_t out = 0; out < K; ++out) {
                uint16_t best_score = NEG_INF_BFLOAT16;
                uint32_t best_index = 0xFFFFFFFF;
                uint32_t best_slot = 0;

                for (uint32_t s = 0; s < stage_num_slots; ++s) {
                    if (heads[s] >= K) {
                        continue;
                    }
                    auto s_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        stage_scores_base + s * CTArgs::topk_scores_slot_bytes);
                    auto s_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        stage_indices_base + s * CTArgs::topk_indices_slot_bytes);
                    if (is_better_candidate(s_scores[heads[s]], s_indices[heads[s]], best_score, best_index)) {
                        best_score = s_scores[heads[s]];
                        best_index = s_indices[heads[s]];
                        best_slot = s;
                    }
                }

                out_scores[out] = best_score;
                out_indices[out] = best_index;
                heads[best_slot]++;
            }
        }
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        template <typename packet_header_t>
        FORCE_INLINE void set_unicast_route(
            volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
            if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
                fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
            } else {
                fabric_set_unicast_route<false>(header, num_hops);
            }
        }

        FORCE_INLINE void write_winner_slot(uint32_t slot_addr, uint16_t score, uint32_t index) {
            auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
            auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
            slot_u16_ptr[0] = score;
            slot_u32_ptr[1] = index;
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        FORCE_INLINE BriscMeshSendMetadata load_mesh_send_metadata(size_t& arg_idx) {
            BriscMeshSendMetadata metadata{};
            metadata.dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
            metadata.dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
            metadata.dst_scores_addr = get_arg_val<uint32_t>(arg_idx++);
            metadata.dst_indices_addr = get_arg_val<uint32_t>(arg_idx++);
            metadata.dst_sem_addr = get_arg_val<uint32_t>(arg_idx++);
            return metadata;
        }

        // Send top-K [scores | indices] from winner CB via fused fabric scatter write
        // + atomic increment.  Chunk 0 → remote scores slot, chunk 1 → remote indices slot.
        FORCE_INLINE void send_mesh_topk_via_fabric_brisc(
            uint32_t final_noc_x,
            uint32_t final_noc_y,
            uint32_t local_src_addr,
            const BriscMeshSendMetadata& metadata,
            size_t& arg_idx) {
            constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
            auto route_id = PacketHeaderPool::allocate_header_n(1);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;
            set_unicast_route(
                packet_header,
                static_cast<uint16_t>(metadata.dst_chip_id),
                static_cast<uint16_t>(metadata.dst_mesh_id),
                1);
            packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                    {get_noc_addr(final_noc_x, final_noc_y, metadata.dst_scores_addr),
                     get_noc_addr(final_noc_x, final_noc_y, metadata.dst_indices_addr)},
                    get_noc_addr(final_noc_x, final_noc_y, metadata.dst_sem_addr),
                    {static_cast<uint16_t>(CTArgs::topk_scores_slot_bytes)},
                    1,
                    true},
                CTArgs::winner_page_bytes);
            auto fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            fabric_sender.open();
            fabric_sender.wait_for_empty_write_slot();
            fabric_sender.send_payload_without_header_non_blocking_from_address(
                local_src_addr, CTArgs::winner_page_bytes);
            fabric_sender.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
            fabric_sender.close();
            noc_async_full_barrier();
        }
#endif

        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            const uint32_t scores_cb_base = get_write_ptr(CTArgs::topk_in_scores_cb);
            const uint32_t indices_cb_base = get_write_ptr(CTArgs::topk_in_indices_cb);

            uint32_t scores_addr = args.scores_addr;
            if constexpr (IsActiveCore && (CTArgs::scores_cb_id != 0xFFFFFFFF)) {
                cb_wait_front(CTArgs::scores_cb_id, CTArgs::scores_num_pages);
                scores_addr = get_read_ptr(CTArgs::scores_cb_id);
            }
            invalidate_l1_cache();

            // Phase 1: per-core local top-K and delivery to the final core.
            if constexpr (IsActiveCore) {
                const uint32_t dst_scores_l1 = scores_cb_base + CTArgs::phase2_scores_byte_offset +
                                               CTArgs::sender_idx * CTArgs::topk_scores_slot_bytes;
                const uint32_t dst_indices_l1 = indices_cb_base + CTArgs::phase2_indices_byte_offset +
                                                CTArgs::sender_idx * CTArgs::topk_indices_slot_bytes;

                if constexpr (CTArgs::topk_k <= 32) {
                    constexpr uint32_t num_input_tiles = CTArgs::phase1_num_input_tiles;

                    auto scores_src = scores_addr;
                    auto indices_src = args.indices_addr;

                    cb_reserve_back(CTArgs::topk_in_scores_cb, num_input_tiles);
                    cb_reserve_back(CTArgs::topk_in_indices_cb, num_input_tiles);
                    auto scores_dst = get_write_ptr(CTArgs::topk_in_scores_cb);
                    auto indices_dst = get_write_ptr(CTArgs::topk_in_indices_cb);
                    auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_dst);
                    auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_dst);
                    noc_async_read(get_noc_addr(scores_src), scores_dst, CTArgs::num_values * sizeof(uint16_t));
                    noc_async_read(get_noc_addr(indices_src), indices_dst, CTArgs::num_values * sizeof(uint32_t));

                    noc_async_read_barrier();

                    cb_push_back(CTArgs::topk_in_scores_cb, num_input_tiles);
                    cb_push_back(CTArgs::topk_in_indices_cb, num_input_tiles);

                    cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                    cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                    phase1_send_topk_to_final(
                        get_read_ptr(CTArgs::topk_out_scores_cb),
                        get_read_ptr(CTArgs::topk_out_indices_cb),
                        dst_scores_l1,
                        dst_indices_l1,
                        args.final_noc_x,
                        args.final_noc_y);

                    cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                    cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                }
            }
            if constexpr (IsActiveCore && (CTArgs::scores_cb_id != 0xFFFFFFFF)) {
                cb_pop_front(CTArgs::scores_cb_id, CTArgs::scores_num_pages);
            }

            // Phase 2: merge per-core top-K arrays into a single device-wide top-K.
            // Output goes to the winner CB in split layout [K scores | K indices].
            // The argmax is global_scores[0] / global_indices[0] (descending order).
            if constexpr (IsFinalCore) {
                auto recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::receiver_semaphore_addr);
                {
                    DeviceZoneScopedN("SP-PHASE2WAIT");
                    wait_and_reset_semaphore(recv_sem_ptr, CTArgs::expected_remote_incs + 1);
                }
                cb_reserve_back(CTArgs::winner_cb_id, 1);
                const uint32_t global_scores = get_write_ptr(CTArgs::winner_cb_id);
                const uint32_t global_indices = global_scores + CTArgs::topk_scores_slot_bytes;

                if constexpr (CTArgs::topk_k <= 32) {
                    constexpr uint32_t p2_tiles = CTArgs::phase2_num_input_tiles;

                    auto all_cores_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        scores_cb_base + CTArgs::phase2_scores_byte_offset);
                    auto all_cores_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        indices_cb_base + CTArgs::phase2_indices_byte_offset);

                    cb_reserve_back(CTArgs::topk_in_scores_cb, p2_tiles);
                    cb_push_back(CTArgs::topk_in_scores_cb, p2_tiles);

                    cb_reserve_back(CTArgs::topk_in_indices_cb, p2_tiles);
                    cb_push_back(CTArgs::topk_in_indices_cb, p2_tiles);

                    cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                    cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                    {
                        DeviceZoneScopedN("SP-PHASE2READ");
                        auto llk_scores_addr = get_read_ptr(CTArgs::topk_out_scores_cb);
                        auto llk_indices_addr = get_read_ptr(CTArgs::topk_out_indices_cb);
                        noc_async_read(get_noc_addr(llk_scores_addr), global_scores, CTArgs::topk_scores_slot_bytes);
                        noc_async_read(get_noc_addr(llk_indices_addr), global_indices, CTArgs::topk_indices_slot_bytes);
                        noc_async_read_barrier();
                    }

                    cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                    cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                } else {
                    auto global_scores_addr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(global_scores);
                    auto global_indices_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_indices);
                    const uint32_t p2_scores_base = scores_cb_base + CTArgs::phase2_scores_byte_offset;
                    const uint32_t p2_indices_base = indices_cb_base + CTArgs::phase2_indices_byte_offset;
                    phase2_merge_global_topk(p2_scores_base, p2_indices_base, global_scores_addr, global_indices_addr);
                }

                // Mesh inter-device reduction stages via LLK (k==32) or scalar merge.
                if constexpr (CTArgs::mesh_mode) {
                    if constexpr (CTArgs::stage1_receiver) {
                        write_topk_slot(
                            CTArgs::scores_scratch_addr +
                                CTArgs::stage1_local_slot_idx * CTArgs::topk_scores_slot_bytes,
                            CTArgs::indices_scratch_addr +
                                CTArgs::stage1_local_slot_idx * CTArgs::topk_indices_slot_bytes,
                            global_scores,
                            global_indices);
                        auto global_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_sem_addr);

                        {
                            DeviceZoneScopedN("SP-MESH1WAIT");
                            wait_and_reset_semaphore(global_sem_ptr, CTArgs::stage1_expected_remote_incs);
                        }

                        if constexpr (CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                            constexpr uint32_t s1_tiles = CTArgs::stage1_num_slots * CTArgs::topk_k;
                            constexpr uint32_t s1_input_tiles = (s1_tiles + 1023) / 1024;
                            cb_reserve_back(CTArgs::mesh_stage_scores_cb, s1_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_scores_cb, s1_input_tiles);
                            cb_reserve_back(CTArgs::mesh_stage_indices_cb, s1_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_indices_cb, s1_input_tiles);

                            cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                            cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                            auto llk_scores = get_read_ptr(CTArgs::topk_out_scores_cb);
                            auto llk_indices = get_read_ptr(CTArgs::topk_out_indices_cb);
                            noc_async_read(get_noc_addr(llk_scores), global_scores, CTArgs::topk_scores_slot_bytes);
                            noc_async_read(get_noc_addr(llk_indices), global_indices, CTArgs::topk_indices_slot_bytes);
                            noc_async_read_barrier();

                            cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                            cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                        }
                    }

                    // Signal BRISC to send via fabric (sends from winner CB directly).

                    if constexpr (CTArgs::stage2_receiver) {
                        write_topk_slot(
                            CTArgs::scores_scratch_addr + CTArgs::scores_scratch_stage2_offset +
                                CTArgs::stage2_local_slot_idx * CTArgs::topk_scores_slot_bytes,
                            CTArgs::indices_scratch_addr + CTArgs::indices_scratch_stage2_offset +
                                CTArgs::stage2_local_slot_idx * CTArgs::topk_indices_slot_bytes,
                            global_scores,
                            global_indices);

                        auto global_stage2_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_stage2_sem_addr);
                        {
                            DeviceZoneScopedN("SP-MESH2WAIT");
                            wait_and_reset_semaphore(global_stage2_sem_ptr, CTArgs::stage2_expected_remote_incs);
                        }
                        if constexpr (CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                            constexpr uint32_t s2_tiles = CTArgs::stage2_num_slots * CTArgs::topk_k;
                            constexpr uint32_t s2_input_tiles = (s2_tiles + 1023) / 1024;
                            cb_reserve_back(CTArgs::mesh_stage_scores_cb, s2_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_scores_cb, s2_input_tiles);
                            cb_reserve_back(CTArgs::mesh_stage_indices_cb, s2_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_indices_cb, s2_input_tiles);

                            cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                            cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                            auto llk_scores = get_read_ptr(CTArgs::topk_out_scores_cb);
                            auto llk_indices = get_read_ptr(CTArgs::topk_out_indices_cb);
                            noc_async_read(get_noc_addr(llk_scores), global_scores, CTArgs::topk_scores_slot_bytes);
                            noc_async_read(get_noc_addr(llk_indices), global_indices, CTArgs::topk_indices_slot_bytes);
                            noc_async_read_barrier();

                            cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                            cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                        }
                    }
                }

                // Pack top-K scores into a bf16 tile for TRISC softmax.
                // In mesh mode, only the stage-2 receiver (absolute final device) does this.
                if constexpr (!CTArgs::mesh_mode || CTArgs::stage2_receiver) {
                    generate_reduce_scaler(CTArgs::scaler_cb, BF16_ONE);
                }
                cb_push_back(CTArgs::winner_cb_id, 1);
            } else if constexpr (IsActiveCore && !IsFinalCore) {
                constexpr uint32_t p2_tiles = CTArgs::phase2_num_input_tiles;
                cb_reserve_back(CTArgs::topk_in_scores_cb, p2_tiles);
                cb_push_back(CTArgs::topk_in_scores_cb, p2_tiles);

                cb_reserve_back(CTArgs::topk_in_indices_cb, p2_tiles);
                cb_push_back(CTArgs::topk_in_indices_cb, p2_tiles);
            }
#elif defined(COMPILE_FOR_BRISC)
            invalidate_l1_cache();
            size_t arg_idx = 0;
            PacketHeaderPool::reset();

            if constexpr (IsFinalCore) {
                cb_wait_front(CTArgs::winner_cb_id, 1);
                const uint32_t global_scores = get_read_ptr(CTArgs::winner_cb_id);
                if constexpr (IsMeshSenderCore) {
                    // Mesh sender: wait for NCRISC to finish, then send via fabric
                    const BriscMeshSendMetadata fabric_meta = load_mesh_send_metadata(arg_idx);
                    {
                        DeviceZoneScopedN("SP-MESHSEND");
                        send_mesh_topk_via_fabric_brisc(
                            args.final_noc_x, args.final_noc_y, global_scores, fabric_meta, arg_idx);
                    }
                } else {
                    DeviceZoneScopedN("SP-FINALCORE");
                    // Top-P filtering + random categorical selection.
                    if constexpr (!CTArgs::mesh_mode || CTArgs::stage2_receiver) {
                        uint32_t K = CTArgs::topk_k;
                        float p = CTArgs::p;
                        if constexpr (CTArgs::enable_metadata) {
                            auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(
                                CTArgs::metadata_output_l1_addr);
                            K = std::min(
                                std::max(static_cast<uint32_t>(metadata_ptr->k), static_cast<uint32_t>(1)),
                                static_cast<uint32_t>(32));
                            p = std::min(std::max(static_cast<float>(metadata_ptr->p), 0.0f), 1.0f);
                            float temperature = std::max(static_cast<float>(metadata_ptr->temperature), 0.01f);
                            if (temperature == 0.0f) {
                                K = 1;
                                p = 1.0f;
                            }
                        }

                        {
                            DeviceZoneScopedN("SP-FC-STAGE");

                            generate_bcast_col_scalar(
                                CircularBuffer(CTArgs::p_bcast_cb), bf16_pack_to_uint32(float_to_bf16_rne(p)));

                            cb_reserve_back(CTArgs::softmax_in_cb, 1);
                            auto tile_u32 =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CTArgs::softmax_in_cb));
                            constexpr uint32_t NEG_INF_BF16_PAIR = 0xFF80FF80;
                            constexpr uint32_t FACE_U32 = 128;  // 256 bf16 per face / 2
                            for (uint32_t i = 0; i < 8; ++i) {
                                tile_u32[i] = NEG_INF_BF16_PAIR;
                            }
                            for (uint32_t i = 0; i < 8; ++i) {
                                tile_u32[FACE_U32 + i] = NEG_INF_BF16_PAIR;
                            }

                            noc_async_read(
                                get_noc_addr(global_scores),
                                get_write_ptr(CTArgs::softmax_in_cb),
                                std::min(K, ELEMS_PER_FACE_ROW) * sizeof(uint16_t));
                            if (K > ELEMS_PER_FACE_ROW) {
                                noc_async_read(
                                    get_noc_addr(global_scores + ELEMS_PER_FACE_ROW * sizeof(uint16_t)),
                                    get_write_ptr(CTArgs::softmax_in_cb) + FACE_ELEMS * sizeof(uint16_t),
                                    std::min(K - ELEMS_PER_FACE_ROW, ELEMS_PER_FACE_ROW) * sizeof(uint16_t));
                            }
                            noc_async_read_barrier();
                            cb_push_back(CTArgs::softmax_in_cb, 1);
                        }
                        uint16_t rand;
                        {
                            DeviceZoneScopedN("SP-FC-RAND-STAGE");
                            cb_wait_front(CTArgs::rand_cb, 1);
                            auto rand_u16 =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::rand_cb));
                            rand = rand_u16[0];
                            generate_bcast_col_scalar(CircularBuffer(CTArgs::rand_bcast_cb), bf16_pack_to_uint32(rand));
                        }

                        {
                            DeviceZoneScopedN("SP-FC-WAIT");
                            cb_wait_front(CTArgs::softmax_out_cb, 1);
                            if constexpr (CTArgs::mask_aliases_scaler) {
                                cb_wait_front(CTArgs::probs_out_cb, 1);
                            } else {
                                cb_wait_front(CTArgs::mask_cb, 1);
                            }
                        }

                        auto mask_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::mask_cb));
                        auto global_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_read_ptr(CTArgs::winner_cb_id) + CTArgs::topk_scores_slot_bytes);

                        uint32_t selected_index;
                        {
                            DeviceZoneScopedN("SP-FC-LOOKUP");
                            if (K == 1) {
                                selected_index = global_indices[0];
                            } else {
                                selected_index = global_indices[K - 1];
                                for (uint32_t i = 0; i < K; ++i) {
                                    const uint32_t tile_idx =
                                        (i < ELEMS_PER_FACE_ROW)
                                            ? (i * ELEMS_PER_FACE_ROW)
                                            : (2 * FACE_ELEMS + (i - ELEMS_PER_FACE_ROW) * ELEMS_PER_FACE_ROW);
                                    if (mask_u16[tile_idx] != 0) {
                                        selected_index = global_indices[i];
                                        break;
                                    }
                                }
                            }
                        }

                        {
                            DeviceZoneScopedN("SP-FC-FINISH");
                            auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::output_addr);
                            output_ptr[0] = selected_index;

                            if constexpr (CTArgs::socket_mode != 0) {
                                cb_reserve_back(CTArgs::socket_cb_id, 1);
                                auto d2h_ptr =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CTArgs::socket_cb_id));
                                d2h_ptr[0] = selected_index;
                                cb_push_back(CTArgs::socket_cb_id, 1);
                            }

                            if constexpr (CTArgs::rand_output_addr != 0) {
                                auto rand_out =
                                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::rand_output_addr);
                                rand_out[0] = rand;
                            }
                        }

                        cb_wait_front(CTArgs::probs_out_cb, 1);
                        if constexpr (CTArgs::copy_probabilities) {
                            DeviceZoneScopedN("SP-CPROBS");
                            constexpr auto scores_field = CTArgs::copy_probabilities_to_q
                                                              ? offsetof(deepseek_b1_ops::DeepseekMetadata, q_scores)
                                                              : offsetof(deepseek_b1_ops::DeepseekMetadata, p_scores);
                            constexpr auto indices_field = CTArgs::copy_probabilities_to_q
                                                               ? offsetof(deepseek_b1_ops::DeepseekMetadata, q_indices)
                                                               : offsetof(deepseek_b1_ops::DeepseekMetadata, p_indices);

                            // Reuse rand_cb's tile slot as a tiny gather scratch -- it's
                            // already popped above and the slot stays reserved for one
                            // more iteration. 32 bf16 lanes = 64 bytes, well under one tile.
                            const uint32_t scratch_l1 = get_read_ptr(CTArgs::rand_cb);
                            auto scratch = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_l1);

                            if (K == 1) {
                                scratch[0] = float_to_bf16_rne(1.0f);
                                for (uint32_t i = 1; i < 2 * ELEMS_PER_FACE_ROW; ++i) {
                                    scratch[i] = 0;
                                }
                            } else {
                                const auto probs_l1 =
                                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::probs_out_cb));
                                for (uint32_t i = 0; i < ELEMS_PER_FACE_ROW; ++i) {
                                    scratch[i] = probs_l1[i * ELEMS_PER_FACE_ROW];
                                    scratch[ELEMS_PER_FACE_ROW + i] = probs_l1[2 * FACE_ELEMS + i * ELEMS_PER_FACE_ROW];
                                }

                                const auto cum_l1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                    get_read_ptr(CTArgs::softmax_out_cb));
                                const uint16_t p_bf16 = float_to_bf16_rne(p);
                                uint32_t num_kept = K;
                                for (uint32_t i = 0; i < K; ++i) {
                                    const uint32_t cum_idx =
                                        (i < ELEMS_PER_FACE_ROW)
                                            ? (i * ELEMS_PER_FACE_ROW)
                                            : (2 * FACE_ELEMS + (i - ELEMS_PER_FACE_ROW) * ELEMS_PER_FACE_ROW);
                                    if (!(cum_l1[cum_idx] < p_bf16)) {
                                        num_kept = i + 1;
                                        break;
                                    }
                                }
                                if (num_kept < 1) {
                                    num_kept = 1;
                                }
                                if (num_kept > K) {
                                    num_kept = K;
                                }
                                for (uint32_t i = num_kept; i < K; ++i) {
                                    scratch[i] = 0;
                                }
                            }

                            const uint32_t scores_dst = CTArgs::metadata_output_l1_addr + scores_field;
                            noc_async_write_one_packet(scratch_l1, get_noc_addr(scores_dst), 32 * sizeof(uint16_t));

                            const uint32_t indices_src_l1 =
                                get_read_ptr(CTArgs::winner_cb_id) + CTArgs::topk_scores_slot_bytes;
                            const uint32_t indices_dst_l1 = CTArgs::metadata_output_l1_addr + indices_field;
                            noc_async_write_one_packet(
                                indices_src_l1, get_noc_addr(indices_dst_l1), 32 * sizeof(uint32_t));

                            noc_async_write_barrier();
                        }
                        cb_pop_front(CTArgs::probs_out_cb, 1);
                        cb_pop_front(CTArgs::softmax_out_cb, 1);
                        if constexpr (!CTArgs::mask_aliases_scaler) {
                            cb_pop_front(CTArgs::mask_cb, 1);
                        }
                        cb_pop_front(CTArgs::rand_cb, 1);
                    }
                }
                if constexpr (!CTArgs::defer_socket_output) {
                    if constexpr (CTArgs::socket_mode == 1) {
                        send_d2h_token_from_cb_brisc(args);
                    }
                    if constexpr (CTArgs::socket_mode == 2) {
                        send_d2d_token_from_cb_brisc(args);
                    }
                }
                persistent_fabric_arg_idx = arg_idx;
                if constexpr (!CTArgs::defer_socket_output) {
                    send_persistent_next_iter_inc_via_fabric_brisc(args, arg_idx);
                }
                cb_pop_front(CTArgs::winner_cb_id, 1);
            }
#elif defined(COMPILE_FOR_TRISC)

            // Matmul leaves the PACK MOP in block-contiguous mode; re-init to standard tile-by-tile
            // so that subsequent pack_tile() calls in run_top32_llk produce correct results.
            ckernel::pack_reconfig_data_format(CTArgs::topk_out_scores_cb);
            PACK((llk_pack_init(CTArgs::topk_out_scores_cb)));

            // Phase 1: LLK top-32 sort (all active cores, k==32 only)
            if constexpr (IsActiveCore && CTArgs::topk_k <= 32) {
                DeviceZoneScopedN("SP-PHASE1LLK");
                run_top32_llk<
                    CTArgs::topk_in_scores_cb,
                    CTArgs::topk_in_indices_cb,
                    CTArgs::topk_out_scores_cb,
                    CTArgs::topk_out_indices_cb>(CTArgs::num_values, CTArgs::phase1_num_input_tiles, 1);
            }

            // Non-final cores: consume dummy pages pushed by NCRISC to keep
            // the topk_in CB write pointer synchronized with the final core.
            if constexpr (IsActiveCore && !IsFinalCore && CTArgs::topk_k <= 32) {
                constexpr uint32_t p2_tiles = CTArgs::phase2_num_input_tiles;
                cb_wait_front(CTArgs::topk_in_scores_cb, p2_tiles);
                cb_pop_front(CTArgs::topk_in_scores_cb, p2_tiles);

                cb_wait_front(CTArgs::topk_in_indices_cb, p2_tiles);
                cb_pop_front(CTArgs::topk_in_indices_cb, p2_tiles);
            }

            // Phase 2: global merge via LLK (final core only, k==32)
            if constexpr (IsFinalCore && CTArgs::topk_k <= 32) {
                {
                    DeviceZoneScopedN("SP-PHASE2LLK");
                    run_top32_llk_presorted_1024_opt<
                        CTArgs::topk_in_scores_cb,
                        CTArgs::topk_in_indices_cb,
                        CTArgs::topk_out_scores_cb,
                        CTArgs::topk_out_indices_cb>(CTArgs::phase2_row_elements, CTArgs::phase2_num_input_tiles, 2);
                }
            }

            // Mesh stage 1 merge via LLK (stage1_receiver, final core, k==32)
            if constexpr (
                IsFinalCore && CTArgs::mesh_mode && CTArgs::stage1_receiver && CTArgs::topk_k <= 32 &&
                CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                DeviceZoneScopedN("SP-MESH1LLK");
                run_top32_llk<
                    CTArgs::mesh_stage_scores_cb,
                    CTArgs::mesh_stage_indices_cb,
                    CTArgs::topk_out_scores_cb,
                    CTArgs::topk_out_indices_cb,
                    true>(CTArgs::stage1_row_elements, CTArgs::stage1_num_input_tiles, 3);
            }

            // Mesh stage 2 merge via LLK (stage2_receiver, final core, k==32)
            if constexpr (
                IsFinalCore && CTArgs::mesh_mode && CTArgs::stage2_receiver && CTArgs::topk_k <= 32 &&
                CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                DeviceZoneScopedN("SP-MESH2LLK");
                run_top32_llk<
                    CTArgs::mesh_stage_scores_cb,
                    CTArgs::mesh_stage_indices_cb,
                    CTArgs::topk_out_scores_cb,
                    CTArgs::topk_out_indices_cb,
                    true>(CTArgs::stage2_row_elements, CTArgs::stage2_num_input_tiles, 4);
            }

            // Fused softmax + cumsum + top-P + inverse-CDF sampling on TRISC
            if constexpr (IsFinalCore && (!CTArgs::mesh_mode || CTArgs::stage2_receiver)) {
                trisc_fused_softmax_top_p_sampling_block<
                    CTArgs::softmax_in_cb,
                    CTArgs::softmax_out_cb,
                    CTArgs::softmax_exp_cb,
                    CTArgs::max_cb,
                    CTArgs::probs_out_cb,
                    CTArgs::scaler_cb,
                    CTArgs::p_bcast_cb,
                    CTArgs::rand_cb,
                    CTArgs::rand_bcast_cb,
                    CTArgs::mask_cb,
                    /*num_tiles=*/1,
                    CTArgs::enable_metadata,
                    CTArgs::metadata_output_l1_addr,
                    CTArgs::inv_temp_bf16>();
            }
#endif
        }
    };
};  // struct TopKSampling

}  // namespace deepseek_b1_ops
