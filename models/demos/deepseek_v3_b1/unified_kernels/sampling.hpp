// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"
#include "api/numeric/bfloat16.h"

#if defined(COMPILE_FOR_TRISC)
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#endif
#endif

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
#include "../metadata/metadata.hpp"
#include "../kernel_includes/tt_metal/dm_utils.hpp"

constexpr uint32_t FACE_ELEMS = 256;
constexpr uint16_t BF16_ONE = 0x3F80;
constexpr uint32_t ELEMS_PER_FACE_ROW = 16;

// Bit-cast helpers without UB
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

// Convert float32 to bf16 bit-pattern using round-to-nearest-even
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

// Fill row 0 (face-0 row-0 lanes 0..15 + face-1 row-0 lanes 0..15) of a 32x32
// bf16 tile with `bf16_val`, repeated. Rows 1..31 are left as-is.
//
// Used by the Stage-B fused TRISC pipeline: TRISC's downstream element-wise
// ops (mul_binary_tile, lt_binary_tile, ...) need the broadcast value at every
// row-0 column, but only row 0 is read by BRISC, so filling rows 1..31 would
// be wasted L1 traffic. The fill takes 16 u32 stores (~32 cy on BRISC).
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
#include "api/compute/transpose_wh.h"
#include "api/compute/transpose_wh_dest.h"
#include "api/compute/cumsum.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rand.h"

#if defined(TRISC_UNPACK)
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_top32_rm_api.h"
#endif
#if defined(TRISC_MATH)
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_top32_rm_api.h"
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_deepseek_top32_rm.h"
#endif

// Sampling-local explicit-fidelity forks of the compute API helpers. Keep
// these local so the core API can continue defaulting to the kernel-wide
// MATH_FIDELITY macro, while sampling can force HiFi4 in only the softmax
// normalization path.
template <MathFidelity math_fidelity>
ALWI void sampling_mul_tiles_bcast_scalar_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::SCALAR, math_fidelity>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

template <MathFidelity math_fidelity>
ALWI void sampling_mul_tiles_bcast_scalar(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          ELWMUL,
          BroadcastType::SCALAR,
          DST_ACCUM_MODE,
          math_fidelity,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

template <MathFidelity math_fidelity>
ALWI void sampling_mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::COL, math_fidelity>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

template <MathFidelity math_fidelity>
ALWI void sampling_mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          math_fidelity,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation, MathFidelity math_fidelity>
ALWI void sampling_reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb, icb_scaler, ocb, call_line);
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, math_fidelity, enforce_fp32_accumulation>()));
    if constexpr (enforce_fp32_accumulation) {
        MATH((tensix_sync()));
        MATH((reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11)));
    }
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
#else
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, math_fidelity>(icb)));
    PACK((llk_pack_reduce_mask_config<reduce_dim>()));
#endif
}

template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation, MathFidelity math_fidelity>
ALWI void sampling_reduce_tile(
    uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, math_fidelity, false, enforce_fp32_accumulation>(
        icb, icb_scaler, idst)));
    UNPACK((llk_unpack_AB_reduce<reduce_type, reduce_dim>(icb, icb_scaler, itile, itile_scaler)));
#else
    MATH((llk_math_reduce(idst)));
    UNPACK((llk_unpack_AB_reduce(icb, icb_scaler, itile, itile_scaler)));
#endif
}

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols>
void softmax_sub_exp_bcast_cols() {
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    // NOTE: <false> selects the accurate (Taylor / range-reduced) FP32 exp
    // path instead of the fast piecewise approximation. Slower, but the
    // approximate path was the dominant source of softmax-tail error in
    // top-P sampling -- see test_sampling p_scores tolerance discussion.
    exp_tile_init<false>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < cols; ++u) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            exp_tile<false>(0);
            tile_regs_commit();
            cb_reserve_back(out_cb, 1);
            tile_regs_wait();
            pack_reconfig_data_format(out_cb);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            tile_regs_release();
        }
    }
    cb_pop_front(in0_cb, rows * cols);
    cb_pop_front(in1_cb, rows);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t out_cb,
    uint32_t rows,
    uint32_t cols>
void softmax_reduce_c() {
    reconfig_data_format(in0_cb, scale_cb);
    sampling_reduce_init<pool_type, reduce_dim, false, MathFidelity::HiFi4>(in0_cb, scale_cb, out_cb);
    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    constexpr uint32_t reduce_dst_idx = 0;
    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        for (uint32_t j = 0; j < cols; j++) {
            sampling_reduce_tile<pool_type, reduce_dim, false, MathFidelity::HiFi4>(
                in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    reduce_uninit();
    UNPACK(tensix_sync());
}

inline void softmax_recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_reconfig_data_format(in_cb);
        pack_tile(0, in_cb);
        cb_push_back(in_cb, 1);
        release_dst();
    }
}


template <uint32_t in_cb, uint32_t out_cb, uint32_t temp_cb, uint32_t num_tiles, >



template <uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t out_cb, uint32_t num_tiles>
void softmax_mul_block_bcast_scalar() {
    reconfig_data_format<false, true>(in0_cb, in1_scalar_cb);
    sampling_mul_tiles_bcast_scalar_init_short<MathFidelity::HiFi4>(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        sampling_mul_tiles_bcast_scalar<MathFidelity::HiFi4>(in0_cb, in1_scalar_cb, 0, 0, 0);
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    // Consume the broadcast scalar tile; otherwise persistent runs can leak scalar CB state.
    cb_pop_front(in1_scalar_cb, 1);
    cb_pop_front(in0_cb, num_tiles);
}

inline void softmax_mul_block_bcast_cols(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    uint32_t num_tiles = rows * cols;
    sampling_mul_bcast_cols_init_short<MathFidelity::HiFi4>(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            sampling_mul_tiles_bcast_cols<MathFidelity::HiFi4>(in0_cb, in1_cb, 0, i, 0);
            cb_reserve_back(out_cb, 1);
            pack_reconfig_data_format(out_cb);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            release_dst();
        }
    }
    cb_pop_front(in1_cb, rows);
    cb_pop_front(in0_cb, num_tiles);
}

void generate_rand_tile(const uint32_t cb_id) {
    uint32_t rand_scale = 0;
    const float one_f = 1.0f;
    std::memcpy(&rand_scale, &one_f, sizeof(uint32_t));
    uint32_t rand_from = 0;
    // if (seed != 0xFFFFFFFF) {
    //     rand_tile_init(seed);
    // }
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

    acquire_dst();

    uint32_t num_faces = 4;
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

    MATH((llk_math_deepseek_top32_rm_init<false>()));
    if constexpr (presorted) {
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing, false)));
    } else {
        MATH((llk_math_deepseek_top32_rm_local_sort<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing)));
    }
    MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, false)));
    MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing, true)));

    for (uint32_t i = 64; i < row_elements; i += 64) {
        if (i + 64 > row_elements) {
            num_faces = 2;
        } else {
            num_faces = 4;
        }

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
            MATH(
                (llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles + 1, decreasing, false)));
        } else {
            MATH((llk_math_deepseek_top32_rm_local_sort<false, DST_ACCUM_MODE>(value_offset_tiles + 1, decreasing)));
        }
        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles + 1, false)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles + 1, increasing, true)));

        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, true)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing, true)));
    }

    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_reconfig_data_format(out_scores_cb);
    ckernel::pack_tile(value_offset_tiles, out_scores_cb);
    ckernel::pack_reconfig_data_format(out_indices_cb);
    ckernel::pack_tile(index_offset_tiles, out_indices_cb);
    PACK(TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0));

    release_dst();

    cb_pop_front(in_scores_cb, num_input_tiles);
    cb_pop_front(in_indices_cb, num_input_tiles);
    // Phase 1's llk_unpack_A_top32_rm_init modifies four unpacker registers for
    // row-major mode. transpose_wh_init_short restores Haloize_mode and X counter,
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

    acquire_dst();

    const uint32_t num_chunks = row_elements / chunk_size;

    // Step 1: load first 1024 values/indices chunk with transpose.
    reconfig_data_format_srca(in_scores_cb);
    transpose_wh_init_short(in_scores_cb);
    transpose_wh_tile(in_scores_cb, 0, value_offset_tiles);

    reconfig_data_format_srca(in_indices_cb);
    transpose_wh_init_short(in_indices_cb);
    transpose_wh_tile(in_indices_cb, 0, index_offset_tiles);

    // Step 2: prepare first chunk for pre-sorted combine pipeline.
    MATH((llk_math_deepseek_top32_rm_init<false>()));
    MATH((llk_math_deepseek_top32_of_1024_rm_pre_sorted_prep<false, DST_ACCUM_MODE, decreasing>(value_offset_tiles)));

    // Steps 3-5: ingest remaining full 1024 chunks and combine.
    for (uint32_t i = 1; i < num_chunks; ++i) {
        reconfig_data_format_srca(in_scores_cb);
        transpose_wh_init_short(in_scores_cb);
        transpose_wh_tile(in_scores_cb, i, value_offset_tiles + 1);

        reconfig_data_format_srca(in_indices_cb);
        transpose_wh_init_short(in_indices_cb);
        transpose_wh_tile(in_indices_cb, i, index_offset_tiles + 1);

        MATH((llk_math_deepseek_top32_of_1024_rm_pre_sorted_prep<false, DST_ACCUM_MODE, increasing>(
            value_offset_tiles + 1)));
        MATH((llk_math_deepseek_top32_of_1024_rm_pre_sorted_combine<false, DST_ACCUM_MODE>((value_offset_tiles))));
    }

    // Step 6: collapse per-face top-32 to a single top-32.
    MATH((llk_math_deepseek_top32_of_1024_rm_pre_sorted_final<false, DST_ACCUM_MODE>(value_offset_tiles)));

    // Steps 7-9: handle trailing (<1024) values in 64-element chunks.
    uint32_t num_faces = 4;
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

        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles + 1, decreasing, false)));
        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles + 1, false)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles + 1, increasing, true)));

        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, true)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing, true)));
    }

    // Step 10: pack final top-32 scores/indices.
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_reconfig_data_format(out_scores_cb);
    ckernel::pack_tile(value_offset_tiles, out_scores_cb);
    ckernel::pack_reconfig_data_format(out_indices_cb);
    ckernel::pack_tile(index_offset_tiles, out_indices_cb);

    PACK(TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0));

    // Reset unpacker state for downstream operations (softmax)
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0)));
    UNPACK(TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));
    UNPACK(TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111));

    release_dst();

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
        uint32_t ReceiverSemaphoreId,
        uint32_t LocalReadySemaphoreId,
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
        static constexpr uint32_t receiver_semaphore_id = ReceiverSemaphoreId;
        static constexpr uint32_t local_ready_semaphore_id = LocalReadySemaphoreId;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage1_sender = Stage1Sender == 1;
        static constexpr bool stage1_receiver = Stage1Receiver == 1;
        static constexpr bool stage2_sender = Stage2Sender == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
        static constexpr uint32_t stage1_slot_base_offset = Stage1SlotBaseOffset;
        static constexpr uint32_t stage1_num_slots = Stage1NumSlots;
        static constexpr uint32_t stage1_expected_remote_incs = Stage1ExpectedRemoteIncs;
        static constexpr uint32_t stage1_local_slot_idx = Stage1LocalSlotOffset;  // slot index (not byte offset)
        static constexpr uint32_t stage2_slot_base_offset = Stage2SlotBaseOffset;
        static constexpr uint32_t stage2_num_slots = Stage2NumSlots;
        static constexpr uint32_t stage2_expected_remote_incs = Stage2ExpectedRemoteIncs;
        static constexpr uint32_t stage2_local_slot_idx = Stage2LocalSlotOffset;  // slot index (not byte offset)
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

        // Per-core gather slot size in bytes: NOC transfer size, address stride between slots,
        // and winner CB layout offset. Padded to 32-byte alignment for NOC efficiency.
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
        uint32_t LocalReadySemaphoreId,
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
        // Stage-B fused-TRISC sampling pipeline reuses two existing compute-side
        // CBs (`softmax_sub_cb`, `sum_cb`) as broadcast inputs from BRISC to TRISC.
        // BRISC pushes `p` and `rand` as broadcast tiles; TRISC pops them.
        uint32_t PBcastCBId = 0xFFFFFFFF,
        uint32_t RandBcastCBId = 0xFFFFFFFF>
    struct WriterCTArgs {
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t local_ready_semaphore_id = LocalReadySemaphoreId;
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
        static constexpr uint32_t inv_temp_bf16 = InvTempBF16;
        static constexpr uint32_t softmax_in_cb = SoftmaxInCBId;
        static constexpr uint32_t temp_cb = TempCBId;
        static constexpr bool defer_socket_output = DeferSocketOutput == 1;
        static constexpr bool enable_metadata = EnableMetadata == 1;
        static constexpr bool copy_probabilities = CopyProbabilities == 1;
        static constexpr bool copy_probabilities_to_q = CopyProbabilitiesToQ == 1;
        static constexpr uint32_t metadata_output_l1_addr = MetadataOutputL1Addr;
        // Stage-B broadcast CBs (mapped onto the existing compute-side
        // `softmax_sub_cb` / `sum_cb` slots, which the softmax tail never
        // touches): BRISC pushes one-tile broadcasts of `p` and `rand` here.
        static constexpr uint32_t p_bcast_cb = PBcastCBId;
        static constexpr uint32_t rand_bcast_cb = RandBcastCBId;
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
        uint32_t TempCBId,
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
        uint32_t Stage2NumInputTiles = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t softmax_in_cb = SoftmaxInCBId;
        static constexpr uint32_t softmax_out_cb = SoftmaxOutCBId;
        static constexpr uint32_t softmax_exp_cb = SoftmaxExpCBId;
        static constexpr uint32_t softmax_sub_cb = SoftmaxSubCBId;
        static constexpr uint32_t max_cb = MaxCBId;
        static constexpr uint32_t sum_cb = SumCBId;
        // Stage-B semantic aliases (same physical CBs as softmax_sub_cb / sum_cb,
        // which the softmax tail never reads). BRISC pushes broadcast `p` to
        // p_bcast_cb and broadcast `rand` to rand_bcast_cb; TRISC pops them
        // in the fused post-softmax pipeline.
        static constexpr uint32_t p_bcast_cb = SoftmaxSubCBId;
        static constexpr uint32_t rand_bcast_cb = SumCBId;
        static constexpr uint32_t scaler_cb = ScalerCBId;
        static constexpr uint32_t temp_cb = TempCBId;
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
            const uint64_t dst_sem_noc_addr =
                final_noc_base | static_cast<uint64_t>(get_semaphore(CTArgs::receiver_semaphore_id));
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


                    {
                        DeviceZoneScopedN("SP-PHASE1SEND");
                        phase1_send_topk_to_final(
                            get_read_ptr(CTArgs::topk_out_scores_cb),
                            get_read_ptr(CTArgs::topk_out_indices_cb),
                            dst_scores_l1,
                            dst_indices_l1,
                            args.final_noc_x,
                            args.final_noc_y);
                    }

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
                auto recv_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::receiver_semaphore_id));
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
                    // Top-P filtering + random categorical selection
                    if constexpr (!CTArgs::mesh_mode || CTArgs::stage2_receiver) {
                        // constexpr uint32_t FACE_ELEMS = 256;
                        // constexpr uint32_t ELEMS_PER_FACE_ROW = 16;
                        uint32_t K = CTArgs::topk_k;
                        uint32_t inv_temp_bf16 = CTArgs::inv_temp_bf16;
                        float p = CTArgs::p;

                        if constexpr (CTArgs::enable_metadata) {
                            auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(
                                CTArgs::metadata_output_l1_addr);
                            // volatile-qualified fields do not deduce with std::max/min literals; load scalars first.
                            float temperature = std::max(static_cast<float>(metadata_ptr->temperature), 0.01f);
                            // Pack two copies of the bf16 scalar into a uint32 so that
                            // generate_bcast_unary_scalar writes a correctly-filled tile word.
                            inv_temp_bf16 = float_to_bf16_packed(1.0f / temperature);
                            K = std::min(
                                std::max(static_cast<uint32_t>(metadata_ptr->k), static_cast<uint32_t>(1)),
                                static_cast<uint32_t>(32));
                            p = std::min(
                                std::max(static_cast<float>(metadata_ptr->p), 0.0f), 1.0f);
                        }

                        // Stage-B fused-TRISC pipeline: BRISC just stages inputs (logits +
                        // broadcast `p` + broadcast `rand`), then consumes TRISC's outputs
                        // (normalized probs + sample_mask). All of the soft-float that used
                        // to live here (Pass 1 sum, Pass 2 top-P, Pass 3 sample +
                        // normalize) now runs on TRISC's SFPU. The only BRISC math left
                        // in SP-FINALCORE is a K-element integer scan for first-1 in the
                        // sample_mask tile (~32 cy).
                        {
                            DeviceZoneScopedN("SP-FC-STAGE");
                            generate_bcast_unary_scalar(CTArgs::temp_cb, inv_temp_bf16);

                            // Broadcast `p` to TRISC for SP-FUSED-THR (mul_binary_tile against
                            // dst[0]=total_exp). Done before the NOC reads so it overlaps with
                            // softmax_mul_block_bcast_scalar on the compute side.
                            generate_row0_bcast(CTArgs::p_bcast_cb, float_to_bf16_rne(p));

                            cb_reserve_back(CTArgs::softmax_in_cb, 1);
                            auto tile_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                get_write_ptr(CTArgs::softmax_in_cb));
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

                        // rand_cb is generated by TRISC just after the softmax tail, so we
                        // can stage rand_bcast_cb the moment it lands -- this push happens
                        // long before TRISC reaches SP-FUSED-NORM, so it's effectively free.
                        uint16_t rand;
                        {
                            DeviceZoneScopedN("SP-FC-RAND-STAGE");
                            cb_wait_front(CTArgs::rand_cb, 1);
                            auto rand_u16 =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::rand_cb));
                            rand = rand_u16[0];
                            generate_row0_bcast(CTArgs::rand_bcast_cb, rand);
                        }

                        {
                            DeviceZoneScopedN("SP-FC-WAIT");
                            // softmax_out_cb now holds the *normalized* probabilities
                            // (TRISC repacked it at the end of SP-FUSED-NORM).
                            cb_wait_front(CTArgs::softmax_out_cb, 1);
                            // softmax_in_cb has been recycled twice on TRISC since BRISC
                            // pushed the staged logits: cumsum (popped in SP-FUSED-THR),
                            // then the sample_mask we now consume.
                            cb_wait_front(CTArgs::softmax_in_cb, 1);
                        }

                        auto prob_u16 =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::softmax_out_cb));
                        // mask_select tile from TRISC: bf16 0x3F80 (=1.0) at the selected
                        // position(s) (positions where cumsum >= rand_scaled AND cumsum <=
                        // cum_kept), bf16 0x0000 elsewhere. The earliest non-zero lane in
                        // row 0 is our sampled token.
                        auto mask_u16 =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::softmax_in_cb));
                        auto global_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_read_ptr(CTArgs::winner_cb_id) + CTArgs::topk_scores_slot_bytes);

                        uint32_t selected_index;
                        {
                            DeviceZoneScopedN("SP-FC-LOOKUP");
                            // Default fallback: last kept token. TRISC's mask construction
                            // guarantees at least one row-0 lane in [0, K) is non-zero (the
                            // top-P cutoff position is always selected when rand_scaled ==
                            // cum_kept), but defensive fallback in case of bf16 noise.
                            selected_index = global_indices[K - 1];
                            for (uint32_t i = 0; i < K; ++i) {
                                const uint32_t tile_idx = (i < 16) ? i : FACE_ELEMS + (i - 16);
                                if (mask_u16[tile_idx] != 0) {
                                    selected_index = global_indices[i];
                                    break;
                                }
                            }
                        }
                        {
                            DeviceZoneScopedN("SP-FC-FINISH");
                            // No need to zero filtered slots: SP-FUSED-NORM already multiplied
                            // prob_u16 by mask_keep, so filtered positions are exactly 0.
                            auto output_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::output_addr);
                            output_ptr[0] = selected_index;

                            if constexpr (CTArgs::socket_mode != 0) {
                                cb_reserve_back(CTArgs::socket_cb_id, 1);
                                auto d2h_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    get_write_ptr(CTArgs::socket_cb_id));
                                d2h_ptr[0] = selected_index;
                                cb_push_back(CTArgs::socket_cb_id, 1);
                            }

                            if constexpr (CTArgs::rand_output_addr != 0) {
                                auto rand_out =
                                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::rand_output_addr);
                                rand_out[0] = rand;
                            }
                        }

                        if constexpr (CTArgs::copy_probabilities) {
                            DeviceZoneScopedN("SP-CPROBS");
                            
                            // Scatter the 32 rescaled top-P probabilities out of the two-face
                            // tile layout and copy the 32 winning indices into the metadata.
                            // When copy_probabilities_to_q is set (spec/verification stage),
                            // write to q_scores/q_indices; otherwise write to p_scores/p_indices.
                            constexpr uint32_t HALF_SCORES_BYTES = 16 * sizeof(uint16_t);
                            constexpr uint32_t FACE_BYTES_OFFSET = FACE_ELEMS * sizeof(uint16_t);

                            constexpr auto scores_field = CTArgs::copy_probabilities_to_q
                                ? offsetof(deepseek_b1_ops::DeepseekMetadata, q_scores)
                                : offsetof(deepseek_b1_ops::DeepseekMetadata, p_scores);
                            constexpr auto indices_field = CTArgs::copy_probabilities_to_q
                                ? offsetof(deepseek_b1_ops::DeepseekMetadata, q_indices)
                                : offsetof(deepseek_b1_ops::DeepseekMetadata, p_indices);

                            const uint32_t scores_src_face0 = get_read_ptr(CTArgs::softmax_out_cb);
                            const uint32_t scores_dst_face0 =
                                CTArgs::metadata_output_l1_addr + scores_field;

                            noc_async_write_one_packet(
                                scores_src_face0, get_noc_addr(scores_dst_face0), HALF_SCORES_BYTES);
                            const uint32_t scores_src_face1 = scores_src_face0 + FACE_BYTES_OFFSET;
                            const uint32_t scores_dst_face1 = scores_dst_face0 + HALF_SCORES_BYTES;
                            noc_async_write_one_packet(
                                scores_src_face1, get_noc_addr(scores_dst_face1), HALF_SCORES_BYTES);

                            const uint32_t indices_src_l1 =
                                get_read_ptr(CTArgs::winner_cb_id) + CTArgs::topk_scores_slot_bytes;
                            const uint32_t indices_dst_l1 =
                                CTArgs::metadata_output_l1_addr + indices_field;
                            noc_async_write_one_packet(
                                indices_src_l1, get_noc_addr(indices_dst_l1), 32 * sizeof(uint32_t));

                            noc_async_write_barrier();
                        }

                        cb_pop_front(CTArgs::softmax_out_cb, 1);
                        cb_pop_front(CTArgs::softmax_in_cb, 1);
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
            PACK((llk_pack_init<false, false, false>(CTArgs::topk_out_scores_cb)));

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

            // Softmax tail + row-wise prefix sum (final core only).
            //
            // We deliberately stop the softmax after `sub_max + exp` and ship the
            // *unnormalized* exponentials `e_i = exp((x_i - max(x))/T)` into
            // `softmax_out_cb`. Then we immediately follow with a single SFPU
            // cumsum across row 0 and pack it back into `softmax_in_cb` (which
            // softmax_mul_block_bcast_scalar just finished consuming).
            //
            // The whole point is to let BRISC finish the math with bit-pattern
            // uint16 compares instead of soft-float adds. `total_exp = cumsum[K-1]`
            // is free, the top-P cut becomes ≤K uint16 compares against
            // `p * total_exp`, and the inverse-CDF sample becomes ≤K uint16
            // compares against `rand_f * cum_kept`. The TRISC normalization
            // constant `Z` cancels in BRISC's rescale by `1/cum_kept` because we
            // stayed in unnormalized space throughout
            // (`(e_i/Z) / (S_norm) = e_i / cum_kept`), so the old
            // `softmax_reduce_c<SUM>` + `softmax_recip_block_inplace` +
            // `softmax_mul_block_bcast_cols` trio remains pure waste.
            if constexpr (IsFinalCore && (!CTArgs::mesh_mode || CTArgs::stage2_receiver)) {
                {
                    DeviceZoneScopedN("SP-SOFTMAX");
                    softmax_mul_block_bcast_scalar<
                        CTArgs::softmax_in_cb,
                        CTArgs::temp_cb,
                        CTArgs::softmax_exp_cb,
                        1>();
                    softmax_reduce_c<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        CTArgs::softmax_exp_cb,
                        CTArgs::scaler_cb,
                        CTArgs::max_cb,
                        1,
                        1>();
                    softmax_sub_exp_bcast_cols<
                        CTArgs::softmax_exp_cb,
                        CTArgs::max_cb,
                        CTArgs::softmax_out_cb,
                        1,
                        1>();
                    // NB: scaler_cb pop is delayed until SP-FUSED-NORM so that
                    // the same 1.0-broadcast tile feeds all three reductions
                    // (softmax_reduce_c, SP-FUSED-THR reduce_max, SP-FUSED-NORM
                    // reduce_min). NCRISC pushes scaler_cb only once per step.

                    generate_rand_tile(CTArgs::rand_cb);
                }

                // Row-wise prefix sum of the unnormalized exponentials e_i that
                // softmax_sub_exp_bcast_cols just produced in softmax_out_cb. We
                // pack the result back into softmax_in_cb -- it's free at this
                // point (softmax_mul_block_bcast_scalar popped it) so we get the
                // staging buffer for free as a return-trip CB.
                //
                // cumsum_tile is a column-wise SFPU scan, so we sandwich it
                // between two transposes to scan row 0 instead.
                {
                    DeviceZoneScopedN("SP-CUMSUM");

                    transpose_wh_init(CTArgs::softmax_out_cb, CTArgs::softmax_in_cb);
                    cumsum_tile_init();

                    cb_wait_front(CTArgs::softmax_out_cb, 1);
                    cb_reserve_back(CTArgs::softmax_in_cb, 1);

                    tile_regs_acquire();
                    transpose_wh_init_short(CTArgs::softmax_out_cb);
                    transpose_wh_tile(CTArgs::softmax_out_cb, 0, 0);
                    cumsum_tile(0, /*first=*/true);
                    transpose_wh_dest_init_short();
                    transpose_wh_dest(0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_reconfig_data_format(CTArgs::softmax_in_cb);
                    pack_tile(0, CTArgs::softmax_in_cb);
                    cb_push_back(CTArgs::softmax_in_cb, 1);
                    tile_regs_release();
                }

                // ── Stage B: fused top-P + inverse-CDF + normalize on SFPU ──
                //
                // Inputs:
                //   softmax_in_cb     row-0 prefix sum (from SP-CUMSUM above)
                //   softmax_out_cb    row-0 unnormalized exponentials e_i
                //   p_bcast_cb        row-0 broadcast of `p`  (BRISC-staged)
                //   rand_bcast_cb     row-0 broadcast of `rand` (BRISC-staged)
                //   scaler_cb         broadcast 1.0 for reduces
                //
                // Outputs:
                //   softmax_out_cb    row-0 normalized probabilities (0 for filtered)
                //   softmax_in_cb     row-0 mask_select tile (BRISC scans for first 1)
                //
                // Math:
                //   threshold   = p * total_exp,  total_exp = max(cumsum)
                //   filt_cumsum = cumsum + (cumsum < threshold) * BIG_VAL
                //   cum_kept    = min(filt_cumsum)
                //   rand_scaled = rand * cum_kept
                //   mask_keep   = cumsum <= cum_kept
                //   mask_select = (cumsum >= rand_scaled) AND mask_keep
                //   prob_i      = (e_i / cum_kept) * mask_keep
                //
                // All scalars live broadcast across row 0 (only row 0 is read
                // by BRISC downstream; rows 1..31 carry through as don't-care
                // garbage). The 1/cum_kept divide is one SFPU recip_tile instead
                // of BRISC's old __divsf3 + K __mulsf3 loop.
                // Both SP-FUSED-THR and SP-FUSED-NORM are written to fit in 4 DST
                // tiles (half-sync mode, the default for bf16 dst with
                // fp32_dest_acc_en=false). Registers are aggressively reused; the
                // comments mark each overwrite point.
                {
                    DeviceZoneScopedN("SP-FUSED-THR");

                    cb_wait_front(CTArgs::softmax_in_cb, 1);   // cumsum tile
                    cb_wait_front(CTArgs::p_bcast_cb, 1);       // p tile

                    tile_regs_acquire();

                    // dst[0] = total_exp broadcast across row 0
                    sampling_reduce_init<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        false,
                        MathFidelity::HiFi4>(
                        CTArgs::softmax_in_cb, CTArgs::scaler_cb, CTArgs::softmax_exp_cb);
                    sampling_reduce_tile<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        false,
                        MathFidelity::HiFi4>(
                        CTArgs::softmax_in_cb, CTArgs::scaler_cb, 0, 0, 0);
                    reduce_uninit();

                    // dst[1] = p (row-0 lanes 0..31 all carry p)
                    copy_tile_to_dst_init_short(CTArgs::p_bcast_cb);
                    copy_tile(CTArgs::p_bcast_cb, 0, 1);

                    // dst[1] = total_exp * p = threshold
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 1);

                    // dst[2] = cumsum reloaded into DST so we can compare against threshold
                    copy_tile_to_dst_init_short(CTArgs::softmax_in_cb);
                    copy_tile(CTArgs::softmax_in_cb, 0, 2);

                    // dst[3] = (cumsum < threshold) ? 1 : 0   (1 = "below cutoff, filter out for reduce_min")
                    lt_binary_tile_init();
                    lt_binary_tile(2, 1, 3);

                    // dst[3] *= BIG_VAL (fp32 ~3.4e38). mul_unary_tile's param1
                    // is a fp32 scalar encoded as uint32 (NOT bf16-packed).
                    constexpr uint32_t BIG_VAL_FP32_U32 = 0x7F7FFFFFu;  // FLT_MAX bit pattern
                    binop_with_scalar_tile_init();
                    mul_unary_tile(3, BIG_VAL_FP32_U32);

                    // dst[2] = filtered cumsum = cumsum + mask_filter_out * BIG_VAL.
                    // Overwrites dst[2]=cumsum, which is no longer needed (we'll
                    // recompute it cheaply in SP-FUSED-NORM if we need it again).
                    add_binary_tile_init();
                    add_binary_tile(2, 3, 2);

                    tile_regs_commit();

                    tile_regs_wait();
                    cb_reserve_back(CTArgs::softmax_exp_cb, 1);
                    pack_reconfig_data_format(CTArgs::softmax_exp_cb);
                    pack_tile(2, CTArgs::softmax_exp_cb);
                    cb_push_back(CTArgs::softmax_exp_cb, 1);
                    tile_regs_release();

                    cb_pop_front(CTArgs::softmax_in_cb, 1);
                    cb_pop_front(CTArgs::p_bcast_cb, 1);
                }

                {
                    DeviceZoneScopedN("SP-FUSED-NORM");

                    cb_wait_front(CTArgs::softmax_exp_cb, 1);   // filtered cumsum
                    cb_wait_front(CTArgs::rand_bcast_cb, 1);     // rand tile

                    tile_regs_acquire();

                    // dst[0] = cum_kept broadcast (min over row-0 lanes of filtered cumsum)
                    sampling_reduce_init<
                        PoolType::MIN,
                        ReduceDim::REDUCE_ROW,
                        false,
                        MathFidelity::HiFi4>(
                        CTArgs::softmax_exp_cb, CTArgs::scaler_cb, CTArgs::softmax_out_cb);
                    sampling_reduce_tile<
                        PoolType::MIN,
                        ReduceDim::REDUCE_ROW,
                        false,
                        MathFidelity::HiFi4>(
                        CTArgs::softmax_exp_cb, CTArgs::scaler_cb, 0, 0, 0);
                    reduce_uninit();

                    // dst[1] = rand (row-0 broadcast)
                    copy_tile_to_dst_init_short(CTArgs::rand_bcast_cb);
                    copy_tile(CTArgs::rand_bcast_cb, 0, 1);

                    // dst[1] = cum_kept * rand = rand_scaled
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 1);

                    // dst[2] = cumsum (recompute from softmax_out_cb -- cheaper than packing
                    // a second copy to a CB during SP-CUMSUM).
                    transpose_wh_init_short(CTArgs::softmax_out_cb);
                    transpose_wh_tile(CTArgs::softmax_out_cb, 0, 2);
                    cumsum_tile_init();
                    cumsum_tile(2, /*first=*/true);
                    transpose_wh_dest_init_short();
                    transpose_wh_dest(2);

                    // dst[3] = mask_keep = (cumsum <= cum_kept). Needed for both the
                    // mask_select AND-mask below AND for zeroing filtered probs.
                    le_binary_tile_init();
                    le_binary_tile(2, 0, 3);

                    // dst[1] = mask_ge_rand = (cumsum >= rand_scaled). Overwrites
                    // dst[1]=rand_scaled, which is no longer needed.
                    ge_binary_tile_init();
                    ge_binary_tile(2, 1, 1);

                    // dst[1] = mask_select_final = mask_ge_rand * mask_keep.
                    // ANDing with mask_keep guarantees BRISC's scan never selects a
                    // filtered token in the rand_scaled == cum_kept edge case.
                    mul_binary_tile_init();
                    mul_binary_tile(1, 3, 1);

                    // dst[0] = 1 / cum_kept (overwrites the cum_kept broadcast).
                    recip_tile_init();
                    recip_tile(0);

                    // dst[2] = e_i (overwrites the cumsum we computed above; cumsum
                    // is no longer needed now that mask_keep and mask_select are built).
                    copy_tile_to_dst_init_short(CTArgs::softmax_out_cb);
                    copy_tile(CTArgs::softmax_out_cb, 0, 2);

                    // dst[2] = e_i * (1/cum_kept) = preliminary normalized prob
                    mul_binary_tile_init();
                    mul_binary_tile(2, 0, 2);

                    // dst[2] *= mask_keep => exact zero in filtered positions
                    mul_binary_tile(2, 3, 2);

                    tile_regs_commit();

                    tile_regs_wait();

                    // Re-pack normalized probs over softmax_out_cb (single-slot CB,
                    // so pop before reserve).
                    cb_pop_front(CTArgs::softmax_out_cb, 1);
                    cb_reserve_back(CTArgs::softmax_out_cb, 1);
                    pack_reconfig_data_format(CTArgs::softmax_out_cb);
                    pack_tile(2, CTArgs::softmax_out_cb);
                    cb_push_back(CTArgs::softmax_out_cb, 1);

                    // softmax_in_cb is free (popped at the end of SP-FUSED-THR); reuse
                    // it as the sample-mask buffer that BRISC will scan.
                    cb_reserve_back(CTArgs::softmax_in_cb, 1);
                    pack_reconfig_data_format(CTArgs::softmax_in_cb);
                    pack_tile(1, CTArgs::softmax_in_cb);
                    cb_push_back(CTArgs::softmax_in_cb, 1);

                    tile_regs_release();

                    cb_pop_front(CTArgs::softmax_exp_cb, 1);
                    cb_pop_front(CTArgs::rand_bcast_cb, 1);
                    cb_pop_front(CTArgs::scaler_cb, 1);
                }
            }
#endif
        }
    };
};  // struct TopKSampling

}  // namespace deepseek_b1_ops
