// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"
#include "api/numeric/bfloat16.h"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "api/socket_api.h"
#include "../micro_ops/host_io/kernels/pcie_noc_utils.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "../../../unified_kernels/kernel_op_api.hpp"

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
    const uint32_t exp_mask  = 0x7F800000u;
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

uint16_t bfloat16_add(uint16_t bf16_a, uint16_t bf16_b) {
    float a = bf16_to_float(bf16_a);
    float b = bf16_to_float(bf16_b);
    float sum = a + b;
    return float_to_bf16_rne(sum);
}

uint16_t bfloat16_div(uint16_t bf16_a, uint16_t bf16_b) {
    uint16_t sign_a = bf16_a & 0x8000;
    uint16_t sign_b = bf16_b & 0x8000;
    int16_t exp_a = (bf16_a & 0x7F80) >> 7;
    int16_t exp_b = (bf16_b & 0x7F80) >> 7;
    uint16_t mant_a = bf16_a & 0x007F;
    uint16_t mant_b = bf16_b & 0x007F;

    int a_is_nan = (exp_a == 0xFF) && (mant_a != 0);
    int b_is_nan = (exp_b == 0xFF) && (mant_b != 0);
    int a_is_inf = (exp_a == 0xFF) && (mant_a == 0);
    int b_is_inf = (exp_b == 0xFF) && (mant_b == 0);
    int a_is_zero = (exp_a == 0) && (mant_a == 0);
    int b_is_zero = (exp_b == 0) && (mant_b == 0);

    if (a_is_nan) {
        return bf16_a;
    }
    if (b_is_nan) {
        return bf16_b | 0x0040;
    }
    if (a_is_inf && b_is_inf) {
        return 0x7FC0;
    }
    if (a_is_inf) {
        return (sign_a ^ sign_b) | 0x7F80;
    }
    if (b_is_inf) {
        return (sign_a ^ sign_b);
    }
    if (a_is_zero && b_is_zero) {
        return 0x7FC0;
    }
    if (a_is_zero) {
        return (sign_a ^ sign_b);
    }
    if (b_is_zero) {
        return (sign_a ^ sign_b) | 0x7F80;
    }

    if (exp_a == 0) {
        exp_a = 1;
        while (mant_a && !(mant_a & 0x80)) {
            mant_a <<= 1;
            exp_a -= 1;
        }
        if (mant_a) {
            mant_a &= 0x7F;
        }
    } else {
        mant_a = mant_a | 0x80;
    }

    if (exp_b == 0) {
        exp_b = 1;
        while (mant_b && !(mant_b & 0x80)) {
            mant_b <<= 1;
            exp_b -= 1;
        }
        if (mant_b) {
            mant_b &= 0x7F;
        }
    } else {
        mant_b = mant_b | 0x80;
    }

    uint16_t sign_res = sign_a ^ sign_b;
    int16_t exp_res = exp_a - exp_b + 127;
    uint32_t mant_res = ((uint32_t)mant_a << 7) / mant_b;

    while (mant_res && !(mant_res & 0x80)) {
        mant_res <<= 1;
        exp_res -= 1;
    }

    if (exp_res >= 0xFF) {
        return sign_res | 0x7F80;
    }

    if (exp_res <= 0) {
        if (exp_res < -7) {
            return sign_res;
        }
        mant_res >>= (1 - exp_res);
        exp_res = 0;
    } else {
        mant_res &= 0x7F;
    }

    uint16_t result = sign_res | (exp_res << 7) | (mant_res & 0x7F);
    return result;
}

#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/debug/dprint_tensix.h"
#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose_wh.h"
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

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols>
void softmax_sub_exp_bcast_cols() {
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < cols; ++u) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            exp_tile<true>(0);
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
    reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
#if defined(TRISC_UNPACK)
    constexpr uint32_t debug_elems = 32;
    auto in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in0_cb).fifo_rd_ptr << cb_addr_shift);
    auto scale_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(scale_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_reduce_c in0[" << i << "]: " << BF16(in0_ptr[i]) << ENDL();
    }
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_reduce_c scale[" << i << "]: " << BF16(scale_ptr[i]) << ENDL();
    }
#endif
    constexpr uint32_t reduce_dst_idx = 0;
    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(reduce_dst_idx, out_cb);
#if defined(TRISC_PACK)
        if (i == 0) {
            constexpr uint32_t debug_elems = 32;
            auto out_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(out_cb).fifo_wr_ptr << cb_addr_shift);
            for (uint32_t j = 0; j < debug_elems; ++j) {
                DPRINT << "softmax_reduce_c out[" << j << "]: " << BF16(out_ptr[j]) << ENDL();
            }
        }
#endif
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
#if defined(TRISC_UNPACK)
    constexpr uint32_t debug_elems = 32;
    auto in_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_recip_block_inplace in[" << i << "]: " << BF16(in_ptr[i]) << ENDL();
    }
#endif
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_reconfig_data_format(in_cb);
        pack_tile(0, in_cb);
#if defined(TRISC_PACK)
        if (i == 0) {
            constexpr uint32_t debug_elems = 32;
            auto out_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in_cb).fifo_wr_ptr << cb_addr_shift);
            for (uint32_t j = 0; j < debug_elems; ++j) {
                DPRINT << "softmax_recip_block_inplace out[" << j << "]: " << BF16(out_ptr[j]) << ENDL();
            }
        }
#endif
        cb_push_back(in_cb, 1);
        release_dst();
    }
}

template <uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t out_cb, uint32_t num_tiles>
void softmax_mul_block_bcast_scalar() {
    reconfig_data_format<false, true>(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
#if defined(TRISC_UNPACK)
    constexpr uint32_t debug_elems = 32;
    auto in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in0_cb).fifo_rd_ptr << cb_addr_shift);
    auto scalar_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
        get_local_cb_interface(in1_scalar_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_mul_block_bcast_scalar in0[" << i << "]: " << BF16(in0_ptr[i]) << ENDL();
    }
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_mul_block_bcast_scalar scalar[" << i << "]: " << BF16(scalar_ptr[i]) << ENDL();
    }
#endif
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, 0, 0, 0);
        // dprint_tensix_dest_reg(0);
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(0, out_cb);
#if defined(TRISC_PACK)
        if (i == 0) {
            constexpr uint32_t debug_elems = 32;
            auto out_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(out_cb).fifo_wr_ptr << cb_addr_shift);
            for (uint32_t j = 0; j < debug_elems; ++j) {
                DPRINT << "softmax_mul_block_bcast_scalar out[" << j << "]: " << BF16(out_ptr[j]) << ENDL();
            }
        }
#endif
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
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
#if defined(TRISC_UNPACK)
    constexpr uint32_t debug_elems = 32;
    auto in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in0_cb).fifo_rd_ptr << cb_addr_shift);
    auto in1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in1_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_mul_block_bcast_cols in0[" << i << "]: " << BF16(in0_ptr[i]) << ENDL();
    }
    for (uint32_t i = 0; i < debug_elems; ++i) {
        DPRINT << "softmax_mul_block_bcast_cols in1[" << i << "]: " << BF16(in1_ptr[i]) << ENDL();
    }
#endif
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
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
    init_sfpu(cb_id, cb_id);
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
    uint32_t in_scores_cb, uint32_t in_indices_cb,
    uint32_t out_scores_cb, uint32_t out_indices_cb,
    bool presorted = false>
void run_top32_llk(uint32_t row_elements, uint32_t num_input_tiles) {
    constexpr uint32_t value_offset_tiles = 0;
    constexpr uint32_t index_offset_tiles = 2;
    constexpr uint32_t decreasing = 0;
    constexpr uint32_t increasing = 1;

    cb_wait_front(in_scores_cb, num_input_tiles);
    cb_wait_front(in_indices_cb, num_input_tiles);
    auto in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in_scores_cb).fifo_rd_ptr << cb_addr_shift);
    auto in1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(in_indices_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < 160; ++i) {
        UNPACK(DPRINT << "local scores[" << i << "]: " << BF16(in0_ptr[i]) << ENDL());
        UNPACK(DPRINT << "local indices[" << i << "]: " << in1_ptr[i] << ENDL());
    }
    UNPACK(DPRINT << "--------------------------------" << ENDL());
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
            MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(value_offset_tiles + 1, decreasing, false)));
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
    auto out0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(out_scores_cb).fifo_wr_ptr << cb_addr_shift);
    auto out1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(out_indices_cb).fifo_wr_ptr << cb_addr_shift);
    
    for (uint32_t i = 0; i < 32; ++i) {
        PACK(DPRINT << "Standard out_scores_cb[" << i << "]: " << BF16(out0_ptr[i]) << ENDL());
        PACK(DPRINT << "Standardout_indices_cb[" << i << "]: " << out1_ptr[i] << ENDL());
    }
    PACK(DPRINT << "--------------------------------" << ENDL());

    cb_push_back(out_scores_cb, 1);
    cb_push_back(out_indices_cb, 1);
}

template <
    uint32_t in_scores_cb, uint32_t in_indices_cb,
    uint32_t out_scores_cb, uint32_t out_indices_cb>
void run_top32_llk_presorted_1024_opt(uint32_t row_elements, uint32_t num_input_tiles) {
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
    // DPRINT values in scores and indices
    auto in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(in_scores_cb).fifo_rd_ptr << cb_addr_shift);
    auto in1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(in_indices_cb).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < 3232; ++i) {
        UNPACK(DPRINT << "scores[" << i << "]: " << BF16(in0_ptr[i]) << ENDL());
        UNPACK(DPRINT << "indices[" << i << "]: " << in1_ptr[i] << ENDL());
    }
    UNPACK(DPRINT << "--------------------------------" << ENDL());
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

        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
            value_offset_tiles + 1, decreasing, false)));
        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles + 1, false)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
            value_offset_tiles + 1, increasing, true)));

        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, true)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
            value_offset_tiles, decreasing, true)));
    }

    // Step 10: pack final top-32 scores/indices.
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_reconfig_data_format(out_scores_cb);
    ckernel::pack_tile(value_offset_tiles, out_scores_cb);
    ckernel::pack_reconfig_data_format(out_indices_cb);
    ckernel::pack_tile(index_offset_tiles, out_indices_cb);
    auto out0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_local_cb_interface(out_scores_cb).fifo_wr_ptr << cb_addr_shift);
    auto out1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(out_indices_cb).fifo_wr_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < 32; ++i) {
        PACK(DPRINT << "Global out_scores_cb[" << i << "]: " << BF16(out0_ptr[i]) << ENDL());
        PACK(DPRINT << "Global out_indices_cb[" << i << "]: " << out1_ptr[i] << ENDL());
    }
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
        uint32_t InvTempBF16 = 0,
        uint32_t TopKInScoresCBId = 0xFFFFFFFF,
        uint32_t TopKInIndicesCBId = 0xFFFFFFFF,
        uint32_t TopKOutScoresCBId = 0xFFFFFFFF,
        uint32_t TopKOutIndicesCBId = 0xFFFFFFFF,
        uint32_t Phase2ScoresByteOffset = 0,
        uint32_t Phase2IndicesByteOffset = 0,
        uint32_t MeshStageScoresCBId = 0xFFFFFFFF,
        uint32_t MeshStageIndicesCBId = 0xFFFFFFFF,
        uint32_t ScoresScratchStage2Offset = 0,
        uint32_t IndicesScratchStage2Offset = 0>
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
        static constexpr uint32_t phase2_num_input_tiles = (NumSenders * TopK + 1023) / 1024;

        // Per-core gather slot sizes (padded to 16-byte alignment for NOC transfers).
        static constexpr uint32_t topk_scores_size = TopK * sizeof(uint16_t);
        static constexpr uint32_t topk_indices_size = TopK * sizeof(uint32_t);
        static constexpr uint32_t topk_scores_stride = (topk_scores_size + 15u) & ~15u;
        static constexpr uint32_t topk_indices_stride = (topk_indices_size + 15u) & ~15u;
        static constexpr uint32_t mesh_stage_scores_cb = MeshStageScoresCBId;
        static constexpr uint32_t mesh_stage_indices_cb = MeshStageIndicesCBId;
        static constexpr uint32_t scores_scratch_stage2_offset = ScoresScratchStage2Offset;
        static constexpr uint32_t indices_scratch_stage2_offset = IndicesScratchStage2Offset;
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
        uint32_t TopKScoresStride = 0,
        uint32_t MeshMode = 0,
        uint32_t Stage2Receiver = 0>
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
        static constexpr uint32_t topk_scores_stride = TopKScoresStride;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
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
        static constexpr uint32_t phase2_row_elements = NumSenders * TopK;
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
        uint32_t scores_scratch_addr;
        uint32_t indices_scratch_addr;
        uint32_t global_sem_addr;
        uint32_t global_stage2_sem_addr;
    };

    struct WriterArgs {
        uint32_t final_noc_x;
        uint32_t final_noc_y;
        uint32_t scratch_addr;
        uint32_t output_addr;
        uint32_t rand_output_addr;
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
        void operator()(const RTArgs& args) { impl(args); }

        void set_seed(uint32_t seed = 0xFFFFFFFF) {
            #if defined(COMPILE_FOR_TRISC)
            if (seed != 0xFFFFFFFF) {
                rand_tile_init(seed);
            }
            #endif
        }

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

            noc_async_write_one_packet<true, true>(local_scores_addr, final_noc_base | dst_scores_l1_addr, CTArgs::topk_scores_size);
            noc_async_write_one_packet<true, true>(local_indices_addr, final_noc_base | dst_indices_l1_addr, CTArgs::topk_indices_size);
            noc_semaphore_inc(dst_sem_noc_addr, 1);
            noc_async_posted_writes_flushed();
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
                        scores_base + c * CTArgs::topk_scores_stride);
                    auto idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        indices_base + c * CTArgs::topk_indices_stride);
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
            uint32_t scores_slot_addr,
            uint32_t indices_slot_addr,
            uint32_t scores_base,
            uint32_t indices_base) {
            constexpr uint32_t K = CTArgs::topk_k;
            // auto dst_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_slot_addr);
            // auto dst_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_slot_addr);
            // for (uint32_t i = 0; i < K; ++i) {
            //     dst_scores[i] = scores[i];
            // }
            // for (uint32_t i = 0; i < K; ++i) {
            //     dst_indices[i] = indices[i];
            // }
            noc_async_read(get_noc_addr(scores_base), scores_slot_addr, K * sizeof(uint16_t));
            noc_async_read(get_noc_addr(indices_base), indices_slot_addr, K * sizeof(uint32_t));
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
                        stage_scores_base + s * CTArgs::topk_scores_stride);
                    auto s_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        stage_indices_base + s * CTArgs::topk_indices_stride);
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
                    {static_cast<uint16_t>(CTArgs::topk_scores_stride)},
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
                const uint32_t dst_scores_l1 =
                    scores_cb_base + CTArgs::phase2_scores_byte_offset +
                    CTArgs::sender_idx * CTArgs::topk_scores_stride;
                const uint32_t dst_indices_l1 =
                    indices_cb_base + CTArgs::phase2_indices_byte_offset +
                    CTArgs::sender_idx * CTArgs::topk_indices_stride;

                if constexpr (CTArgs::topk_k <= 32) {
                    constexpr uint32_t num_input_tiles = CTArgs::phase1_num_input_tiles;
                    auto scores_src = scores_addr;
                    auto indices_src = args.indices_addr;

                    cb_reserve_back(CTArgs::topk_in_scores_cb, num_input_tiles);
                    cb_reserve_back(CTArgs::topk_in_indices_cb, num_input_tiles);
                    auto scores_dst = get_write_ptr(CTArgs::topk_in_scores_cb);
                    auto indices_dst = get_write_ptr(CTArgs::topk_in_indices_cb);
                    // for (uint32_t i = 0; i < CTArgs::num_values; ++i) {
                    //     scores_dst[i] = scores_src[i];
                    // }
                    // use NoC to copy sorces_src to scores_dst
                    noc_async_read(get_noc_addr(scores_src), scores_dst, CTArgs::num_values * sizeof(uint16_t));
                    // for (uint32_t i = 0; i < CTArgs::num_values; ++i) {
                    //     indices_dst[i] = indices_src[i];
                    // }
                    // use NoC to copy indices_src to indices_dst
                    noc_async_read(get_noc_addr(indices_src), indices_dst, CTArgs::num_values * sizeof(uint32_t));

                    noc_async_read_barrier();
                    cb_push_back(CTArgs::topk_in_scores_cb, num_input_tiles);
                    cb_push_back(CTArgs::topk_in_indices_cb, num_input_tiles);

                    cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                    cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                    phase1_send_topk_to_final(
                        get_read_ptr(CTArgs::topk_out_scores_cb),
                        get_read_ptr(CTArgs::topk_out_indices_cb),
                        dst_scores_l1, dst_indices_l1, args.final_noc_x, args.final_noc_y);

                    cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                    cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                } else {
                    auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
                    auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.indices_addr);

                    if constexpr (IsFinalCore) {
                        auto out_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_scores_l1);
                        auto out_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_indices_l1);
                        phase1_reduce_local_topk(scores_ptr, indices_ptr, out_scores, out_indices);
                    } else {
                        const uint32_t local_scores_addr = scores_cb_base;
                        const uint32_t local_indices_addr = indices_cb_base;
                        auto out_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_scores_addr);
                        auto out_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_indices_addr);
                        phase1_reduce_local_topk(scores_ptr, indices_ptr, out_scores, out_indices);
                        phase1_send_topk_to_final(
                            local_scores_addr, local_indices_addr,
                            dst_scores_l1, dst_indices_l1, args.final_noc_x, args.final_noc_y);
                    }
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
                wait_and_reset_semaphore(recv_sem_ptr, CTArgs::expected_remote_incs + 1);

                const uint32_t global_scores = get_write_ptr(CTArgs::winner_cb_id);
                const uint32_t global_indices = global_scores + CTArgs::topk_scores_stride;

                // DPRINT scores and indices

                if constexpr (CTArgs::topk_k <= 32) {
                    constexpr uint32_t p2_tiles = CTArgs::phase2_num_input_tiles;
                    
                    auto all_cores_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        scores_cb_base + CTArgs::phase2_scores_byte_offset);
                    auto all_cores_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        indices_cb_base + CTArgs::phase2_indices_byte_offset);
                    constexpr uint32_t total_gathered = CTArgs::num_senders * CTArgs::topk_k;
                    // for (uint32_t i = 0; i < total_gathered; ++i) {
                    //     DPRINT << "all_cores_scores[" << i << "] = " << BF16(all_cores_scores[i]) << ENDL();
                    //     DPRINT << "all_cores_indices[" << i << "] = " << all_cores_indices[i] << ENDL();
                    // }
                    cb_reserve_back(CTArgs::topk_in_scores_cb, p2_tiles);
                    cb_push_back(CTArgs::topk_in_scores_cb, p2_tiles);

                    cb_reserve_back(CTArgs::topk_in_indices_cb, p2_tiles);
                    cb_push_back(CTArgs::topk_in_indices_cb, p2_tiles);

                    cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                    cb_wait_front(CTArgs::topk_out_indices_cb, 1);


                    auto llk_scores_addr = get_read_ptr(CTArgs::topk_out_scores_cb);
                    auto llk_indices_addr = get_read_ptr(CTArgs::topk_out_indices_cb);
                    noc_async_read(get_noc_addr(llk_scores_addr), global_scores, CTArgs::topk_k * sizeof(uint16_t));
                    noc_async_read(get_noc_addr(llk_indices_addr), global_indices, CTArgs::topk_k * sizeof(uint32_t));
                    noc_async_read_barrier();

                    cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                    cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                } else {
                    auto global_scores_addr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(global_scores);
                    auto global_indices_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        global_indices);
                    const uint32_t p2_scores_base =
                        scores_cb_base + CTArgs::phase2_scores_byte_offset;
                    const uint32_t p2_indices_base =
                        indices_cb_base + CTArgs::phase2_indices_byte_offset;
                    phase2_merge_global_topk(p2_scores_base, p2_indices_base, global_scores_addr, global_indices_addr);
                }

                // Mesh inter-device reduction stages via LLK (k==32) or scalar merge.
                if constexpr (CTArgs::mesh_mode) {
                    if constexpr (CTArgs::stage1_receiver) {
                        write_topk_slot(
                            args.scores_scratch_addr +
                                CTArgs::stage1_local_slot_idx * CTArgs::topk_scores_stride,
                            args.indices_scratch_addr +
                                CTArgs::stage1_local_slot_idx * CTArgs::topk_indices_stride,
                            global_scores,
                            global_indices);
                        auto global_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_sem_addr);
                        wait_and_reset_semaphore(global_sem_ptr, CTArgs::stage1_expected_remote_incs);

                        if constexpr (CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                            constexpr uint32_t s1_tiles = CTArgs::stage1_num_slots * CTArgs::topk_k;
                            constexpr uint32_t s1_input_tiles = (s1_tiles + 1023) / 1024;
                            cb_reserve_back(CTArgs::mesh_stage_scores_cb, s1_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_scores_cb, s1_input_tiles);
                            cb_reserve_back(CTArgs::mesh_stage_indices_cb, s1_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_indices_cb, s1_input_tiles);

                            cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                            cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                            auto llk_scores =
                                get_read_ptr(CTArgs::topk_out_scores_cb);
                            auto llk_indices =
                                get_read_ptr(CTArgs::topk_out_indices_cb);
                            noc_async_read(get_noc_addr(llk_scores), global_scores, CTArgs::topk_k * sizeof(uint16_t));
                            noc_async_read(get_noc_addr(llk_indices), global_indices, CTArgs::topk_k * sizeof(uint32_t));
                            noc_async_read_barrier();

                            cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                            cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                        } else {
                            phase3_merge_mesh_stage_topk_slots(
                                args.scores_scratch_addr,
                                args.indices_scratch_addr,
                                CTArgs::stage1_num_slots,
                                global_scores,
                                global_indices);
                        }
                    }

                    // Signal BRISC to send via fabric (sends from winner CB directly).
                    if constexpr (IsMeshSenderCore && (CTArgs::stage1_sender || CTArgs::stage2_sender)) {
                        cb_reserve_back(CTArgs::winner_cb_id, 1);
                        cb_push_back(CTArgs::winner_cb_id, 1);
                    }

                    if constexpr (CTArgs::stage2_receiver) {
                        write_topk_slot(
                            args.scores_scratch_addr + CTArgs::scores_scratch_stage2_offset +
                                CTArgs::stage2_local_slot_idx * CTArgs::topk_scores_stride,
                            args.indices_scratch_addr + CTArgs::indices_scratch_stage2_offset +
                                CTArgs::stage2_local_slot_idx * CTArgs::topk_indices_stride,
                            global_scores,
                            global_indices);
                        auto global_stage2_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_stage2_sem_addr);
                        wait_and_reset_semaphore(global_stage2_sem_ptr, CTArgs::stage2_expected_remote_incs);

                        if constexpr (CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                            constexpr uint32_t s2_tiles = CTArgs::stage2_num_slots * CTArgs::topk_k;
                            constexpr uint32_t s2_input_tiles = (s2_tiles + 1023) / 1024;
                            cb_reserve_back(CTArgs::mesh_stage_scores_cb, s2_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_scores_cb, s2_input_tiles);
                            cb_reserve_back(CTArgs::mesh_stage_indices_cb, s2_input_tiles);
                            cb_push_back(CTArgs::mesh_stage_indices_cb, s2_input_tiles);

                            cb_wait_front(CTArgs::topk_out_scores_cb, 1);
                            cb_wait_front(CTArgs::topk_out_indices_cb, 1);

                            auto llk_scores =
                                get_read_ptr(CTArgs::topk_out_scores_cb);
                            auto llk_indices =
                                get_read_ptr(CTArgs::topk_out_indices_cb);
                            noc_async_read(get_noc_addr(llk_scores), global_scores, CTArgs::topk_k * sizeof(uint16_t));
                            noc_async_read(get_noc_addr(llk_indices), global_indices, CTArgs::topk_k * sizeof(uint32_t));
                            noc_async_read_barrier();

                            cb_pop_front(CTArgs::topk_out_scores_cb, 1);
                            cb_pop_front(CTArgs::topk_out_indices_cb, 1);
                        } else {
                            phase3_merge_mesh_stage_topk_slots(
                                args.scores_scratch_addr + CTArgs::scores_scratch_stage2_offset,
                                args.indices_scratch_addr + CTArgs::indices_scratch_stage2_offset,
                                CTArgs::stage2_num_slots,
                                global_scores,
                                global_indices);
                        }
                    }
                }

                // Pack top-K scores into a bf16 tile for TRISC softmax.
                // In mesh mode, only the stage-2 receiver (absolute final device) does this.
                if constexpr (!CTArgs::mesh_mode || CTArgs::stage2_receiver) {
                    constexpr uint32_t K = CTArgs::topk_k;
                    constexpr uint32_t FACE_ELEMS = 256;
                    constexpr uint16_t BF16_ONE = 0x3F80;
                    constexpr uint32_t ELEMS_PER_FACE_ROW = 16;

                    DPRINT << "Top-" << K << " scores (before softmax):" << ENDL();
                    auto global_scores_addr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(global_scores);
                    for (uint32_t i = 0; i < K; ++i) {
                        DPRINT << "  [" << i << "] " << BF16(global_scores_addr[i]) << ENDL();
                    }

                    constexpr uint32_t BF16_ONE_PACKED = 0x3F803F80;
                    generate_reduce_scaler(CTArgs::scaler_cb, BF16_ONE_PACKED);

                    {
                        DPRINT << "Inv temp BF16: " << BF16((uint16_t)(CTArgs::inv_temp_bf16 >> 16)) << ENDL();
                        generate_bcast_unary_scalar(CTArgs::temp_cb, CTArgs::inv_temp_bf16);
                        auto temp_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CTArgs::temp_cb));
                        DPRINT << "Value in temp CB: " << BF16(temp_ptr[0]) << ENDL();
                    }

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
                    // auto tile_u16 =
                    //     reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CTArgs::softmax_in_cb));
                    // for (uint32_t i = 0; i < 16 && i < K; ++i) {
                    //     tile_u16[i] = global_scores_addr[i];
                    // }
                    // for (uint32_t i = 0; i < 16 && (i + 16) < K; ++i) {
                    //     tile_u16[FACE_ELEMS + i] = global_scores_addr[16 + i];
                    // }
                    noc_async_read(get_noc_addr(global_scores), get_write_ptr(CTArgs::softmax_in_cb), std::min(K, ELEMS_PER_FACE_ROW) * sizeof(uint16_t));
                    if (K > ELEMS_PER_FACE_ROW) {
                        noc_async_read(get_noc_addr(global_scores + ELEMS_PER_FACE_ROW*sizeof(uint16_t)), get_write_ptr(CTArgs::softmax_in_cb) + FACE_ELEMS*sizeof(uint16_t), std::min(K - ELEMS_PER_FACE_ROW, ELEMS_PER_FACE_ROW) * sizeof(uint16_t));
                    }
                    noc_async_read_barrier();
                    cb_push_back(CTArgs::softmax_in_cb, 1);
                }
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
                if constexpr (IsMeshSenderCore) {
                    // Mesh sender: wait for NCRISC to finish, then send via fabric
                    cb_wait_front(CTArgs::winner_cb_id, 1);
                    const BriscMeshSendMetadata metadata = load_mesh_send_metadata(arg_idx);
                    const uint32_t winner_addr = get_write_ptr(CTArgs::winner_cb_id);
                    send_mesh_topk_via_fabric_brisc(args.final_noc_x, args.final_noc_y, winner_addr, metadata, arg_idx);
                    cb_pop_front(CTArgs::winner_cb_id, 1);
                } else {
                    // Top-P filtering + random categorical selection
                    if constexpr (CTArgs::topk_k > 1 && (!CTArgs::mesh_mode || CTArgs::stage2_receiver)) {
                        constexpr uint32_t K = CTArgs::topk_k;
                        constexpr uint32_t FACE_ELEMS = 256;

                        cb_wait_front(CTArgs::softmax_out_cb, 1);
                        cb_wait_front(CTArgs::rand_cb, 1);

                        auto prob_u16 =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::softmax_out_cb));
                        auto rand_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::rand_cb));
                        auto global_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_write_ptr(CTArgs::winner_cb_id) + CTArgs::topk_scores_stride);

                        uint16_t rand = rand_u16[0];

                        DPRINT << "BRISC sampling: rand=" << BF16(rand) << " p=" << CTArgs::p << ENDL() << ENDL();

                        DPRINT << "Probabilities: " << ENDL();
                        for (uint32_t i = 0; i < K; ++i) {
                            DPRINT << "Index " << global_indices[i] << " probability "
                                   << BF16(prob_u16[i < 16 ? i : FACE_ELEMS + (i - 16)]) << ENDL();
                        }
                        DPRINT << ENDL();

                        float cum_prob = 0.0f;
                        uint32_t kept_tokens = K;
                        for (uint32_t i = 0; i < K; ++i) {
                            uint16_t prob = (i < 16) ? prob_u16[i] : prob_u16[FACE_ELEMS + (i - 16)];
                            DPRINT << "Accumulating probability " << BF16(prob) << " sum so far " << cum_prob << ENDL();
                            cum_prob += bf16_to_float(prob);
                            DPRINT << "Cumulative probability " << cum_prob << ENDL();
                            if (cum_prob > CTArgs::p) {
                                kept_tokens = i + 1;
                                break;
                            }
                        }

                        DPRINT << "Top-P kept=" << kept_tokens << " cum_prob=" << cum_prob << ENDL();

                        float cum_sum = 0.0f;
                        uint32_t selected_index = global_indices[0];
                        for (uint32_t i = 0; i < kept_tokens; ++i) {
                            uint16_t prob = (i < 16) ? prob_u16[i] : prob_u16[FACE_ELEMS + (i - 16)];
                            cum_sum += bf16_to_float(prob) / cum_prob;
                            if (cum_sum > bf16_to_float(rand)) {
                                selected_index = global_indices[i];
                                break;
                            }
                        }

                        DPRINT << "Selected index=" << selected_index << ENDL();

                        auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.output_addr);
                        output_ptr[0] = selected_index;

                        if (args.rand_output_addr != 0) {
                            auto rand_out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(args.rand_output_addr);
                            rand_out[0] = rand;
                        }

                        cb_pop_front(CTArgs::softmax_out_cb, 1);
                        cb_pop_front(CTArgs::rand_cb, 1);
                    }
                }

                if constexpr (CTArgs::socket_mode == 1) {
                    send_d2h_token_from_cb_brisc(args);
                }
                if constexpr (CTArgs::socket_mode == 2) {
                    send_d2d_token_from_cb_brisc(args);
                }

                send_persistent_next_iter_inc_via_fabric_brisc(args, arg_idx);
            }
#elif defined(COMPILE_FOR_TRISC)
            // Phase 1: LLK top-32 sort (all active cores, k==32 only)
            if constexpr (IsActiveCore && CTArgs::topk_k <= 32) {
                run_top32_llk<
                    CTArgs::topk_in_scores_cb, CTArgs::topk_in_indices_cb,
                    CTArgs::topk_out_scores_cb, CTArgs::topk_out_indices_cb>(
                    CTArgs::num_values, CTArgs::phase1_num_input_tiles);
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
                run_top32_llk_presorted_1024_opt<
                    CTArgs::topk_in_scores_cb, CTArgs::topk_in_indices_cb,
                    CTArgs::topk_out_scores_cb, CTArgs::topk_out_indices_cb>(
                    CTArgs::phase2_row_elements, CTArgs::phase2_num_input_tiles);
            }

            // Mesh stage 1 merge via LLK (stage1_receiver, final core, k==32)
            if constexpr (IsFinalCore && CTArgs::mesh_mode && CTArgs::stage1_receiver &&
                          CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                run_top32_llk<
                    CTArgs::mesh_stage_scores_cb, CTArgs::mesh_stage_indices_cb,
                    CTArgs::topk_out_scores_cb, CTArgs::topk_out_indices_cb,
                    true>(CTArgs::stage1_row_elements, CTArgs::stage1_num_input_tiles);
            }

            // Mesh stage 2 merge via LLK (stage2_receiver, final core, k==32)
            if constexpr (IsFinalCore && CTArgs::mesh_mode && CTArgs::stage2_receiver &&
                          CTArgs::topk_k <= 32 && CTArgs::mesh_stage_scores_cb != 0xFFFFFFFF) {
                run_top32_llk<
                    CTArgs::mesh_stage_scores_cb, CTArgs::mesh_stage_indices_cb,
                    CTArgs::topk_out_scores_cb, CTArgs::topk_out_indices_cb,
                    true>(CTArgs::stage2_row_elements, CTArgs::stage2_num_input_tiles);
            }

            // Softmax + random (final core only)
            if constexpr (IsFinalCore && (!CTArgs::mesh_mode || CTArgs::stage2_receiver)) {
                // deepseek_compute_kernel_hw_startup<true>(CTArgs::softmax_in_cb, CTArgs::temp_cb, CTArgs::softmax_out_cb);
                softmax_mul_block_bcast_scalar<CTArgs::softmax_in_cb, CTArgs::temp_cb, CTArgs::softmax_exp_cb, 1>();
                softmax_reduce_c<
                    PoolType::MAX, ReduceDim::REDUCE_ROW,
                    CTArgs::softmax_exp_cb, CTArgs::scaler_cb, CTArgs::max_cb, 1, 1>();
                softmax_sub_exp_bcast_cols<CTArgs::softmax_exp_cb, CTArgs::max_cb, CTArgs::softmax_sub_cb, 1, 1>();
                softmax_reduce_c<
                    PoolType::SUM, ReduceDim::REDUCE_ROW,
                    CTArgs::softmax_sub_cb, CTArgs::scaler_cb, CTArgs::sum_cb, 1, 1>();
                // `scaler_cb` is reused across both reductions above; consume it once both are done.
                cb_pop_front(CTArgs::scaler_cb, 1);
                softmax_recip_block_inplace(CTArgs::sum_cb, 1);
                softmax_mul_block_bcast_cols(CTArgs::softmax_sub_cb, CTArgs::sum_cb, CTArgs::softmax_out_cb, 1, 1);

                if constexpr (CTArgs::topk_k > 1) {
                    generate_rand_tile(CTArgs::rand_cb);
                }
            }
#endif
        }
    };
};  // struct TopKSampling

}  // namespace deepseek_b1_ops
