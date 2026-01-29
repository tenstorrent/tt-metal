// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tools/profiler/kernel_profiler.hpp>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "../../deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

// Minimal init to switch from matmul mode to copy_tile mode.
// This is lighter weight than init_sfpu() because it skips the pack-related
// initialization (which is already done by mm_block_init/custom_mm_block_init if output CB is the same).
// Also skips llk_math_pack_sync_init which is already done by mm_block_init/custom_mm_block_init.
template <uint32_t icb>
FORCE_INLINE void init_copy_tile_after_matmul() {
    // Reconfigure unpacker hardware for unary operation (same CB for both A and B)
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb)));
#ifdef TRISC_UNPACK

    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    if (enable_unpack_to_dest) {
        // Initialize unpacker A for copy operation
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    } else {
        // Initialize unpacker A for copy operation
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    }
#endif
    // Switch math from matmul mode to datacopy mode
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    // Reconfigure math HW for unary operation (same CB for both srcA and srcB)
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    uint32_t NumTilesK,
    uint32_t OutputTileId = 0,
    bool PopA = false,
    bool UseCustomMM = true>
FORCE_INLINE void matmul_with_relu_block() {
    DeviceZoneScopedN("matmul_with_relu_block");

    cb_wait_front(CbA, NumTilesK);
    cb_wait_front(CbB, NumTilesK);
    constexpr uint32_t num_output_tiles = 1;
    cb_reserve_back(CbOut, num_output_tiles);

    tile_regs_acquire();

    {
        DeviceZoneScopedN("matmul_tiles");
        if constexpr (UseCustomMM) {
            custom_mm_block(CbA, CbB, 0, 0, 0, false, NumTilesK);
        } else {
            for (uint32_t k = 0; k < NumTilesK; k++) {
                matmul_tiles(CbA, CbB, k, k, 0);
            }
        }
    }
    {
        DeviceZoneScopedN("relu_tile");
        relu_tile_init();
        relu_tile(0);
    }
    tile_regs_commit();

    if constexpr (PopA) {
        cb_pop_front(CbA, NumTilesK);
    }
    cb_pop_front(CbB, NumTilesK);  // Pop weight CB to advance to next layer's weights

    tile_regs_wait();
    pack_tile(0, CbOut, OutputTileId);  // Pack at offset OutputTileId
    tile_regs_release();

    cb_push_back(CbOut, num_output_tiles);
}

constexpr uint32_t MATMUL_ACC_REG_ID = 0;
constexpr uint32_t BIAS_REG_ID = 1;

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbBias,
    uint32_t CbOut,
    uint32_t NumTilesK,
    uint32_t NumTilesBias,
    uint32_t OutputTileId = 0,
    bool PopA = false,
    bool PopBias = false,
    bool UseCustomMM = true>
FORCE_INLINE void matmul_with_bias_block(uint32_t bias_tile_index) {
    DeviceZoneScopedN("matmul_with_bias_block");

    constexpr uint32_t num_output_tiles = 1;
    cb_reserve_back(CbOut, num_output_tiles);
    cb_wait_front(CbA, NumTilesK);
    cb_wait_front(CbB, NumTilesK);
    cb_wait_front(CbBias, NumTilesBias);

    tile_regs_acquire();

    {
        DeviceZoneScopedN("matmul_tiles");
        if constexpr (UseCustomMM) {
            custom_mm_block(CbA, CbB, 0, 0, MATMUL_ACC_REG_ID, false, NumTilesK);
        } else {
            for (uint32_t k = 0; k < NumTilesK; k++) {
                matmul_tiles(CbA, CbB, k, k, MATMUL_ACC_REG_ID);
            }
        }
    }

    {
        DeviceZoneScopedN("copy_tile_init");

        // Alternative to init_sfpu() + copy_tile_to_dst_init_short_with_dt()
        init_copy_tile_after_matmul<CbBias>();
        copy_tile(CbBias, bias_tile_index, BIAS_REG_ID);
    }

    {
        DeviceZoneScopedN("add_binary_tile_init");
        add_binary_tile_init();
        add_binary_tile(
            MATMUL_ACC_REG_ID, BIAS_REG_ID, MATMUL_ACC_REG_ID);  // Accumulate the bias into the matmul result
    }

    tile_regs_commit();

    if constexpr (PopA) {
        cb_pop_front(CbA, NumTilesK);
    }
    if constexpr (PopBias) {
        cb_pop_front(CbBias, NumTilesBias);
    }
    cb_pop_front(CbB, NumTilesK);  // Pop weight CB to advance to next layer's weights

    tile_regs_wait();
    pack_tile(MATMUL_ACC_REG_ID, CbOut, OutputTileId);  // Pack at offset OutputTileId
    tile_regs_release();

    cb_push_back(CbOut, num_output_tiles);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t mm1_full_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t intermediate_pregather_cb = get_compile_time_arg_val(4);
    constexpr uint32_t mm2_full_cb = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(6);
    constexpr bool fp32_dest_acc_en = get_compile_time_arg_val(7);
    constexpr uint32_t num_layers = get_compile_time_arg_val(8);
    constexpr bool use_custom_mm = get_compile_time_arg_val(9);

    const uint32_t bias_tile_index = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_output_tiles = 1;
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 1;
    constexpr uint32_t in0_block_w = 1;

    if constexpr (use_custom_mm) {
        custom_mm_block_init(mm1_full_cb, weight0_cb, intermediate_pregather_cb, false, num_tiles_k);
    } else {
        mm_block_init(
            mm1_full_cb, weight0_cb, intermediate_pregather_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
    }

    // All layers use the same pattern: MM1_FULL_CB -> matmul+relu, then MM2_FULL_CB (bias MM1_FULL_CB) -> matmul+bias
    // The ping-pong mcast restores MM1_FULL_CB after each layer
    for (uint32_t layer = 0; layer < num_layers; layer++) {
        if constexpr (use_custom_mm) {
            custom_mm_block_init(mm1_full_cb, weight0_cb, intermediate_pregather_cb, false, num_tiles_k);
        } else {
            mm_block_init(
                mm1_full_cb, weight0_cb, intermediate_pregather_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
        }
        // MM1_FULL_CB -> matmul+relu -> INTERMEDIATE_PREGATHER_CB
        // Don't pop MM1 yet - needed for bias
        matmul_with_relu_block<
            mm1_full_cb,
            weight0_cb,
            intermediate_pregather_cb,
            num_tiles_k,
            0,
            false,
            use_custom_mm>();

        // MM2_FULL_CB (with MM1_FULL_CB bias) -> matmul+bias -> INTERMEDIATE_PREGATHER_CB
        // Pop both MM2 (input) and MM1 (bias/residual) after this
        matmul_with_bias_block<
            mm2_full_cb,
            weight1_cb,
            mm1_full_cb,
            intermediate_pregather_cb,
            num_tiles_k,
            num_tiles_k,
            0,
            true,
            true,
            use_custom_mm>(bias_tile_index);
    }
}
}  // namespace NAMESPACE
