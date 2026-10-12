// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/constants.hpp>

#include "qkv_causal_conv1d_silu_bw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

inline constexpr uint32_t qkv_conv_bw_tap_count = 4;
inline constexpr uint32_t qkv_conv_bw_window_rows = tt::constants::TILE_HEIGHT + qkv_conv_bw_tap_count - 1;

inline uint64_t qkv_conv_bw_window_bytes(uint64_t block_ct, uint64_t element_bytes) {
    return qkv_conv_bw_window_rows * block_ct * tt::constants::TILE_WIDTH * element_bytes;
}

// grad_preact: act_rm 2, act_tile 4, weights 4, partial 2, grad_out 2, preact 1, dc_tile 1, dc_rm 2, grad_tap 4
// (bf16 blocks), sigmoid 1 tile, acc 8 blocks of fp32 tiles.
inline uint64_t qkv_conv_grad_preact_l1_bytes(uint64_t block_ct, uint64_t tile_bytes, uint64_t element_bytes) {
    return (22 * block_ct + 1) * tile_bytes + 8 * block_ct * 2 * tile_bytes +
           qkv_conv_bw_window_bytes(block_ct, element_bytes);
}

// grad_input: act_rm 2, act_tile 1, weights 4, partial 2, output 2 (bf16 blocks).
inline uint64_t qkv_conv_grad_input_l1_bytes(uint64_t block_ct, uint64_t tile_bytes, uint64_t element_bytes) {
    return 11 * block_ct * tile_bytes + qkv_conv_bw_window_bytes(block_ct, element_bytes);
}

struct QkvCausalConv1dSiluGradPreactProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluGradPreactInputs&, std::vector<Tensor>&);
};

struct QkvCausalConv1dSiluGradInputProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluGradInputInputs&, Tensor&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
