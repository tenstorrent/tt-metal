// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/constants.hpp>

#include "qkv_causal_conv1d_silu_bw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

inline constexpr uint32_t qkv_conv_bw_tap_count = 4;
inline constexpr uint32_t qkv_conv_bw_dst_tiles = 4;
inline constexpr uint32_t qkv_conv_bw_window_rows = tt::constants::TILE_HEIGHT + qkv_conv_bw_tap_count - 1;

inline uint64_t qkv_conv_bw_window_bytes(uint64_t block_ct, uint64_t element_bytes) {
    return qkv_conv_bw_window_rows * block_ct * tt::constants::TILE_WIDTH * element_bytes;
}

// Tile budget per core: 36 bf16 tile slots per block_ct, 4 sigmoid tiles, 1 ones tile, 8 fp32 acc tiles per
// block_ct (= 16 bf16), plus the input and dc row windows.
inline uint64_t qkv_conv_bw_l1_bytes(uint64_t block_ct, uint64_t tile_bytes, uint64_t element_bytes) {
    return (52 * block_ct + qkv_conv_bw_dst_tiles + 1) * tile_bytes +
           2 * qkv_conv_bw_window_bytes(block_ct, element_bytes);
}

struct QkvCausalConv1dSiluBackwardProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluBackwardInputs&, std::vector<Tensor>&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
