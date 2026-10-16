// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "qkv_causal_conv1d_silu_bw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

inline constexpr uint32_t qkv_conv_bw_tap_count = 4;
inline constexpr uint32_t qkv_conv_bw_dst_tiles = 4;
inline constexpr uint32_t qkv_conv_bw_window_rows =
    tt::constants::TILE_HEIGHT + qkv_conv_bw_tap_count - 1;  // 3 rows above + 32

struct QkvCausalConv1dSiluBackwardDataflowBuffer {
    std::string_view name;
    uint32_t tiles;
    tt::DataFormat format;
};

// Every L1 buffer of the program, sized in tiles. Used by both the program factory and the L1 budget check.
inline std::array<QkvCausalConv1dSiluBackwardDataflowBuffer, 14> qkv_conv_bw_dataflow_buffers(
    uint32_t channel_tiles_per_block, tt::DataFormat input_format) {
    const auto bf16 = input_format;
    constexpr auto fp32 = tt::DataFormat::Float32;
    return {{
        {"act_rm", 8 * channel_tiles_per_block, bf16},    // 4 shifted row-major views of the input window, x2
        {"act_tile", 4 * channel_tiles_per_block, bf16},  // the 4 views tilized
        {"weights", 4 * channel_tiles_per_block, bf16},   // taps [4][channel_tiles_per_block]
        {"grad_out", 2 * channel_tiles_per_block, bf16},
        {"preact", channel_tiles_per_block, bf16},
        {"sigmoid", qkv_conv_bw_dst_tiles, bf16},
        {"dc_tile", channel_tiles_per_block, bf16},
        {"dc_rm", 2 * channel_tiles_per_block, bf16},
        {"acc", 8 * channel_tiles_per_block, fp32},  // grad_tap running sum [4][channel_tiles], previous and next
        {"grad_tap", 4 * channel_tiles_per_block, fp32},
        {"dcv_rm", 4 * channel_tiles_per_block, bf16},  // 4 shifted row-major views of the dc window
        {"dcv_tile", 4 * channel_tiles_per_block, bf16},
        {"output", 2 * channel_tiles_per_block, bf16},
        {"ones", 1, bf16},
    }};
}

inline uint64_t qkv_conv_bw_window_bytes(uint64_t channel_tiles_per_block, uint64_t bytes_per_element) {
    return qkv_conv_bw_window_rows * channel_tiles_per_block * tt::constants::TILE_WIDTH * bytes_per_element;
}

// L1 per core: every dataflow buffer plus the input and dc row windows.
inline uint64_t qkv_conv_bw_l1_bytes(
    uint32_t channel_tiles_per_block, tt::DataFormat input_format, uint64_t bytes_per_element) {
    uint64_t bytes = 0;
    for (const auto& buffer : qkv_conv_bw_dataflow_buffers(channel_tiles_per_block, input_format)) {
        bytes += static_cast<uint64_t>(buffer.tiles) * tt::tile_size(buffer.format);
    }
    return bytes + 2 * qkv_conv_bw_window_bytes(channel_tiles_per_block, bytes_per_element);
}

struct QkvCausalConv1dSiluBackwardProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluBackwardInputs&, std::vector<Tensor>&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
