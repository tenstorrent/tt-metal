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
inline constexpr uint32_t qkv_conv_bw_window_rows = tt::constants::TILE_HEIGHT + qkv_conv_bw_tap_count - 1;

// One dataflow buffer of the program; entries are tiles of `format`. This single table sizes the buffers in the
// program factory and the per-core L1 budget checked in validation, so the two cannot drift apart.
struct QkvCausalConv1dSiluBackwardDataflowBuffer {
    std::string_view name;
    uint32_t tiles;
    tt::DataFormat format;
};

inline std::array<QkvCausalConv1dSiluBackwardDataflowBuffer, 14> qkv_conv_bw_dfbs(
    uint32_t block_ct, tt::DataFormat bf16) {
    constexpr auto fp32 = tt::DataFormat::Float32;
    return {{
        {"act_rm", 8 * block_ct, bf16},    // 4 shifted row-major views of the input window, double buffered
        {"act_tile", 4 * block_ct, bf16},  // the 4 views tilized
        {"weights", 4 * block_ct, bf16},   // taps [4][block_ct]
        {"grad_out", 2 * block_ct, bf16},
        {"preact", block_ct, bf16},
        {"sigmoid", qkv_conv_bw_dst_tiles, bf16},
        {"dc_tile", block_ct, bf16},
        {"dc_rm", 2 * block_ct, bf16},
        {"acc", 8 * block_ct, fp32},  // grad_tap running sum [4][block_ct], previous and next
        {"grad_tap", 4 * block_ct, fp32},
        {"dcv_rm", 4 * block_ct, bf16},  // 4 shifted row-major views of the dc window
        {"dcv_tile", 4 * block_ct, bf16},
        {"output", 2 * block_ct, bf16},
        {"ones", 1, bf16},
    }};
}

inline uint64_t qkv_conv_bw_window_bytes(uint64_t block_ct, uint64_t element_bytes) {
    return qkv_conv_bw_window_rows * block_ct * tt::constants::TILE_WIDTH * element_bytes;
}

// L1 per core: every dataflow buffer plus the input and dc row windows.
inline uint64_t qkv_conv_bw_l1_bytes(uint32_t block_ct, tt::DataFormat bf16, uint64_t element_bytes) {
    uint64_t bytes = 0;
    for (const auto& dfb : qkv_conv_bw_dfbs(block_ct, bf16)) {
        bytes += static_cast<uint64_t>(dfb.tiles) * tt::tile_size(dfb.format);
    }
    return bytes + 2 * qkv_conv_bw_window_bytes(block_ct, element_bytes);
}

struct QkvCausalConv1dSiluBackwardProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluBackwardInputs&, std::vector<Tensor>&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
