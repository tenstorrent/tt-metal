// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <vector>

#include "hostdevcommon/qkv_causal_conv1d_silu_bw_layout.hpp"
#include "qkv_causal_conv1d_silu_bw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

inline constexpr uint32_t qkv_conv_bw_tap_count = qkv_conv_bw::tap_count;

struct QkvCausalConv1dSiluBackwardDataflowBuffer {
    std::string_view name;
    uint32_t tiles;
    tt::DataFormat format;
};

// Every L1 buffer of the program, sized in tiles. Used by both the program factory and the L1 budget check.
inline std::vector<QkvCausalConv1dSiluBackwardDataflowBuffer> qkv_conv_bw_dataflow_buffers(
    uint32_t channel_tiles_per_block, tt::DataFormat input_format, bool compute_tap_grads = true) {
    const auto bf16 = input_format;
    constexpr auto fp32 = tt::DataFormat::Float32;
    std::vector<QkvCausalConv1dSiluBackwardDataflowBuffer> buffers{
        // the tile row copied 4x, each one row lower, as tiles [tap][ct]; x2 so the reader builds the next step's
        // copies while compute consumes these
        {"x_shifted_tile", 8 * channel_tiles_per_block, bf16},
        {"weights", 8 * channel_tiles_per_block, bf16},  // taps [4][channel_tiles_per_block], x2 to prefetch
        {"grad_out", 2 * channel_tiles_per_block, bf16},
        {"preact", channel_tiles_per_block, bf16},
        // fp32: silu_bw needs preact * (1 - s), and a bf16 s near 1 leaves (1 - s) with few bits. One block, so
        // every step ends on the buffer boundary: silu_bw works in DST-sized batches and the last one is short
        // when channel_tiles_per_block is not a multiple of 4, which with a 4-tile buffer made the next batch
        // straddle the end.
        {"sigmoid", channel_tiles_per_block, fp32},
        // grad_preact tiles for the writer, which builds the 4 shifted copies out of them; x2 so compute runs a step
        // ahead
        {"grad_preact_out", 2 * channel_tiles_per_block, bf16},
        // the 4 shifted copies as tiles [tap][ct]; x2 because compute consumes them one step behind the writer (see
        // the compute kernel)
        {"grad_preact_shifted_tile", 8 * channel_tiles_per_block, bf16},
        {"output", 2 * channel_tiles_per_block, bf16},
        {"ones", 1, bf16},
    };
    if (compute_tap_grads) {
        // a second copy of the grad_preact tiles, consumed by the tap-gradient accumulation
        buffers.push_back({"grad_preact_tile", channel_tiles_per_block, bf16});
        // grad_w [4][channel_tiles], L1 accumulated. One item deep: a core's next item waits for the writer to drain
        // it, but two items deep gained 1% at 8 splits and nothing at 16 for 128 KB more L1 at chunk 256.
        buffers.push_back({"partial", 4 * channel_tiles_per_block, fp32});
    }
    return buffers;
}

// L1 per core: every dataflow buffer and the reader's and the writer's strip scratch.
inline uint64_t qkv_conv_bw_l1_bytes(
    uint32_t channel_tiles_per_block, tt::DataFormat input_format, bool compute_tap_grads = true) {
    uint64_t bytes = 0;
    for (const auto& buffer : qkv_conv_bw_dataflow_buffers(channel_tiles_per_block, input_format, compute_tap_grads)) {
        bytes += static_cast<uint64_t>(buffer.tiles) * tt::tile_size(buffer.format);
    }
    return bytes + qkv_conv_bw::ReaderScratch{channel_tiles_per_block}.bytes() +
           qkv_conv_bw::WriterScratch{channel_tiles_per_block}.bytes();
}

struct QkvCausalConv1dSiluBackwardProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluBackwardParams&, const QkvCausalConv1dSiluBackwardInputs&, std::vector<Tensor>&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
