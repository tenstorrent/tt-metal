// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants shared by the host and the kernels of qkv_causal_conv1d_silu_bw.
#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"

namespace qkv_conv_bw {

// The backward stages the input the way the forward does: the same taps, face and strip constants and the same
// ReaderScratch, all from the forward's header.
using namespace qkv_conv;

constexpr uint32_t halo_rows = history_rows;  // rows of history above a tile row

// The writer stages each grad_preact tile as a left and a right strip of face rows too, with the halo below: the
// tile's 32 rows, then the first 3 rows of the next tile row (zero after the last one), since grad_input row r reads
// grad_preact rows r .. r + 3. The tile's faces land at the strip start, so the strips need no phase.
struct WriterScratch {
    uint32_t channel_tiles_per_block;
    constexpr uint32_t strips() const {
        return 2 * channel_tiles_per_block * 2 * strip_stride;
    }
    // Slack for aligning the base.
    constexpr uint32_t bytes() const {
        return strips() + read_alignment;
    }
};
// Tiles per DST acquire in the compute kernel (fp32 DST); the sigmoid buffer is one channel block, so every step ends
// on its boundary whatever the batch.
constexpr uint32_t dest_tiles = max_dest_tiles_fp32;

}  // namespace qkv_conv_bw
