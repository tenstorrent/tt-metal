// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mx_common.hpp"

#include <tt_stl/assert.hpp>

#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"

namespace tt::tt_metal::mx {

namespace {

uint32_t compute_exp_count(uint32_t elem_count, const FormatParams& params) {
    TT_ASSERT(params.block_size > 0, "MX block size must be > 0");
    TT_ASSERT(elem_count % params.block_size == 0, "MX element count must be divisible by block size");
    return elem_count / params.block_size;
}

uint32_t compute_exp_bytes(uint32_t exp_count, uint32_t l1_alignment) { return tt::round_up(exp_count, l1_alignment); }

uint32_t compute_elem_words(uint32_t elem_count, const FormatParams& params) {
    TT_ASSERT(params.elem_width_storage_bits > 0, "MX element storage width must be > 0");
    uint32_t bits = elem_count * params.elem_width_storage_bits;
    return (bits + 31) / 32;
}

}  // namespace

TileWordCounts compute_tile_word_counts(uint32_t elem_count, const FormatParams& params) {
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    TileWordCounts counts;
    counts.exp_count = compute_exp_count(elem_count, params);
    counts.exp_bytes = compute_exp_bytes(counts.exp_count, l1_alignment);
    counts.exp_words = counts.exp_bytes / 4;
    counts.elem_words = compute_elem_words(elem_count, params);
    return counts;
}

}  // namespace tt::tt_metal::mx
