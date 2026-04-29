// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "internal/risc_attribs.h"

namespace deepseek_b1_ops {

// Full byte size of the DeepseekMetadata struct. MUST stay in sync with
// `METADATA_TENSOR_BYTES` in metadata.py — both Python and C++ size the LM-head
// sampling source/destination buffers from this constant.
inline constexpr uint32_t kMetadataTensorBytes = 256;

// Layout (total: 256 bytes):
//   bytes 0..51   : 13 scalar fields (existing metadata, 13 * 4B = 52B)
//   bytes 52..63  : padding to 64B (3 * 4B)
//   bytes 64..191 : p_indices[32]        (32 * uint32_t, 128B)
//   bytes 192..255: p_scores[32]         (32 * packed bf16 as uint16_t, 64B)
//
// The `p_indices` / `p_scores` arrays hold the final top-P rescaled results
// (after softmax + temperature + top-P filter).  Entries at positions [0, k)
// are valid; entries at [k, 32) are garbage.  Inside the valid range, entries
// at [kept_tokens, k) are zeroed out (filtered by the top-P cutoff).
struct DeepseekMetadata {
    // Output fields
    uint32_t tok0_id;
    uint32_t tok0_type;
    uint32_t tok0_pos;
    uint32_t tok1_id;
    uint32_t tok1_type;
    uint32_t tok1_pos;
    // Input fields
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position_id;
    uint32_t prefill_token_id;
    float temperature;
    uint32_t k;
    float probability_mass_threshold;
    // Padding to bring the header up to 64B.
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
    // Top-P outputs.
    uint32_t p_indices[32];
    uint16_t p_scores[32];
};

static_assert(
    sizeof(DeepseekMetadata) == kMetadataTensorBytes, "DeepseekMetadata size must equal kMetadataTensorBytes");
static_assert(offsetof(DeepseekMetadata, p_indices) == 64, "p_indices must start at offset 64");
static_assert(offsetof(DeepseekMetadata, p_scores) == 192, "p_scores must start at offset 192");

}  // namespace deepseek_b1_ops
