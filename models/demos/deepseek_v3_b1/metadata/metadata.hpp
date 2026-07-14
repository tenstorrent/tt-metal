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
inline constexpr uint32_t kMetadataTensorBytes = 512;

struct DeepseekMetadata {
    // Input / token-type fields
    uint32_t lane_id;  // 0 for Base Token, i > 0 for each MTP Level
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position_id;
    uint32_t output_token_ids[5];   // 5 output token slots (1 Base Token + 4 Speculative Tokens per MTP level)
    uint32_t prefill_token_ids[4];  // 4 prefill token slots (1 for each MTP Level)
    float temperature;
    uint32_t k;
    float p;

    // Q and P outputs.
    uint32_t p_indices[32];
    uint16_t p_scores[32];
    uint32_t q_indices[32];
    uint16_t q_scores[32];

    // Padding
    uint32_t padding[16];
};

static_assert(
    sizeof(DeepseekMetadata) == kMetadataTensorBytes, "DeepseekMetadata size must equal kMetadataTensorBytes");
static_assert(offsetof(DeepseekMetadata, p_indices) == 64, "p_indices must start at offset 64");
static_assert(offsetof(DeepseekMetadata, p_scores) == 192, "p_scores must start at offset 192");
static_assert(offsetof(DeepseekMetadata, q_indices) == 256, "q_indices must start at offset 256");
static_assert(offsetof(DeepseekMetadata, q_scores) == 384, "q_scores must start at offset 384");

}  // namespace deepseek_b1_ops
