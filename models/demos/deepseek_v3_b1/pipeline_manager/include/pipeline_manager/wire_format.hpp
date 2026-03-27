// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

static constexpr uint32_t PAGE_SIZE_BYTES = 64;
static constexpr uint32_t PAGE_SIZE_WORDS = PAGE_SIZE_BYTES / sizeof(uint32_t);

// 64-byte wire format for InjectDescriptor (host → device).
//
//   Word  Field
//   [0]   user_id
//   [1]   token_id
//   [2]   position
//   [3]   mode          (0=PREFILL, 1=DECODE)
//   [4]   spec_flag     (0 or 1)
//   [5]   temperature   (reinterpret as float)
//   [6]   top_p         (reinterpret as float)
//   [7]   top_k
//   [8-15] reserved     (zero)
using PageBuffer = std::array<uint32_t, PAGE_SIZE_WORDS>;

inline PageBuffer serialize_inject(const InjectDescriptor& desc) {
    PageBuffer page = {};
    page[0] = static_cast<uint32_t>(desc.user_id);
    page[1] = static_cast<uint32_t>(desc.token_id);
    page[2] = static_cast<uint32_t>(desc.position);
    page[3] = static_cast<uint32_t>(desc.mode);
    page[4] = desc.spec_flag ? 1u : 0u;
    std::memcpy(&page[5], &desc.temperature, sizeof(float));
    std::memcpy(&page[6], &desc.top_p, sizeof(float));
    page[7] = static_cast<uint32_t>(desc.top_k);
    return page;
}

inline InjectDescriptor deserialize_inject(const PageBuffer& page) {
    InjectDescriptor desc;
    desc.user_id = static_cast<int32_t>(page[0]);
    desc.token_id = static_cast<int32_t>(page[1]);
    desc.position = static_cast<int32_t>(page[2]);
    desc.mode = static_cast<TokenMode>(page[3]);
    desc.spec_flag = (page[4] != 0);
    std::memcpy(&desc.temperature, &page[5], sizeof(float));
    std::memcpy(&desc.top_p, &page[6], sizeof(float));
    desc.top_k = static_cast<int32_t>(page[7]);
    return desc;
}

// 64-byte wire format for ResultDescriptor (device → host).
//
//   Word  Field
//   [0]   user_id
//   [1]   actual_token
//   [2]   predicted_token
//   [3]   mode          (0=PREFILL, 1=DECODE)
//   [4]   position
//   [5]   spec_flag     (0 or 1)
//   [6]   sampled       (0 or 1)
//   [7-15] reserved     (zero)
inline PageBuffer serialize_result(const ResultDescriptor& desc) {
    PageBuffer page = {};
    page[0] = static_cast<uint32_t>(desc.user_id);
    page[1] = static_cast<uint32_t>(desc.actual_token);
    page[2] = static_cast<uint32_t>(desc.predicted_token);
    page[3] = static_cast<uint32_t>(desc.mode);
    page[4] = static_cast<uint32_t>(desc.position);
    page[5] = desc.spec_flag ? 1u : 0u;
    page[6] = desc.sampled ? 1u : 0u;
    return page;
}

inline ResultDescriptor deserialize_result(const PageBuffer& page) {
    ResultDescriptor desc;
    desc.user_id = static_cast<int32_t>(page[0]);
    desc.actual_token = static_cast<int32_t>(page[1]);
    desc.predicted_token = static_cast<int32_t>(page[2]);
    desc.mode = static_cast<TokenMode>(page[3]);
    desc.position = static_cast<int32_t>(page[4]);
    desc.spec_flag = (page[5] != 0);
    desc.sampled = (page[6] != 0);
    return desc;
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
