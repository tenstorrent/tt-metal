// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace clock_rollover {
constexpr uint32_t magic = 0x43524f4c;
constexpr uint32_t version = 1;
constexpr uint32_t lanes = 5;
constexpr uint32_t capacity = 256;
constexpr uint32_t header_words = 16;
constexpr uint32_t record_words = 12;
constexpr uint32_t lane_words = header_words + capacity * record_words;
constexpr uint32_t core_bytes = lanes * lane_words * sizeof(uint32_t);
// Header: magic,version,lane,capacity,total,next,initial_hi,final_hi,transition_seq,
// retries,complete,first_hi,first_lo,last_hi,last_lo,reserved.
// Record: seq,slot,before_hi,before_lo,start_packed_hi,start_lo,end_packed_hi,end_lo,
// after_hi,after_lo,retries,reserved. Raw words are never repaired.
}  // namespace clock_rollover
