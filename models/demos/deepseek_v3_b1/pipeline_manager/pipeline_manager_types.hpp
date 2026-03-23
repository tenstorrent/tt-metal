// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

static constexpr int MAX_USERS = 64;
static constexpr int MAX_SEQ_LEN = 4096;
static constexpr int DEFAULT_CHUNK_SIZE = 24;
static constexpr int32_t EMPTY_TOKEN = -1;
// TODO: Make this a runtime parameter
static constexpr int32_t EOS_TOKEN = 2;

enum class UserState : uint8_t {
    INACTIVE = 0,
    PREFILL = 1,
    DECODE = 2,
    COMPLETE = 3,
};

enum class TokenMode : uint8_t {
    PREFILL = 0,
    DECODE = 1,
};

enum class RequestType : uint8_t {
    ALLOCATE = 1,
    SUBMIT = 2,
    CONTINUE = 3,
    CANCEL = 4,
};

struct ISRequest {
    RequestType type = RequestType::ALLOCATE;
    int32_t request_id = 0;
    int32_t user_id = -1;
    int32_t token_count = 0;
    std::array<int32_t, MAX_SEQ_LEN> tokens = {};
    int32_t max_new_tokens = 0;
    bool spec_decode = false;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;
};

struct PMResponse {
    int32_t request_id = 0;
    int32_t user_id = -1;
    int32_t error_code = 0;
};

struct OutputMessage {
    int32_t user_id = -1;
    int32_t token_id = EMPTY_TOKEN;
    bool is_eos = false;
    bool is_complete = false;
    int32_t tokens_generated = 0;
    uint32_t generation = 0;
};

struct InjectDescriptor {
    int32_t user_id = -1;
    int32_t token_id = EMPTY_TOKEN;
    int32_t position = 0;
    TokenMode mode = TokenMode::PREFILL;
    bool spec_flag = false;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;
};

struct ResultDescriptor {
    int32_t user_id = -1;
    int32_t actual_token = EMPTY_TOKEN;
    int32_t predicted_token = EMPTY_TOKEN;
    TokenMode mode = TokenMode::PREFILL;
    int32_t position = 0;
    bool spec_flag = false;
    bool sampled = false;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
