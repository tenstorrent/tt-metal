# SPDX-License-Identifier: Apache-2.0
"""Worker-facing KV budget for this autoport's normalized hybrid cache."""

import math

TARGET_TOKENS_PER_USER = 32768
PRIOR_TURN_TOKENS_PER_USER = 24576


def get_pool_token_budget(*, max_model_len, max_num_seqs, b8_pool_ids=None):
    global_blocks = max_num_seqs * math.ceil((TARGET_TOKENS_PER_USER + PRIOR_TURN_TOKENS_PER_USER) / 64)
    budget = max(max_model_len, 32 * global_blocks)
    if b8_pool_ids is None:
        return budget
    if b8_pool_ids != "20480":
        raise ValueError("GEMMA4_B8_KV_POOL_IDS supports only 20480; unset it for the default pool")
    if max_num_seqs != 8:
        return budget
    if max_model_len != 262144:
        raise ValueError("GEMMA4_B8_KV_POOL_IDS=20480 requires the advertised 262144-token context")
    # The pinned worker adds one output block and eight 1024-token sliding
    # allowances per slot, then divides by its 32-token block size. This is
    # allocation sizing, not a change to the model or scheduler context limit.
    worker_output_tokens = 32 * max_num_seqs
    worker_sliding_tokens = 8 * 1024 * max_num_seqs
    return 20480 * 32 - worker_output_tokens - worker_sliding_tokens
