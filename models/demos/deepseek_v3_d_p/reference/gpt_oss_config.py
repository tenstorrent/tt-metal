# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS Model Configuration.

Mirrors DeepSeekV3Config but with the dimensions used by the open-source
GPT-OSS reference (EMB_SIZE/MOE_INTERMEDIATE_SIZE = 2880, 128 routed experts,
top-4 routing). Kept side-by-side with DeepSeekV3Config so test files can
import whichever they need without modifying production code.
"""


class GptOssConfig:
    """GPT-OSS model dimensions."""

    # Core dimensions
    EMB_SIZE = 2880  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload (kept name-compatible with DeepSeekV3Config)
    MOE_INTERMEDIATE_SIZE = 2880  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 2880  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 1
    # The TTNN gate currently routes via expert groups (DeepSeek-style). 128 experts split
    # cleanly into 8 groups of 16, with 4 limited groups giving 16 candidate experts >= top-4.
    NUM_EXPERT_GROUPS = 8
    NUM_LIMITED_GROUPS = 4

    # Model architecture (placeholders — not used by the MoE tests but kept for parity)
    NUM_LAYERS = 36
    NUM_DENSE_LAYERS = 0
    VOCAB_SIZE = 201088

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 1.0
