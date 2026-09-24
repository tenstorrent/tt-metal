# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Static model-dimension constants for ERNIE-4.5-21B-A3B (the prefill engine's `model_config`)."""


class Ernie45Config:
    NUM_LAYERS = 28
    EMB_SIZE = 2560
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload (engine/migration contract)
    NUM_ATTENTION_HEADS = 20
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 128
    ROTARY_DIM = 128
    INTERMEDIATE_SIZE = 12288
    MOE_INTERMEDIATE_SIZE = 1536
    NUM_ROUTED_EXPERTS = 64
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 2
    VOCAB_SIZE = 103424
    ROPE_THETA = 500000.0
