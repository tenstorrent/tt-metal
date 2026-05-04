# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.6 B1 logical model dimensions (HF / full-model shapes).

Single source of truth for dimension constants used by prepare_weights,
weight_provider, stage, and tests.
"""


class LogicalModelDimensions:
    """HF / logical tensor dimensions for Kimi K2.6 B1."""

    HIDDEN_SIZE = 7168
    VOCAB_SIZE = 163840
    NUM_LAYERS = 61
    NUM_ATTENTION_HEADS = 64
    Q_A_DIM = 1536
    Q_B_OUT = 12288
    KV_A_DIM = 576
    KV_B_LORA_RANK = 512
    KV_B_PROJ_OUT = 16384
    O_PROJ_OUT = 8192
    MOE_INTERMEDIATE_SIZE = 2048
    INTERMEDIATE_SIZE = 18432
    GATE_NUM_INDICES = 384
    N_GROUP = 1
    TOPK_GROUP = 1
    ROUTED_SCALING_FACTOR = 2.827
    FIRST_K_DENSE_REPLACE = 1
    RMS_NORM_EPS = 1e-5
    ROPE_THETA = 50000
    ROPE_SCALING_FACTOR = 64
    MAX_POSITION_EMBEDDINGS = 262144
    NUM_NEXTN_PREDICT_LAYERS = 0


class RoutedExpert:
    """MoE routed-expert kernel layout constants."""

    M = 1
    K = 7168
    N_PER_CORE = 32
    NUM_CORES = 8
    GATE_PROJ_N = 2048
    GATE_EPS = 1e-20
    GATE_SCALING_FACTOR = 2.827
    TILE_W = 32
    FINAL_OUTPUT_WIDTH_PER_CORE = 32 * 32  # 1024
    INPUT_CORE_Y = 9
    SEED = 0
    GATE_PROJ_EXPERT_SEED = 0
    UP_PROJ_EXPERT_SEED = 384
    DOWN_PROJ_EXPERT_SEED = 768


class SharedExpert:
    """MoE shared-expert kernel layout constants."""

    K_PARALLEL = 8
    N_PARALLEL = 8
    N_PER_CORE = 64
    SEED = 100
