# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 B1 logical model dimensions (HF / full-model shapes).

Single source of truth for dimension constants used by prepare_weights,
weight_provider, stage, and tests.
"""


class LogicalModelDimensions:
    """HF / logical tensor dimensions for DeepSeek V3 B1."""

    HIDDEN_SIZE = 7168
    VOCAB_SIZE = 129280
    Q_A_DIM = 1536
    Q_B_OUT = 24576
    KV_A_DIM = 576
    KV_B_LORA_RANK = 512
    KV_B_PROJ_OUT = 32768
    O_PROJ_OUT = 16384
    MOE_INTERMEDIATE_SIZE = 2048
    INTERMEDIATE_SIZE = 18432
    GATE_NUM_INDICES = 256


class RoutedExpert:
    """MoE routed-expert kernel layout constants."""

    M = 1
    K = 7168
    N_PER_CORE = 32
    NUM_CORES = 8
    GATE_PROJ_N = 2048
    GATE_EPS = 1e-20
    GATE_SCALING_FACTOR = 2.5
    TILE_W = 32
    FINAL_OUTPUT_WIDTH_PER_CORE = 32 * 32  # 1024
    INPUT_CORE_Y = 9
    SEED = 0
    GATE_PROJ_EXPERT_SEED = 0
    UP_PROJ_EXPERT_SEED = 256
    DOWN_PROJ_EXPERT_SEED = 512


class SharedExpert:
    """MoE shared-expert kernel layout constants."""

    K_PARALLEL = 8
    N_PARALLEL = 8
    N_PER_CORE = 64
    SEED = 100
