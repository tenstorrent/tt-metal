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
