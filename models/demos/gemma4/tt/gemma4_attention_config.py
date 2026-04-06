# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4-specific attention ProgramConfig for TT matmul operations.

Generates optimal program configs for the attention QKV projections,
softmax, and output projection based on device capabilities and
layer type (sliding vs global).
"""


def get_attention_program_config(config, mesh_config, is_decode):
    """
    Get matmul program configs for attention operations.

    Args:
        config: Gemma4AttentionConfig (has head_dim, num_heads, etc.)
        mesh_config: MeshConfig
        is_decode: True for decode mode

    Returns:
        dict of program configs for each attention matmul
    """
    # Skeleton: will return optimal ProgramConfigs during implementation
    return {}
