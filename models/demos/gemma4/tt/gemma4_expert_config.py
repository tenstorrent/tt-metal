# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4-specific expert ProgramConfig for TT matmul operations.

Generates optimal program configs for the expert gate/up/down projections
based on device capabilities (moe_intermediate_size=704 per expert).
"""


def get_expert_program_config(config, mesh_config, is_decode):
    """
    Get matmul program configs for expert operations.

    Args:
        config: Gemma4ExpertConfig (has moe_intermediate_size, num_experts, etc.)
        mesh_config: MeshConfig
        is_decode: True for decode mode

    Returns:
        dict of program configs for each expert matmul
    """
    # Skeleton: will return optimal ProgramConfigs during implementation
    return {}
