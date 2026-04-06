# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 routed experts.

GeGLU activation with no bias. Each expert has gate_proj, up_proj, down_proj.
"""


def load_expert_weights(state_dict, config, mesh_device, mesh_config, dtype, tensor_cache_path=None):
    """
    Load expert weights from state_dict.

    Args:
        state_dict: layer-level state dict with expert weight keys
        config: Gemma4ExpertConfig
        mesh_device: TT mesh device
        mesh_config: MeshConfig for parallelization
        dtype: weight dtype
        tensor_cache_path: optional path for weight caching

    Returns:
        dict of loaded TT tensors
    """
    # Skeleton: will be implemented when expert compute is added
    return {}
