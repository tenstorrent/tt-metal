# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode expert forward pass for Gemma4.

Pure function — all CCL operations are gated by mesh_config.
"""


def decode_forward(
    hidden_states, expert_indices, expert_weights, weights, config, mesh_config, mesh_device, ccl_manager
):
    """
    Single-token expert computation (decode mode).

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        expert_indices: [1, 1, batch, top_k] expert assignments
        expert_weights: [1, 1, batch, top_k] routing weights
        weights: dict of TT weight tensors
        config: Gemma4ExpertConfig
        mesh_config: MeshConfig
        mesh_device: TT mesh device
        ccl_manager: CCLManager (used only if EP > 1)

    Returns:
        output: [1, 1, batch, hidden_size] on device
    """
    # Skeleton: will be implemented
    raise NotImplementedError("Gemma4 decode experts not yet implemented")
