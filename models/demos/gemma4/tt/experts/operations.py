# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation and expert reduction helpers.
"""


def geglu(gate_output, up_output):
    """
    GeGLU activation: GELU(gate) * up

    Gemma4 uses gelu_pytorch_tanh variant.
    """
    # Skeleton: will be implemented
    raise NotImplementedError("GeGLU not yet implemented")


def reduce_expert_outputs(expert_outputs, expert_weights):
    """
    Weighted sum of expert outputs.

    Args:
        expert_outputs: list of [1, 1, seq_len, hidden_size] tensors
        expert_weights: [1, 1, seq_len, top_k] routing weights
    """
    # Skeleton: will be implemented
    raise NotImplementedError("Expert reduction not yet implemented")
