# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np


# Reconstructed function based on test usage and partial RoPE concept
def get_rot_transformation_mat(head_dim, partial_rotary_factor):
    """
    Generates a transformation matrix for partial rotary embeddings.

    Args:
        head_dim (int): The dimension of the attention head.
        partial_rotary_factor (float): The fraction of the head dimension subject to rotation.

    Returns:
        np.ndarray: A (1, 1, head_dim, head_dim) matrix applying rotation to the
                    first `rotary_dim` dimensions and identity otherwise.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim % 2 != 0:
        raise ValueError("Rotary dimension must be even.")

    # Initialize as identity matrix
    rot_mat = np.eye(head_dim, dtype=np.float32)

    # Apply RoPE-like transformation structure to the rotary dimensions
    # This structure ensures the non-zero pattern checked in test_glm4_rope_structure
    for i in range(0, rotary_dim, 2):
        # Set the diagonal elements for this pair to 0
        rot_mat[i, i] = 0
        rot_mat[i + 1, i + 1] = 0
        # Set the off-diagonal elements to mimic rotation pair
        rot_mat[i, i + 1] = 1.0
        rot_mat[i + 1, i] = -1.0

    # Reshape to the expected (1, 1, head_dim, head_dim)
    rot_mat = rot_mat.reshape(1, 1, head_dim, head_dim)

    return rot_mat
