# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

import torch


def sequential_prefix_scan(A, x):
    """Does sequential version of parallel scan algorithm.
        h(t) = A(t)*h(t-1) + x(t), where h(0) = 0.
    Args:
        A: shape (1, 1, b*l, d_in*n) -> (1, 1, 1*1024, 5120*32)
        x: shape (1, 1, b*l, d_in*n) -> (1, 1, 1*1024, 5120*32)

    Returns:
        output: shape (1, 1, b*l, d_in*n) -> (1, 1, 1*1024, 5120*32)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

    """
    (_, _, l, d_in_n) = x.shape

    hidden_states = torch.zeros((1, 1, l, d_in_n), device=A.device)

    for i in range(l):
        hidden_states[:, :, i] = A[:, :, i] * hidden_states[:, :, i - 1] + x[:, :, i]

    return hidden_states
