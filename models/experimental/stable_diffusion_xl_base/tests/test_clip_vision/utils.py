# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}")

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
