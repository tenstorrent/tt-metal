# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 bringup tests."""

import torch


def randomize_norm_weights(module: torch.nn.Module, *, scale: float = 0.5) -> torch.nn.Module:
    """Give every `nn.RMSNorm` in `module` a non-trivial affine weight, in place.

    `nn.RMSNorm` initialises `weight` to all ones, so a reference model built with random weights
    (rather than loaded from the checkpoint) has an *identity* affine in every norm. That makes the
    norm weights invisible to a PCC comparison: a port that loaded the wrong norm weight, swapped two
    of them, or never loaded them at all would still match the reference exactly.

    MiniMax-H3 is full of RMSNorms -- `norm1`, `norm2`, the per-head `norm_q`/`norm_k`, the refiner's
    `final_norm` -- so this blind spot covers most of the model's non-matmul parameters. Measured on
    the token refiner at real dims, randomizing the norms moves "norm weights never loaded" from PCC
    1.000000 (undetectable) to 0.887, and "norm1/norm2 swapped" from 1.000000 to 0.986.

    Call this on the torch reference *before* taking its `state_dict`, so the TT module under test
    loads the same non-trivial values.
    """
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.RMSNorm) and submodule.weight is not None:
            submodule.weight.data = 1.0 + scale * torch.randn_like(submodule.weight.data)
    return module
