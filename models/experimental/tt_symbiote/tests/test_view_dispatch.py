# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal test for view dispatch (prepare_args + handle_view) without full tt_symbiote deps."""
import operator
from functools import reduce

import torch


def _prepare_args_for_view(func_args, func_kwargs):
    """Same logic as device_management.prepare_args_for_torch_dispatch for aten::view."""
    if len(func_args) < 2:
        return func_args, func_kwargs
    t, shape = func_args[0], func_args[1]
    if not isinstance(t, torch.Tensor) or not isinstance(shape, (list, tuple)) or len(shape) == 0:
        return func_args, func_kwargs
    target_numel = reduce(operator.mul, shape, 1)
    if t.numel() <= target_numel or target_numel <= 0:
        return func_args, func_kwargs
    func_args = list(func_args)
    func_args[0] = t.flatten()[:target_numel].clone()
    return tuple(func_args), func_kwargs


def test_view_padded_logic():
    """Padded tensor (32 elements) viewed to (2,3,4) should trim to 24 then reshape."""
    padded = torch.randn(32)
    shape = (2, 3, 4)
    args, kw = _prepare_args_for_view((padded, shape), {})
    assert args[0].numel() == 24
    out = args[0].reshape(shape)
    assert out.shape == (2, 3, 4)


def test_view_padded_large():
    """Same scenario as test_gr00t: 2965872 elements, view (1, 1152, 196, 4) = 903168."""
    padded = torch.randn(2965872, dtype=torch.bfloat16)
    shape = (1, 1152, 196, 4)
    target_numel = 1 * 1152 * 196 * 4
    assert target_numel == 903168
    args, kw = _prepare_args_for_view((padded, shape), {})
    assert args[0].numel() == target_numel
    out = args[0].reshape(shape)
    assert out.shape == shape
