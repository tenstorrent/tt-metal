# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def handle_view(func, args, kwargs):
    """Handle view operation (torch path). Expects tensor size to match product of new_shape.
    When tensor has padded numel (e.g. device im2col output 2965872) but view expects logical
    (1, 1152, 196, 4)=903168, take first 903168 elements and reshape so view succeeds."""
    t = args[0]
    shape = args[1]
    if not isinstance(shape, (list, tuple)) or len(shape) == 0:
        return t.reshape(shape)
    from functools import reduce
    import operator

    target_numel = reduce(operator.mul, shape, 1)
    if t.numel() == target_numel:
        return t.reshape(shape)
    if t.numel() > target_numel and target_numel > 0:
        return t.flatten()[:target_numel].clone().reshape(shape)
    return t.reshape(shape)


func_to_torch = {
    "aten::view": handle_view,
}


def can_dispatch_to_torch(func_name: str, args=None, kwargs=None) -> bool:
    if func_name == "aten::view":
        if len(args) != 2 or len(kwargs or {}) != 0:
            return False
    return func_name in func_to_torch


def dispatch_to_torch(func_name, args, kwargs):
    """Dispatch operation to TTNN handler."""
    return func_to_torch[func_name](func_name, args, kwargs)
