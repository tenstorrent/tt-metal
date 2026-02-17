# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def handle_view(func, args, kwargs):
    """Handle view operation."""
    return args[0].reshape(args[1])


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
