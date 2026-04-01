# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def handle_view(func, args, kwargs):
    """Handle view operation (CPU fallback when TTNN view is not used).

    Mesh-col-sharded activations are often concatenated to full hidden width on host while a
    preceding op still requests a 2D shape with the per-device shard width (e.g. 2048 vs 256).
    If the requested 2D product does not match numel but the leading dimension does divide numel,
    infer the trailing dimension (same idea as a single ``-1`` in ``reshape``).
    """
    inp = args[0]
    shape = args[1]
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        r0, r1 = int(shape[0]), int(shape[1])
        n = inp.numel()
        if r0 > 0 and r0 * r1 != n and n % r0 == 0:
            return inp.reshape(r0, n // r0)
    return inp.reshape(shape)


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
