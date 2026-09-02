# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic in-place promotion of Linear-family modules to their LoRA variants.

Rather than thread a ``lora_enabled`` flag through every module constructor,
walk a built model once and upgrade each plain ``Linear``/``ColParallelLinear``/
``RowParallelLinear`` to the matching ``LoRAMixin`` subclass. The subclasses are
thin (mixin + base + ``_init_lora_state``), so swapping ``__class__`` and calling
``_init_lora_state`` post-construction is equivalent to having built the LoRA
variant directly. Modules already LoRA-aware are left untouched.

This keeps the swap surface identical — ``bind_active`` / ``set_active_lora`` /
``reapply_after_load`` act on the promoted modules exactly as before — while
making every Linear in the model a LoRA target, not just the attn/ffn subset.
"""
from __future__ import annotations

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.lora import LoRAMixin

# Base types eligible for in-place promotion. Exact-type match (not isinstance)
# so an unknown Linear subclass isn't silently mis-promoted.
_PROMOTABLE = (Linear, ColParallelLinear, RowParallelLinear)

# Per-base promotion classes, built lazily. Base FIRST, mixin SECOND: Python
# forbids ``__class__`` assignment when the mixin comes first (the instance
# "solid base" changes -> 'object layout differs'). With the base first the
# layout is identical to the original, so the swap is allowed. The pre-defined
# LoRA*Linear classes are mixin-first (so their forward override wins in runtime
# mode) and can't be used here; fuse mode never needs that override, and every
# bind method is LoRAMixin-exclusive, so ordering is otherwise irrelevant.
_PROMOTED_CACHE: dict[type, type] = {}


def _promoted_class(base_cls: type) -> type:
    cls = _PROMOTED_CACHE.get(base_cls)
    if cls is None:
        cls = type(f"LoRA{base_cls.__name__}", (base_cls, LoRAMixin), {})
        _PROMOTED_CACHE[base_cls] = cls
    return cls


def _iter_modules(root):
    yield root
    for _, child in root.named_children():
        yield from _iter_modules(child)


def promote_to_lora(root, *, mode: str = "fuse") -> int:
    """Upgrade every plain Linear-family descendant of ``root`` to a LoRA-aware
    class in place. Returns the number promoted. Idempotent: modules already
    ``LoRAMixin`` (e.g. built via ``lora_enabled``) are skipped."""
    promoted = 0
    for module in _iter_modules(root):
        if isinstance(module, LoRAMixin):
            continue
        if type(module) not in _PROMOTABLE:
            continue
        module.__class__ = _promoted_class(type(module))
        module._init_lora_state(mode=mode)
        promoted += 1
    return promoted
