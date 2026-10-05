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
# mode) and can't be used here.
#
# Base-first has a sharp edge, and it cost a bring-up: it puts ``Module`` BEFORE
# ``LoRAMixin`` in the MRO, so every method the mixin overrides from the base is
# SHADOWED on a promoted instance -- ``deallocate_weights``, ``forward`` and
# ``forward_fused_addcmul``. ``deallocate_weights`` is the one that silently
# changes an answer: without the mixin's version, a page-out never clears
# ``_delta_applied``, so the next ``reapply_after_load`` takes its "already
# applied" early return and the adapter is simply gone from the reloaded weight.
# Every pipeline that evicts its transformer between requests (a ``coresident:
# False`` preset, or ``dynamic_load``) then runs the BASE model under an
# adapter's name, with no error anywhere.
#
# So the weight lifecycle is wired explicitly here instead of being left to the
# MRO: the wrappers below call ``LoRAMixin``'s uniquely-named hooks and then the
# base implementation. They cannot use the mixin's own overrides, because those
# reach the base through a zero-argument ``super()`` that resolves past
# ``LoRAMixin`` to ``object`` on a promoted class.
_PROMOTED_CACHE: dict[type, type] = {}


def _lifecycle_overrides(base_cls: type) -> dict:
    """Weight page-out / page-in wiring for a base-first promoted class."""
    base_deallocate = base_cls.deallocate_weights
    base_load = base_cls.load
    base_save = base_cls.save
    base_mark_loaded = base_cls._mark_loaded  # noqa: SLF001

    def deallocate_weights(self) -> None:
        LoRAMixin._lora_on_unload(self)  # noqa: SLF001
        base_deallocate(self)

    def load(self, directory, /, *, prefix: str = "") -> None:
        base_load(self, directory, prefix=prefix)
        LoRAMixin._lora_on_load(self)  # noqa: SLF001

    def _mark_loaded(self) -> None:
        base_mark_loaded(self)
        LoRAMixin._lora_on_load(self)  # noqa: SLF001

    def save(self, directory, /, *, prefix: str = "") -> None:
        LoRAMixin._lora_guard_save(self)  # noqa: SLF001
        base_save(self, directory, prefix=prefix)

    return {"deallocate_weights": deallocate_weights, "load": load, "_mark_loaded": _mark_loaded, "save": save}


def _promoted_class(base_cls: type) -> type:
    cls = _PROMOTED_CACHE.get(base_cls)
    if cls is None:
        cls = type(f"LoRA{base_cls.__name__}", (base_cls, LoRAMixin), _lifecycle_overrides(base_cls))
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
    if mode == "runtime":
        # ``LoRAMixin.forward`` is shadowed by the base's on a base-first promoted class (see the
        # comment above ``_PROMOTED_CACHE``), so a runtime-mode adapter would add nothing to any
        # forward and report success. Construct ``LoRA*Linear`` directly for runtime mode.
        raise ValueError(
            "promote_to_lora cannot deliver lora_mode='runtime': the promoted class is base-first, "
            "so LoRAMixin.forward never runs. Build LoRALinear/LoRAColParallelLinear/"
            "LoRARowParallelLinear directly instead."
        )
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
