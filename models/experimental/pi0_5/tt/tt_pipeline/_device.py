# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, collapsed ``set_device`` recursive bind walk for the standalone pi0.5 port.

Distillation of ``tt_symbiote.utils.device_management.set_device`` (459 L -> ~90 L).
KEPT: the ``_``-prefixed-attr SKIP (so ``set_device(stage, mesh)`` never rebinds the
stage's pre-bound ``_prefix_kv`` tensors nor descends into ``D2DBridge._module_a``; the
stage's PUBLIC ``self.blocks`` / ``self.suffix`` DO get bound); per-module ``to_device``
+ ``preprocess_weights()`` + ``move_weights_to_device()``; ``_bypass_tensor_wrapping``
propagation; walking ``__dict__`` / dict / list / tuple children. DROPPED: the nn.Module
branch, the arch-mismatch torch-fallback swap, from_pretrained flags, recipe/make_kv_cache,
visualization, distributed init.

ZERO tt_symbiote imports. Imports with tt_symbiote NOT installed.
"""
from __future__ import annotations

import os
import warnings

from ._module import MeshShapeToDeviceArch, TTNNModule

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

__all__ = ["set_device"]


def _active_device_arch():
    mesh = os.environ.get("MESH_DEVICE")
    if mesh is None:
        return None
    return MeshShapeToDeviceArch.get(mesh)


def _module_allowed_archs(module: TTNNModule):
    forward = getattr(type(module), "forward", None)
    if forward is None:
        return None
    return getattr(forward, "__tt_allowed_archs__", None)


def _is_arch_supported(module: TTNNModule) -> bool:
    allowed = _module_allowed_archs(module)
    if allowed is None:
        return True
    active = _active_device_arch()
    if active is None:
        # No MESH_DEVICE set; let the call-time @run_on_devices check raise if it runs.
        return True
    return active in allowed


def set_device(obj, device) -> None:
    """Bind every ``TTNNModule`` reachable from ``obj`` to ``device``, then preprocess +
    move weights. ``_``-prefixed attributes are skipped (pre-bound children / private
    tensors are not re-walked)."""
    initialized_modules: list = []

    def _bind(child: TTNNModule) -> None:
        if not _is_arch_supported(child):
            warnings.warn(
                f"{child.module_name}: device arch unsupported on {_active_device_arch()}; "
                f"leaving TTNN module in place (no torch fallback in the standalone port).",
                stacklevel=2,
            )
        child.to_device(device)
        if device.get_num_devices() > 1:
            child.set_device_state()
        child._tt_pipeline_device_set = True
        initialized_modules.append(child)

    def _walk(current_obj, parent_is_ttnn: bool = False) -> None:
        if not isinstance(current_obj, TTNNModule):
            return
        if not getattr(current_obj, "_bypass_tensor_wrapping", False):
            current_obj._bypass_tensor_wrapping = parent_is_ttnn
        for attr_name, value in list(current_obj.__dict__.items()):
            if attr_name.startswith("_"):
                continue  # skip private/pre-bound children (e.g. _prefix_kv, _module_a)
            if isinstance(value, TTNNModule):
                _bind(value)
                _walk(value, parent_is_ttnn=True)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, TTNNModule):
                        _bind(v)
                        _walk(v, parent_is_ttnn=True)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, TTNNModule):
                        _bind(v)
                        _walk(v, parent_is_ttnn=True)

    if isinstance(obj, TTNNModule):
        if _is_arch_supported(obj):
            _bind(obj)
        else:
            warnings.warn(
                f"Root {obj.module_name}: device arch unsupported; cannot swap root in place.",
                stacklevel=2,
            )
    _walk(obj)

    for module in initialized_modules:
        try:
            module.preprocess_weights()
            module.move_weights_to_device()
        except Exception as e:
            warnings.warn(
                f"set_device: failed to (preprocess|move) weights for {module.module_name}: {e!r}",
                stacklevel=2,
            )

    try:
        setattr(obj, "_tt_pipeline_device_set", True)
    except Exception:
        pass
