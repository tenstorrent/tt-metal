# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy (deferred) parameter initialization for Python TTML modules."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

from ttml.modules.parameter import Parameter, TensorMetadata

_lazy_init_enabled: ContextVar[bool] = ContextVar("ttml_lazy_init_enabled", default=False)


def is_lazy_init_enabled() -> bool:
    return _lazy_init_enabled.get()


@contextmanager
def lazy_init():
    """Context: `ttml.init.*` factories return :class:`TensorMetadata` instead of allocating.

    :class:`~ttml.modules.parameter.Parameter` then holds metadata until
    :func:`materialize_module` runs. C++ parameter registration is deferred until
    materialization.
    """
    token = _lazy_init_enabled.set(True)
    try:
        yield
    finally:
        _lazy_init_enabled.reset(token)


def _replicate_tensor_to_mesh_mapper() -> Any | None:
    """Return a TensorToMesh replicate mapper for non-sharded mesh parameters."""
    import ttml

    device = ttml.autograd.AutoContext.get_instance().get_device()
    return ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)


def _collect_lazy_parameter_slots(root: Any) -> list[tuple[Any, str]]:
    from ttml.modules.module_base import AbstractModuleBase

    slots: list[tuple[Any, str]] = []
    for _prefix, mod in root.named_modules():
        if not isinstance(mod, AbstractModuleBase):
            continue
        for attr_name in list(mod.__dict__.keys()):
            val = mod.__dict__.get(attr_name)
            if isinstance(val, Parameter) and isinstance(val.peek_tensor(), TensorMetadata):
                slots.append((mod, attr_name))
    return slots


def materialize_module(
    root: Any,
    mesh_device: Optional[Any] = None,
    layout_plan: Optional[Callable[[str, TensorMetadata], Any]] = None,
    on_device_init: bool = False,
) -> Any:
    """Replace lazy :class:`TensorMetadata` with real tensors and register parameters.

    Args:
        root: Model root (subclass of :class:`~ttml.modules.module_base.AbstractModuleBase`).
        mesh_device: Optional mesh; when ``None``, uses :func:`ttml.maybe_mesh`. When the
            mesh has multiple devices and a parameter has no mapper, a TensorToMesh replicate
            mapper is applied so weights are not left single-device.
        layout_plan: Optional ``(full_param_path, metadata) -> mapper | None``. A non-``None``
            return overrides the metadata's mapper for that parameter.
        on_device_init: Reserved for future on-device random init parity with host path.

    Returns:
        ``root`` for chaining.
    """
    from ttml._mesh import maybe_mesh
    from ttml.modules.module_base import AbstractModuleBase

    if not isinstance(root, AbstractModuleBase):
        raise TypeError(f"materialize_module expects AbstractModuleBase, got {type(root)!r}")

    mesh = mesh_device if mesh_device is not None else maybe_mesh()

    lazy_slots = _collect_lazy_parameter_slots(root)
    if not lazy_slots:
        return root

    def _full_path(mod: AbstractModuleBase, attr_name: str) -> str:
        for prefix, m in root.named_modules():
            if m is mod:
                return f"{prefix}.{attr_name}" if prefix else attr_name
        return attr_name

    def _resolve_mapper(full_path: str, meta: TensorMetadata) -> Any | None:
        mapper = meta.mapper
        if layout_plan is not None:
            override = layout_plan(full_path, meta)
            if override is not None:
                mapper = override
        if mapper is None and mesh is not None and mesh.num_devices() > 1:
            return _replicate_tensor_to_mesh_mapper()
        return mapper

    tensor_by_metadata_id: dict[int, Any] = {}

    for mod, attr_name in lazy_slots:
        param = getattr(mod, attr_name)
        meta = param.peek_tensor()
        if not isinstance(meta, TensorMetadata):
            continue
        mid = id(meta)
        if mid in tensor_by_metadata_id:
            tensor = tensor_by_metadata_id[mid]
        else:
            full_path = _full_path(mod, attr_name)
            mapper = _resolve_mapper(full_path, meta)
            tensor = meta.materialize(mapper_override=mapper)
            tensor_by_metadata_id[mid] = tensor
        object.__setattr__(param, "tensor", tensor)

    for mod, attr_name in lazy_slots:
        param = getattr(mod, attr_name)
        inner = param.peek_tensor()
        if isinstance(inner, TensorMetadata):
            raise RuntimeError(
                f"Parameter {attr_name!r} on {mod.get_name() if hasattr(mod, 'get_name') else mod!r} "
                "is still lazy after materialize_module; check init_fn or layout_plan."
            )
        mod._bind_parameter(inner, attr_name)
        # Generic post-materialize hook. Features like FSDP register callbacks
        # via ``parameter.add_post_materialize_callback(...)`` to attach
        # metadata onto the now-materialized autograd tensor (e.g. mirror
        # ``_fsdp_managed`` markers from the wrapper to ``inner`` so
        # ``sync_gradients`` / ``ttml.fsdp.is_fsdp_managed`` can find them).
        # ``materialize_module`` itself stays feature-agnostic.
        param._run_post_materialize_callbacks()

    if on_device_init:
        # Placeholder: host NumPy init remains default; on-device rand path can be wired here.
        pass

    return root
