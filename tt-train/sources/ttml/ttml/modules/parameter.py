# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter and Buffer wrappers for tensor registration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable


LAZY_PARAMETER_ACCESS_MSG = (
    "This parameter has not been materialized yet. "
    "Build the model inside `with ttml.lazy_init():` and then call "
    "`ttml.materialize_module(model)` before accessing `.tensor` or running forward. "
    "In-place initializers (`ttml.init.uniform_`, etc.) also require a materialized tensor."
)


@dataclass(frozen=True)
class TensorMetadata:
    """Deferred parameter: shape + factory; no device storage until materialization.

    Do not use empty `ttml.autograd.Tensor` as a stand-in; keep allocation in
    `init_fn` until `materialize()` runs.

    ``mapper`` (optional ``ttnn.CppTensorToMesh``): the distribution plan used
    when ``materialize`` runs. Downstream consumers — notably
    ``ttml.fsdp.fully_shard`` — can introspect the existing sharding via
    ``mapper.config().placements`` and build a new combined mapper without
    round-tripping the not-yet-allocated tensor through host. ``None`` means
    "no explicit mapper; treat as fully replicated" (default replicate mapper
    is installed by :func:`ttml.materialize_module`).
    """

    shape: tuple[int, ...]
    init_fn: Callable[..., Any]
    mapper: Any | None = None
    requires_grad: bool = True

    def materialize(self, mapper_override: Any | None = None) -> Any:
        """Allocate the autograd tensor using optional mapper override."""
        mapper = self.mapper if mapper_override is None else mapper_override
        return self.init_fn(self.shape, mapper)


def replace_lazy_mapper(parameter: "Parameter", new_mapper: Any) -> None:
    """Replace a lazy ``parameter``'s mapper.

    Used by :func:`ttml.fsdp.fully_shard` to install an FSDP shard placement on
    a lazy parameter without materializing. Errors if ``parameter`` has already
    been materialized — by then the in-memory tensor has the old mapper baked in
    and a host roundtrip is the only way to redistribute (use the eager FSDP
    code path instead).
    """
    inner = parameter.peek_tensor()
    if not isinstance(inner, TensorMetadata):
        raise RuntimeError(
            "replace_lazy_mapper called on a materialized Parameter; only lazy "
            "parameters can have their mapper rewritten without a host roundtrip."
        )
    new_meta = replace(inner, mapper=new_mapper)
    object.__setattr__(parameter, "tensor", new_meta)


class Parameter:
    """Wrapper marking a tensor as a trainable parameter."""

    def __init__(self, tensor: Any) -> None:
        object.__setattr__(self, "tensor", tensor)
        # Generic "fire after .tensor becomes a real autograd tensor" hook.
        # Used by features like FSDP to attach metadata onto the materialized
        # autograd tensor without requiring materialize_module to know about
        # them. See ``add_post_materialize_callback``.
        object.__setattr__(self, "_post_materialize_callbacks", [])

    def __getattribute__(self, name: str) -> Any:
        if name == "tensor":
            t = object.__getattribute__(self, "tensor")
            if isinstance(t, TensorMetadata):
                raise RuntimeError(LAZY_PARAMETER_ACCESS_MSG)
            return t
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def peek_tensor(self) -> Any:
        """Return stored tensor or :class:`TensorMetadata` without raising."""
        return object.__getattribute__(self, "tensor")

    @property
    def is_lazy(self) -> bool:
        return isinstance(self.peek_tensor(), TensorMetadata)

    def add_post_materialize_callback(self, fn: Callable[["Parameter"], None]) -> None:
        """Register ``fn(self)`` to run once ``self.tensor`` is a real autograd tensor.

        If this Parameter is already materialized at registration time, ``fn``
        runs immediately (the eager case). Otherwise the call is deferred
        until :func:`ttml.materialize_module` materializes the parameter and
        drains the callback list via :meth:`_run_post_materialize_callbacks`.

        This is the generic decoupling layer between lazy init and any feature
        that needs to attach metadata to the materialized tensor — e.g. FSDP's
        ``_fsdp_managed`` markers (looked up by ``sync_gradients`` and the
        Muon guard via ``ttml.fsdp.is_fsdp_managed``). Without it, every such
        feature would need a hand-coded propagation step inside
        ``materialize_module``.
        """
        if not isinstance(self.peek_tensor(), TensorMetadata):
            fn(self)
            return
        cbs = object.__getattribute__(self, "_post_materialize_callbacks")
        cbs.append(fn)

    def _run_post_materialize_callbacks(self) -> None:
        """Fire and clear all registered post-materialize callbacks.

        Called by :func:`ttml.materialize_module` immediately after binding
        the materialized tensor. Idempotent; subsequent calls are no-ops
        because the list has been drained.
        """
        cbs = object.__getattribute__(self, "_post_materialize_callbacks")
        for fn in cbs:
            fn(self)
        cbs.clear()

    def __repr__(self) -> str:
        inner = self.peek_tensor()
        return f"Parameter({inner})"


class Buffer:
    """Wrapper marking a tensor as a non-trainable buffer."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"Buffer({self.tensor})"
