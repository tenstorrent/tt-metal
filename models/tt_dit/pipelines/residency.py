# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils import cache

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ..layers.module import Module
    from .pipeline_api import PipelineAPIMixin


class Offload:
    """Evicting the module deallocates its weights; loading it reads them from the cache."""

    def evict(self, module: Module) -> None:
        module.deallocate_weights()


@dataclass(frozen=True)
class Shard:
    """Evicting the module shards its weights over `axis`; loading it gathers them back."""

    axis: int
    ccl_manager: Any

    def evict(self, module: Module) -> None:
        module.shard(self.axis)


@dataclass
class _Entry:
    spec: dict[str, Any]
    evicts: tuple[Module, ...]
    mode: Offload | Shard
    on_evict: tuple[Callable[[], None], ...]


class Residency:
    """Owns which modules are on device.

    Each module is registered once with the modules that must be gone before it is resident (`evicts`), how it is
    evicted when another module lists it (`mode`), and what runs after it is evicted (`on_evict`). A module that no
    registration lists is never evicted. Residency releases traces only through `on_evict`, never on a load.

    `on_evict` is the designer's declaration of what an eviction releases; it is trusted, not checked. The default,
    the pipeline's `release_traces`, suits a new module. `()` keeps a pipeline whose eviction released nothing,
    `(tracer.release_trace,)` one that released exactly that tracer, and `(pipeline.release_traces, extra, ...)`
    composes per-module cleanup.
    """

    def __init__(self, pipeline: PipelineAPIMixin) -> None:
        self._pipeline = pipeline
        self._entries: dict[Module, _Entry] = {}

    def register(
        self,
        module: Module,
        spec: dict[str, Any],
        *,
        evicts: Iterable[Module] = (),
        mode: Offload | Shard | None = None,
        on_evict: Iterable[Callable[[], None]] | None = None,
    ) -> None:
        """Register `module` with `spec`, the keywords `cache.load_model` takes besides the module."""
        assert module not in self._entries, f"{type(module).__name__} is already registered"
        self._entries[module] = _Entry(
            spec=spec,
            evicts=tuple(evicts),
            mode=Offload() if mode is None else mode,
            on_evict=(self._pipeline.release_traces,) if on_evict is None else tuple(on_evict),
        )

    def load(self, module: Module) -> None:
        """Evict what `module` excludes, then make `module` resident."""
        entry = self._entries[module]
        for other in entry.evicts:
            if not other.is_loaded():
                continue
            other_entry = self._entries[other]
            other_entry.mode.evict(other)
            for callback in other_entry.on_evict:
                callback()

        if module.is_transiently_sharded():
            module.gather(entry.mode.ccl_manager)
        elif not module.is_loaded():
            cache.load_model(module, **entry.spec)
