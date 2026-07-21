# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTML Python package.

This package provides Python bindings and implementations for the TTML
(Tenstorrent Machine Learning) library. C++ symbols from the _ttml nanobind
extension are explicitly re-exported here and in subpackage __init__.py files.
"""

import sys
from contextlib import contextmanager

import ttnn

# Try to import _ttml from the build directory first (when using .pth file with
# build_metal.sh --build-tt-train), then fall back to local package (standalone pip install)
try:
    import _ttml

    # Ensure _ttml is also visible as a submodule of this package for relative imports
    sys.modules[__name__ + "._ttml"] = _ttml

except ImportError:
    from . import _ttml

# --- Top-level symbols from _ttml ---
from ._ttml import NamedParameters

# --- Python subpackages ---
from . import autograd
from . import lazy
from . import init
from . import models
from . import modules

# Lazy / deferred parameter initialization (Python-side)
from .lazy import is_lazy_init_enabled, lazy_init, materialize_module

# --- Re-export _ttml submodules that have no Python package counterpart ---
# These are pure C++ nanobind submodules; making them attributes of ttml
# and registering in sys.modules allows both attribute access (ttml.ops.loss.*)
# and import statements (from ttml import ops).
ops = _ttml.ops
sys.modules[f"{__name__}.ops"] = ops

core = _ttml.core
sys.modules[f"{__name__}.core"] = core

optimizers = _ttml.optimizers
sys.modules[f"{__name__}.optimizers"] = optimizers

from ._mesh import Mesh, open_device_mesh, maybe_mesh, mesh, sync_gradients

from . import fsdp

from .sharding import Sharding


def manual_seed(seed: int) -> None:
    """Seed all of ttml's RNGs from a single call."""
    init.manual_seed(seed)
    autograd.AutoContext.get_instance().set_seed(seed)


class DramFootprintScope:
    """Live DRAM footprint (bytes per device) yielded by :func:`track_dram_footprint`.

    ``peak_allocated_bytes`` is the highest usage; ``min_largest_free_bytes`` is the largest single buffer
    still allocatable at the tightest point (the contiguity that gates OOM under fragmentation, and
    runs out before total free does). Both read live while the scope is open, frozen once it closes.
    """

    def __init__(self) -> None:
        self._final = None  # DramFootprint, set once the scope closes

    def _snapshot(self):
        return self._final if self._final is not None else core.utils.DramFootprintTracker.footprint()

    @property
    def peak_allocated_bytes(self) -> int:
        """Peak DRAM usage in bytes per device."""
        return self._snapshot().peak_allocated_bytes

    @property
    def min_largest_free_bytes(self) -> int:
        """Largest single buffer still allocatable at the tightest point, in bytes per device."""
        return self._snapshot().min_largest_free_bytes

    @property
    def reserved_bytes(self) -> int:
        """DRAM reserved outside the allocator arena (physical - arena), in bytes per device.

        A static device property, so it reads the same whether or not the scope is open.
        """
        return core.utils.dram_reserved_bytes()

    @property
    def arena_bytes(self) -> int:
        """DRAM arena (allocatable) size, in bytes per device -- the budget peak usage competes for.

        A static device property. ``peak_allocated_bytes`` is a fraction of this; OOM is gated by it.
        """
        return core.utils.dram_arena_bytes()


@contextmanager
def track_dram_footprint():
    """Measure the real peak DRAM footprint (per device) of the enclosed block, at ~zero cost.

    Reads the device allocator directly rather than capturing the op graph, so it reflects true
    usage and fragmentation without perturbing op timings. The yielded scope exposes ``peak_allocated_bytes``
    and ``min_largest_free_bytes`` (the OOM-limiting contiguity), live while open and final once closed.

    Example::

        with ttml.track_dram_footprint() as dram:
            loss = model(batch); loss.backward(); optimizer.step()
        print(dram.peak_allocated_bytes, dram.min_largest_free_bytes)   # finals, bytes per device

    Not nestable and not thread safe: there is one tracking session per device.
    """
    scope = DramFootprintScope()
    core.utils.DramFootprintTracker.begin()
    try:
        yield scope
    finally:
        scope._final = core.utils.DramFootprintTracker.end()
