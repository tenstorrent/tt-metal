# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The torch reference models must not load when the converted-weight cache hits.

``cache.load_model`` takes ``get_torch_state_dict`` as a callable and never invokes it on
a hit, so a warm boot needs no source safetensors at all: 17 MB of configs and tokenizer
is enough, against ~106 GB of weights. That only holds while nothing materialises the
reference models anyway.

Two ways to lose it, and both are quiet — the pipeline still works, it just reads a
hundred gigabytes it does not need:

* rebuilding them in ``__init__`` rather than on first use;
* passing ``self._torch_x.state_dict`` instead of ``lambda: self._torch_x.state_dict()``.
  The first evaluates the property to fetch the bound method, so the checkpoint loads
  before ``load_model`` can decide it does not need it. That one got past a review and a
  hardware run here, and was only caught by deleting the weights and trying to boot.

No device: the pipeline is built with ``__new__`` and only the attributes these two paths
touch are populated.
"""

from __future__ import annotations

import pytest

import ttnn
from models.tt_dit.parallel.config import DiTGParallelConfigNoCFG, Flux2VaeParallelConfig, ParallelFactor
from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline


class _Mesh:
    """Just the surface `_prepare_*` reads: the cache key includes the mesh shape."""

    shape = (2, 2)


class _LoadedModule:
    """A submodule whose weights are already on device, so ``load_model`` returns early."""

    def is_loaded(self) -> bool:
        return True


class _ExplodingReference:
    """Stands in for a torch reference model that must never be built.

    Accessing anything on it fails the test where the access happens, which points at the
    line that broke laziness rather than at a boot that is merely slower than it should be.
    """

    def __getattr__(self, name: str):
        msg = (
            f"the torch reference model was materialised to reach .{name} — on a cache hit "
            "nothing should touch it, and doing so makes the source weights a hard "
            "requirement of every boot"
        )
        raise AssertionError(msg)


def _pipeline_with_warm_cache(monkeypatch) -> Flux2Pipeline:
    """A Flux2Pipeline with every submodule already loaded and no device behind it."""
    p = Flux2Pipeline.__new__(Flux2Pipeline)
    p.checkpoint_name = "black-forest-labs/FLUX.2-dev"
    p.dynamic_load = False
    p.is_fsdp = True
    p._mesh_device = _Mesh()
    p._parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=2, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    p._vae_parallel = Flux2VaeParallelConfig(
        tp_parallel=ParallelFactor(factor=2, mesh_axis=1),
        h_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    p.transformer = _LoadedModule()
    p._vae_decoder = _LoadedModule()
    # The properties resolve through these, so a cache hit must leave them untouched.
    p._torch_transformer_cached = _ExplodingReference()
    p._torch_vae_cached = _ExplodingReference()
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *_a, **_k: None)
    return p


def test_a_warm_transformer_never_touches_the_torch_reference(monkeypatch):
    _pipeline_with_warm_cache(monkeypatch)._prepare_transformer()


def test_a_warm_vae_never_touches_the_torch_reference(monkeypatch):
    _pipeline_with_warm_cache(monkeypatch)._prepare_vae()


@pytest.mark.parametrize("name", ["_torch_transformer", "_torch_vae"])
def test_the_reference_models_are_lazy_properties(name):
    """A plain attribute assigned in __init__ would load the checkpoint before anything
    had a chance to consult the cache."""
    attr = getattr(Flux2Pipeline, name, None)
    assert isinstance(attr, property), f"{name} must be a property so it can stay unloaded"


def test_a_cache_miss_does_reach_the_reference(monkeypatch):
    """The mirror of the tests above: laziness must not turn into never loading at all."""

    class _NotLoaded:
        def is_loaded(self) -> bool:
            return False

    p = _pipeline_with_warm_cache(monkeypatch)
    p.transformer = _NotLoaded()

    reached = []
    monkeypatch.setattr(
        "models.tt_dit.utils.cache.load_model",
        lambda **kw: reached.append(kw["get_torch_state_dict"]),
    )
    p._prepare_transformer()

    assert reached, "a cache miss must still call load_model"
    # Passed as a zero-argument callable, so load_model decides whether to pay for it.
    assert callable(reached[0])
