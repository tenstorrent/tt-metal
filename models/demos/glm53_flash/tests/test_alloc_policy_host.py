# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device; ``tt/mesh.py`` imports ttnn, nothing is opened): the allocation discipline under live traces
(``doc/serving/prefix_caching.md`` s.6). The freeze flag and the transient list are per mesh device, shared by every
``GLMMesh`` built over it (the per-layer meshes of ``FunctionalDecoder.from_state_dict`` included); a lazily built
device constant is cached before the first capture and a chunk transient after it, released at the chunk end.

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest \\
        models/demos/glm53_flash/tests/test_alloc_policy_host.py -q -p no:cacheprovider
"""

from __future__ import annotations

import dataclasses
import pathlib
from types import SimpleNamespace

from models.demos.glm53_flash.tt import mesh as mesh_mod
from models.demos.glm53_flash.tt.mesh import GLMMesh, lazy_const


class _Buf:
    def __init__(self):
        self.allocated = True

    def is_allocated(self) -> bool:
        return self.allocated


def _device(ident: int):
    return SimpleNamespace(shape=(2, 4), id=lambda: ident)


def test_frozen_flag_and_transients_are_shared_per_device(monkeypatch):
    monkeypatch.setattr(mesh_mod, "ttnn", SimpleNamespace(deallocate=lambda t: setattr(t, "allocated", False)))
    monkeypatch.setattr(mesh_mod, "_RUNTIME", {})
    dev = _device(4242)
    a = GLMMesh.from_device(dev)
    b = GLMMesh.from_device(dev)
    c = GLMMesh(dev, 2, 4)
    d = dataclasses.replace(a, topology=None)
    assert a.allocations_frozen is b.allocations_frozen is c.allocations_frozen is d.allocations_frozen
    other = GLMMesh.from_device(_device(7))
    assert other.allocations_frozen is not a.allocations_frozen
    plain = GLMMesh.from_device(SimpleNamespace(shape=(1, 1)))  # a device without id(): keyed by the object
    assert plain.allocations_frozen is not a.allocations_frozen and not plain.frozen
    cache: dict = {}
    made: list = []

    def make():
        made.append(_Buf())
        return made[-1]

    x = a.lazy(cache, ("k", 1), make)
    assert a.lazy(cache, ("k", 1), make) is x and cache[("k", 1)] is x and len(made) == 1, "cached before the freeze"
    assert a.release_transients() == 0
    a.freeze_allocations()
    assert b.frozen and c.frozen and d.frozen and not other.frozen
    y = b.lazy(cache, ("k", 2), make)
    assert ("k", 2) not in cache, "built under live traces: a chunk transient, never cached"
    y2 = b.lazy(cache, ("k", 2), make)
    assert y2 is y and len(made) == 2, "served within the step (a chunk's geometry asked for by every layer)"
    lst = c.lazy(cache, ("list",), lambda: [_Buf(), _Buf()])
    assert isinstance(lst, list) and ("list",) not in cache and c.lazy(cache, ("list",), make) is lst
    assert a.lazy(cache, ("k", 1), make) is x, "the pre-capture entry still serves"
    assert d.release_transients() == 3
    assert all(not t.allocated for t in [y] + lst) and x.allocated
    assert a.release_transients() == 0 and other.release_transients() == 0
    y3 = b.lazy(cache, ("k", 2), make)
    assert y3 is not y and y3.allocated and ("k", 2) not in cache, "rebuilt after the step's release"
    y3.allocated = False  # a caller freed it early: the next asker gets a fresh build, not a dead buffer
    y4 = b.lazy(cache, ("k", 2), make)
    assert y4 is not y3 and y4.allocated and a.release_transients() == 1 and not y4.allocated
    # mesh.const follows the same rule; lazy_const without a mesh (single-chip tests) caches
    k = a.const(("c",), make)
    assert ("c",) not in a._consts and k.allocated
    assert a.release_transients() == 1 and not k.allocated
    plain_cache: dict = {}
    v = lazy_const(None, plain_cache, "q", make)
    assert plain_cache["q"] is v and lazy_const(None, plain_cache, "q", make) is v
    assert lazy_const(b, plain_cache, "q", make) is v, "a cached entry is served frozen or not"
    # every eager step of GLM53MeshModel releases the transients it built: the chunk (prefill_streams), the decode
    # step and the wave (prefill_wave; its omission grew the DRAM 0.34 MiB per bank per 2048-row wave on the 1 M
    # direct run of 2026-10-30, ~16 geometry constants x 11 DSA layers never freed -> OOM at 555 K per row)
    import ast

    from models.demos.glm53_flash import tt as tt_pkg

    tree = ast.parse(pathlib.Path(tt_pkg.__path__[0], "mesh_model.py").read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GLM53MeshModel")
    steps = {n.name: n for n in model.body if isinstance(n, ast.FunctionDef)}
    for name in ("prefill_streams", "decode_step", "prefill_wave"):
        calls = [
            n
            for n in ast.walk(steps[name])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "release_transients"
        ]
        assert calls, f"GLM53MeshModel.{name} must release the transients GLMMesh.lazy built under live traces"
