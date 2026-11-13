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


class _T:
    """a stand-in device tensor of the AllocPool tests: shape / padded_shape, the buffer it views (base)"""

    def __init__(self, shape, dt, layout, mem, base=None):
        self.shape, self.dt, self.layout, self.mem = tuple(int(v) for v in shape), dt, layout, mem
        r = lambda v: -(-v // 32) * 32  # noqa: E731
        self.padded_shape = self.shape[:-2] + (r(self.shape[-2]), r(self.shape[-1])) if layout == "TILE" else self.shape
        self.base = base if base is not None else self


def _fake_ttnn(made: list):
    def empty(shape, dt, layout, dev, mem):
        made.append(_T(shape, dt, layout, mem))
        return made[-1]

    def reshape(base, want, padded):
        v = _T(want, base.dt, base.layout, base.mem, base=base.base)
        v.padded_shape = tuple(padded)
        return v

    return SimpleNamespace(empty=empty, reshape=reshape, Shape=tuple, TILE_LAYOUT="TILE")


def test_dsa_pool_borrows_the_kda_pool(monkeypatch):
    """GLM53_DSA_POOL_BORROW (mk_stack.AllocPool.lend_from, the B 16 x 1 M L1 diet): the DSA pool and its hand-off child
    take the KDA stacks' pool tensors of the same key before they allocate; each lent tensor goes to ONE borrower (s1
    never shares a buffer with an MLA / MoE intermediate); the lender's lists and counts are untouched; a narrow
    (alias) request takes a lent tensor of its padded key as a view; without a lender nothing changes."""
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_stack as MS

    made: list = []
    monkeypatch.setattr(MS, "ttnn", _fake_ttnn(made))
    BF, FP, L1, DEV = "bf16", "fp32", "L1", None
    req = lambda pool, shape, dt: pool(shape, dt, "TILE", DEV, L1)  # noqa: E731
    kda = MS.AllocPool()
    with kda:  # a KDA stack build at t = 16: hn / hn2 / a (bf16), s1 (fp32), post (fp32)
        k_bf = [req(kda, (1, 1, 16, 4096), BF) for _ in range(3)]
        k_s1 = req(kda, (1, 1, 16, 16384), FP)
        k_post = req(kda, (1, 4, 16, 1), FP)
    lists_before = {k: list(v) for k, v in kda.lists.items()}
    n0 = len(made)
    dsa = MS.AllocPool()
    dsa.lend_from(kda)
    with dsa.child():  # the MLA -> MoE hand-off first (stack_model._dsa_layer order)
        s1 = req(dsa.child(), (1, 1, 16, 16384), FP)
    assert s1 is k_s1 and len(made) == n0, "s1 takes the KDA pool's s1 buffer: no new L1"
    for _ in range(2):  # the MLA build, then the MoE build (each its own `with`: same requests -> same tensors)
        with dsa:
            got = [req(dsa, (1, 1, 16, 4096), BF) for _ in range(4)]
            g_s1 = req(dsa, (1, 1, 16, 16384), FP)
            g_post = req(dsa, (1, 4, 16, 1), FP)
    assert got[:3] == k_bf and got[3] not in k_bf and g_post is k_post, "lent first, then its own"
    assert g_s1 is not k_s1, "a tensor the child holds is never lent to the parent (s1 stays exclusive)"
    assert len(made) == n0 + 2, "only the requests the lender cannot serve allocate (one bf16, one fp32)"
    assert {k: list(v) for k, v in kda.lists.items()} == lists_before, "the lender's lists are untouched"
    with kda:  # the lender's own builds still get exactly their tensors (its counts / alias bases unchanged)
        assert [req(kda, (1, 1, 16, 4096), BF) for _ in range(3)] == k_bf
    # narrow widths (alias): a padded-key match in the lender is lent as a view of the requested rows
    kda.alias = dsa.alias = True
    with kda:
        k_pc = req(kda, (4, 1, 4, 1), FP)  # a narrow KDA build's stacked post: padded (4, 1, 32, 32), its own tensor
    n1 = len(made)
    with dsa:
        d_pc = req(dsa, (4, 1, 4, 1), FP)
    assert d_pc.base is k_pc.base and d_pc.shape == (4, 1, 4, 1) and len(made) == n1, "lent as a view, no new L1"
    # no lender: the pool allocates as before
    plain = MS.AllocPool()
    with plain:
        p = [req(plain, (1, 1, 16, 4096), BF) for _ in range(2)]
    assert all(t not in k_bf for t in p) and len(made) == n1 + 2
    # GLM53_KDA_POOL_BORROW: the two pools lend to each other (stack_model._dsa_pools: KDA <- DSA without the KDA child)
    made.clear()
    kda2, dsa2 = MS.AllocPool(), MS.AllocPool()
    kda2.child()  # (the KDA stacks' persistent hand-offs: exists before the DSA builds)
    with kda2:  # the dense KDA stack: the bf16 trio
        k2 = [req(kda2, (1, 1, 16, 4096), BF) for _ in range(3)]
    dsa2.lend_from(kda2)
    kda2.lend_from(dsa2, child=False)
    with dsa2:  # the first DSA layer: 4 bf16 (3 lent + its own), its MoE keys (router / latent: DSA-only so far)
        d2 = [req(dsa2, (1, 1, 16, 4096), BF) for _ in range(4)]
        d_lat = req(dsa2, (1, 1, 16, 512), FP)
        d_ids = req(dsa2, (1, 1, 16, 8), "u16")
    n2 = len(made)
    assert d2[:3] == k2 and d2[3] not in k2 and n2 == 3 + 3, "DSA <- KDA first, then its own"
    with kda2:  # the sparse KDA stack: the DSA MoE's keys again + one more bf16
        s2 = [req(kda2, (1, 1, 16, 4096), BF) for _ in range(4)]
        s_lat = req(kda2, (1, 1, 16, 512), FP)
        s_ids = req(kda2, (1, 1, 16, 8), "u16")
    assert s_lat is d_lat and s_ids is d_ids and s2[3] is d2[3], "KDA <- DSA: the DSA-only keys and its own bf16"
    assert s2[:3] == k2 and len(made) == n2, "no new L1: the sparse KDA stack's requests are all served"
    with kda2:  # a fifth bf16: the lender's lent-in tensors are never borrowed back (each serves one request per build)
        again = [req(kda2, (1, 1, 16, 4096), BF) for _ in range(5)]
    assert len({id(t) for t in again}) == 5 and len(made) == n2 + 1, "a family's tensor never comes back to it"
    with kda2.child():  # the KDA child (persistent hand-offs, state across programs) never borrows
        h = req(kda2.child(), (1, 1, 16, 512), FP)
    assert h is not d_lat and len(made) == n2 + 2
    with dsa2.child():  # the DSA child (s1) borrows the KDA pool's, never a DSA tensor lent to it
        s1b = req(dsa2.child(), (1, 1, 16, 512), FP)
    assert s1b is not d_lat and len(made) == n2 + 3
