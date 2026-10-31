# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fused_tail's host plans (C++, bound in ttnn.experimental.moe_block as fused_tail_*) against the Python reference of
the trimmed tree (mstaletovic/moe-trim: models/demos/moe_block/tt/{tail_engine,tail_a2a}.py and
ttnn/ttnn/bringup/combine_rf/fabric_all_gather_py.py) on a table of inputs. No device.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/fused_tail/test_fused_tail_host.py

The reference modules are loaded from $MOE_BLOCK_REF_TREE (default /data/mstaletovic/tt-metal-trimb); the test skips
when that tree is absent. Their device-only imports (prepared_program, se_engine3) are stubbed; the TailEngine
geometry runs the reference __init__ on a stand-in mesh (its gather plan, which needs a device, is skipped).
"""

import ast
import importlib.util
import itertools
import os
import sys
import types
from pathlib import Path

import pytest

import ttnn

mb = ttnn.experimental.moe_block

REF_TREE = Path(os.environ.get("MOE_BLOCK_REF_TREE", "/data/mstaletovic/tt-metal-trimb"))
GS = (2, 3, 4, 6, 8)


def _defs_from(path: Path, names: set) -> dict:
    """The named top-level functions of a source file, executed alone (no module imports)."""
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def ref():
    """The reference modules: fag (fabric_all_gather_py), tail_a2a, tail_engine."""
    bringup = REF_TREE / "ttnn" / "ttnn" / "bringup"
    tt = REF_TREE / "models" / "demos" / "moe_block" / "tt"
    if not (tt / "tail_engine.py").exists():
        pytest.skip(f"reference tree not found: {REF_TREE}")
    stubs = {}

    def pkg(name, path=None):
        m = types.ModuleType(name)
        m.__path__ = [str(path)] if path else []
        stubs[name] = m
        return m

    br = pkg("ttnn.bringup", bringup)
    br.combine_rf = pkg("ttnn.bringup.combine_rf", bringup / "combine_rf")
    pp = pkg("ttnn.bringup.prepared_program")
    pp.generic_op = None
    for name in ("models", "models.demos", "models.demos.moe_block", "models.demos.moe_block.tt"):
        pkg(name)
    se3 = types.ModuleType("models.demos.moe_block.tt.se_engine3")
    se3.__dict__.update(_defs_from(tt / "se_engine3.py", {"_ring_pages"}))
    se3.l1_free_bytes = lambda mesh_device: mesh_device.l1_free  # (the stand-in mesh's free L1)
    stubs[se3.__name__] = se3
    saved = {k: sys.modules.get(k) for k in stubs}
    had_attr = hasattr(ttnn, "bringup")
    sys.modules.update(stubs)
    if not had_attr:
        ttnn.bringup = br

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    fag = load("ttnn.bringup.combine_rf.fabric_all_gather_py", bringup / "combine_rf" / "fabric_all_gather_py.py")
    a2a = load("fused_tail_ref_tail_a2a", tt / "tail_a2a.py")
    eng = load("fused_tail_ref_tail_engine", tt / "tail_engine.py")
    yield types.SimpleNamespace(fag=fag, a2a=a2a, eng=eng)
    for k in list(sys.modules):
        if k.startswith("ttnn.bringup") or k.startswith("fused_tail_ref_"):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
        else:
            sys.modules.pop(k, None)
    if not had_attr:
        del ttnn.bringup


def _units(us):
    return [tuple(u) for u in us]


@pytest.mark.parametrize("G", GS)
@pytest.mark.parametrize("ring", (False, True))
@pytest.mark.parametrize("balance", (False, True))
def test_gather_schedule(ref, G, ring, balance):
    for p in range(G):
        fwd, bwd = mb.fused_tail_gather_schedule(p, G, ring, balance)
        rf, rb = ref.fag._schedule(p, G, ring, balance)
        assert (_units(fwd), _units(bwd)) == (_units(rf), _units(rb)), (p, G, ring, balance)


@pytest.mark.parametrize("G", GS)
@pytest.mark.parametrize("ring", (False, True))
def test_chunk_order(ref, G, ring):
    for p in range(G):
        assert list(mb.fused_tail_chunk_order(p, G, ring)) == ref.eng.chunk_order(p, G, ring)


A2A_SHAPES = [  # (H, rows a mesh row, TP): GLM-5.3, K2.7, GLM-Flash, DSv4.1-Flash and odd ones
    (6144, 640, 4),
    (7168, 1024, 4),
    (4096, 640, 4),
    (5120, 640, 4),
    (6144, 256, 2),
    (6144, 384, 3),
    (6144, 1536, 6),
]


@pytest.mark.parametrize("H, rows, G", A2A_SHAPES)
@pytest.mark.parametrize("ring", (False, True))
@pytest.mark.parametrize("L", (1, 2))
@pytest.mark.parametrize("NBK, RP", [(8, 7), (8, 1), (12, 3)])
def test_a2a_schedule(ref, H, rows, G, ring, L, NBK, RP):
    ring = ring and G > 2
    NTB, RT = H // 32 // G, rows // G // 32
    # (the far side's ties as the fabric might route them: alternating, to exercise both)
    ties = {(p, (p + G // 2) % G): ("bwd" if p % 2 else "fwd") for p in range(G)} if ring and G % 2 == 0 else {}
    got = mb.fused_tail_a2a_schedule(G, ring, L, NTB, RT, NBK, RP, ties)
    sc = ref.a2a.TailSchedule(G=G, ring=ring, L=L, NTB=NTB, RT=RT, NBK=NBK, RP=RP, ties=ties)
    assert {tuple(k): v for k, v in got["expect"].items()} == sc.expect
    assert {tuple(k): [tuple(x) for x in v] for k, v in got["readies"].items()} == {
        k: sorted(v) for k, v in sc.readies.items()
    }
    for p, d in itertools.product(range(G), ("fwd", "bwd")):
        assert _units(got["dests"][(p, d)]) == _units(sc.dests(p, d)), (p, d)
    for part, l in itertools.product(range(3), range(L)):
        assert [tuple(x) for x in got["run_list"][(part, l)]] == sc.run_list(part, l), (part, l)


class _Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Mesh:
    """What TailEngine.__init__ asks of the mesh before its gather plan."""

    def __init__(self, gx, gy, l1_free):
        self.grid, self.l1_free = _Grid(gx, gy), l1_free

    def compute_with_storage_grid_size(self):
        return self.grid


ENGINE_SHAPES = [  # (rows a mesh row, Hr, Ho, TP, grid x): K3 at chunk 5120 / 8192 on 8x4, 2x4 rows, a narrow grid
    (640, 3584, 1792, 4, 12),
    (1024, 3584, 1792, 4, 12),
    (2560, 3584, 1792, 4, 12),
    (128, 3584, 1792, 4, 12),
    (640, 2048, 1024, 4, 9),
    (768, 3584, 3584, 2, 12),
]


def _reference_engine(ref, monkeypatch, rows, Hr, Ho, G, gx, cap, wtb):
    monkeypatch.setattr(ref.eng.TailEngine, "_gather_plan", lambda self, topology, num_links: None)
    dtype = {1088: ttnn.bfloat8_b, 2048: ttnn.bfloat16}[wtb]
    mesh = _Mesh(gx, 10, cap + ref.eng.KEPT_LATER)
    return ref.eng.TailEngine(mesh, rows=rows, Hr=Hr, Ho=Ho, chunks=G, eps=1e-5, w_dtype=dtype)


@pytest.mark.parametrize("rows, Hr, Ho, G, gx", ENGINE_SHAPES)
@pytest.mark.parametrize("cap_kb", (1400, 900, 600))
@pytest.mark.parametrize("wtb", (1088, 2048))
def test_engine_geometry(ref, monkeypatch, rows, Hr, Ho, G, gx, cap_kb, wtb):
    cap = cap_kb * 1024
    try:
        e = _reference_engine(ref, monkeypatch, rows, Hr, Ho, G, gx, cap, wtb)
    except AssertionError:  # (the reference rejects the shape or the L1: so must the C++ plan)
        try:
            mb.fused_tail_engine_geometry(rows, Hr, Ho, G, gx - 1, 10, wtb, cap)
        except RuntimeError:
            return
        raise AssertionError("the reference rejects this engine, the C++ plan accepts it")
    got = mb.fused_tail_engine_geometry(rows, Hr, Ho, G, gx - 1, 10, wtb, cap)
    pairs = lambda v: [tuple(x) for x in v]  # noqa: E731
    for k in ("KT", "NTO", "RT", "PAIRS", "NSTMAX", "NOWNMAX", "KH", "PNMAX", "FNR", "SHMAX", "NSLOT", "arena_bytes"):
        assert got[k] == getattr(e, k), k
    for k in ("nst", "nown", "pn", "n0", "FNS"):
        assert list(got[k]) == list(getattr(e, k)), k
    for k in ("kh", "wk", "ak"):
        assert pairs(got[k]) == [tuple(x) for x in getattr(e, k)], k
    assert [tuple(tuple(h) for h in f) for f in got["fin"]] == [tuple(tuple(h) for h in f) for f in e.fin]
    off, _ = e._layout(e.NSLOT)
    assert list(got["layout"]) == [off[k] for k in off]
    assert got["region_tiles"] == max(k1 - k0 for k0, k1 in e.wk) * e.PNMAX


def _chips(ref, G, ring):
    """fabric_all_gather_py._plan's rings for a mesh row of G chips (balanced), with the engine's chunk orders."""
    chips = {}
    for p in range(G):
        nxt = (0, (p + 1) % G) if (ring or p < G - 1) else None
        prv = (0, (p - 1) % G) if (ring or p > 0) else None
        fwd, bwd = ref.fag._schedule(p, G, ring, True)
        chips[(0, p)] = dict(rings=[dict(p=p, next=nxt, prev=prv, fwd=fwd, bwd=bwd)])
    for coord, ch in chips.items():
        rg = ch["rings"][0]
        for d, peer in (("fwd", rg["next"]), ("bwd", rg["prev"])):
            if peer is not None and (rg[d] or ref.fag._needs_ready(chips, coord, 0, d)):
                rg[f"{d}_links"] = [0, 1]
        ch["order"] = ref.eng.chunk_order(rg["p"], G, ring)
    return chips


@pytest.mark.parametrize("rows, Hr, Ho, G, gx", ENGINE_SHAPES)
@pytest.mark.parametrize("ring", (False, True))
def test_engine_relays(ref, monkeypatch, rows, Hr, Ho, G, gx, ring):
    ring = ring and G > 2
    try:
        e = _reference_engine(ref, monkeypatch, rows, Hr, Ho, G, gx, ref.eng.ARENA_MAX, 1088)
    except AssertionError:
        pytest.skip("the reference has no engine of this shape (test_engine_geometry checks the C++ agrees)")
    e.chips, e.ring = _chips(ref, G, ring), ring
    for p in range(G):
        got = mb.fused_tail_engine_relays(p, G, ring, rows, Hr, Ho, gx - 1, 10)
        assert {tuple(k): list(v) for k, v in got.items()} == e._relays((0, p)), p
