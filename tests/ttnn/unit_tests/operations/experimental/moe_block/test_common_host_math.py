# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_block/common host math (C++, bound in ttnn.experimental.moe_block) against the Python reference functions of
the trimmed tree (mstaletovic/moe-trim) on a table of inputs. No device.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/test_common_host_math.py

The reference modules are loaded from $MOE_BLOCK_REF_TREE (default /data/mstaletovic/tt-metal-trimb) as
ttnn.bringup.combine_rf.* without running that tree's package __init__s; they import this tree's ttnn.
"""

import importlib
import itertools
import os
import struct
import sys
import types
from pathlib import Path

import pytest

import ttnn

mb = ttnn.experimental.moe_block

REF_TREE = Path(os.environ.get("MOE_BLOCK_REF_TREE", "/data/mstaletovic/tt-metal-trimb"))
MESHES = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (3, 5), (4, 4), (8, 4), (4, 8)]
WRAPS = list(itertools.product((False, True), repeat=2))
TIES = (0, 2)


@pytest.fixture(scope="module")
def ref():
    """The reference modules: sparse_combine_l1, sparse_combine_rf, sparse_all_gather_2d, combine_plan."""
    bringup = REF_TREE / "ttnn" / "ttnn" / "bringup"
    if not (bringup / "combine_rf" / "sparse_combine_l1.py").exists():
        pytest.skip(f"reference tree not found: {REF_TREE}")
    saved = {k: sys.modules.get(k) for k in ("ttnn.bringup", "ttnn.bringup.combine_rf")}
    had_attr = hasattr(ttnn, "bringup")
    pkg = types.ModuleType("ttnn.bringup")
    pkg.__path__ = [str(bringup)]
    sub = types.ModuleType("ttnn.bringup.combine_rf")
    sub.__path__ = [str(bringup / "combine_rf")]
    pkg.combine_rf = sub
    sys.modules["ttnn.bringup"], sys.modules["ttnn.bringup.combine_rf"] = pkg, sub
    if not had_attr:
        ttnn.bringup = pkg
    names = ("sparse_combine_l1", "sparse_combine_rf", "sparse_all_gather_2d", "combine_plan")
    mods = types.SimpleNamespace(**{n: importlib.import_module(f"ttnn.bringup.combine_rf.{n}") for n in names})
    yield mods
    for k in list(sys.modules):
        if k.startswith("ttnn.bringup"):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
    if not had_attr:
        del ttnn.bringup


def tie_rule(t):
    return mb.FabricTieRule.SourceParity if t == 2 else mb.FabricTieRule.NorthEast


def chips(R, C):
    return [(r, c) for r in range(R) for c in range(C)]


def test_route_and_first_hop(ref):
    for (R, C), (wx, wy), t in itertools.product(MESHES, WRAPS, TIES):
        for src, home in itertools.product(chips(R, C), repeat=2):
            exp = ref.sparse_combine_l1.route(src, home, R, C, wx, wy, 0, t)
            got = mb.route(src, home, R, C, wx, wy, tie_rule(t))
            assert [(tuple(ch), d) for ch, d in got] == exp, (R, C, wx, wy, t, src, home)
            assert mb.first_hop(src, home, R, C, wx, wy, tie_rule(t)) == ref.sparse_combine_rf.first_hop(
                src, home, R, C, wx, wy, t
            )


def test_first_hop_idx(ref, monkeypatch):
    for t in TIES:
        monkeypatch.setenv("TT_FABRIC_TIE_PARITY", "1" if t == 2 else "0")
        assert ref.sparse_combine_l1.tie_mode() == t
        for (R, C), (wx, wy) in itertools.product(MESHES, WRAPS):
            N = R * C
            for s, d in itertools.product(range(N), repeat=2):
                if s == d:
                    continue
                exp = ref.combine_plan.first_hop_idx(s, d, C, R, (wx, wy))
                assert mb.first_hop_idx(s, d, R, C, wx, wy, tie_rule(t)) == exp, (R, C, wx, wy, t, s, d)


def test_link_of_hbit_route_words(ref):
    for acc, src, home, L in itertools.product(range(16), range(8), range(8), (1, 2)):
        assert mb.link_of(acc, src, home, L) == ref.combine_plan.link_of(acc, src, home, L)
    for t in list(range(300)) + [2**16 - 1, 2**20 + 7, 2**31 - 1, 2**32 - 1]:
        for k in range(32):
            assert mb.hbit(t, k) == ref.sparse_all_gather_2d.hbit(t, k), (t, k)
    for N in range(1, 65):
        assert mb.route_words(N) == ref.sparse_all_gather_2d.route_words(N)


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("wrap", (False, True))
def test_span_tables(ref, n, wrap):
    sag = ref.sparse_all_gather_2d
    assert mb.span_table(n, wrap) == sag.span_table(n, wrap)
    assert mb.ring_table(n, wrap) == sag.ring_table(n, wrap)
    assert mb.axis_reach(n, wrap) == sag.axis_reach(n, wrap)


def test_span_subs(ref):
    for (R, C), (wx, wy) in itertools.product(MESHES, WRAPS):
        exp = ref.sparse_all_gather_2d.span_subs(R, C, wx, wy)
        got = mb.span_subs(R, C, wx, wy)
        assert exp == {
            "__WRAPX__": "true" if got["wrap_x"] else "false",
            "__WRAPY__": "true" if got["wrap_y"] else "false",
            "__XRING__": ", ".join(map(str, got["xring"])),
            "__YRING__": ", ".join(map(str, got["yring"])),
        }, (R, C, wx, wy)


def test_manifest_layout(ref):
    for (R, C), (wx, wy), S in itertools.product(MESHES, WRAPS, (32, 640, 2560)):
        wx_, wy_ = wx and C > 2, wy and R > 2  # (as mesh_plan derives them)
        try:
            exp = ref.sparse_all_gather_2d.manifest_layout(R, C, wx_, wy_, S)
        except AssertionError:
            with pytest.raises(Exception):
                mb.manifest_layout(R, C, wx_, wy_, S)
            continue
        assert mb.manifest_layout(R, C, wx_, wy_, S) == exp, (R, C, wx_, wy_, S)


class _FakeMesh:
    """The bits of a MeshDevice the reference's _mesh_plan / port_luts touch."""

    def __init__(self, R, C):
        self.shape = (R, C)

    def get_fabric_node_id(self, coord):
        return coord


def test_mesh_plan_and_port_luts(ref, monkeypatch):
    sag = ref.sparse_all_gather_2d
    monkeypatch.setattr(ttnn, "get_neighbor_eth_directions", lambda mesh, a, b: [0], raising=False)
    monkeypatch.setattr(ttnn, "get_forwarding_link_indices", lambda a, b: [0, 1, 2, 3], raising=False)
    for (R, C), (tx, ty), L in itertools.product(MESHES, WRAPS, (1, 2)):
        exp_chips, _, exp_wrap = sag._mesh_plan(_FakeMesh(R, C), L, (tx, ty))
        got = mb.mesh_plan_uniform(R, C, tx, ty, L)
        assert {tuple(k): {d: dict(peer=tuple(p["peer"]), links=list(p["links"])) for d, p in v.items()} for k, v in got.items()} == exp_chips, (R, C, tx, ty, L)  # fmt: skip
        exp_luts, exp_n = ref.combine_plan.port_luts(_FakeMesh(R, C), L, (tx, ty))
        got_l = mb.port_luts_uniform(R, C, tx, ty, L)
        assert {tuple(k): list(v) for k, v in got_l["lut"].items()} == exp_luts
        assert {tuple(k): v for k, v in got_l["num_ports"].items()} == exp_n


def test_snap_bits(ref):
    for q in (None, 2.0**-20, 2.0**-14, 2.0**-10, 0.5, 1.0, 4.0):
        assert mb.snap_bits(q) == ref.sparse_combine_rf._snap_bits(q), q
    assert mb.MoEBlockConfig(combine_snap=2.0**-14).snap_bits == struct.unpack("<I", struct.pack("<f", 1.5 * 2.0**9))[0]
    with pytest.raises(Exception):
        mb.snap_bits(0.3)


def test_config_defaults(ref):
    py = ref.sparse_combine_rf.CombineTuning()
    cpp = mb.CombineTuning()
    assert (cpp.credthr, cpp.credidle, cpp.credage, cpp.credauto) == (int(py.credthr), py.credidle, py.credage, py.credauto)
    assert (cpp.far, cpp.la, cpp.gbgate, cpp.snoc) == (py.far, py.la, py.gbgate, int(py.snoc))
    assert cpp.place == mb.PortPlacement.Noc1
    cfg = mb.MoEBlockConfig()
    assert cfg.tie_rule == mb.FabricTieRule.NorthEast  # (main's tables; the reference defaults to parity)
    assert (cfg.experts_during_dispatch, cfg.deterministic, cfg.combine_snap) == (False, True, None)
    assert (cfg.head_dram_fallback, cfg.router_split, cfg.residual) == (True, False, mb.ResidualLayout.SpTp)
    assert (cfg.num_links, cfg.persist_l1_small, cfg.handoff_bytes) == (2, False, 16)
    cfg2 = mb.MoEBlockConfig(persist_l1_small=True, combine_tuning=mb.CombineTuning(la=8, far=2))
    assert cfg2.handoff_bytes == 4 and cfg2.combine_tuning.la == 8 and cfg2 != cfg
    with pytest.raises(AttributeError):
        mb.MoEBlockConfig(no_such_field=1)
