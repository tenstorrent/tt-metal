# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the matmul sweep must open the RUN'S mesh, not a hardcoded 1x1.

The sweep used to open ttnn.MeshShape(1, 1) with no fabric. Interleaved between the loop's NxM
FABRIC_1D opens, that 1x1<->NxM format switch left the fabric control plane inconsistent and
deadlocked the next open_mesh_device (the op-sig probe wedged forever). The fix opens the SAME
topology resolve_mesh_shape reports (FABRIC_1D when multi-chip), so there is no format switch.

These risks are about device lifecycle, not matmul math, so ttnn is faked and the sweep body mocked:

  s1  TOPOLOGY: across many TT_PERF_MESH_ROWS/COLS permutations, the shape opened == resolve_mesh_shape
  s2  FABRIC GATED: multi-chip opens set FABRIC_1D; a 1-chip open touches fabric ZERO times
  s3  LIFECYCLE ORDER: exactly [fabric FABRIC_1D, open, close, fabric DISABLED] for multi-chip
  s4  BALANCE: over many repeated pre-passes every FABRIC_1D is matched by a DISABLED (no leak)
  s5  ISOLATION: a sweep that raises still closes the mesh AND disables fabric (finally), no leak
  s6  STANDALONE: env unset -> 1x1 and ZERO fabric calls, so a bare run is unchanged
  s7  NO REGRESSION: the 1x1 hardcode is gone from source and resolve_mesh_shape is wired in
"""

import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _load_sweep():
    spec = importlib.util.spec_from_file_location("cc_matmul_sweep_mesh_stress", str(_CC / "matmul_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_ttnn():
    t = types.ModuleType("ttnn")
    t.calls = []

    class MeshShape:
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols

    class FabricConfig:
        FABRIC_1D = "FABRIC_1D"
        DISABLED = "DISABLED"

    def set_fabric_config(cfg):
        t.calls.append(("fabric", cfg))

    def open_mesh_device(shape, **kw):
        t.calls.append(("open", shape.rows, shape.cols))
        return ("dev", shape.rows, shape.cols)

    def close_mesh_device(dev):
        t.calls.append(("close",))

    t.MeshShape = MeshShape
    t.FabricConfig = FabricConfig
    t.set_fabric_config = set_fabric_config
    t.open_mesh_device = open_mesh_device
    t.close_mesh_device = close_mesh_device
    return t


def _fake_perf_adapter():
    m = types.ModuleType("agent.perf_adapter")

    def resolve_mesh_shape(default_rows=1, default_cols=1):
        r = (os.environ.get("TT_PERF_MESH_ROWS") or "").strip()
        c = (os.environ.get("TT_PERF_MESH_COLS") or "").strip()
        if r or c:
            rr = int(r) if r else int(default_rows)
            cc = int(c) if c else int(default_cols)
            if rr >= 1 and cc >= 1:
                return rr, cc
        return int(default_rows), int(default_cols)

    m.resolve_mesh_shape = resolve_mesh_shape
    return m


def _wire(monkeypatch, boom=None):
    """Fake ttnn + resolve_mesh_shape, and mock the sweep body so no real device is touched.
    Returns (module, fake_ttnn). fake_ttnn.calls is the ordered device-lifecycle trace."""
    m = _load_sweep()
    tt = _fake_ttnn()
    monkeypatch.setitem(sys.modules, "ttnn", tt)
    monkeypatch.setitem(sys.modules, "agent.perf_adapter", _fake_perf_adapter())
    monkeypatch.setattr(m, "enumerate_matmul_sigs", lambda *a, **k: ["sig"])
    monkeypatch.setattr(m, "parse_matmul_sigs", lambda sigs: [{"m": 32, "k": 64, "n": 128}])
    monkeypatch.setattr(m, "summarize", lambda table, pcc: {"shapes": 1, "seeded": 1, "improved": 0})

    def _sweep(mesh, matmuls, pcc_threshold=0.99, iters=5):
        if boom is not None:
            raise boom
        return [{"m": 32, "k": 64, "n": 128, "best": {"fidelity": "LoFi", "dtype": "bfloat8_b"}}]

    monkeypatch.setattr(m, "sweep_matmuls", _sweep)
    return m, tt


def _set_mesh(monkeypatch, rows, cols):
    if rows is None:
        monkeypatch.delenv("TT_PERF_MESH_ROWS", raising=False)
    else:
        monkeypatch.setenv("TT_PERF_MESH_ROWS", str(rows))
    if cols is None:
        monkeypatch.delenv("TT_PERF_MESH_COLS", raising=False)
    else:
        monkeypatch.setenv("TT_PERF_MESH_COLS", str(cols))


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (8, 4), (4, 2)])
def test_s1_opens_resolved_topology(monkeypatch, tmp_path, rows, cols):
    m, tt = _wire(monkeypatch)
    _set_mesh(monkeypatch, rows, cols)
    m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    opens = [c for c in tt.calls if c[0] == "open"]
    assert opens == [
        ("open", rows, cols)
    ], f"{rows}x{cols} -> opened {opens} (must match resolve_mesh_shape, never 1x1)"


def test_s1_100_random_topologies(monkeypatch, tmp_path):
    rng = random.Random(20260730)
    for _ in range(100):
        r, c = rng.randint(1, 8), rng.randint(1, 8)
        with monkeypatch.context() as mp:
            m, tt = _wire(mp)
            _set_mesh(mp, r, c)
            m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
            assert ("open", r, c) in tt.calls, f"{r}x{c} not opened; calls={tt.calls}"


# --------------------------------------------------------------------------- s2
@pytest.mark.parametrize("rows,cols,multichip", [(1, 1, False), (1, 2, True), (1, 8, True), (2, 4, True)])
def test_s2_fabric_only_when_multichip(monkeypatch, tmp_path, rows, cols, multichip):
    m, tt = _wire(monkeypatch)
    _set_mesh(monkeypatch, rows, cols)
    m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    fabric = [c for c in tt.calls if c[0] == "fabric"]
    if multichip:
        assert ("fabric", "FABRIC_1D") in fabric and ("fabric", "DISABLED") in fabric, f"{rows}x{cols}: {fabric}"
    else:
        assert fabric == [], f"single chip must not touch fabric, got {fabric}"


# --------------------------------------------------------------------------- s3
def test_s3_lifecycle_order_multichip(monkeypatch, tmp_path):
    m, tt = _wire(monkeypatch)
    _set_mesh(monkeypatch, 1, 8)
    m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    assert tt.calls == [
        ("fabric", "FABRIC_1D"),
        ("open", 1, 8),
        ("close",),
        ("fabric", "DISABLED"),
    ], f"lifecycle out of order: {tt.calls}"


# --------------------------------------------------------------------------- s4
def test_s4_fabric_balanced_across_many_prepasses(monkeypatch, tmp_path):
    """The wedge came from an unbalanced fabric transition. Over many pre-passes on a multi-chip
    mesh, FABRIC_1D enables and DISABLED disables must stay 1:1 with opens/closes."""
    m, tt = _wire(monkeypatch)
    _set_mesh(monkeypatch, 1, 8)
    n = 50
    for _ in range(n):
        m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    ons = sum(1 for c in tt.calls if c == ("fabric", "FABRIC_1D"))
    offs = sum(1 for c in tt.calls if c == ("fabric", "DISABLED"))
    opens = sum(1 for c in tt.calls if c[0] == "open")
    closes = sum(1 for c in tt.calls if c[0] == "close")
    assert ons == offs == opens == closes == n, f"unbalanced: on={ons} off={offs} open={opens} close={closes} n={n}"


# --------------------------------------------------------------------------- s5
def test_s5_failure_still_closes_and_disables_fabric(monkeypatch, tmp_path):
    m, tt = _wire(monkeypatch, boom=RuntimeError("sweep blew up on device"))
    _set_mesh(monkeypatch, 1, 8)
    with pytest.raises(RuntimeError):  # allow-pytest.raises: no expect_error fixture
        m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    assert ("close",) in tt.calls, "mesh not closed after sweep raised (device leak)"
    assert (
        "fabric",
        "DISABLED",
    ) in tt.calls, "fabric not disabled after sweep raised (fabric leak -> next open wedges)"


def test_s5_failure_balanced_across_cycles(monkeypatch, tmp_path):
    n = 25
    for _ in range(n):
        with monkeypatch.context() as mp:
            m, tt = _wire(mp, boom=RuntimeError("boom"))
            _set_mesh(mp, 1, 8)
            with pytest.raises(RuntimeError):  # allow-pytest.raises: no expect_error fixture
                m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
            assert [c[0] for c in tt.calls] == ["fabric", "open", "close", "fabric"], tt.calls


# --------------------------------------------------------------------------- s6
@pytest.mark.parametrize("env", [(None, None), (None, "8"), ("1", None)])
def test_s6_standalone_unset_is_1x1_no_fabric(monkeypatch, tmp_path, env):
    m, tt = _wire(monkeypatch)
    _set_mesh(monkeypatch, env[0], env[1])
    m.run_prepass("n::t", out_path=str(tmp_path / "o.json"))
    # A fully-unset pair -> 1x1 no fabric. A half-set pair defaults the missing side to 1.
    opens = [c for c in tt.calls if c[0] == "open"]
    if env == (None, None):
        assert opens == [("open", 1, 1)] and [c for c in tt.calls if c[0] == "fabric"] == [], tt.calls
    else:
        assert len(opens) == 1 and opens[0][0] == "open", tt.calls


# --------------------------------------------------------------------------- s7
def test_s7_source_has_no_1x1_hardcode_and_wires_resolver():
    src = (_CC / "matmul_sweep.py").read_text()
    assert "MeshShape(1, 1)" not in src, "the 1x1 hardcode is back -- the fabric-transition wedge will return"
    assert "resolve_mesh_shape" in src, "run_prepass must derive its topology from resolve_mesh_shape"
    # the fabric guard must be present so single-chip standalone stays fabric-free
    assert "FabricConfig.FABRIC_1D" in src and "FabricConfig.DISABLED" in src, "fabric enable/disable guard missing"


# --------------------------------------------------------------------------- s8
def test_s8_repo_root_resolves_to_actual_repo():
    """_REPO_ROOT is the fallback the CLI uses when no --repo-root is passed. It must be the REPO
    root (parent of models/), not models/ itself: a relative perf-test node resolved against models/
    points at models/models/... which does not exist, so the op-sig probe enumerates ZERO matmuls
    (the exact 0-shapes regression). Off-by-one here silently empties the sweep."""
    m = _load_sweep()
    assert m._REPO_ROOT.name != "models", f"_REPO_ROOT points at models/ (off-by-one): {m._REPO_ROOT}"
    assert (m._REPO_ROOT / "models" / "experimental" / "perf_automation").is_dir(), m._REPO_ROOT


# --------------------------------------------------------------------------- s9
def test_s9_empty_enumeration_keeps_prior_table(monkeypatch, tmp_path):
    """A probe that crashes / a node that fails to resolve enumerates zero. That must NEVER overwrite
    a previously-good table -- doing so erases every seed. Keep the prior non-empty table."""
    import json

    m = _load_sweep()
    monkeypatch.setattr(m, "enumerate_matmul_sigs", lambda *a, **k: [])
    monkeypatch.setattr(m, "parse_matmul_sigs", lambda sigs: [])
    out = tmp_path / "matmul_sweep.json"
    prior = {"ok": True, "shapes": 76, "seeded": 76, "improved": 54, "seeds": [{"m": 32}]}
    out.write_text(json.dumps(prior))
    res = m.run_prepass("n::t", out_path=str(out))
    assert res.get("shapes") == 76, f"empty pass did not keep prior table: {res}"
    assert json.loads(out.read_text()).get("shapes") == 76, "prior table on disk was clobbered by zeros"


def test_s9_empty_enumeration_writes_zero_when_no_prior(monkeypatch, tmp_path):
    import json

    m = _load_sweep()
    monkeypatch.setattr(m, "enumerate_matmul_sigs", lambda *a, **k: [])
    monkeypatch.setattr(m, "parse_matmul_sigs", lambda sigs: [])
    out = tmp_path / "matmul_sweep.json"  # nothing prior
    res = m.run_prepass("n::t", out_path=str(out))
    assert res.get("shapes") == 0 and json.loads(out.read_text()).get("shapes") == 0, res
