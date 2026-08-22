"""Deterministic topology guard: --mesh parse parity, manifest roundtrip, and the emit-e2e hard-assert.

These are the provable fail-safe properties: emit-e2e must parse --mesh identically to auto-up (no
silent 1-chip collapse on comma), the graduated topology must roundtrip through the manifest, and a
--mesh that disagrees with the graduated split must be caught (never silently proceed).
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _emit():
    return importlib.import_module("scripts.tt_hw_planner.commands.emit_e2e")


def _par():
    return importlib.import_module("scripts.tt_hw_planner.parallelism")


@dataclass
class _PC:
    tp: int
    dp: int

    @property
    def chips(self) -> int:
        return self.tp * self.dp


def test_mesh_chip_count_comma_and_x_are_identical():
    mc = _emit()._mesh_chip_count
    # comma, x, and bare number all mean the same chip count (the auto-up-parity fix)
    assert mc("8x4") == 32
    assert mc("8,4") == 32
    assert mc("4,8") == 32
    assert mc("32") == 32
    assert mc("1x8") == 8
    assert mc("1,8") == 8
    assert mc(None) == 1
    assert mc("") == 1


def test_manifest_roundtrip(tmp_path):
    par = _par()
    assert par.read_parallelism_manifest(tmp_path) is None  # cold
    p = par.write_parallelism_manifest(tmp_path, chips=32, tp=8, dp=4)
    assert p is not None and p.name == par.PARALLELISM_MANIFEST
    m = par.read_parallelism_manifest(tmp_path)
    assert m == {"chips": 32, "tp": 8, "dp": 4, "mesh": [4, 8]}


def test_manifest_read_malformed_returns_none(tmp_path):
    par = _par()
    (tmp_path / par.PARALLELISM_MANIFEST).write_text("not json")
    assert par.read_parallelism_manifest(tmp_path) is None
    (tmp_path / par.PARALLELISM_MANIFEST).write_text('{"no_chips": 1}')
    assert par.read_parallelism_manifest(tmp_path) is None


def test_topology_guard_matches_pass():
    tm = _emit()._topology_mismatch
    m = {"chips": 32, "tp": 8, "dp": 4, "mesh": [4, 8]}
    assert tm(m, _PC(tp=8, dp=4), 32) is None  # exact match → no error


def test_topology_guard_no_manifest_is_permissive():
    tm = _emit()._topology_mismatch
    assert tm(None, _PC(tp=8, dp=4), 32) is None  # older bring-up, nothing recorded → don't block


def test_topology_guard_single_device_graduation_not_enforced():
    tm = _emit()._topology_mismatch
    m = {"chips": 1, "tp": 1, "dp": 1, "mesh": [1, 1]}
    assert tm(m, None, 1) is None
    # even if given a bigger mesh, a tp<=1 graduation has no split to reproduce
    assert tm(m, _PC(tp=8, dp=4), 32) is None


def test_topology_guard_catches_single_device_when_sharded_graduated():
    tm = _emit()._topology_mismatch
    m = {"chips": 32, "tp": 8, "dp": 4, "mesh": [4, 8]}
    err = tm(m, None, 1)  # graduated TP=8 but --mesh gave 1 chip (the comma-bug scenario)
    assert err and "1 chip" in err and "TP=8" in err


def test_topology_guard_catches_different_split():
    tm = _emit()._topology_mismatch
    m = {"chips": 32, "tp": 8, "dp": 4, "mesh": [4, 8]}
    err = tm(m, _PC(tp=4, dp=2), 8)  # wrong chip count + TP
    assert err and "mismatch" in err.lower() and "TP=8" in err


def test_resolve_demo_dir_slug_matches_scaffold(tmp_path, monkeypatch):
    """emit-e2e must resolve the SAME slug the scaffold created — dotted names (HunyuanImage-3.0)
    map '.' and '-' both to '_' (hunyuanimage_3_0), not keep the dot (hunyuanimage_3.0)."""
    import types

    emit = _emit()
    scaffold = importlib.import_module("scripts.tt_hw_planner.scaffold_demo_folder")
    monkeypatch.chdir(tmp_path)  # no matching dir on disk → falls to the default slug path
    args = types.SimpleNamespace(model_id="tencent/HunyuanImage-3.0", output=None)
    resolved = emit._resolve_demo_dir(args)
    assert resolved.name == scaffold._slug("HunyuanImage-3.0") == "hunyuanimage_3_0"
