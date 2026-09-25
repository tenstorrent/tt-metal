# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Overlay `apply_for` drops patches independently, so a `.py.last_good_native` snapshot can land
next to a stub patch that regressed the stub to a torch wrapper. `_prune_stale_graduation_snapshots`
must delete the snapshot in that case."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


TORCH_WRAPPER = "class C:\n    def __call__(self, *a, **k):\n        return self._torch_module(*a, **k)\n"
NATIVE = "import ttnn\nclass C:\n    def forward(self, x):\n        return ttnn.matmul(x, x)\n"
OP_SYNTH_STUB = (
    "class C:\n"
    "    def __call__(self, *args, **kwargs):\n"
    "        args = tuple(_coerce_to_torch(a) for a in args)\n"
    "        kwargs = {k: _coerce_to_torch(v) for k, v in kwargs.items()}\n"
    "        return self._op(*args, **kwargs)\n"
)


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@e"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
    (root / "README.md").write_text("x")
    subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(root), check=True)


def _place_stub_and_snapshot(root: Path, comp: str, stub_body: str) -> tuple[Path, Path]:
    stub_dir = root / "models" / "demos" / "test_model" / "_stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / f"{comp}.py"
    stub.write_text(stub_body)
    snap = stub_dir / f"{comp}.py.last_good_native"
    snap.write_text("")
    return stub, snap


def _pruner():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")._prune_stale_graduation_snapshots


def test_prune_removes_snapshot_when_stub_is_torch_wrapper(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _, snap = _place_stub_and_snapshot(tmp_path, "comp", TORCH_WRAPPER)
    _pruner()(["models/demos/test_model/_stubs/comp.py.last_good_native"], "test/model")
    assert not snap.is_file()


def test_prune_preserves_snapshot_when_stub_is_native(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _, snap = _place_stub_and_snapshot(tmp_path, "comp", NATIVE)
    _pruner()(["models/demos/test_model/_stubs/comp.py.last_good_native"], "test/model")
    assert snap.is_file()


def test_prune_ignores_non_snapshot_paths(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    stub, snap = _place_stub_and_snapshot(tmp_path, "comp", TORCH_WRAPPER)
    _pruner()(["models/demos/test_model/_stubs/comp.py"], "test/model")
    assert snap.is_file()
    assert stub.is_file()


def test_prune_also_covers_sharded_suffix(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    stub_dir = tmp_path / "models" / "demos" / "test_model" / "_stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "comp.py").write_text(TORCH_WRAPPER)
    sharded = stub_dir / "comp.py.last_good_sharded"
    sharded.write_text("")
    _pruner()(["models/demos/test_model/_stubs/comp.py.last_good_sharded"], "test/model")
    assert not sharded.is_file()


def test_prune_handles_missing_snapshot_gracefully(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _pruner()(["models/demos/test_model/_stubs/ghost.py.last_good_native"], "test/model")


def test_prune_removes_snapshot_over_op_synth_coerce_to_torch(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _, snap = _place_stub_and_snapshot(tmp_path, "comp", OP_SYNTH_STUB)
    _pruner()(["models/demos/test_model/_stubs/comp.py.last_good_native"], "test/model")
    assert not snap.is_file()
