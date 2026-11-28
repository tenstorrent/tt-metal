# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paths into the shared transformers stack resolve to the layout the checkout actually has.

2026-11-10: upstream moved the stack from models/tt_transformers to models/ttt_compat and left only an import
alias at the old path. The planner still pointed its demo, its registries and its edit scope at the old files,
so on a current checkout the template backend's demo was "missing", every reuse entry named a file that was
gone, and the mesh-label check could not open the demo it invokes. A checkout from before the move must keep
working too.
"""

import re
from pathlib import Path

import pytest

from scripts.tt_hw_planner.discovery import TTT_DIRS, mentions_ttt, ttt_dir, ttt_path

NEW, OLD = TTT_DIRS


def _checkout(root: Path, layout: str) -> Path:
    for rel in ("tt/rope.py", "demo/simple_text_demo.py"):
        (root / layout / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / layout / rel).write_text("")
    return root


@pytest.fixture
def new_layout(tmp_path):
    root = _checkout(tmp_path / "new", NEW)
    stray = root / OLD / "demo" / "glm" / "_stubs" / "attention.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("")
    return root


@pytest.fixture
def old_layout(tmp_path):
    return _checkout(tmp_path / "old", OLD)


def test_the_stack_dir_is_the_newest_layout_present(new_layout, old_layout, tmp_path):
    assert ttt_dir(new_layout) == NEW
    assert ttt_dir(old_layout) == OLD
    assert ttt_dir(tmp_path / "empty") == NEW


def test_a_path_written_for_either_layout_lands_in_the_checkouts_own(new_layout, old_layout):
    for written in (f"{NEW}/tt/rope.py", f"{OLD}/tt/rope.py"):
        assert ttt_path(written, new_layout) == f"{NEW}/tt/rope.py"
        assert ttt_path(written, old_layout) == f"{OLD}/tt/rope.py"
    assert ttt_path(OLD, new_layout) == NEW
    assert ttt_path(Path(f"{OLD}/tt/rope.py"), new_layout) == Path(f"{NEW}/tt/rope.py")
    assert ttt_path(f"{OLD}/tt/rope.py (wraps models/common/rmsnorm.py)", new_layout) == (
        f"{NEW}/tt/rope.py (wraps models/common/rmsnorm.py)"
    )


def test_a_file_left_at_the_old_location_keeps_its_path(new_layout):
    stray = f"{OLD}/demo/glm/_stubs/attention.py"
    assert ttt_path(stray, new_layout) == stray


def test_paths_outside_the_stack_are_untouched(new_layout):
    for p in (None, "", "models/common/rmsnorm.py", f"{OLD}_v2/tt/x.py", "HF AutoTokenizer.from_pretrained(HF_MODEL)"):
        assert ttt_path(p, new_layout) == p


def test_a_traceback_from_either_layout_names_the_stack():
    for d in TTT_DIRS:
        assert mentions_ttt(f"File '/home/u/tt-metal/{d}/tt/rope.py', line 1")
    assert not mentions_ttt("File '/home/u/tt-metal/models/common/rmsnorm.py', line 1")


def test_the_registries_point_into_the_checkouts_layout(monkeypatch, old_layout):
    from scripts.tt_hw_planner.compatibility import BuildingBlock, Effort, Status
    from scripts.tt_hw_planner.family_backends import FamilyBackend
    from scripts.tt_hw_planner.reuse_registry import ReuseEntry

    monkeypatch.setenv("TT_HW_PLANNER_BRINGUP_CWD", str(old_layout))
    demo = f"{NEW}/demo/simple_text_demo.py"
    backend = FamilyBackend("LLM", "t", demo, "template", None, smoke_test_entry=demo)
    assert backend.demo_path == backend.smoke_test_entry == f"{OLD}/demo/simple_text_demo.py"
    entry = ReuseEntry(("llama",), re.compile("Rotary"), "rope", f"{NEW}/tt/rope.py", "RotarySetup", "reuse")
    assert entry.tt_path == f"{OLD}/tt/rope.py"
    block = BuildingBlock("rope", "", lambda _: True, f"{NEW}/tt/rope.py", Status.SUPPORTED, Effort.DROP_IN)
    assert block.tt_path == f"{OLD}/tt/rope.py" and block.registry_tt_path is None


def _code(tmp_path, checkout):
    yaml = pytest.importorskip("yaml")
    from scripts.tt_hw_planner.commands import publish_hf as ph

    out = tmp_path / "tt-model.yaml"
    ph._write_tt_model_yaml(
        out,
        state={"batch": 32, "env": {"arch": "blackhole"}},
        slug="m",
        repo_id="org/repo",
        checkout=str(checkout),
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        mesh_device="(1, 1)",
        kind="vllm-plugin",
        plugin_ref="0123abc",
        vllm_version="0.31.0",
        extra_models_dir="models/demos/m/vllm_bundle",
        commit=None,
    )
    return yaml.safe_load(out.read_text())["source"]["code"]


def test_a_package_ships_the_stack_and_its_alias(tmp_path, new_layout, old_layout):
    assert {NEW, OLD} <= set(_code(tmp_path, new_layout))
    code = _code(tmp_path, old_layout)
    assert OLD in code and NEW not in code
