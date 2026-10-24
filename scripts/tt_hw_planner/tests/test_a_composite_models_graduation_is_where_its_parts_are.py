# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A composite model's bring-up state lives with its parts, and the trace gate must find it there.

`read_graduation` read `bringup_status.json` out of the demo dir and nothing else. A composite e2e
package has no status file of its own -- its parts were brought up separately, each keeping its status
beside its own stubs -- so the read came back empty, `trace_policy` reported `known: False`, and
`classify_trace_verdict` returned a hard FAIL. Correct on its own terms (nothing licenses skipping a
trace on no evidence) but the evidence existed one directory away: 51 graduated modules read as 0,
and the agent was told to "make the bring-up status readable from this demo dir" -- a dir it is never
written to. It cost every round of a 25-hour run, and no edit to the model could have fixed it.

Where the parts are is not guessable and must not be typed, so the PIPELINE is asked: the imports it
actually makes are walked, and any directory along the way carrying a status file is a component.

That walk had a hole of its own. `from pkg import a, b` may be importing SUBMODULES, and for a
namespace package (no __init__.py -- which is how the bring-up tool leaves its stub packages) that is
the only form it can be reached by: `pkg.py` does not exist and neither does `pkg/__init__.py`. One
whole component was imported that way, so it was invisible to the walk AND absent from the correctness
fingerprint, which means editing its stubs did not invalidate a cached pass.
"""

from __future__ import annotations

import json

import pytest

from scripts.tt_hw_planner import trace_gate as TG
from scripts.tt_hw_planner.commands import emit_e2e as E


def _bringup_dir(root, rel, modules, graduated=True, kind="sharded"):
    """A component bring-up dir: a status file, and a stub snapshot per graduated module."""
    d = root / rel
    (d / "_stubs").mkdir(parents=True)
    (d / "bringup_status.json").write_text(json.dumps({"components": [{"name": m} for m in modules]}))
    for m in modules:
        (d / "_stubs" / f"{m}.py").write_text("def build(device, torch_module):\n    return None\n")
        if graduated:
            (d / "_stubs" / f"{m}.py.last_good_{kind}").write_text("snapshot")
    return d


@pytest.fixture
def composite(tmp_path, monkeypatch):
    """A demo whose parts live elsewhere, reached the way the real one reaches them."""
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)

    a = _bringup_dir(tmp_path, "models/parts/first", ["alpha", "beta"])
    b = _bringup_dir(tmp_path, "models/elsewhere/second", ["gamma"])

    # dotted module path (resolves through the file itself) and bare-submodule import (does not)
    (demo / "tt" / "pipeline.py").write_text(
        "from models.parts.first._stubs.alpha import thing\n" "from models.elsewhere.second._stubs import gamma\n"
    )

    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    return demo, a, b


# --- the gate finds a composite's state ----------------------------------------------------------


def test_a_composite_no_longer_reads_as_zero_graduated(composite):
    """THE FAILURE: an empty read became `known: False`, which is a hard FAIL with no way out."""
    demo, _, _ = composite
    assert not (demo / TG._STATUS_FILE).is_file(), "the demo dir has no status file; that is the case"
    g = TG.read_graduation(demo)
    pol = TG.trace_policy(g)
    assert len(g) == 3, g
    assert pol["known"] is True and pol["all_graduated"] is True
    assert len(pol["graduated_modules"]) == 3


def test_the_component_dirs_come_from_what_the_pipeline_imports(composite):
    demo, a, b = composite
    found = TG._component_status_dirs(demo)
    assert a.resolve() in found and b.resolve() in found


def test_a_part_reached_only_as_a_bare_submodule_is_still_found(composite):
    """THE WALK'S HOLE: `from pkg import mod` on a namespace package resolved to nothing."""
    demo, _, b = composite
    assert not (b / "__init__.py").exists() and not (b / "_stubs" / "__init__.py").exists()
    assert b.resolve() in TG._component_status_dirs(demo), "a namespace-package component went missing"
    assert "gamma" in " ".join(TG.read_graduation(demo))


def test_an_ungraduated_part_is_reported_not_hidden(composite, tmp_path):
    demo, _, _ = composite
    _bringup_dir(tmp_path, "models/parts/third", ["delta"], graduated=False)
    (demo / "tt" / "pipeline.py").write_text(
        (demo / "tt" / "pipeline.py").read_text() + "from models.parts.third._stubs import delta\n"
    )
    pol = TG.trace_policy(TG.read_graduation(demo))
    assert pol["known"] is True
    assert pol["all_graduated"] is False, "an ungraduated part must not be rounded up to done"
    assert any("delta" in m for m in pol["eager_eligible_modules"])


def test_the_same_module_name_in_two_parts_is_not_dropped(tmp_path, monkeypatch):
    """Merging dirs must not lose a module because another component happens to share its name."""
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    _bringup_dir(tmp_path, "models/parts/one", ["shared"])
    _bringup_dir(tmp_path, "models/parts/two", ["shared"], graduated=False)
    (demo / "tt" / "pipeline.py").write_text(
        "from models.parts.one._stubs import shared\nfrom models.parts.two._stubs import shared\n"
    )
    g = TG.read_graduation(demo)
    assert len(g) == 2, f"one of the two `shared` modules was overwritten: {g}"
    assert TG.trace_policy(g)["all_graduated"] is False


# --- a single-dir model is untouched -------------------------------------------------------------


def test_a_model_with_its_own_status_file_behaves_exactly_as_before(tmp_path, monkeypatch):
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    demo = _bringup_dir(tmp_path, "models/demos/solo", ["only"], kind="native")
    g = TG.read_graduation(demo)
    assert g == {"only": "native"}, "names must stay unqualified for a model that keeps its own status"


def test_a_demo_with_no_state_anywhere_still_reports_nothing_known(tmp_path):
    """Absence of evidence stays absence of evidence: this must NOT become a silent pass."""
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")
    assert TG.read_graduation(demo) == {}
    assert TG.trace_policy(TG.read_graduation(demo))["known"] is False


# --- the fingerprint hole the same walk was hiding ----------------------------------------------


def test_editing_a_part_reached_as_a_bare_submodule_invalidates_a_cached_pass(composite, monkeypatch):
    """THE STALE PASS: those sources were never hashed, so an edit to them changed no key."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-1")
    demo, _, b = composite
    before = E._source_fingerprint(demo)
    assert any("second" in str(f) for f in E._import_closure(demo)), "the part is still outside the walk"
    (b / "_stubs" / "gamma.py").write_text("def build(device, torch_module):\n    return 'edited'\n")
    assert E._source_fingerprint(demo) != before


def test_an_attribute_import_resolves_to_no_file_and_is_harmless(tmp_path):
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (tmp_path / "models" / "pkg").mkdir(parents=True)
    (tmp_path / "models" / "pkg" / "mod.py").write_text("VALUE = 1\n")
    (demo / "tt" / "pipeline.py").write_text("from models.pkg.mod import VALUE\n")
    files = {f.name for f in E._import_closure(demo)}
    assert "mod.py" in files and "VALUE" not in files


# --- the constraints -----------------------------------------------------------------------------


def test_it_names_no_model_or_stage():
    """Nothing here may assume a component's name, a package layout, or where a model lives."""
    import ast
    import inspect
    import textwrap

    for fn in (TG._component_status_dirs, TG._graduation_in, TG.read_graduation):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "vae", "transformer", "text_encoder", "tt_dit", "demos/", "pipelines"):
            assert name not in lowered, f"{name!r} in {fn.__name__} assumes a layout"
