# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--box names a BOX, and a name that is not one must say so.

WHAT THE OPERATOR SEES ON THIS HOST. tt-smi reports the board series of all four chips as `p300c`:

    UMD Chip 0..3 | Blackhole | p300c

So `--box p300c` is the obvious thing to type, and the CLI help used to suggest exactly that
("e.g. p300c, T3K, Galaxy"). But the planner's boxes are a different vocabulary -- a closed table of
nine names in which `p300c` does not appear, and neither does `Galaxy` (it is GalaxyWH / GalaxyBH).
Four p300c Blackhole chips are the four-chip box QB2.

WHY IT MATTERED. --box is the only input to _derive_mesh_device_env, which sets MESH_DEVICE -- the
variable that tells the MODEL which board profile to load. An unresolvable name fell into the same
blanket `except: return` as a failed import, so the run set nothing, printed nothing, and loaded a
default profile. The operator asked for a specific board and got silence, with no line in the log to
notice was missing.

The valid names are a closed set, so this is checkable, and a wrong one is a typo in an argument --
loud is the only correct response.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))


def _args(box, mesh="1x1"):
    class A:
        pass

    a = A()
    a.box, a.mesh = box, mesh
    return a


def _derive(monkeypatch, box, mesh="1x1"):
    from scripts.tt_hw_planner.commands.optimize import _derive_mesh_device_env

    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _derive_mesh_device_env(_args(box, mesh))
    import os

    return os.environ.get("MESH_DEVICE")


def _refusal(monkeypatch, box):
    """The message a bad --box produces. Fails the test if it produced none.

    A plain try/except, because this package sets its own pytest rootdir: the repo-root conftest,
    and with it the expect_error fixture, is never loaded for these tests.
    """
    try:
        _derive(monkeypatch, box)
    except SystemExit as exc:
        return str(exc)
    raise AssertionError("--box %r was accepted silently" % box)


def test_the_board_series_tt_smi_prints_is_refused_by_name(monkeypatch):
    """THE BUG. `p300c` is what the hardware calls itself and what the help used to suggest."""
    assert "unknown --box" in _refusal(monkeypatch, "p300c")


def test_the_error_points_at_the_box_these_chips_actually_are(monkeypatch):
    assert "QB2" in _refusal(monkeypatch, "p300c")


def test_the_error_lists_what_would_have_worked(monkeypatch):
    """Otherwise the operator has to go and find the table themselves."""
    msg = _refusal(monkeypatch, "nonsense-box")
    for name in ("N150", "N300", "T3K", "P100", "P150", "P300", "QB2", "GalaxyWH", "GalaxyBH"):
        assert name in msg, "%s missing from the list of valid boxes" % name


def test_a_real_box_still_resolves(monkeypatch):
    assert _derive(monkeypatch, "QB2", "1x1") == "P150"
    assert _derive(monkeypatch, "QB2", "2x2") == "P150x4_2x2"


def test_case_does_not_matter(monkeypatch):
    assert _derive(monkeypatch, "qb2", "1x1") == "P150"


def test_no_box_is_still_a_no_op(monkeypatch):
    """Omitting --box is a choice, not a typo: detection handles it and nothing should fail."""
    assert _derive(monkeypatch, None) is None


def test_an_explicit_mesh_device_is_never_overridden(monkeypatch):
    from scripts.tt_hw_planner.commands.optimize import _derive_mesh_device_env

    monkeypatch.setenv("MESH_DEVICE", "P300")
    _derive_mesh_device_env(_args("QB2", "2x2"))
    import os

    assert os.environ["MESH_DEVICE"] == "P300", "the operator's own export was replaced"


def test_the_help_no_longer_suggests_a_name_that_does_not_exist():
    src = (_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    i = src.index('popt.add_argument(\n        "--box"')
    block = src[i : i + 700]
    assert "e.g. p300c" not in block, "the help still suggests a box name that fails"
    assert "QB2" in block and "GalaxyWH" in block, "the help does not list the real names"


def test_argparse_cannot_catch_this_so_the_check_must():
    """--box is a free-form string to argparse -- there is no `choices=`, because the box table is
    imported lazily. So nothing upstream rejects a bad name and this check is the only gate."""
    src = (_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    i = src.index('popt.add_argument(\n        "--box"')
    assert "choices" not in src[i : i + 700]

    opt = (_ROOT / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    j = opt.index("def _derive_mesh_device_env(")
    body = opt[j : opt.index("\ndef ", j + 1)]
    assert "raise SystemExit" in body, "an unresolvable box still returns quietly"


def test_an_import_failure_is_still_tolerated():
    """Only an unknown NAME is fatal. If the box table cannot be imported at all there is nothing to
    validate against, and that must not take the run down -- it is the tool's problem, not the
    operator's typo."""
    opt = (_ROOT / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    j = opt.index("def _derive_mesh_device_env(")
    body = opt[j : opt.index("\ndef ", j + 1)]
    assert body.count("except Exception:") >= 2, "the import guard and the name check share one handler"
