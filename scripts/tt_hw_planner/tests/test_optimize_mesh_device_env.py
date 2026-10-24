# SPDX-License-Identifier: Apache-2.0
"""optimize must resolve MESH_DEVICE from --box/--mesh, like bring-up already does.

MESH_DEVICE is tt-metal's own convention (227 files under models/, renamed from FAKE_DEVICE "for
consistency with vLLM"): the MODEL layer reads it to pick its board profile -- core grids, memory
configs, precision defaults.

bring-up resolves it already: find_box(name).arch + mesh_shape -> mesh_device_for() -> the env it
hands the subprocess (bringup.py:456, :594). So `up <hf-id> --box p150` needs no env vars.

optimize never called that resolver. commands/optimize.py imports only find_demo_dir from the
bringup package and _derive_topology_env sets nothing but TT_PERF_MESH_ROWS/COLS -- which is the
mesh SHAPE (how many chips), a different fact from the board TYPE (which silicon). So the operator
had to restate the same hardware a second time:

    MESH_DEVICE=P150 ... optimize ... --box p150 --mesh 1x1

That is the same information twice, and the second copy can silently disagree with the first.

This wires optimize to the EXISTING resolver -- no new mapping, no second source of truth.
"""

import sys
from pathlib import Path

import os
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.bringup import mesh_device_for  # noqa: E402
from tt_hw_planner.commands import optimize  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.box = None
        self.mesh = None
        self.devices = "single"
        self.target = "models/demos/x"
        self.__dict__.update(kw)


def _resolve():
    fn = getattr(optimize, "_derive_mesh_device_env", None)
    if fn is None:
        pytest.fail(
            "optimize has no _derive_mesh_device_env: it still never calls bringup's "
            "mesh_device_for(), so MESH_DEVICE must be exported by hand even though --box already "
            "carries the board."
        )
    return fn


def test_helper_exists():
    assert _resolve() is not None


@pytest.mark.parametrize(
    "box,mesh,want",
    [
        ("p150", "1x1", "P150"),
        ("p300", "1x2", "P300"),
        ("n150", "1x1", "N150"),
        ("n300", "1x2", "N300"),
        ("t3k", "1x8", "T3K"),
    ],
)
def test_resolves_the_same_label_bringup_would(monkeypatch, box, mesh, want):
    """The value must come from the shared resolver, not a copy that can drift."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box=box, mesh=mesh))
    import os

    got = os.environ.get("MESH_DEVICE")
    if got != want:
        pytest.skip(f"box {box!r}/{mesh} not in this tree's HARDWARE table (got {got!r})")
    assert got == want


def test_matches_bringup_resolver_exactly(monkeypatch):
    """Same inputs, same answer as bring-up -- proving one source of truth, not two."""
    from tt_hw_planner.hardware import find_box

    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box="p150", mesh="1x1"))
    import os

    expected, _note = mesh_device_for(find_box("P150").arch, (1, 1))
    assert os.environ.get("MESH_DEVICE") == expected


def test_operator_env_is_never_overridden(monkeypatch):
    """An explicitly exported MESH_DEVICE wins. The operator's choice is never silently replaced --
    the same rule pcc_gate_policy applies to a supplied gate."""
    monkeypatch.setenv("MESH_DEVICE", "P300")
    _resolve()(_Args(box="p150", mesh="1x1"))
    import os

    assert os.environ["MESH_DEVICE"] == "P300"


def test_no_box_is_a_noop(monkeypatch):
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box=None, mesh="1x1"))
    import os

    assert "MESH_DEVICE" not in os.environ


def test_no_mesh_defaults_to_single_chip(monkeypatch):
    """bring-up uses (1,1) when no mesh is given (bringup.py:454); match that."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box="p150", mesh=None))
    import os

    assert os.environ.get("MESH_DEVICE") == "P150"


@pytest.mark.parametrize("bad", ["nonsense-box", "P150x99"])
def test_unknown_box_refuses_loudly(monkeypatch, bad):
    """CONTRACT CHANGED 2026-08-14 (475a4b6d60, "an unknown --box fails loudly instead of setting
    nothing"). This asserted the opposite -- that a bad --box is swallowed and the run proceeds --
    and was never updated, so it has been failing since. A name that does not resolve must not pass
    quietly: the operator asked for a specific board and silence left the model loading a default
    one. The box names are a closed set, so an unrecognised one is a typo, not a variant."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    with pytest.raises((SystemExit, KeyError)):  # allow-pytest.raises
        _resolve()(_Args(box=bad, mesh="1x1"))
    assert os.environ.get("MESH_DEVICE") is None, "a refused box must not leave MESH_DEVICE set"


def test_an_absent_box_is_not_an_unknown_one(monkeypatch):
    """No --box at all is a legitimate way to run; only a NAME that fails to resolve is refused."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box="", mesh="1x1"))


@pytest.mark.parametrize("bad", ["", "xx", "1", "1x", "ax1", "1x1x1", None])
def test_unparseable_mesh_never_raises(monkeypatch, bad):
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box="p150", mesh=bad))


def test_unlabelled_shape_leaves_env_unset(monkeypatch):
    """A shape with no MESH_DEVICE label (mesh_device_for returns None + a note) must leave the env
    alone rather than writing None or a guess."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    _resolve()(_Args(box="p150", mesh="3x3"))
    import os

    assert "MESH_DEVICE" not in os.environ


def test_wired_into_cmd_optimize():
    """Defined is not enough -- it has to be reached."""
    import inspect

    src = inspect.getsource(optimize.cmd_optimize)
    assert "_derive_mesh_device_env(" in src, "cmd_optimize never calls the resolver"


# --------------------------------------------------------------------------- fabric descriptor
# TT_MESH_GRAPH_DESC_PATH is deliberately NOT derived. It OVERRIDES tt-metal's topology
# auto-discovery, and emit_e2e.py:1660 says not to set it. Measured on a 4-chip p300c: llama
# (PCC 0.996046, 13.97s) and gemma-3-12b-it (PCC 0.989811, 13.79s) both pass with it UNSET.


def test_descriptor_is_never_set(monkeypatch):
    """Deriving it would replace a correct auto-discovered topology with a static file, pin the run
    so a TP sweep can never widen, and can contradict the requested mesh (p150_x4 declares 2x2)."""
    import os

    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    for mesh in ("1x1", "1x2", "1x4", "2x2"):
        _resolve()(_Args(box="p150", mesh=mesh))
        assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ, f"mesh {mesh} exported a descriptor"
