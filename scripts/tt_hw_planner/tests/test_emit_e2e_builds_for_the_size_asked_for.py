# SPDX-License-Identifier: Apache-2.0
"""--area on emit-e2e: the input SIZE a model is BUILT for, stated by the operator.

A model whose inputs have a size -- a diffusion pipeline's condition image, a vision tower's resolution --
runs at whatever it is handed, so nothing in the checkpoint says which size to build for and the builder
picks a default. That default then decides how the emitted pipeline lays its stages out over the mesh, so
a demo built for a small default does not merely run slower at a large size: it aborts. The operator is
the only one who knows which size they want, and before this flag they could not say.

What has to hold:
  - asked for, the size reaches the BUILDER (so the emitted demo is built for it), the GATE's process,
    and the gate's own verdict key (a pass at one size is not a pass at another);
  - not asked for, nothing changes at all -- no block in the prompt, no variable set, and the cached
    verdict of an earlier run still matches, because a model with no size axis must not be handed one;
  - the flag and its refusals are the ones `optimize` already has, not a second spelling.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
_REPO = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_REPO))

from models.experimental.perf_automation.agent.perf_adapter import AREA_ENV  # noqa: E402
from tt_hw_planner._cli_helpers.workload import add_area_arg, workload_value  # noqa: E402
from tt_hw_planner.commands import emit_e2e as E  # noqa: E402
from tt_hw_planner.commands import optimize as O  # noqa: E402

AREA = 1024


# ---- what the builder is told ------------------------------------------------------------------


def test_no_size_asked_for_tells_the_builder_nothing():
    """A model with no size axis must not be handed one."""
    for none in (None, 0):
        assert E._area_prompt_block(none) == ""


def test_the_size_asked_for_reaches_the_builder():
    block = E._area_prompt_block(AREA)
    assert str(AREA) in block
    # and it names the variable a later run can change the size by, without re-emitting
    assert AREA_ENV in block


def test_the_builder_is_told_the_size_shapes_the_BUILD_not_only_the_inputs():
    """The point of the flag: what a stage sizes its layout by has to follow the size, or a run at it
    aborts rather than running slower."""
    block = E._area_prompt_block(AREA).lower()
    assert "built" in block and ("mesh" in block or "layout" in block or "lays" in block)


# ---- what the gate is handed -------------------------------------------------------------------


def test_the_gate_process_is_handed_the_request():
    """The emitted gate runs in its own process, so the request reaches it by a named variable."""
    assert E.E2E_MCP_AREA_ENV and E.E2E_MCP_AREA_ENV != E.E2E_MCP_BATCH_ENV


def test_the_gate_server_passes_the_size_to_the_gate(monkeypatch):
    pytest.importorskip("mcp")
    import importlib

    monkeypatch.setenv(E.E2E_MCP_AREA_ENV, str(AREA))
    monkeypatch.setenv("E2E_MCP_DEMO_DIR", "/nonexistent-demo")
    sys.modules.pop("scripts.tt_hw_planner.e2e_mcp", None)
    M = importlib.import_module("scripts.tt_hw_planner.e2e_mcp")
    seen = {}
    monkeypatch.setattr(
        M,
        "_run_deterministic_gates",
        lambda d, p, t, batch=1, area=None: seen.update(area=area) or (False, ["x"]),
    )
    fn = getattr(M.termination_check, "fn", M.termination_check)
    assert fn()["can_stop"] is False
    assert seen["area"] == AREA
    sys.modules.pop("scripts.tt_hw_planner.e2e_mcp", None)


# ---- what the verdict is scoped to --------------------------------------------------------------


def _key(tmp_path, area):
    demo = tmp_path / "demo"
    demo.mkdir(exist_ok=True)
    (demo / "model.py").write_text("x = 1\n")
    return E._correctness_key(demo, 0.99, 1, area)


def test_a_pass_at_one_size_is_not_a_pass_at_another(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "run_stamp", lambda: "stamp")
    a, b = _key(tmp_path, 256), _key(tmp_path, 1024)
    if a is None:
        pytest.skip("no source fingerprint available here")
    assert a != b


def test_asking_for_no_size_keeps_the_key_it_had(tmp_path, monkeypatch):
    """So an existing run's cached verdict is not thrown away by this change."""
    monkeypatch.setattr(E, "run_stamp", lambda: "stamp")
    none, zero = _key(tmp_path, None), _key(tmp_path, 0)
    if none is None:
        pytest.skip("no source fingerprint available here")
    assert none == zero


# ---- the flag is the one optimize already has ----------------------------------------------------


def test_both_commands_take_the_same_flag():
    """Not a second spelling: emit-e2e and optimize add the SAME argument."""
    p1, p2 = ArgumentParser(), ArgumentParser()
    add_area_arg(p1)
    O.add_workload_flags(p2)
    help1 = {a.dest: a.help for a in p1._actions}.get("area")
    help2 = {a.dest: a.help for a in p2._actions}.get("area")
    assert help1 and help1 == help2


@pytest.mark.parametrize("bad", [0, -1])
def test_a_size_that_cannot_be_one_is_refused(bad):
    v, err = workload_value(SimpleNamespace(area=bad), "area")
    assert v is None and err and "area" in err


def test_a_size_not_asked_for_is_not_a_refusal():
    assert workload_value(SimpleNamespace(area=None), "area") == (None, None)


def test_the_command_threads_the_size_from_the_flag_to_the_builder_and_the_gate():
    """The wiring itself, which each test above sees only one end of."""
    import inspect

    # cmd_emit_e2e delegates; the phase that builds the prompt is where the flags are read.
    src = inspect.getsource(E._emit_e2e_phase_a)
    assert 'workload_value(args, "area")' in src  # read off the flag, validated like optimize's
    assert "_area_prompt_block(" in src  # handed to the builder
    assert "area=_area" in src  # handed to the gate's own process


def test_the_emit_e2e_command_carries_the_flag():
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "emit-e2e", "--help"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--area" in out.stdout and AREA_ENV in out.stdout
