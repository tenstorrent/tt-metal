# SPDX-License-Identifier: Apache-2.0
"""--batch and --area: the WORKLOAD the optimize run profiles, as flags rather than exports.

Both reached the profiled test only as environment variables, so an operator had to know the spellings
and export them around the command. A run that forgot one profiled a different workload than the one it
reported -- and for a model whose cost follows an input SIZE (a diffusion pipeline's condition area) the
size also decides how the pipeline LAYS ITSELF OUT, so a missing export is not a slower run but an abort.

What has to hold:
  - the flags parse and land in the variables the tests actually read, under the names declared once in
    perf_adapter (not a second spelling typed here);
  - a flag that is not passed leaves the environment ALONE, so exporting still works and a supervised
    run started with an environment keeps it;
  - a value that cannot be a workload is refused with a message, not written.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
_REPO = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_REPO))

from models.experimental.perf_automation.agent.perf_adapter import AREA_ENV, BATCH_ENV  # noqa: E402
from tt_hw_planner.commands import optimize  # noqa: E402

ENVS = (BATCH_ENV, AREA_ENV)


@pytest.fixture
def clean_env(monkeypatch):
    for e in ENVS:
        monkeypatch.delenv(e, raising=False)
    return monkeypatch


def _parse(argv):
    p = ArgumentParser()
    optimize.add_workload_flags(p)
    return p.parse_args(argv)


def test_the_flags_parse():
    a = _parse(["--batch", "1", "--area", "1024"])
    assert (a.batch, a.area) == (1, 1024)


def test_they_land_in_the_variables_the_tests_read(clean_env):
    assert optimize._apply_workload_flags(_parse(["--batch", "1", "--area", "1024"])) is None
    assert os.environ[BATCH_ENV] == "1"
    assert os.environ[AREA_ENV] == "1024"


def test_the_names_are_the_ones_declared_once(clean_env):
    """Not a second spelling typed into the tool: the same constants the adapter publishes."""
    assert (BATCH_ENV, AREA_ENV) == ("TT_PERF_BATCH", "TT_PERF_AREA")


@pytest.mark.parametrize("argv", [[], ["--batch", "4"], ["--area", "512"]])
def test_a_flag_not_passed_leaves_the_environment_alone(clean_env, argv):
    """Exporting still works, and a supervised run that inherited an environment keeps it."""
    for e in ENVS:
        clean_env.setenv(e, "7")
    optimize._apply_workload_flags(_parse(argv))
    passed = {BATCH_ENV: "--batch" in argv, AREA_ENV: "--area" in argv}
    for e in ENVS:
        assert os.environ[e] != "7" if passed[e] else os.environ[e] == "7"


@pytest.mark.parametrize("bad", [0, -1, -32])
@pytest.mark.parametrize("name", ["batch", "area"])
def test_a_value_that_cannot_be_a_workload_is_refused(clean_env, name, bad):
    args = SimpleNamespace(batch=None, area=None)
    setattr(args, name, bad)
    msg = optimize._apply_workload_flags(args)
    assert msg and name in msg
    assert BATCH_ENV not in os.environ and AREA_ENV not in os.environ


def test_the_defaults_ask_for_nothing():
    a = _parse([])
    assert a.batch is None and a.area is None


def test_the_optimize_command_carries_them():
    """The flags are wired onto the REAL command, not only onto a parser a test builds. The parser is
    built inside main(), so the only way to see it is to ask the command itself."""
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "optimize", "--help"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--batch" in out.stdout and "--area" in out.stdout
    # and they say which variable they set, so an operator reading --help can still export instead
    assert BATCH_ENV in out.stdout and AREA_ENV in out.stdout
