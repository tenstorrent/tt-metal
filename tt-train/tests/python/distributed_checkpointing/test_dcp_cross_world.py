# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint saved on one parallelism strategy loads onto another: each world in its own process.

A tensor-parallel model needs the process-wide ParallelismContext, which cannot be reset, so every world runs
through ``dcp_driver.py`` (this directory) in a child process. The save child trains a tiny Llama (vocab
130, so the padded vocab differs between TP sizes), saves it and writes a reference of every tensor in canonical
layout plus the logits on a fixed batch; the load child builds the destination world, loads, and compares. This
module holds no device fixture of its own: the children need the chips.
"""

import os
import re
import subprocess
import sys

import pytest
import ttnn

import ttml
from dcp_driver import MESH_OPEN_FAILED

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(3600)]

MGD_DIR = os.path.normpath(os.path.join(os.path.dirname(ttml.__file__), "..", "..", "..", "configs", "mgd"))
DRIVER = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcp_driver.py")]


def _world(mesh: str, axes: str, flags: tuple[str, ...], mgd: str | None):
    args = ["--mesh", mesh, "--axes", axes, *flags]
    if mgd:
        args += ["--mgd", os.path.join(MGD_DIR, mgd)]
    return args


CASES = {
    "2x2_dp_tp->2x2_dp_fsdp": (
        _world("2x2", "dp,tp", ("--tp",), "bh_2_2_line_line.textproto"),
        _world("2x2", "dp,fsdp", ("--fsdp",), "bh_2_2_line_line.textproto"),
        4,
    ),
    "2x2_dp_fsdp->2x2_dp_tp": (
        _world("2x2", "dp,fsdp", ("--fsdp",), "bh_2_2_line_line.textproto"),
        _world("2x2", "dp,tp", ("--tp",), "bh_2_2_line_line.textproto"),
        4,
    ),
    "2x2_dp_tp->single": (
        _world("2x2", "dp,tp", ("--tp",), "bh_2_2_line_line.textproto"),
        _world("1x1", "dp,tp", (), None),
        4,
    ),
    "2x2_dp_tp->1x4_fsdp": (
        _world("2x2", "dp,tp", ("--tp",), "bh_2_2_line_line.textproto"),
        _world("1x4", "dp,fsdp", ("--fsdp",), "bh_qb_1_4_ring_ring.textproto"),
        4,
    ),
    "1x2_tp_tied->1x2_fsdp": (
        _world("1x2", "dp,tp", ("--tp", "--tie"), "bh_1_2_line_line.textproto"),
        _world("1x2", "dp,fsdp", ("--fsdp", "--tie"), "bh_1_2_line_line.textproto"),
        2,
    ),
}


def _release_devices() -> None:
    """The child processes open the chips; make sure this process holds none."""
    for release in (ttml.close_device_mesh, ttml.core.distributed.release_metal_env):
        try:
            release()
        except Exception:  # noqa: BLE001
            pass


_NOISE = re.compile(
    r"nanobind|leaked (instance|type|function|keep_alive)|refleaks|skipped remainder|\| DEBUG |^Config\{"
)


def _run(args: list[str], what: str) -> str:
    env = dict(os.environ)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    proc = subprocess.run(DRIVER + args, capture_output=True, text=True, timeout=1500, env=env)
    # The interpreter's exit prints a long nanobind leak report after any traceback; keep the lines that matter.
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if not _NOISE.search(ln)]
    if proc.returncode == MESH_OPEN_FAILED:  # e.g. the QuietBox 1x4 ring descriptor on a loudbox
        reason = next((ln for ln in lines if ln.startswith("MESH_OPEN_FAILED")), "no reason given")
        pytest.skip(f"{what}: this host cannot open the mesh: {reason}")
    tail = "\n".join(lines[-60:])
    assert proc.returncode == 0, f"{what} failed with exit {proc.returncode}:\n{tail}"
    return proc.stdout


_SAVED: dict[tuple[str, ...], str] = {}


def _saved_checkpoint(save_world: list[str], tmp_path_factory, what: str) -> str:
    """Save once per save world: the cases that start from the same world share the checkpoint."""
    key = tuple(save_world)
    if key not in _SAVED:
        ckpt = str(tmp_path_factory.mktemp("dcp") / "step_0000002")
        _run(["save", "--dir", ckpt, "--steps", "2", *save_world], what)
        _SAVED[key] = ckpt
    return _SAVED[key]


@pytest.mark.parametrize("case", list(CASES))
def test_checkpoint_loads_onto_another_world(tmp_path_factory, case):
    save_world, load_world, needed = CASES[case]
    available = ttnn.GetNumAvailableDevices()
    if available < needed:
        pytest.skip(f"{case} needs {needed} devices, {available} available")
    for args in (save_world, load_world):
        if "--mgd" in args and not os.path.isfile(args[args.index("--mgd") + 1]):
            pytest.skip(f"{case}: no descriptor {args[args.index('--mgd') + 1]}")
    _release_devices()
    ckpt = _saved_checkpoint(save_world, tmp_path_factory, f"{case} save")
    out = _run(["load", "--dir", ckpt, "--steps", "2", *load_world], f"{case} load")

    assert "RESULT: OK" in out, out.splitlines()[-5:]


def test_same_world_through_the_driver_matches_exactly(tmp_path):
    if ttnn.GetNumAvailableDevices() < 2:
        pytest.skip("needs two devices")
    _release_devices()
    world = _world("1x2", "dp,tp", ("--tp",), "bh_1_2_line_line.textproto")
    ckpt = str(tmp_path / "step_0000002")
    _run(["save", "--dir", ckpt, "--steps", "2", *world], "save")
    out = _run(["load", "--dir", ckpt, "--steps", "2", "--same-layout", *world], "load")
    assert "RESULT: OK" in out
