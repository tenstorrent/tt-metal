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
import shutil
import socket
import sys

import pytest
import ttnn

import ttml
from dcp_driver import MESH_OPEN_FAILED
from dcp_harness import run_world

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(3600)]

MGD_DIR = os.path.normpath(os.path.join(os.path.dirname(ttml.__file__), "..", "..", "..", "configs", "mgd"))
# Two MPI ranks on one host, each with its own [2, 2] mesh (mesh ids 0 and 1), eight chips in all. The shared-mesh
# layout (the ttnn dual-rank descriptor) is not used for cross-world checks: its TP collectives do not reach across
# the ranks, so replicas drift and the forward is not consistent (test_dcp_multirank covers the protocol on it).
TWO_MESHES_MGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bh_two_2x2_meshes.textproto")
DRIVER = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcp_driver.py")]
TT_RUN = shutil.which("tt-run")


def _world(mesh: str, axes: str, flags: tuple[str, ...], mgd: str | None, ranks: int = 1):
    """The driver arguments of one world; ``ranks > 1`` launches it through tt-run with the dual-rank descriptor."""
    args = ["--mesh", mesh, "--axes", axes, *flags]
    if mgd:
        args += ["--mgd", os.path.join(MGD_DIR, mgd)]
    if ranks > 1:
        args += ["--mgd", TWO_MESHES_MGD, "--ranks", str(ranks)]
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
    # Two ranks with one [2, 2] mesh each train the same model (the hierarchical shape): every key is a replica
    # across the ranks, one of them is elected to write it, rank 0 merges the index. Eight chips, so a four-chip
    # host skips these.
    "2ranks_2x2_dp_tp->2x2_dp_fsdp": (
        _world("2x2", "dp,tp", ("--tp",), None, ranks=2),
        _world("2x2", "dp,fsdp", ("--fsdp",), "bh_2_2_line_line.textproto"),
        8,
    ),
    "2ranks_2x2_dp_tp->single": (
        _world("2x2", "dp,tp", ("--tp",), None, ranks=2),
        _world("1x1", "dp,tp", (), None),
        8,
    ),
    "2x2_dp_fsdp->2ranks_2x2_dp_tp": (
        _world("2x2", "dp,fsdp", ("--fsdp",), "bh_2_2_line_line.textproto"),
        _world("2x2", "dp,tp", ("--tp",), None, ranks=2),
        8,
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


def _command(args: list[str]) -> list[str]:
    """The driver command line; a world with ``--ranks N`` runs under tt-run, every rank on this host."""
    if "--ranks" not in args:
        return DRIVER + args
    args = list(args)
    at = args.index("--ranks")
    del args[at : at + 2]
    mgd = args[args.index("--mgd") + 1]
    return [
        TT_RUN,
        "--mesh-graph-descriptor",
        mgd,
        "--hosts",
        socket.gethostname(),
        "--mpi-args",
        "--tag-output",
        *DRIVER,
        *args,
    ]


def _run(args: list[str], what: str) -> str:
    env = dict(os.environ)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    if "--ranks" in args and TT_RUN is None:
        pytest.skip(f"{what}: tt-run is not installed")
    proc = run_world(_command(args), env, 1500)
    # The interpreter's exit prints a long nanobind leak report after any traceback; keep the lines that matter.
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if not _NOISE.search(ln)]
    opened = not any("MESH_OPEN_FAILED" in ln for ln in lines)
    if proc.returncode == MESH_OPEN_FAILED or not opened:  # e.g. the QuietBox 1x4 ring descriptor on a loudbox
        reason = next((ln for ln in lines if "MESH_OPEN_FAILED" in ln), "no reason given")
        pytest.skip(f"{what}: this host cannot open the mesh: {reason}")
    tail = "\n".join(lines[-60:])
    assert proc.returncode == 0, f"{what} failed with exit {proc.returncode}:\n{tail}"
    if "--ranks" in args and what.endswith("save"):
        # tt-run does not always carry the ranks' exit status; the save prints its summary only when it succeeded.
        assert '"saved"' in proc.stdout, f"{what} printed no summary:\n{tail}"
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
