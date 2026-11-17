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

import numpy as np
import pytest
from dcp_driver import LOGITS_TOLERANCE, MESH_OPEN_FAILED, RESUME, compare

# Each child is one process opening a mesh, training or loading a tiny Llama: about a minute warm, a few cold (the
# whole module took 5 minutes warm and 13 cold on a loudbox). A hung child fails after 10 minutes, so one hang still
# leaves a 25-minute CI leg the time to report the tests after it.
CHILD_TIMEOUT_S = 600
pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(2 * CHILD_TIMEOUT_S + 60)]

DRIVER = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcp_driver.py")]


def _world(mesh: str, axes: str, flags: tuple[str, ...], mgd: str | None):
    """The driver arguments of a world; ``mgd`` names a bundled descriptor, resolved by ``_resolve``."""
    args = ["--mesh", mesh, "--axes", axes, *flags]
    if mgd:
        args += ["--mgd", mgd]
    return args


def _resolve(world: list[str], mgd_path) -> list[str]:
    """``world`` with its descriptor name replaced by the descriptor's path; skips if the descriptor is missing."""
    if "--mgd" not in world:
        return world
    at = world.index("--mgd") + 1
    path = mgd_path(world[at])
    if not os.path.isfile(path):
        pytest.skip(f"no descriptor {path}")
    return world[:at] + [path] + world[at + 1 :]


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
        _world("1x4", "dp,fsdp", ("--fsdp",), "bh_1_4_line_line.textproto"),
        4,
    ),
    # Two-device cases, which a 2-chip CI host can run. TP1 -> TP2: the fused rows spread over the shards and the vocab
    # padding grows from 160 to 192 rows.
    "1x2_fsdp->1x2_tp": (
        _world("1x2", "dp,fsdp", ("--fsdp",), "bh_1_2_line_line.textproto"),
        _world("1x2", "dp,tp", ("--tp",), "bh_1_2_line_line.textproto"),
        2,
    ),
    # TP2 -> one device, with the fused qkv bias and LoRA's fused lora_B trained.
    "1x2_tp_bias_lora->single": (
        _world("1x2", "dp,tp", ("--tp", "--bias", "--lora"), "bh_1_2_line_line.textproto"),
        _world("1x1", "dp,tp", ("--bias", "--lora"), None),
        2,
    ),
    "1x2_tp_tied->1x2_fsdp": (
        _world("1x2", "dp,tp", ("--tp", "--tie"), "bh_1_2_line_line.textproto"),
        _world("1x2", "dp,fsdp", ("--fsdp", "--tie"), "bh_1_2_line_line.textproto"),
        2,
    ),
}


_NOISE = re.compile(
    r"nanobind|leaked (instance|type|function|keep_alive)|refleaks|skipped remainder|\| DEBUG |^Config\{"
)


def _run(args: list[str], what: str, expect_ok: bool = True) -> str:
    env = dict(os.environ)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    proc = subprocess.run(DRIVER + args, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S, env=env)
    # The interpreter's exit prints a long nanobind leak report after any traceback; keep the lines that matter.
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if not _NOISE.search(ln)]
    if proc.returncode == MESH_OPEN_FAILED:  # only the descriptor-fabric mismatch: the QuietBox 1x4 ring on a loudbox
        reason = next((ln for ln in lines if ln.startswith("MESH_OPEN_FAILED")), "no reason given")
        pytest.skip(f"{what}: this host cannot open the mesh: {reason}")
    for line in lines:  # the measured differences, for the log
        if "max |diff| / max |ref|" in line:
            print(f"{what}: {line}")
    tail = "\n".join(lines[-60:])
    assert proc.returncode == (0 if expect_ok else 1), f"{what} failed with exit {proc.returncode}:\n{tail}"
    return "\n".join(lines)


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
def test_checkpoint_loads_onto_another_world(
    tmp_path_factory, release_metal_env, skip_if_host_too_small, mgd_path, case
):
    save_world, load_world, needed = CASES[case]
    skip_if_host_too_small((needed,), case)
    save_world, load_world = _resolve(save_world, mgd_path), _resolve(load_world, mgd_path)
    release_metal_env()  # the children open the chips
    ckpt = _saved_checkpoint(save_world, tmp_path_factory, f"{case} save")
    out = _run(["load", "--dir", ckpt, "--steps", "2", *load_world], f"{case} load")

    assert "RESULT: OK" in out, out.splitlines()[-5:]


def test_same_world_through_the_driver_matches_exactly(tmp_path, release_metal_env, skip_if_host_too_small, mgd_path):
    skip_if_host_too_small((2,), "the same-world driver round trip")
    release_metal_env()
    world = _resolve(_world("1x2", "dp,tp", ("--tp",), "bh_1_2_line_line.textproto"), mgd_path)
    ckpt = str(tmp_path / "step_0000002")
    _run(["save", "--dir", ckpt, "--steps", "2", *world], "save")
    out = _run(["load", "--dir", ckpt, "--steps", "2", "--same-layout", *world], "load")
    assert "RESULT: OK" in out


def test_a_tp2_checkpoint_with_its_fused_declarations_stripped_is_refused_on_one_device(
    tmp_path, release_metal_env, skip_if_host_too_small, mgd_path
):
    """Without its fused_segments declarations a TP2 checkpoint holds its fused rows in TP2's per-shard order and its
    index says nothing about it; loaded onto one device it would scramble Q, K and V, so the load refuses it."""
    skip_if_host_too_small((2,), "the stripped-declarations case")
    release_metal_env()
    save_world = _resolve(_world("1x2", "dp,tp", ("--tp", "--strip-fused"), "bh_1_2_line_line.textproto"), mgd_path)
    ckpt = str(tmp_path / "step_0000002")
    _run(["save", "--dir", ckpt, "--steps", "2", "--no-resume-step", *save_world], "save")
    out = _run(["load", "--dir", ckpt, "--steps", "2", *_world("1x1", "dp,tp", (), None)], "load", False)
    assert "LayoutError" in out and "its name suggests a fused projection" in out, out.splitlines()[-8:]


def test_a_scrambled_fused_weight_fails_the_logits_check(tmp_path, release_metal_env, skip_if_host_too_small, mgd_path):
    """The negative control of the logits tolerance: a TP2 checkpoint loads onto one device, then the first block's K
    and V rows are swapped, what reading the fused weight in the wrong order does, and the logits must catch it."""
    skip_if_host_too_small((2,), "the logits negative control")
    release_metal_env()
    save_world = _resolve(_world("1x2", "dp,tp", ("--tp",), "bh_1_2_line_line.textproto"), mgd_path)
    ckpt = str(tmp_path / "step_0000002")
    _run(["save", "--dir", ckpt, "--steps", "2", "--no-resume-step", *save_world], "save")
    out = _run(
        ["load", "--dir", ckpt, "--steps", "2", *_world("1x1", "dp,tp", ("--scramble-qkv",), None)], "load", False
    )
    assert "MISMATCH: logits differ" in out, out.splitlines()[-8:]


def test_non_finite_values_fail_the_comparison_on_either_side():
    """NaN compares false with everything, so a tolerance alone would pass a world whose logits or post-step moments
    are NaN. The comparison names non-finite values instead, exact or not, in the loaded world or the reference."""
    reference = {"logits": np.ones((2, 3), np.float32), RESUME + "state:m": np.ones(4, np.float32)}
    assert compare(reference, {k: v.copy() for k, v in reference.items()}, LOGITS_TOLERANCE) == []
    for key in reference:
        for tolerance in (0.0, LOGITS_TOLERANCE):
            for value in (np.nan, np.inf):
                broken = {k: v.copy() for k, v in reference.items()}
                broken[key].flat[0] = value
                assert any("non-finite" in p for p in compare(broken, reference, tolerance)), (key, tolerance, value)
                assert any("non-finite" in p for p in compare(reference, broken, tolerance)), (key, tolerance, value)
