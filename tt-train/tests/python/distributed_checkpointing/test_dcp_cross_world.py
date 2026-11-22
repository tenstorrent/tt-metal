# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint saved on one parallelism strategy loads onto another: each world in its own process.

A tensor-parallel model needs the process-wide ParallelismContext, which cannot be reset, so every world runs
through ``dcp_driver.py`` (this directory) in a child process. The save child trains a tiny Llama (vocab
130, so the padded vocab differs between TP sizes), saves it and writes a reference of every tensor in canonical
layout plus the logits on a fixed batch; the load child builds the destination world, loads, and compares. This
module holds no device fixture of its own: the children need the chips.

The tt-run modules (``test_dcp_multirank``, ``test_dcp_multirank_uneven_failure``, ``test_dcp_multimesh``) skip under
plain pytest; this module runs each of them whole, in a tt-run world of its own, on two ranks of this host, so CI runs
every one of their tests.
"""

import json
import os
import socket
import sys

import numpy as np
import pytest
from dcp_driver import LOGITS_TOLERANCE, RESUME, compare
from dcp_harness import DRIVER, HERE, TT_RUN, output_lines, run_checked, run_world, tt_run_command

# Each child is one process opening a mesh, training or loading a tiny Llama: in CI the whole module takes about
# 3.5 minutes and its slowest case (two children) 78 s, while on a host reading its files over NFS one child has taken
# up to 135 s. A hung child fails after 3 minutes, so one hang still leaves the 30-minute QB2 python leg the time to
# report the tests after it.
CHILD_TIMEOUT_S = 180
pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(2 * CHILD_TIMEOUT_S + 60)]

# Two MPI ranks on one host, each with its own [2, 2] mesh (mesh ids 0 and 1), eight chips in all. One [1, 4] mesh
# spanning both processes (the ttnn dual-rank descriptor) is not used here: the driver's Llama hangs on it in a
# reduce-scatter whose line crosses the processes, as in #65032 (test_dcp_multirank covers the protocol on it).
TWO_MESHES_MGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bh_two_2x2_meshes.textproto")


def _world(mesh: str, axes: str, flags: tuple[str, ...], mgd: str | None, ranks: int = 1):
    """The driver arguments of one world; ``mgd`` names a bundled descriptor, resolved by ``_resolve``. ``ranks > 1``
    launches the world through tt-run with ``TWO_MESHES_MGD``, one [2, 2] mesh per rank."""
    args = ["--mesh", mesh, "--axes", axes, *flags]
    if mgd:
        args += ["--mgd", mgd]
    if ranks > 1:
        args += ["--mgd", TWO_MESHES_MGD, "--ranks", str(ranks)]
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
    # Two ranks with one [2, 2] mesh each run the same DP x TP training independently (same seed, same batches, no
    # gradient sync between them), so every key is an identical replica on both ranks, as after a cross-mesh gradient
    # sync; one rank is elected to write each box, and rank 0 merges the index. Eight chips, so a four-chip host skips
    # these.
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


def _command(args: list[str]) -> list[str]:
    """The driver command line; a world with ``--ranks N`` runs under tt-run, every rank on this host."""
    if "--ranks" not in args:
        return DRIVER + args
    args = list(args)
    at = args.index("--ranks")
    del args[at : at + 2]
    return tt_run_command(args[args.index("--mgd") + 1], socket.gethostname(), DRIVER + args)


def _run(args: list[str], what: str, expect_ok: bool = True) -> str:
    lines = run_checked(
        _command(args),
        what,
        CHILD_TIMEOUT_S,
        needs_tt_run="--ranks" in args,
        save=args[0] == "save",
        expect_ok=expect_ok,
    )
    for line in lines:  # the measured differences, for the log
        if "max |diff| / max |ref|" in line:
            print(f"{what}: {line}")
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


def test_stripped_fused_declarations_are_refused(tmp_path, release_metal_env, skip_if_host_too_small, mgd_path):
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


DUAL_RANK_MGD = "tests/ttnn/distributed/config/bh_1x4_dual_rank_mesh_graph_descriptor.textproto"
TWO_MESH_MGD = os.path.join(HERE, "bh_two_1x2_meshes_one_host.textproto")
# A tt-run module runs its two-rank tests in one world; on a QuietBox 2 each module takes under three minutes.
MODULE_TIMEOUT_S = 600


def _run_module_on_two_ranks(module: str, mgd: str, env: dict, results: str) -> None:
    """Run the tt-run test module ``module`` whole on two ranks of this host, and require every one of its tests to
    pass on both ranks: none may fail or skip. Each rank reports its outcome in a file in ``results`` (the
    ``pytest_sessionfinish`` of this directory's conftest), since tt-run can interleave the ranks' output lines."""
    if TT_RUN is None:
        pytest.skip("tt-run is not installed")
    if not os.path.isfile(mgd):
        pytest.skip(f"no descriptor {mgd}")
    os.makedirs(results, exist_ok=True)
    env = dict(env, DCP_RANK_RESULTS_DIR=results)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    command = [sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", os.path.join(HERE, module)]
    proc = run_world(tt_run_command(mgd, socket.gethostname(), command), env, MODULE_TIMEOUT_S)
    tail = "\n".join(output_lines(proc)[-60:])
    for rank in (0, 1):
        path = os.path.join(results, f"rank-{rank}.json")
        assert os.path.isfile(path), f"{module}: rank {rank} reported no outcome (exit {proc.returncode}):\n{tail}"
        with open(path) as f:
            outcome = json.load(f)
        print(f"{module} rank {rank}: {outcome}")
        bad = {k: outcome[k] for k in ("failed", "error", "skipped", "xfailed", "xpassed") if outcome[k]}
        assert outcome["exitstatus"] == 0 and outcome["passed"] and not bad, f"{module} rank {rank}: {outcome}\n{tail}"


@pytest.mark.timeout(MODULE_TIMEOUT_S + 2 * CHILD_TIMEOUT_S + 60)
def test_two_ranks_on_one_mesh_load_on_one_process(tmp_path, release_metal_env, skip_if_host_too_small):
    """The test_dcp_multirank module, whole: two ranks co-owning one [1, 4] mesh, every test passing on both. Its
    source test saves a checkpoint whose keys' chunks split between the ranks' files (the other two-rank cases give
    each rank a whole mesh, so each key is written by one rank); one process with one device then loads it, and every
    parameter must equal the global array rebuilt from the two ranks' own shards."""
    skip_if_host_too_small((4,), "the dual-rank module")
    release_metal_env()
    mgd = os.path.join(os.environ.get("TT_METAL_HOME", os.getcwd()), DUAL_RANK_MGD)
    env = dict(os.environ, DCP_CROSS_WORLD_DIR=str(tmp_path))
    _run_module_on_two_ranks("test_dcp_multirank.py", mgd, env, str(tmp_path / "results"))
    assert os.path.isfile(tmp_path / "shards_1.npz"), "the source test wrote no shards"
    out = _run(["load-tp-block", "--dir", str(tmp_path), "--mesh", "1x1", "--axes", "dp,tp"], "load")
    assert "RESULT: OK" in out, out.splitlines()[-5:]


@pytest.mark.timeout(MODULE_TIMEOUT_S + 60)
def test_the_two_mesh_module_passes_on_two_ranks(tmp_path, release_metal_env, skip_if_host_too_small):
    """The test_dcp_multimesh module, whole: two ranks with one [1, 2] mesh each, every test passing on both."""
    skip_if_host_too_small((4,), "the two-mesh module")
    release_metal_env()
    _run_module_on_two_ranks("test_dcp_multimesh.py", TWO_MESH_MGD, dict(os.environ), str(tmp_path))


@pytest.mark.timeout(MODULE_TIMEOUT_S + 60)
def test_the_uneven_load_failure_module_passes_on_two_ranks(tmp_path, release_metal_env, skip_if_host_too_small):
    """The test_dcp_multirank_uneven_failure module: its load leaves the two ranks' device allocations out of step, so
    it runs in a tt-run world of its own, where nothing runs after it."""
    skip_if_host_too_small((4,), "the uneven-failure module")
    release_metal_env()
    mgd = os.path.join(os.environ.get("TT_METAL_HOME", os.getcwd()), DUAL_RANK_MGD)
    _run_module_on_two_ranks("test_dcp_multirank_uneven_failure.py", mgd, dict(os.environ), str(tmp_path))


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
