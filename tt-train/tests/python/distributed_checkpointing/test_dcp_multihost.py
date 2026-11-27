# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints across hosts: a world of several hosts saves, a world with another host count, mesh or parallelism
loads, and every tensor comes back. Launched by hand, never in CI: it needs hosts to run on and a shared directory.

    DCP_HOSTS=hostA,hostB DCP_SHARED_TMP=/path/to/a/shared/dir \\
        pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multihost.py

``DCP_HOSTS`` lists the hosts tt-run launches on (one rank per host); ``DCP_SHARED_TMP`` is a directory every host
sees. In a two-node Slurm allocation, ``DCP_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)``
lists them.

Two shapes of multi-host world, each a parametrized set of save/load pairs through ``dcp_driver.py``:

* **One mesh per host** (two loudboxes, no fabric between them): each host opens its own ``[2, 2]`` mesh under
  ``bh_two_2x2_meshes.textproto`` and trains its own copy of the tiny Llama (same seed and batches, no gradient sync
  between the hosts), so the ranks hold identical replicas of every key, one of them elected to write it. The load
  worlds have one host or two, DP×TP or FSDP, so the host count, the mesh and the parallelism all change under the
  checkpoint.
* **One mesh over several hosts** (linked galaxies, Big-Mesh): set ``DCP_BIGMESH_MGD`` to the descriptor,
  ``DCP_BIGMESH_SHAPE`` to its mesh (``16x4``) and ``DCP_BIGMESH_HOSTS`` to the hosts; the pairs then save on the big
  mesh with DP×TP and load on the first host alone with FSDP, and back. Skipped unless those are set.

Each pair runs the save world and the load world as separate tt-run launches; rank 0 of the load world compares
every parameter and moment in canonical layout against the saving world's shards, so a two-host checkpoint is
checked against a one-host world and the other way round. A world that overruns its budget is terminated through
its process group, so mpirun takes the ranks down on every host; a device that hung needs a reset before the next
launch (``TT_METAL_OPERATION_TIMEOUT_SECONDS``, forwarded through ``DCP_MPI_ARGS``, turns a hang into an error
that names the stuck operation). Without ``DCP_HOSTS`` the module skips.
"""

import os
import shutil
import uuid

import pytest

from dcp_harness import DRIVER, HERE, MGD_DIR, run_checked, tt_run_command

pytestmark = [pytest.mark.timeout(7200)]

TWO_MESHES = os.path.join(HERE, "bh_two_2x2_meshes.textproto")
HOSTS = [h for h in os.environ.get("DCP_HOSTS", "").split(",") if h]
SHARED = os.environ.get("DCP_SHARED_TMP")
BIGMESH_MGD = os.environ.get("DCP_BIGMESH_MGD")
BIGMESH_SHAPE = os.environ.get("DCP_BIGMESH_SHAPE", "16x4")
BIGMESH_HOSTS = [h for h in os.environ.get("DCP_BIGMESH_HOSTS", ",".join(HOSTS)).split(",") if h]


def _world(mesh, axes, flags, mgd, hosts):
    """A world: driver arguments plus the hosts that run it (one rank per host; one host means one process)."""
    args = ["--mesh", mesh, "--axes", axes, *flags, "--mgd", mgd]
    return {"args": args, "hosts": list(hosts)}


def _pairs():
    if len(HOSTS) < 2:
        return {}
    one, two = HOSTS[:1], HOSTS[:2]
    mgd22 = os.path.join(MGD_DIR, "bh_2_2_line_line.textproto")
    pairs = {
        "2hosts_2x2_dp_tp->1host_2x2_fsdp": (
            _world("2x2", "dp,tp", ("--tp",), TWO_MESHES, two),
            _world("2x2", "dp,fsdp", ("--fsdp",), mgd22, one),
        ),
        "2hosts_2x2_fsdp->2hosts_2x2_dp_tp": (
            _world("2x2", "dp,fsdp", ("--fsdp",), TWO_MESHES, two),
            _world("2x2", "dp,tp", ("--tp",), TWO_MESHES, two),
        ),
        "1host_2x2_dp_tp->2hosts_2x2_fsdp": (
            _world("2x2", "dp,tp", ("--tp",), mgd22, one),
            _world("2x2", "dp,fsdp", ("--fsdp",), TWO_MESHES, two),
        ),
        "2hosts_2x2_dp_tp->2hosts_2x2_dp_tp_same": (
            _world("2x2", "dp,tp", ("--tp",), TWO_MESHES, two),
            _world("2x2", "dp,tp", ("--tp", "--same-layout"), TWO_MESHES, two),
        ),
    }
    if BIGMESH_MGD and len(BIGMESH_HOSTS) >= 2:
        rows, cols = (int(x) for x in BIGMESH_SHAPE.lower().split("x"))
        first = BIGMESH_HOSTS[:1]
        single = f"{rows * cols // len(BIGMESH_HOSTS)}x1"
        # The galaxy model: hidden 1024, 8 heads, 8 KV heads, vocab 1000, so every 32-way FSDP shard is tile-aligned
        # and the heads divide by the TP size, the big mesh's column count.
        sizes = ("--hidden", "1024", "--heads", "8", "--kv-heads", "8", "--vocab", "1000")
        single_mgd = os.environ.get(
            "DCP_BIGMESH_SINGLE_MGD", os.path.join(MGD_DIR, "bh_galaxy_32_1_ring_ring.textproto")
        )
        # The load world only runs a forward pass on the big mesh; the save world's gradient sync spans the hosts,
        # and a collective that hangs there leaves the chips needing a reset, so the pair that loads runs first.
        # The training step the driver takes after a save or a load syncs gradients over the DP axis, whose lines
        # cross the hosts on the big mesh, and that collective hangs there today (#65032): the pairs that touch the
        # big mesh skip the step on both of their worlds. The pair that saves on the big mesh cannot skip it: its
        # save world trains first, so it is skipped until #65032 is fixed (see KNOWN_HANGS).
        no_step = ("--no-resume-step",)
        pairs["1host_fsdp->bigmesh_dp_tp"] = (
            _world(single, "fsdp,dp", ("--fsdp", *sizes, *no_step), single_mgd, first),
            _world(BIGMESH_SHAPE, "dp,tp", ("--tp", *no_step), BIGMESH_MGD, BIGMESH_HOSTS),
        )
        pairs["bigmesh_dp_tp->1host_fsdp"] = (
            _world(BIGMESH_SHAPE, "dp,tp", ("--tp", *sizes, *no_step), BIGMESH_MGD, BIGMESH_HOSTS),
            _world(single, "fsdp,dp", ("--fsdp", *no_step), single_mgd, first),
        )
    return pairs


PAIRS = _pairs()
# Pairs that cannot pass today, skipped with the reason instead of hanging until the budget and leaving the chips to
# be reset: the save world on the big mesh trains, and its gradient sync across the hosts hangs.
KNOWN_HANGS = {
    "bigmesh_dp_tp->1host_fsdp": "#65032: training on a mesh spanning hosts hangs in the cross-host gradient sync, "
    "so the big-mesh save world never reaches its save",
}


def _command(world, command, ckpt, steps):
    args = [command, "--dir", ckpt, "--steps", str(steps), *world["args"]]
    if len(world["hosts"]) == 1:
        return DRIVER + args
    mgd = world["args"][world["args"].index("--mgd") + 1]
    # DCP_MPI_ARGS adds mpirun options, e.g. "-x TT_METAL_OPERATION_TIMEOUT_SECONDS" to forward a dispatch timeout
    # (and its triage command) to every rank; see tools/tt-triage.py.
    return tt_run_command(mgd, ",".join(world["hosts"]), DRIVER + args, os.environ.get("DCP_MPI_ARGS", "--tag-output"))


def _run(world, command, ckpt, steps, what):
    command_line = _command(world, command, ckpt, steps)
    lines = run_checked(command_line, what, 3000, needs_tt_run=len(world["hosts"]) > 1, save=command == "save")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def shared_root():
    if len(HOSTS) < 2:
        pytest.skip("set DCP_HOSTS=hostA,hostB (and DCP_SHARED_TMP) to run the multi-host cases")
    if not SHARED:
        pytest.skip("set DCP_SHARED_TMP to a directory every host sees")
    root = os.path.join(SHARED, f"dcp-multihost-{uuid.uuid4().hex[:8]}")
    os.makedirs(root, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


_SAVED: dict[tuple, str] = {}


def _saved(save_world, shared_root, what):
    key = (tuple(save_world["args"]), tuple(save_world["hosts"]))
    if key not in _SAVED:
        ckpt = os.path.join(shared_root, f"save{len(_SAVED)}", "step_0000002")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        _run(save_world, "save", ckpt, 2, what)
        _SAVED[key] = ckpt
    return _SAVED[key]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=pytest.mark.skip(reason=KNOWN_HANGS[case])) if case in KNOWN_HANGS else case
        for case in PAIRS
    ],
)
def test_checkpoint_crosses_hosts(shared_root, case):
    save_world, load_world = PAIRS[case]
    ckpt = _saved(save_world, shared_root, f"{case} save")
    out = _run(load_world, "load", ckpt, 2, f"{case} load")
    # Kept in the report (-rA): whether the saving world's replicas agreed across its ranks, and the logits.
    for line in out.splitlines():
        if line.startswith(("NOTE", "logits:", "RESULT")) or "<stdout>: NOTE" in line or "<stdout>: logits:" in line:
            print(f"{case}: {line.split('<stdout>: ')[-1]}")
    assert "RESULT: OK" in out, "\n".join(out.splitlines()[-12:])
