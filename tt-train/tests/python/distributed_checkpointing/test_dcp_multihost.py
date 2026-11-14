# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints across hosts: a world of several hosts saves, a world with another host count, mesh or parallelism
loads, and every tensor comes back. Launched by hand, never in CI: it needs hosts to run on and a shared directory.

    DCP_HOSTS=hostA,hostB DCP_SHARED_TMP=/data/<you>/dcp_runs \\
        pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multihost.py

``DCP_HOSTS`` lists the hosts tt-run launches on (one rank per host); ``DCP_SHARED_TMP`` is a directory every host
sees. Under Slurm, ``/data/awliu/dcp_multihost.sbatch`` asks for two nodes and sets both.

Two shapes of multi-host world, each a parametrized set of save/load pairs through ``dcp_driver.py``:

* **One mesh per host** (two loudboxes, no fabric between them; the pipeline-parallel and hierarchical shape):
  each host opens its own ``[2, 2]`` mesh under ``bh_two_2x2_meshes.textproto``, trains the tiny Llama with the
  same seed, and the ranks hold replicas of every key that one of them is elected to write. The load worlds have
  one host or two, DP×TP or FSDP, so the host count, the mesh and the parallelism all change under the checkpoint.
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
import re
import shutil
import sys
import uuid

import pytest

import ttml
from dcp_driver import MESH_OPEN_FAILED
from dcp_harness import run_world

pytestmark = [pytest.mark.timeout(7200)]

HERE = os.path.dirname(os.path.abspath(__file__))
MGD_DIR = os.path.normpath(os.path.join(os.path.dirname(ttml.__file__), "..", "..", "..", "configs", "mgd"))
TWO_MESHES = os.path.join(HERE, "bh_two_2x2_meshes.textproto")
DRIVER = [sys.executable, os.path.join(HERE, "dcp_driver.py")]
TT_RUN = shutil.which("tt-run")
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
        # cross the hosts on the big mesh, and that collective hangs there today (the open fabric issue): the pairs
        # that touch the big mesh skip the step on both of their worlds.
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
_NOISE = re.compile(
    r"nanobind|leaked (instance|type|function|keep_alive)|refleaks|skipped remainder|\| DEBUG |^Config\{"
)


def _command(world, command, ckpt, steps):
    args = [command, "--dir", ckpt, "--steps", str(steps), *world["args"]]
    if len(world["hosts"]) == 1:
        return DRIVER + args
    mgd = world["args"][world["args"].index("--mgd") + 1]
    # DCP_MPI_ARGS adds mpirun options, e.g. "-x TT_METAL_OPERATION_TIMEOUT_SECONDS" to forward a dispatch timeout
    # (and its triage command) to every rank; see tools/tt-triage.py.
    mpi_args = os.environ.get("DCP_MPI_ARGS", "--tag-output")
    hosts = ",".join(world["hosts"])
    return [TT_RUN, "--mesh-graph-descriptor", mgd, "--hosts", hosts, "--mpi-args", mpi_args, *DRIVER, *args]


def _run(world, command, ckpt, steps, what):
    if len(world["hosts"]) > 1 and TT_RUN is None:
        pytest.skip(f"{what}: tt-run is not installed")
    env = dict(os.environ)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    proc = run_world(_command(world, command, ckpt, steps), env, 3000)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if not _NOISE.search(ln)]
    if proc.returncode == MESH_OPEN_FAILED or any("MESH_OPEN_FAILED" in ln for ln in lines):
        reason = next((ln for ln in lines if "MESH_OPEN_FAILED" in ln), "no reason given")
        pytest.skip(f"{what}: a host cannot open the mesh: {reason}")
    tail = "\n".join(lines[-80:])
    assert proc.returncode == 0, f"{what} failed with exit {proc.returncode}:\n{tail}"
    if command == "save":
        assert '"saved"' in proc.stdout, f"{what} printed no summary:\n{tail}"
    return proc.stdout


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


@pytest.mark.parametrize("case", list(PAIRS))
def test_checkpoint_crosses_hosts(shared_root, case):
    save_world, load_world = PAIRS[case]
    ckpt = _saved(save_world, shared_root, f"{case} save")
    out = _run(load_world, "load", ckpt, 2, f"{case} load")
    # Kept in the report (-rA): whether the saving world's replicas agreed across its ranks, and the logits.
    for line in out.splitlines():
        if line.startswith(("NOTE", "logits:", "RESULT")) or "<stdout>: NOTE" in line or "<stdout>: logits:" in line:
            print(f"{case}: {line.split('<stdout>: ')[-1]}")
    assert "RESULT: OK" in out, "\n".join(out.splitlines()[-12:])
