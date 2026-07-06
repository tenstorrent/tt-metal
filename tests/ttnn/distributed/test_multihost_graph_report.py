# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-host TTNN graph-report test.

This is the unit-test form of the earlier ``models/demos/multihost_poc`` proof of
concept. It verifies that the TTNN graph report (the data ttnn-visualizer reads)
is correctly captured and *merged* across more than one physical host.

What it does on every rank (one MPI process per host):
  1. Builds an identical input sharded across the full global mesh.
  2. Captures a graph (``begin_graph_capture`` / ``end_graph_capture_to_file``) while
     running a few per-device ops (``to_device``, ``mul``, ``add``, ``gelu``) plus one
     cross-host collective (``all_gather``).
  3. Writes its own per-rank capture JSON into a shared report dir.
  4. Rank 0 merges all per-rank JSONs into a single ``db.sqlite`` via ``import_report``.
  5. Rank 0 asserts the merged DB actually contains data from BOTH hosts: operations
     (per rank), tensors, and the buffer / buffer_chunks tables.

The test manages its own capture instead of relying on the autouse
``ttnn_graph_report`` conftest fixture, so it can assert on the resulting DB inside
the test body (the fixture only writes the report at teardown). Do NOT set
``enable_graph_report`` in the TTNN config when running this test, or the fixture
would open a competing capture.

Launch (2-host Blackhole loudbox), see ``run_multihost_graph_report.sh``:

    tt-run --rank-binding tests/ttnn/distributed/config/bh_lbx2_1x16_rank_bindings.yaml \\
           --mpi-args "--hostfile $HOSTFILE" \\
           pytest tests/ttnn/distributed/test_multihost_graph_report.py

When run on a single host it skips cleanly.

Note: the cross-host ``all_gather`` (a CCL collective) runs correctly, but its
graph ``function_end`` is currently not recorded by the capture (it completes on a
different thread than the one that owns the thread-local capture). It therefore
shows up as an ``incomplete_operation`` rather than a finished op, so we do not
assert it as a completed op here.
"""

import os
import shutil
import sqlite3
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.graph_report import import_report

NUM_DEVICES = 16  # 1x16 global mesh: 8 BH chips on each of 2 hosts
CLUSTER_AXIS = 1
GATHER_DIM = 3
SHARD_WIDTH = 32  # one tile column per device

# Per-device ops we expect to be captured as complete operations (in dispatch order).
EXPECTED_OPS = ("ttnn.to_device", "ttnn.multiply", "ttnn.add", "ttnn.gelu")


def _report_dir() -> Path:
    """Shared (NFS) report dir all ranks write into; rank 0 reads them all back."""
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    return Path(home) / "generated" / "ttnn" / "reports" / "test_multihost_graph_report"


def _table_count(cur: sqlite3.Cursor, table: str) -> int:
    return int(cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, NUM_DEVICES), id="1x16_grid")], indirect=True)
def test_multihost_graph_report(mesh_device):
    # --- 0. Only meaningful across >=2 hosts ------------------------------------
    if not ttnn.using_distributed_env():
        pytest.skip(
            "Multi-host graph report test: launch across >=2 hosts with tt-run "
            "(see tests/ttnn/distributed/run_multihost_graph_report.sh)."
        )

    rank = int(ttnn.distributed_context_get_rank())
    world_size = int(ttnn.distributed_context_get_size())
    assert world_size >= 2, f"expected at least 2 hosts, got world_size={world_size}"

    num_devices = mesh_device.get_num_devices()  # global mesh size (16)
    report_dir = _report_dir()

    # --- 1. Start from a clean shared report dir (rank 0 owns it) ----------------
    ttnn.distributed_context_barrier()
    if rank == 0:
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir.mkdir(parents=True, exist_ok=True)
    ttnn.distributed_context_barrier()  # dir exists before anyone writes

    # --- 2. Build identical, sharded input on every rank ------------------------
    torch.manual_seed(0)
    torch_input = torch.cat(
        [torch.rand(1, 1, 32, SHARD_WIDTH).bfloat16() for _ in range(num_devices)],
        dim=GATHER_DIM,
    )  # [1, 1, 32, 32*16]

    # Multi-host safe placement: build the sharded tensor on HOST first (no device=),
    # then to_device. Passing device= to from_torch would distribute to all 16 devices
    # in one shot, touching the other host's remote chips.
    host_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=GATHER_DIM),
    )

    # --- 3. Capture a graph over a few real ops ---------------------------------
    # NORMAL mode actually runs the ops on device so buffers are allocated and the
    # detailed buffer report (buffers / buffer_chunks) has something to record.
    ttnn.graph.enable_detailed_buffer_tracing()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        tt_input = ttnn.to_device(host_input, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.mul(tt_input, 2.0)
        x = ttnn.add(x, tt_input)  # == 3 * input
        x = ttnn.gelu(x)  # == gelu(3 * input)
        ttnn.synchronize_device(mesh_device)

        # Realistic cross-host collective: reconstructs the full tensor on every device,
        # which requires the two hosts to move data over TT-fabric.
        gathered = ttnn.all_gather(
            x,
            GATHER_DIM,
            cluster_axis=CLUSTER_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=2,
        )
        ttnn.synchronize_device(mesh_device)
    finally:
        json_path = report_dir / f"graph_capture_{rank + 1}_of_{world_size}.json"
        ttnn.graph.end_graph_capture_to_file(str(json_path))
        ttnn.graph.disable_detailed_buffer_tracing()
    logger.info(f"[rank {rank}] wrote capture -> {json_path}")

    # --- 4. Merge all per-host captures into one DB (rank 0 only) ----------------
    ttnn.distributed_context_barrier()  # every rank's JSON is on disk
    if rank == 0:
        import_report(report_dir, report_dir)
    ttnn.distributed_context_barrier()  # merge done before anyone inspects / exits

    # --- 5. Light correctness check (local shards only; no collective) ----------
    torch_reference = torch.nn.functional.gelu(3.0 * torch_input.float())
    local_outputs = ttnn.get_device_tensors(gathered)
    assert len(local_outputs) > 0, f"rank {rank}: expected at least one local device tensor"
    for i, tt_out in enumerate(local_outputs):
        out = ttnn.to_torch(tt_out).float()
        assert out.shape == torch_reference.shape, f"rank {rank} dev {i}: {out.shape} != {torch_reference.shape}"

    # --- 6. Assert the merged report actually has multi-host data (rank 0) -------
    # This is the whole point: data from BOTH hosts present in one db.sqlite.
    # Runs after all barriers so a failure here can never deadlock the other ranks.
    if rank != 0:
        return

    db_path = report_dir / "db.sqlite"
    assert db_path.exists(), f"merged report DB not created at {db_path}"

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        world = int(cur.execute("SELECT value FROM report_metadata WHERE key = 'world_size'").fetchone()[0])
        assert world == world_size, f"report world_size={world}, expected {world_size}"

        op_rows = cur.execute("SELECT rank, name FROM operations").fetchall()
        ranks_with_ops = {r for (r, _) in op_rows}
        assert ranks_with_ops == set(range(world_size)), (
            f"expected captured operations from every rank {set(range(world_size))}, " f"got {sorted(ranks_with_ops)}"
        )

        # Every host must have captured each expected per-device op.
        for r in range(world_size):
            names = {name for (rr, name) in op_rows if rr == r}
            missing = [op for op in EXPECTED_OPS if op not in names]
            assert not missing, f"rank {r} missing expected ops {missing}; captured: {sorted(names)}"

        n_tensors = _table_count(cur, "tensors")
        n_buffers = _table_count(cur, "buffers")
        n_chunks = _table_count(cur, "buffer_chunks")
        logger.info(
            f"[merged report] {len(op_rows)} operations, {n_tensors} tensors, "
            f"{n_buffers} buffers, {n_chunks} buffer_chunks"
        )

        assert n_tensors > 0, "tensors table is empty"
        assert n_buffers > 0, "buffers table is empty"
        assert n_chunks > 0, "buffer_chunks table is empty"
    finally:
        conn.close()

    logger.info(f"[rank 0] multi-host graph report verified at {db_path}")
