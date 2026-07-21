# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-host TTNN graph-report test for CCL / collective ops.

This is the single-host analog of ``test_multihost_graph_report.py``. It exists to
validate the fix for ttnn-visualizer issue #1684 *without* needing a multi-host
(fabric-cabled) setup: the underlying bug is in graph *capture*, not in cross-host
data movement, so it reproduces on any single host with >= 2 devices.

Root cause (see ``GraphTracker::wrap_with_current_context``): graph-capture state is
thread_local. CCL collectives such as ``all_gather`` produce heterogeneous
MeshWorkloads whose compile/dispatch is offloaded onto the dispatch thread pool.
Before the fix, those worker threads ran with an empty thread-local processor list,
so the collective's events (including its ``function_end``) were silently dropped —
leaving an unbalanced ``function_start`` that the importer records as an
``incomplete_operation`` error instead of a finished op.

This test runs a small ``all_gather`` (plus a few per-device ops) inside a graph
capture on a single multi-device mesh, imports the report into SQLite, and asserts
that the collective appears as a completed operation with no ``incomplete_operation``
error. It auto-skips on single-device machines (the ``mesh_device`` fixture skips
when more devices are requested than are physically present).

Run on any host with >= 2 Tenstorrent devices (e.g. n300, T3000, multi-chip BH):

    pytest tests/ttnn/distributed/test_singlehost_graph_report_ccl.py
"""

import sqlite3

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.graph_report import import_report

GATHER_DIM = 3
SHARD_WIDTH = 32  # one tile column per device

# Ops we expect to be captured as complete operations. The ``ttnn.all_gather``
# collective is the point of this test: its dispatch is offloaded to worker
# threads, but with context propagation (#1684) it is recorded as a finished op
# rather than an incomplete_operation error.
EXPECTED_OPS = ("ttnn.to_device", "ttnn.multiply", "ttnn.add", "ttnn.gelu", "ttnn.all_gather")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_singlehost_graph_report_ccl(mesh_device, tmp_path):
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip(f"CCL graph-report test needs >= 2 devices, got {num_devices}")

    # --- Build an input sharded across the (line) mesh along the gather dim -----
    torch.manual_seed(0)
    torch_input = torch.cat(
        [torch.rand(1, 1, 32, SHARD_WIDTH).bfloat16() for _ in range(num_devices)],
        dim=GATHER_DIM,
    )  # [1, 1, 32, 32*num_devices]

    host_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=GATHER_DIM),
    )

    report_path = tmp_path / "graph_capture.json"
    output_dir = tmp_path / "output"

    # --- Capture a graph over a few real ops plus the collective ----------------
    ttnn.graph.enable_detailed_buffer_tracing()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        tt_input = ttnn.to_device(host_input, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.mul(tt_input, 2.0)
        x = ttnn.add(x, tt_input)  # == 3 * input
        x = ttnn.gelu(x)  # == gelu(3 * input)
        ttnn.synchronize_device(mesh_device)

        # Line-topology all_gather across the whole mesh. Dispatch of this
        # collective is offloaded to worker threads; the fix ensures its capture
        # events are still recorded (see module docstring).
        gathered = ttnn.all_gather(x, GATHER_DIM, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.graph.end_graph_capture_to_file(str(report_path))
        ttnn.graph.disable_detailed_buffer_tracing()
    logger.info(f"wrote capture -> {report_path}")

    # --- Light correctness check ------------------------------------------------
    torch_reference = torch.nn.functional.gelu(3.0 * torch_input.float())
    local_outputs = ttnn.get_device_tensors(gathered)
    assert len(local_outputs) > 0, "expected at least one local device tensor"
    for i, tt_out in enumerate(local_outputs):
        out = ttnn.to_torch(tt_out).float()
        assert out.shape == torch_reference.shape, f"dev {i}: {out.shape} != {torch_reference.shape}"

    # --- Import the report and assert the collective is a completed op ----------
    db_path = import_report(report_path, output_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        op_names = {name for (name,) in cur.execute("SELECT name FROM operations").fetchall()}
        missing = [op for op in EXPECTED_OPS if op not in op_names]
        assert not missing, f"missing expected ops {missing}; captured: {sorted(op_names)}"

        # The whole point of #1684: the collective must not surface as an
        # incomplete_operation (its function_end is now captured on the worker
        # thread via context propagation).
        incomplete = cur.execute(
            "SELECT operation_name FROM errors WHERE error_type = 'incomplete_operation'"
        ).fetchall()
        assert not incomplete, f"unexpected incomplete_operation errors (collective end dropped?): {incomplete}"
    finally:
        conn.close()

    logger.info(f"single-host CCL graph report verified at {db_path}")
