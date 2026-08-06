# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-report regression test for work offloaded to dispatch worker threads.

Graph-capture state (``GraphTracker``'s processor stack) is ``thread_local``. Work that a
mesh operation hands to the dispatch thread pool therefore used to run with an empty
processor list and silently drop every tracking event it fired. The clearest instance is
``MeshWorkloadImpl::compile``, which offloads per-device program compilation to the thread
pool whenever a ``MeshWorkload`` holds more than one program, and emits ``track_allocate_cb``
from those workers. When it does offload, *every* program compiles on a worker and none on
the calling thread, so the capture came back with zero circular-buffer allocations for the
op rather than merely fewer.

A collective is used here because that is where the gap was reported (ttnn-visualizer #1684),
but it is not collective-specific: the ``create_at`` adapter in ``ttnn/api/ttnn/device_operation.hpp``
adds one program per mesh coordinate, so any multi-device op on that path is affected.

``ThreadPool::enqueue`` now installs the enqueuing thread's processors on the worker for the
duration of each task, so those events land in the capture. This test asserts that the
``all_gather`` operation's captured subgraph contains circular-buffer allocation nodes, which
is the directly observable consequence of the fix. It also asserts the capture is balanced
(no ``incomplete_operation`` rows), which is the symptom reported in ttnn-visualizer #1684.

Runs on any single host with >= 2 local devices, and skips otherwise.

    pytest tests/ttnn/distributed/test_singlehost_graph_report_ccl.py
"""

import json
import sqlite3

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.graph_report import import_report

GATHER_DIM = 3
SHARD_WIDTH = 32  # one tile column per device

# The collective is the subject of the test; the elementwise ops around it just keep the
# capture representative of a real report.
COLLECTIVE_OP = "ttnn.all_gather"


@pytest.fixture(autouse=True)
def require_two_local_devices():
    """Skip before the ``mesh_device`` fixture tries to open the 1x2 mesh.

    conftest only skips an oversized mesh request when ``using_distributed_env()`` is
    false, since under multi-host MPI a mesh legitimately spans hosts. On a single-device
    box that happens to have a multi-rank context initialized, that carve-out lets the
    request through to ``open_mesh_device``, which raises TT_FATAL at setup instead of
    skipping. This test is single-host by construction, so gate it on the local count.

    Autouse so it is ordered ahead of ``mesh_device``, which fails during its own setup
    and would leave no opportunity to skip from the test body.
    """
    available = ttnn.get_num_devices()
    if available < 2:
        pytest.skip(f"needs >= 2 local devices to build a 1x2 mesh, found {available}")


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=["device_params"], ids=["fabric_1d"]
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_singlehost_graph_report_ccl(mesh_device, tmp_path):
    num_devices = mesh_device.get_num_devices()

    torch.manual_seed(0)
    torch_input = torch.cat(
        [torch.rand(1, 1, 32, SHARD_WIDTH).bfloat16() for _ in range(num_devices)],
        dim=GATHER_DIM,
    )  # [1, 1, 32, 32*num_devices]

    host_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=GATHER_DIM),
    )

    report_path = tmp_path / "graph_capture.json"
    output_dir = tmp_path / "output"

    # Detailed buffer tracing is a process-global flag, so each acquire gets its own finally:
    # nesting them keeps a failure in begin_graph_capture from leaking the flag into the rest
    # of the pytest session, and from calling end_graph_capture without a matching begin.
    ttnn.graph.enable_detailed_buffer_tracing()
    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            tt_input = ttnn.to_device(host_input, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.multiply(tt_input, 2.0)
            x = ttnn.add(x, tt_input)  # == 3 * input
            x = ttnn.gelu(x)
            ttnn.synchronize_device(mesh_device)

            # Each device holds one shard along GATHER_DIM, so after the gather every device
            # holds the full tensor. Compiling this collective produces a heterogeneous
            # MeshWorkload, whose per-device compile runs on dispatch worker threads.
            gathered = ttnn.all_gather(x, GATHER_DIM, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(mesh_device)
        finally:
            ttnn.graph.end_graph_capture_to_file(str(report_path))
    finally:
        ttnn.graph.disable_detailed_buffer_tracing()
    logger.info(f"wrote capture -> {report_path}")

    # Every device must come back with the whole gathered tensor.
    torch_reference = torch.nn.functional.gelu(3.0 * torch_input.float())
    local_outputs = ttnn.get_device_tensors(gathered)
    assert len(local_outputs) == num_devices, f"expected {num_devices} shards, got {len(local_outputs)}"
    for device_index, tt_out in enumerate(local_outputs):
        assert_with_pcc(torch_reference, ttnn.to_torch(tt_out).float(), 0.99)
        logger.debug(f"device {device_index} output matches reference")

    db_path = import_report(report_path, output_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        op_names = {name for (name,) in cur.execute("SELECT name FROM operations").fetchall()}
        assert COLLECTIVE_OP in op_names, f"{COLLECTIVE_OP} missing from operations; captured: {sorted(op_names)}"
        assert len(op_names) > 1, f"only the collective was recorded; captured: {sorted(op_names)}"

        # The regression: circular-buffer allocations for the collective are emitted from
        # dispatch worker threads during heterogeneous MeshWorkload compile. Without context
        # propagation those workers have no processors installed and the nodes never appear.
        rows = cur.execute(
            """
            SELECT captured_graph.captured_graph
            FROM captured_graph
            JOIN operations ON operations.operation_id = captured_graph.operation_id
            WHERE operations.name = ?
            """,
            (COLLECTIVE_OP,),
        ).fetchall()
        assert rows, f"no captured subgraph recorded for {COLLECTIVE_OP}"

        cb_allocations = 0
        for (subgraph,) in rows:
            for node in json.loads(subgraph):
                if node.get("node_type") == "circular_buffer_allocate":
                    cb_allocations += 1

        assert cb_allocations > 0, (
            f"{COLLECTIVE_OP} captured no circular_buffer_allocate nodes: events fired on "
            "dispatch worker threads were dropped by the capture"
        )
        logger.info(f"{COLLECTIVE_OP} captured {cb_allocations} circular-buffer allocations")

        incomplete = cur.execute(
            "SELECT operation_name FROM errors WHERE error_type = 'incomplete_operation'"
        ).fetchall()
        assert not incomplete, f"capture was unbalanced, incomplete_operation rows: {incomplete}"
    finally:
        conn.close()

    logger.info(f"graph report verified at {db_path}")
