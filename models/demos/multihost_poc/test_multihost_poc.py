# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal multi-host proof-of-concept for the TTNN graph report feature.

This is intentionally the smallest possible "model-like" run that exercises a real
multi-host (2 host) Blackhole setup. Its only job is to:

  1. Prove the two hosts are in lockstep and talking to each other over MPI.
  2. Run a handful of TTNN ops on each host (so each host produces report data).
  3. Do one realistic cross-host collective (all_gather over the host boundary).
  4. Let the autouse ``ttnn_graph_report`` fixture (root conftest.py) emit a single,
     merged multi-host TTNN report.

It is run like the other model demos, but it MUST be launched across 2 hosts with
``tt-run`` (which wraps ``mpirun``). When run on a single host it skips cleanly.

See ``models/demos/multihost_poc/README.md`` and ``run_multihost_poc.sh`` for the
exact slurm / tt-run launch command. Issue: tenstorrent/ttnn-visualizer#1335.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Unified 1x16 mesh spanning two BH "loudbox" hosts (8 chips per host). This matches
# tests/tt_metal/distributed/config/bh_lbx2_1x16_rank_bindings.yaml, which is the
# binding the launch script passes to tt-run.
NUM_DEVICES = 16

# All-gather over the second mesh axis -> data crosses the host boundary (devices
# 0-7 live on host 0, devices 8-15 on host 1).
CLUSTER_AXIS = 1
GATHER_DIM = 3
SHARD_WIDTH = 32  # one tile column per device


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, NUM_DEVICES), id="1x16_grid")], indirect=True)
def test_multihost_poc(mesh_device):
    # --- 0. This PoC only means anything across >=2 hosts -------------------------
    if not ttnn.using_distributed_env():
        pytest.skip(
            "Multi-host PoC: launch across >=2 hosts with tt-run "
            "(see models/demos/multihost_poc/run_multihost_poc.sh)."
        )

    rank = int(ttnn.distributed_context_get_rank())
    world_size = int(ttnn.distributed_context_get_size())
    assert world_size >= 2, f"PoC expects at least 2 hosts, got world_size={world_size}"
    logger.info(f"[multihost-poc] rank {rank}/{world_size}, local devices = {mesh_device.get_num_devices()}")

    # --- 1. Host-to-host MPI handshake (the "are we really multi-host" check) -----
    # Pure MPI over the host network: every rank contributes its rank id and they
    # must all agree on the full set. This is the same coordination layer the real
    # multi-host models use to stay in lockstep.
    ttnn.distributed_context_barrier()
    gathered_ranks = sorted(int(r) for r in ttnn.distributed_context_allgather_int(rank))
    logger.info(f"[multihost-poc] MPI all_gather of ranks -> {gathered_ranks}")
    assert gathered_ranks == list(range(world_size)), gathered_ranks

    num_devices = mesh_device.get_num_devices()  # 16 (global mesh size)

    # --- 2. Build identical input on every rank, sharded across all 16 devices ----
    # SPMD: every rank seeds the same so the per-device shards line up across hosts.
    torch.manual_seed(0)
    torch_input = torch.cat(
        [torch.rand(1, 1, 32, SHARD_WIDTH).bfloat16() for _ in range(num_devices)],
        dim=GATHER_DIM,
    )  # [1, 1, 32, 32*16]

    # Multi-host safe placement (mirrors tests/.../test_multi_device.py::test_multihost_sanity):
    # build the sharded tensor on HOST first (no device=), then to_device. Passing
    # device= to from_torch would distribute to all 16 devices in one shot, touching the
    # other host's remote chips and throwing "Attempted to access remote device...".
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=GATHER_DIM),
    )
    tt_input = ttnn.to_device(tt_input, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # each local device now holds [1, 1, 32, 32]

    # --- 3. A few plain per-device ops (each shows up in this host's report) -------
    x = ttnn.mul(tt_input, 2.0)
    x = ttnn.add(x, tt_input)  # == 3 * input
    x = ttnn.gelu(x)  # == gelu(3 * input)
    ttnn.synchronize_device(mesh_device)

    # --- 4. The realistic cross-host collective -----------------------------------
    # all_gather along the sharded dim reconstructs the full tensor on every device,
    # which requires the two hosts to move data to each other over TT-fabric.
    gathered = ttnn.all_gather(
        x,
        GATHER_DIM,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Linear,
        num_links=2,
    )
    ttnn.synchronize_device(mesh_device)

    # --- 5. Light correctness check -----------------------------------------------
    # After the gather every local device should hold the full eltwise result.
    torch_reference = torch.nn.functional.gelu(3.0 * torch_input.float())
    local_outputs = ttnn.get_device_tensors(gathered)
    assert len(local_outputs) > 0, "expected at least one local device tensor on this rank"
    for i, tt_out in enumerate(local_outputs):
        ok, msg = comp_pcc(torch_reference, ttnn.to_torch(tt_out), pcc=0.99)
        assert ok, f"rank {rank} local device {i}: {msg}"

    # --- 6. Finish together -------------------------------------------------------
    ttnn.distributed_context_barrier()
    logger.info(
        f"[multihost-poc] rank {rank} done. " f"Rank 0 merges the per-host captures into: {ttnn.CONFIG.report_path}"
    )
