# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-stage, one-galaxy fabric check: two processes, one 4x4 half each, no model.

Checks what a two-stage K3 prefill run needs from the fabric before any model code runs:

1. each mesh comes up with the per-axis topology the fabric mode promises (a declared RING that the
   fabric does not realize only logs, so this asserts it);
2. all_gather on both mesh axes, with the topology the prefill model would pick, matches torch;
3. a tensor sharded [rows on SP, hidden on TP] (the runner's D2D layout) crosses from rank 0's mesh
   to rank 1's mesh intact, repeatedly, interleaved with CCLs on both meshes.

Launch (rank binding from `gen_pipeline_binding.py --stage-shape 4x4`; PREFILL_FABRIC_MODE one of
2d_torus_x, 2d_torus_y, 2d):

    PREFILL_FABRIC_MODE=2d_torus_x tt-run --rank-binding <binding.yaml> \
        python3 models/demos/deepseek_v3_d_p/tests/kimi_k3/two_stage_fabric_check.py

Exit code 0 on both ranks means pass; each rank prints `TWO_STAGE_CHECK rank=<r> PASS`.
"""

import os
import sys

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import open_mesh_device
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology

SP_AXIS, TP_AXIS = 0, 1
MESH_SHAPE = (4, 4)
# The axes each fabric mode wraps on a top/bottom half of a galaxy (SP = mesh axis 0, TP = axis 1).
EXPECTED_RING = {
    "2d_torus_x": (False, True),
    "2d_torus_y": (True, False),
    "2d": (False, False),
}
ROWS, HIDDEN = 2560, KimiK3Config.EMB_SIZE  # one 2560-token chunk, as the first 4x4 bring-up runs it
ITERATIONS = int(os.getenv("TWO_STAGE_CHECK_ITERATIONS", "10"))


def _mapper_dims():
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return tuple(dims)


def check_topology(mesh, mode):
    probe = ttnn.from_torch(
        torch.zeros(1, 1, 32 * MESH_SHAPE[0], 32 * MESH_SHAPE[1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=_mapper_dims(), mesh_shape=MESH_SHAPE),
    )
    realized = tuple(ttnn.get_usable_topology(probe, ttnn.Topology.Ring, axis) == ttnn.Topology.Ring for axis in (0, 1))
    model = per_axis_topology()
    logger.info(f"realized ring per axis {realized}, model picks {model}, expected ring {EXPECTED_RING[mode]}")
    assert realized == EXPECTED_RING[mode], f"fabric mode {mode}: realized ring axes {realized}"
    for axis, ring in enumerate(realized):
        assert (model[axis] == ttnn.Topology.Ring) == ring, f"model topology {model} disagrees with axis {axis}"
    return model


def check_all_gather(mesh, topology, seed):
    torch.manual_seed(seed)
    for axis in (SP_AXIS, TP_AXIS):
        full = torch.randn(1, 1, 128 * MESH_SHAPE[0], 256 * MESH_SHAPE[1])
        tt_in = ttnn.from_torch(
            full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=_mapper_dims(), mesh_shape=MESH_SHAPE),
        )
        gather_dim = 2 if axis == SP_AXIS else 3
        out = ttnn.all_gather(
            tt_in,
            dim=gather_dim,
            cluster_axis=axis,
            num_links=get_num_links(mesh, axis),
            topology=topology[axis],
        )
        # After gathering along `axis`, every device on that axis holds the full extent of gather_dim.
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        want_rows = full.shape[2] // MESH_SHAPE[0] if axis == TP_AXIS else full.shape[2]
        want_cols = full.shape[3] // MESH_SHAPE[1] if axis == SP_AXIS else full.shape[3]
        want = full[:, :, :want_rows, :want_cols]
        assert torch.allclose(got.float(), want.to(torch.bfloat16).float()), f"all_gather on axis {axis} mismatch"
        logger.info(f"all_gather axis {axis} ({topology[axis]}) OK")


def make_socket_config(sender_rank, receiver_rank):
    connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)), ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0))
        )
        for coord in ttnn.MeshCoordinateRange(ttnn.MeshShape(*MESH_SHAPE))
    ]
    return ttnn.SocketConfig(connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096), sender_rank, receiver_rank)


def main():
    mode = os.environ.get("PREFILL_FABRIC_MODE", "").strip().lower()
    assert mode in EXPECTED_RING, f"set PREFILL_FABRIC_MODE to one of {sorted(EXPECTED_RING)}"
    mesh = open_mesh_device(MESH_SHAPE, KimiK3Config, l1_small_size=KimiK3Config.L1_SMALL_SIZE)
    rank = int(ttnn.distributed_context_get_rank())
    assert int(ttnn.distributed_context_get_size()) == 2, "run with exactly 2 ranks"

    topology = check_topology(mesh, mode)
    check_all_gather(mesh, topology, seed=rank)

    socket = ttnn.MeshSocket(mesh, make_socket_config(0, 1))
    mapper = ttnn.ShardTensor2dMesh(mesh, dims=_mapper_dims(), mesh_shape=MESH_SHAPE)
    for it in range(ITERATIONS):
        torch.manual_seed(1000 + it)
        payload = torch.randn(1, 1, ROWS, HIDDEN).to(torch.bfloat16)
        tt_payload = ttnn.from_torch(
            payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper
        )
        if rank == 0:
            ttnn.experimental.send_async(tt_payload, socket)
        else:
            received = ttnn.allocate_tensor_on_device(tt_payload.spec, mesh)
            ttnn.experimental.recv_async(received, socket)
            got = ttnn.to_torch(
                received, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=_mapper_dims(), mesh_shape=MESH_SHAPE)
            )
            assert torch.equal(got, payload), f"iteration {it}: received tensor differs from what rank 0 sent"
        # CCLs on both meshes between transfers: the handoff shares fabric routers with them in the runner.
        check_all_gather(mesh, topology, seed=it)
        ttnn.synchronize_device(mesh)
        logger.info(f"rank {rank} iteration {it} OK")

    ttnn.distributed_context_barrier()
    ttnn.close_mesh_device(mesh)
    print(f"TWO_STAGE_CHECK rank={rank} mode={mode} PASS", flush=True)


if __name__ == "__main__":
    sys.exit(main())
