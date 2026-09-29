# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Standalone ``all_to_all_async_generic`` at the V4.1 attention reshard shapes (bead 8y7.9.1).

``TtV41Attention.forward`` reshards q head->sequence before ``sparse_sdpa`` and the output sequence->head after it,
over TP (cluster_axis 1). On the LoudBox 2x4 mesh TP has 4 chips; on 4x2 it has 2, where the block test hung.
Each case runs one direction on one mesh axis and checks the result bit-exact against the torch reshard:
global tensor sharded [in_dim over the axis, out_dim over the other axis] -> the same global tensor sharded
[out_dim over the axis, ...], i.e. the op must concatenate ``in_dim`` and split ``out_dim`` across the axis.

Finding (4x2, cluster_axis 1): the op's writer stops in ``fail_stop_invalid_fabric_route`` during its multicast
initialization (it picks east/west for mesh axis 1; the LoudBox 4x2 TP axis is wired north-south), so the collective
hangs. ``test_v41_tp_all_to_all`` covers the V4.1 reshard ``V41Collectives.tp_all_to_all`` that attention uses,
which routes TP=2 around the op (gather + ``mesh_partition``).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives

# per-chip q at the attention call site: [1, heads/tp, seq/sp, head_dim]; seq/sp of the whole chunk
SIZES = {
    "small": dict(heads=32, seq=512, head_dim=128),  # SmallV41Config, SMALL_SEQ
    "production": dict(heads=64, seq=2048, head_dim=512),  # V4.1 Flash, block test SEQ
}
DIRECTIONS = {"head_to_seq": (1, 2), "seq_to_head": (2, 1)}  # (in_dim, out_dim)
NUM_LINKS = (1, 2)


def _mesh(shape):
    rows, cols = shape
    return pytest.param(
        shape,
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=f"mesh-{rows}x{cols}"),
        id=f"fabric2d-mesh-{rows}x{cols}",
    )


def _make_input(mesh_device, cluster_axis, size, direction):
    """Global torch tensor and its mesh tensor with the attention call site's per-chip shape."""
    dims = SIZES[size]
    tp, sp = mesh_device.shape[1], mesh_device.shape[0]
    axis_size, other_size = mesh_device.shape[cluster_axis], mesh_device.shape[1 - cluster_axis]
    in_dim, out_dim = DIRECTIONS[direction]
    # per-chip input of the attention reshard (head->seq: [1, H/tp, S/sp, D]; seq->head: [1, H, S/(sp*tp), D])
    if direction == "head_to_seq":
        heads, rows = dims["heads"] // tp, dims["seq"] // sp
    else:
        heads, rows = dims["heads"], dims["seq"] // (sp * tp)
    local = [1, heads, rows, dims["head_dim"]]
    global_shape = list(local)
    global_shape[in_dim] *= axis_size
    global_shape[out_dim] *= other_size
    torch.manual_seed(0)
    host = torch.randn(global_shape).to(torch.bfloat16)
    placements = [None, None]
    placements[cluster_axis] = ttnn.PlacementShard(in_dim)
    placements[1 - cluster_axis] = ttnn.PlacementShard(out_dim)
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements, mesh_device.shape))
    x = ttnn.from_torch(
        host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    assert list(ttnn.get_device_tensors(x)[0].shape) == local
    return host, x, local


def _check(mesh_device, y, host, local, cluster_axis, direction):
    """Device (axis a, other o) must hold the full ``in_dim`` and global ``out_dim`` block o * axis_size + a."""
    in_dim, out_dim = DIRECTIONS[direction]
    axis_size = mesh_device.shape[cluster_axis]
    expect_local = list(local)
    expect_local[in_dim] *= axis_size
    expect_local[out_dim] //= axis_size
    block = expect_local[out_dim]
    for idx, shard in enumerate(ttnn.get_device_tensors(y)):
        coord = divmod(idx, mesh_device.shape[1])
        a, o = coord[cluster_axis], coord[1 - cluster_axis]
        got = ttnn.to_torch(shard)
        assert list(got.shape) == expect_local, f"device {coord}: shape {list(got.shape)} != {expect_local}"
        want = host.narrow(out_dim, (o * axis_size + a) * block, block)
        assert torch.equal(got, want), f"device {coord}: reshard mismatch"


@pytest.mark.timeout(600)
@pytest.mark.parametrize("num_links", NUM_LINKS, ids=[f"links{n}" for n in NUM_LINKS])
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("cluster_axis", [1, 0], ids=["tp_axis", "sp_axis"])
@pytest.mark.parametrize("mesh_device, device_params", [_mesh((2, 4)), _mesh((4, 2))], indirect=True)
def test_v41_all_to_all(mesh_device, device_params, cluster_axis, size, direction, num_links):
    """The shared op itself. On 4x2 it runs only with V41_A2A_REPRO_4X2=1: the tp_axis case hangs the mesh (the safe
    runner then resets it; see module docstring) and the sp_axis cases were not run."""
    if tuple(mesh_device.shape) == (4, 2) and os.environ.get("V41_A2A_REPRO_4X2") != "1":
        pytest.skip(
            "4x2 all_to_all_async_generic: known TP-axis hang (bead 8y7.9.1); set V41_A2A_REPRO_4X2=1 to reproduce"
        )
    t0 = time.time()
    host, x, local = _make_input(mesh_device, cluster_axis, size, direction)
    topology = per_axis_topology()[cluster_axis]
    logger.info(
        f"a2a {tuple(mesh_device.shape)} axis={cluster_axis} ({mesh_device.shape[cluster_axis]} chips) {direction} "
        f"local={local} links={num_links} topology={topology} input ready {time.time() - t0:.1f}s"
    )
    in_dim, out_dim = DIRECTIONS[direction]
    t1 = time.time()
    y = ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=in_dim,
        out_dim=out_dim,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"a2a device done {time.time() - t1:.2f}s")
    _check(mesh_device, y, host, local, cluster_axis, direction)
    logger.info(f"a2a PASS total {time.time() - t0:.1f}s")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("mesh_device, device_params", [_mesh((2, 4)), _mesh((4, 2))], indirect=True)
def test_v41_tp_all_to_all(mesh_device, device_params, size, direction):
    """``V41Collectives.tp_all_to_all`` (the attention reshard) is bit exact and repeatable on both meshes."""
    t0 = time.time()
    host, x, local = _make_input(mesh_device, 1, size, direction)
    ccl = V41Collectives(mesh_device)
    in_dim, out_dim = DIRECTIONS[direction]
    outputs = []
    for rep in range(2):  # the second call reuses the program cache and cycles the CCL semaphores
        t1 = time.time()
        outputs.append(ccl.tp_all_to_all(x, in_dim, out_dim))
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"tp_all_to_all {tuple(mesh_device.shape)} {direction} local={local} rep {rep} {time.time() - t1:.2f}s"
        )
    for y in outputs:
        _check(mesh_device, y, host, local, 1, direction)
    logger.info(f"tp_all_to_all PASS total {time.time() - t0:.1f}s")
