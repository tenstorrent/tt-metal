# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.width_sharded_all_reduce`` (TP4) on DSpark blocks: a width-sharded [h, N] block in
(h x 32) tiles is summed elementwise over its shard bytes, so the output keeps the input's tiles and spec.

Production: o_b's [rows, 5120] on 80 cores (num_links=2, 640 B packets) and Engram's wkv [rows, 25600] on 80
cores (num_links=2), on a 1x4 submesh of the 8x4 Galaxy.

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_width_sharded_all_reduce_rows.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config

PARENT_MESH = (8, 4)
TP = 4
CASES = {"ob": (5120, 640), "engram": (25600, None)}  # width, max_packet_bytes
DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "l1_small_size": 10240}]


def _submesh(mesh_device):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH} mesh, got {tuple(mesh_device.shape)}")
    return mesh_device.create_submesh(ttnn.MeshShape(1, TP), ttnn.MeshCoordinate(0, 0))


def _input(submesh, rows, h, width, seed=0):
    """Per-rank [rows, width] partials; h == 0 is today's ROW_MAJOR row block."""
    torch.manual_seed(seed)
    parts = torch.randn(TP, 1, rows, width, dtype=torch.bfloat16)
    kwargs = dict(dtype=ttnn.bfloat16, device=submesh, mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0))
    if h == 0:
        tt = ttnn.from_torch(
            parts,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=width_sharded_l1_config(rows, width, submesh, num_cores=80, tile_height=1),
            **kwargs,
        )
    else:
        tt = ttnn.from_torch(
            parts,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((h, 32)),
            memory_config=width_sharded_l1_config(h, width, submesh, num_cores=80, tile_height=h),
            **kwargs,
        )
    return parts, tt


def _all_reduce(x, max_packet_bytes):
    return ttnn.experimental.deepseek.width_sharded_all_reduce(
        x, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear, max_packet_bytes=max_packet_bytes
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("rows, h", [(4, 4), (1, 1), (5, 8)], ids=str)
def test_width_sharded_all_reduce_rows(mesh_device, case, rows, h, expect_error):
    submesh = _submesh(mesh_device)
    width, packet = CASES[case]
    if case == "engram" and h == 8:
        # 6400 pages over 2 links need > 40 reducer cores per link at the fabric's max payload: unsupported.
        _, x = _input(submesh, rows, h, width)
        with expect_error(RuntimeError, "coalesced path"):
            _all_reduce(x, packet)
        return
    for step in range(3):  # odd / even steps use the two halves of the ping-pong scratch
        parts, x = _input(submesh, rows, h, width, seed=step)
        out = _all_reduce(x, packet)
        assert out.layout == ttnn.TILE_LAYOUT and tuple(out.tile.tile_shape) == (h, 32)
        assert out.memory_config() == x.memory_config()
        expected = parts.float().sum(0)[0]
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)).float()
        for rank in range(TP):
            for j in range(rows):
                passing, pcc = comp_pcc(expected[j], got[rank, 0, j], 0.999)
                assert passing, f"step {step} rank {rank} row {j}: PCC {pcc}"
            assert torch.equal(got[rank], got[0]), f"rank {rank} differs from rank 0"
        logger.info(f"step {step}: PCC {pcc}")
        ttnn.deallocate(x)
        ttnn.deallocate(out)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_width_sharded_all_reduce_rows_perf(mesh_device):
    """Each case on a cleared program cache, so its scratch is placed as it would be alone."""
    from tracy import signpost

    submesh = _submesh(mesh_device)
    runs = [
        ("ob_s1", 1, 0, 5120, 640),
        ("ob_s4", 4, 4, 5120, 640),
        # Larger packets than L1_SMALL holds: the block's scratch goes to L1.
        ("ob_s4_pkt1280", 4, 4, 5120, 1280),
        ("ob_s4_pkt2560", 4, 4, 5120, None),
        ("engram_s1", 1, 0, 25600, None),
        ("engram_s4", 4, 4, 25600, None),
    ]
    for name, rows, h, width, packet in runs:
        if True:
            submesh.clear_program_cache()
            _, x = _input(submesh, rows, h, width)
            ttnn.deallocate(_all_reduce(x, packet))
            ttnn.synchronize_device(submesh)
            signpost(name)
            for _ in range(20):
                ttnn.deallocate(_all_reduce(x, packet))
            ttnn.synchronize_device(submesh)
            ttnn.deallocate(x)
