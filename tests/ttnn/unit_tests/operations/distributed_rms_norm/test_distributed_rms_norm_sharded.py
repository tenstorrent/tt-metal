# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — native sharded placement (HEIGHT_SHARDED / WIDTH_SHARDED, L1), zero-copy.

Inputs are built exactly as the golden suite builds them (eval.sharding.auto_shard_config over the live grid,
output memory_config = the input's), through the golden helper, so a buffer OOM of the harness-chosen shard
skips as infeasible. Each case pins one regime branch of the sharded path:

  HEIGHT
    (1, 1, 640, 1792)  bf16 + bf16 gamma  rows of 1 tile row per core, prescale path, row 0 shared with ports
    (1, 1, 100, 512)   fp32, no gamma     h_non_aligned, last shard holds fewer rows than the shard height
    (1, 1, 32, 8192)   bf16 + fp32 gamma  gamma streamed in chunks (1 MB of gamma tiles does not fit next to
                                          the 2 x 512 KB in/out shards); fused normalize
    (1, 1, 128, 4096)  fp32 + fp32 gamma  gamma chunks on 4 cores
  WIDTH (R5 device-local pre-combine)
    (1, 1, 32, 1792)   bf16 + bf16 gamma  56 W chunks, one combiner
    (1, 1, 640, 1792)  bf16 + bf16 gamma  20 combiners over 56 chunks
    (2, 4, 128, 512)   fp32 + fp32 gamma  rank 4
    (1, 1, 2048, 2048) bf16 + bf16 gamma  64 chunks x 64 tile rows: stats stream in stat_rows batches
    (1, 1, 4096, 1024) fp32 + fp32 gamma  128 tile rows per core
    (1, 1, 32, 8192)   bf16, no gamma     ragged last shard (3-tile shards, the last holds 1 valid tile)
"""

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.ccl import Topology

from .test_distributed_rms_norm import _FABRIC, _MESH

H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
F32, BF16 = ttnn.float32, ttnn.bfloat16

CASES = [
    ("height", (1, 1, 640, 1792), H, BF16, BF16),
    ("height", (1, 1, 100, 512), H, F32, None),
    ("height", (1, 1, 32, 8192), H, BF16, F32),
    ("height", (1, 1, 128, 4096), H, F32, F32),
    ("width", (1, 1, 32, 1792), W, BF16, BF16),
    ("width", (1, 1, 640, 1792), W, BF16, BF16),
    ("width", (2, 4, 128, 512), W, F32, F32),
    ("width", (1, 1, 2048, 2048), W, BF16, BF16),
    ("width", (1, 1, 4096, 1024), W, F32, F32),
    ("width", (1, 1, 32, 8192), W, BF16, None),
]


def _run(mesh_device, shape, layout, dtype, gamma_dtype, cluster_axis=1, topology=Topology.Linear, num_links=None):
    if tuple(mesh_device.shape)[cluster_axis] < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has size 1 on mesh {tuple(mesh_device.shape)}")
    run_distributed_rms_norm(
        (shape,),
        mesh_device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        gamma_mode="gamma" if gamma_dtype is not None else "no_gamma",
        gamma_dtype=gamma_dtype if gamma_dtype is not None else "none",
        memory_layout=layout,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
    )


@_MESH
@_FABRIC
@pytest.mark.parametrize(
    "kind,shape,layout,dtype,gamma_dtype",
    CASES,
    ids=[f"{k}-{'x'.join(map(str, s))}-{d}-{g}" for k, s, _, d, g in CASES],
)
def test_sharded(mesh_device, kind, shape, layout, dtype, gamma_dtype):
    _run(mesh_device, shape, layout, dtype, gamma_dtype)


@_MESH
@_FABRIC
@pytest.mark.parametrize("layout", [H, W], ids=["height", "width"])
def test_sharded_axis0_two_links(mesh_device, layout):
    """Along a mesh column, two links requested (HEIGHT spreads row groups over both links)."""
    _run(mesh_device, (1, 1, 640, 1536), layout, BF16, F32, cluster_axis=0, num_links=2)
