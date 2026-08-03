# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gate the fused DistributedGroupNorm at the MiniMax-H3 encoder's real GN sites.

Per-frame GroupNorm with T folded into the batch axis and H sharded across a cluster
axis. Decides whether H/W spatial sharding of the encoder is viable: the fused op must
(a) match torch at batch=T, and (b) not deadlock as spatial collapses to H/f=2, W=16
with C=1024.

Run:  ./python_env/bin/python -m pytest <this file> -q --no-header
"""
from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.normalization import DistributedGroupNorm
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

NUM_GROUPS = 32
EPS = 1e-6

# (C, T, H, W) at every encoder GroupNorm site, shallow -> deep.
SITES = [
    (128, 17, 256, 256),
    (128, 17, 128, 128),
    (256, 17, 128, 128),
    (256, 9, 64, 64),
    (512, 9, 64, 64),
    (512, 5, 32, 32),
    (512, 5, 16, 16),
    (1024, 5, 16, 16),
]

MESHES = [
    pytest.param(
        (4, 8),
        1,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="sp8_axis1",
    ),
    pytest.param(
        (4, 8),
        0,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="sp4_axis0",
    ),
]


@pytest.mark.parametrize(
    ("mesh_device", "cluster_axis", "device_params"), MESHES, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("site", SITES, ids=[f"c{c}_t{t}_{h}x{w}" for c, t, h, w in SITES])
def test_encoder_site(mesh_device, cluster_axis, site):
    channels, frames, height, width = site
    cluster = tuple(mesh_device.shape)[cluster_axis]
    if height % cluster:
        pytest.skip(f"H={height} not divisible by cluster {cluster}")

    torch.manual_seed(0)
    ref = torch.nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=channels, eps=EPS)
    torch.nn.init.normal_(ref.weight)
    torch.nn.init.normal_(ref.bias)
    ref.eval()

    # T as the batch axis: no statistic crosses a frame boundary.
    x_TCHW = torch.randn(frames, channels, height, width, dtype=torch.bfloat16)
    with torch.no_grad():
        expected_TCHW = ref(x_TCHW)

    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    norm = DistributedGroupNorm.from_torch(
        torch_ref=ref,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        mesh_axis=None,
        ccl_manager=ccl,
        core_grid=ttnn.CoreGrid(x=8, y=8),
    )

    x_THWC = x_TCHW.permute(0, 2, 3, 1).contiguous()
    tt_in = bf16_tensor(x_THWC, device=mesh_device, mesh_axis=cluster_axis, shard_dim=1)
    logger.info(f"site C={channels} T={frames} {height}x{width} -> local H={height // cluster}")

    tt_out = norm(tt_in)

    dims = [None, None]
    dims[cluster_axis] = 1
    dims[1 - cluster_axis] = 0
    actual = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )
    if actual.ndim == 5:
        actual = actual[0]
    actual = actual[:frames]
    actual_TCHW = actual.permute(0, 3, 1, 2)

    assert actual_TCHW.shape == expected_TCHW.shape, f"{actual_TCHW.shape} != {expected_TCHW.shape}"
    assert_quality(expected_TCHW, actual_TCHW, pcc=0.999)
