# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ..tt.group_norm import TtGroupNorm, TtGroupNormParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality, from_torch_fast, to_torch


@pytest.mark.parametrize("device_params", [{"trace_region_size": 40960}], indirect=True)
@pytest.mark.parametrize("use_tracing", [True])
@pytest.mark.parametrize(
    ("channels", "height", "width", "group_count"),
    [
        (512, 128, 128, 32),
        (512, 256, 256, 32),
        (256, 512, 512, 32),
        (512, 512, 512, 32),
        (128, 1024, 1024, 32),
        (256, 1024, 1024, 32),
    ],
)
def test_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    use_tracing: bool,
    channels: int,
    height: int,
    width: int,
    group_count: int,
) -> None:
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    batch_size = mesh_device.get_num_devices()

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    torch_model.eval()

    parameters = TtGroupNormParameters.from_torch(torch_model.state_dict(), device=mesh_device)
    tt_model = TtGroupNorm(parameters, eps=torch_model.eps, num_groups=group_count)

    torch.manual_seed(0)

    inp = torch.randn([batch_size, channels, height, width], dtype=torch_dtype)

    tt_inp_host = from_torch_fast(
        inp.permute(0, 2, 3, 1),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, 0),
    )

    with torch.no_grad():
        out = torch_model(inp)

    tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)

    if use_tracing:
        # cache
        logger.info("caching...")
        tt_model(tt_inp)

        # trace
        logger.info("tracing...")
        tid = ttnn.begin_trace_capture(mesh_device)
        tt_out = tt_model(tt_inp)
        ttnn.end_trace_capture(mesh_device, tid)

        # execute
        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_inp_host, tt_inp)
        ttnn.execute_trace(mesh_device, tid)
        logger.info("done...")
    else:
        logger.info("compiling...")
        tt_model(tt_inp)

        logger.info("executing...")
        ttnn.copy_host_to_device_tensor(tt_inp_host, tt_inp)
        tt_out = tt_model(tt_inp)
        logger.info("done...")

    tt_out_torch = to_torch(tt_out, shard_dim=0).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
