# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_conv2d import vae_conv2d, TtConv2dParameters
from ...tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager
import time


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.max()}, {data_.min()}]"


@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear],
        # [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        #    "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 15210496}],
    indirect=True,
)

# @pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize(
    ("batch", "height", "width", "in_channels", "out_channels"),
    [
        (1, 256, 256, 512, 512)
        # (1, 128, 128, 16, 512), #slice -> 8
        # (1, 128, 128, 512, 512), #slice -> 4
        # (1, 256, 256, 512, 512), #slice -> 4, (8)
        # (1, 512, 512, 512, 512), #(16) - 64. How does data Move affect?
        # (1, 512, 512, 512, 256), #16
        # (1, 512, 512, 256, 256), #(4),8
        # (1, 1024, 1024, 256, 256), #16 Need to try height activation override. Data will be significant to move
        # (1, 1024, 1024, 256, 128), #16
        # (1, 1024, 1024, 128, 128), #16. Verify this
        # (1, 1024, 1024, 128, 3), #(8) 16
    ],
)
def test_fun_conv2d(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    cfg,
    sp,
    tp,
    topology,
) -> None:
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )
    logger.info(" Test started ... ")
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    device = parallel_manager.vae_parallel_config.device  # mesh_device
    logger.info(f"Device: {device}, {device.core_grid}")

    # mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

    # Construct test tensor of data; 8 chunks of 32x32
    """
    compute_grid_size = device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    ccl_semaphore_handles = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)
    """

    torch_model = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    torch_model.eval()

    parameters = TtConv2dParameters.from_torch(
        torch_conv=torch_model, dtype=ttnn_dtype, parallel_config=parallel_manager.vae_parallel_config
    )

    inp = torch.randn(batch, in_channels, height, width)
    torch_input_padded = inp.permute(0, 2, 3, 1)
    # torch_input_padded = torch.nn.functional.pad(inp.permute(0, 2, 3, 1), (0, 8-in_channels)) #channel dimension is padded to 8

    tt_inp = ttnn.from_torch(torch_input_padded, dtype=ttnn_dtype, device=device)
    # print(f" Shard spec {tt_inp.shard_spec().value().grid}")

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = vae_conv2d(tt_inp, parameters)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    logger.info(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))

    total_time = 0
    num_itr = 10
    for _ in range(num_itr):
        tt_inp = ttnn.from_torch(torch_input_padded.clone().detach(), dtype=ttnn_dtype, device=device)
        start_time = time.time()
        tt_out = vae_conv2d(tt_inp, parameters)
        total_time += time.time() - start_time
        logger.info(f"time: {(time.time() - start_time)}")

    logger.info(f" conv time: {total_time/num_itr}")

    # print(f" in sem: {ccl_semaphore_handles}")
    print([i for i in mesh_device.get_device_ids()])
