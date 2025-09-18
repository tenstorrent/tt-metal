# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_conv2d import vae_conv2d, TtConv2dParameters
from ...tt.utils import assert_quality, to_torch
from models.common.utility_functions import comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager
import time


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = to_torch(data, mesh_composer=None)
    return (
        f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}, shape:{data_.shape}]"
    )


# TODO: Move to parallel manager
def gn_all_gather(x, parallel_config):
    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=3,
        multi_device_global_semaphore=parallel_config.new_gather_handles,
        topology=ttnn.Topology.Linear,
        mesh_device=parallel_config.device,
        cluster_axis=1,
        num_links=1,
    )
    ttnn.synchronize_device(parallel_config.device)
    return x_g


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
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
    indirect=True,
)

# @pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize(
    (
        "batch",
        "height",
        "width",
        "in_channels",
        "out_channels",
        "mesh_sharded_output",
        "mesh_sharded_input",
        "kernel",
        "padding",
        "stride",
    ),
    [
        (1, 128, 128, 16, 512, False, False, 3, 1, 1),
        (1, 128, 128, 512, 512, False, False, 3, 1, 1),
        (1, 256, 256, 512, 512, False, False, 3, 1, 1),
        (1, 512, 512, 512, 512, False, False, 3, 1, 1),
        (1, 512, 512, 512, 256, False, False, 3, 1, 1),
        (1, 512, 512, 256, 256, False, False, 3, 1, 1),
        (
            1,
            1024,
            1024,
            256,
            256,
            False,
            False,
            3,
            1,
            1,
        ),  # 16 Need to try height activation override. Data will be significant to move
        (1, 1024, 1024, 256, 128, False, False, 3, 1, 1),
        (1, 1024, 1024, 128, 128, False, False, 3, 1, 1),
        (1, 1024, 1024, 128, 3, True, False, 3, 1, 1),  # (8) 16
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
    mesh_sharded_output: bool,
    mesh_sharded_input: bool,
    kernel: int,
    padding: int,
    stride: int,
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
    logger.info(f"Device: {device}, {device.core_grid} , ids:[{list(device.get_device_ids())}]")

    # mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

    # Construct test tensor of data; 8 chunks of 32x32
    """
    compute_grid_size = device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    ccl_semaphore_handles = ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0)
    """

    torch_model = torch.nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, stride=stride
    )
    torch_model.eval()

    parameters = TtConv2dParameters.from_torch(
        torch_conv=torch_model,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.vae_parallel_config,
        mesh_sharded_output=mesh_sharded_output,
        mesh_sharded_input=mesh_sharded_input,
    )

    inp = torch.randn(batch, in_channels, height, width)
    torch_input_padded = inp.permute(0, 2, 3, 1)
    # torch_input_padded = torch.nn.functional.pad(inp.permute(0, 2, 3, 1), (0, 8-in_channels)) #channel dimension is padded to 8

    tt_inp = ttnn.from_torch(
        torch_input_padded,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if mesh_sharded_input else None,
    )
    # print(f" Shard spec {tt_inp.shard_spec().value().grid}")

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = vae_conv2d(tt_inp, parameters)

    if mesh_sharded_output:
        tt_out = gn_all_gather(tt_out, parallel_manager.vae_parallel_config)

    tt_final_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)
    result, output = comp_pcc(out, tt_final_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}")
    assert_quality(tt_final_out_torch, out, pcc=0.94, ccc=0.94)

    total_time = 0
    num_itr = 2
    for _ in range(num_itr):
        # logger.info(f" Getting data")
        tt_inp = ttnn.from_torch(
            torch_input_padded.clone().detach(),
            dtype=ttnn_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if mesh_sharded_input else None,
        )
        # tt_inp = ttnn.from_torch(torch_input_padded.clone().detach(), dtype=ttnn_dtype, device=device)
        start_time = time.time()
        tt_out = vae_conv2d(tt_inp, parameters)
        total_time += time.time() - start_time
        logger.info(f"time: {(time.time() - start_time)}")

    logger.info(f" conv time: {total_time/num_itr}")
