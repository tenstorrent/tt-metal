# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_unet_mid_block import unet_mid_block, TtUNetMidBlock2DParameters
from ...tt.utils import assert_quality, to_torch
from models.common.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager
from ...reference.vae_decoder import UNetMidBlock2D


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}]"


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
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 15210496}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch", "in_channels", "height", "width", "num_groups", "num_heads", "sharded_input"),
    [
        (1, 512, 128, 128, 32, 4, True),
        (1, 512, 128, 128, 32, 4, False),
    ],
)
def test_unetmid_block(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    height: int,
    width: int,
    num_groups: int,
    num_heads: int,
    sharded_input: bool,
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
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    device = parallel_manager.vae_parallel_config.device
    logger.info(f"Device: {device}, {device.core_grid}")

    attention_head_dim = in_channels // num_heads
    torch_model = UNetMidBlock2D(
        in_channels=in_channels, resnet_groups=num_groups, attention_head_dim=attention_head_dim
    )
    # sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    # print(sd_vae.decoder)
    # torch_model = sd_vae.decoder.mid_block
    torch_model.eval()

    parameters = TtUNetMidBlock2DParameters.from_torch(
        unet_mid_block=torch_model,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.vae_parallel_config,
        mesh_sharded_input=sharded_input,
    )

    # inp = torch.randn(batch, in_channels, height, width)
    inp = torch.normal(1, 2, (batch, in_channels, height, width))

    # tt_inp = ttnn.from_torch(inp.permute(0, 2, 3, 1), dtype=ttnn_dtype, device=device,mesh_mapper=ttnn.ShardTensorToMesh(device,dim=-1) if sharded_input else None)
    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if sharded_input else None,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = unet_mid_block(tt_inp, parameters)

    if sharded_input:
        tt_out = gn_all_gather(tt_out, parallel_manager.vae_parallel_config)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))
