# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_updecoder_block import updecoder_block, TtUpDecoderBlock2DParameters
from ...reference.vae_decoder import UpDecoderBlock2D
from ...tt.utils import assert_quality, to_torch
from models.common.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}]"


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
    (
        "batch",
        "in_channels",
        "out_channels",
        "height",
        "width",
        "num_groups",
        "num_layers",
        "add_upsample",
        "mesh_sharded_input",
        "allow_gn_sharded_compute",
    ),
    [
        # (1, 512, 512, 128, 128, 32, 3, True,False,True),
        # (1, 512, 512, 128, 128, 32, 3, True,True,True),
        # (1, 512, 512, 256, 256, 32, 3, True,False,True),
        # (1, 512, 512, 256, 256, 32, 3, True,True,True),
        # (1, 512, 256, 512, 512, 32, 3, True,False,True),
        # (1, 512, 256, 512, 512, 32, 3, True,True,True),
        # (1, 256, 128, 1024, 1024, 32, 3, False,False,True),
        (1, 256, 128, 1024, 1024, 32, 3, False, True, False),  # All gather currently hanging after gn. So do it before.
    ],
)
def test_updecoder_block(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    num_groups: int,
    num_layers: int,
    add_upsample,
    cfg,
    sp,
    tp,
    topology,
    mesh_sharded_input,
    allow_gn_sharded_compute,
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
    device = parallel_manager.vae_parallel_config.device
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    logger.info(f"Device: {device}, {device.core_grid}")

    torch_model = UpDecoderBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        resnet_groups=num_groups,
        num_layers=num_layers,
        add_upsample=add_upsample,
    )
    torch_model.eval()

    parameters = TtUpDecoderBlock2DParameters.from_torch(
        updecoder_block=torch_model,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.vae_parallel_config,
        mesh_sharded_input=mesh_sharded_input,
        gn_allow_sharded_compute=allow_gn_sharded_compute,
    )

    # inp = torch.randn(batch, in_channels, height, width)
    inp = torch.normal(1, 2, (batch, in_channels, height, width))

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if mesh_sharded_input else None,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = updecoder_block(tt_inp, parameters)

    if mesh_sharded_input:
        tt_out = ttnn.experimental.all_gather_async(
            input_tensor=tt_out,
            dim=3,
            multi_device_global_semaphore=parallel_manager.vae_parallel_config.new_gather_handles,
            topology=ttnn.Topology.Linear,
        )

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
