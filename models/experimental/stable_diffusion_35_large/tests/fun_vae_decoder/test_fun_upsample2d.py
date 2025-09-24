# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_upsample2d import vae_upsample2d, TtUpsample2DParameters
from ...reference.vae_decoder import Upsample2D
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
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.max()}, {data_.min()}]"


# TODO: Move to parallel manager
def gn_all_gather(x, parameters: TtUpsample2DParameters):
    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=3,
        multi_device_global_semaphore=parameters.parallel_config.new_gather_handles,
        topology=ttnn.Topology.Linear,
        mesh_device=parameters.parallel_config.device,
        cluster_axis=1,
        num_links=1,
    )
    ttnn.synchronize_device(parameters.parallel_config.device)
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
    ("batch", "in_channels", "out_channels", "height", "width", "sharded_input"),
    [
        (1, 512, 512, 128, 128, False),
        (1, 512, 512, 256, 256, False),
        (1, 256, 256, 512, 512, False),
        # (1, 256, 128, 128, 128),
        # (1, 512, 256, 256, 256),
        # (1, 512, 256, 256, 256),
        # (1, 512, 512, 16, 16),
        # (512, 256, 256, 32),
        # (256, 512, 512, 32),
        # (512, 512, 512, 32),
        # (128, 1024, 1024, 32),
        # (256, 1024, 1024, 32),
    ],
)
def test_fun_upsample2d(
    *,
    mesh_device: ttnn.MeshDevice,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
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
    device = parallel_manager.vae_parallel_config.device
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    logger.info(f"Device: {device}, {device.core_grid}")

    torch_model = Upsample2D(channels=in_channels, out_channels=out_channels)
    torch_model.eval()

    parameters = TtUpsample2DParameters.from_torch(
        torch_upsample=torch_model,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.vae_parallel_config,
        mesh_sharded_input=sharded_input,
    )

    inp = torch.randn(batch, in_channels, height, width)
    torch_input_padded = inp.permute(0, 2, 3, 1)

    tt_inp = ttnn.from_torch(
        torch_input_padded,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if sharded_input else None,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = vae_upsample2d(tt_inp, parameters)

    if sharded_input:
        tt_out = gn_all_gather(tt_out, parameters)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))
