# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_group_norm import TtGroupNormParameters, vae_group_norm
from ...tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager


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
@pytest.mark.parametrize(
    ("batch", "channels", "height", "width", "group_count", "sharded_input"),
    [
        # Not sharded input
        (1, 128, 128, 128, 32, False),
        # (1, 512, 256, 256, 32,False),
        # (1, 512, 512, 512, 32,False),
        # (1, 256, 512, 512, 32,False),
        # (1, 256, 1024, 1024, 32,False),
        # (1, 128, 1024, 1024, 32,False),
        # (1, 3, 1024, 1024, 32,False),
        # Sharded input
        # (1, 512, 128, 128, 32,True),
        # (1, 512, 256, 256, 32,True),
        # (1, 512, 512, 512, 32,True),
        # (1, 256, 512, 512, 32,True),
        # (1, 256, 1024, 1024, 32,True),
        # (1, 128, 1024, 1024, 32,True),
    ],
)
def test_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    batch: int,
    channels: int,
    height: int,
    width: int,
    group_count: int,
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

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    torch_model.eval()

    inp = torch.randn((batch, channels, height, width), dtype=torch_dtype)

    parameters = TtGroupNormParameters.from_torch(
        torch_model,
        parallel_config=parallel_manager.vae_parallel_config,
        mesh_sharded_input=sharded_input,
        allow_sharded_compute=True,
    )

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=parallel_manager.vae_parallel_config.device,
        # mesh_mapper=ttnn.ReplicateTensorToMesh(parallel_manager.vae_parallel_config.device),
        mesh_mapper=ttnn.ShardTensorToMesh(parallel_manager.vae_parallel_config.device, dim=-1)
        if sharded_input
        else None,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    logger.info("vae group norm start ... ")
    tt_out = vae_group_norm(tt_inp, parameters)
    logger.info("vae group norm end ... ")
    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    logger.info("Computing PCC ... ")
    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))
    logger.info(print_stats("in", inp))
