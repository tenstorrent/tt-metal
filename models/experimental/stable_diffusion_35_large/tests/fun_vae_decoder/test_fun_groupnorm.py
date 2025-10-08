# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.fun_vae_decoder.fun_group_norm import TtGroupNormParameters, vae_group_norm
from ...tt.utils import assert_quality, to_torch
from models.common.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager, create_vae_parallel_config
import tracy
import time


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.max()}, {data_.min()}]"


def signpost_name(shape, core_grid, sharded, time=None):
    tracy.signpost(f"shape:{shape},core_grid:{core_grid},sharded:{sharded},time:{time}")


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
    ("batch", "height", "width", "channels", "group_count", "sharded_input", "core_grid"),
    [
        # Not sharded input
        # (1, 128, 128, 128, 32, False, ttnn.CoreGrid(y=4, x=8)),
        (1, 128, 128, 512, 32, False, None),
        (1, 128, 128, 512, 32, True, None),
        (1, 256, 256, 512, 32, False, None),
        (1, 256, 256, 512, 32, True, None),
        (1, 512, 512, 512, 32, False, None),
        (1, 512, 512, 512, 32, True, None),
        (1, 512, 512, 256, 32, False, None),
        (1, 512, 512, 256, 32, True, None),
        (1, 1024, 1024, 256, 32, False, None),
        (1, 1024, 1024, 256, 32, True, None),
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
    core_grid: ttnn.CoreGrid,
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
    vae_device = parallel_manager.submesh_devices[0]

    if parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape[1] != 4:
        cfg_shape = parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape
        assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
        logger.info(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP + T5")
        vae_device.reshape(ttnn.MeshShape(1, 4))
    vae_parallel_config = create_vae_parallel_config(vae_device, parallel_manager)

    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    torch_model.eval()

    in_shape = (batch, channels, height, width)
    inp = torch.randn(in_shape, dtype=torch_dtype)

    parameters = TtGroupNormParameters.from_torch(
        torch_model,
        parallel_config=vae_parallel_config,
        mesh_sharded_input=sharded_input,
        allow_sharded_compute=True,
        core_grid=core_grid,
    )

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=vae_parallel_config.device,
        # mesh_mapper=ttnn.ReplicateTensorToMesh(parallel_manager.vae_parallel_config.device),
        mesh_mapper=ttnn.ShardTensorToMesh(vae_parallel_config.device, dim=-1) if sharded_input else None,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        out = torch_model(inp)

    tracy.signpost("Caching")
    tt_out = vae_group_norm(tt_inp, parameters)
    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    logger.info("Computing PCC ... ")
    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")

    start_time = time.time()
    for i in range(10):
        tt_out = vae_group_norm(tt_inp, parameters)
    end_time = time.time()
    signpost_name(in_shape, parameters.core_grid, parameters.mesh_sharded_input, time=(end_time - start_time) / 10)
    # logger.info(f"Time taken: {end_time - start_time} seconds")
