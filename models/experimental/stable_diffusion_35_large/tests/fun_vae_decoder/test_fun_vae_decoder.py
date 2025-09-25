# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
import time
from ...reference.vae_decoder import VaeDecoder
from ...tt.fun_vae_decoder.fun_vae_decoder import sd_vae_decode, TtVaeDecoderParameters
from ...tt.utils import assert_quality, to_torch
from models.common.utility_functions import comp_allclose, comp_pcc
from ...tt.parallel_config import StableDiffusionParallelManager, create_vae_parallel_config
import tracy


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}, shape:{data_.shape},stats in shape: {data.shape}]"


@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(8, 4), (2, 0), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
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
        "layers_per_block",
        "height",
        "width",
        "norm_num_groups",
        "block_out_channels",
    ),
    [
        (1, 16, 3, 2, 128, 128, 32, (128, 256, 512, 512)),  # slice 128, output blocks 32. Need to parametize
        # (1, 16, 3, 2, 128, 128, 32, (128, 256, 512, 512)),  # slice 128, output blocks 32. Need to parametize
    ],
)
# @pytest.mark.usefixtures("use_program_cache")
def test_vae_decoder(
    *,
    mesh_device: ttnn.MeshDevice,
    batch: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    height: int,
    width: int,
    norm_num_groups: int,
    block_out_channels: list[int] | tuple[int, ...],
    cfg,
    sp,
    tp,
    topology,
    num_links,
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
        num_links=num_links,
    )
    # mesh_device = device
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    torch_model = VaeDecoder(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
    )

    # print(torch_model)
    # logger.info(summary(torch_model, input_size=(batch, in_channels, height, width), depth=10,row_settings=("ascii_only","var_names",)))
    # return

    # sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    # print(sd_vae.decoder)
    # torch_model = sd_vae.decoder
    torch_model.eval()

    vae_device = parallel_manager.submesh_devices[0]

    if parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape[1] != 4:
        cfg_shape = parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape
        assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
        print(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP + T5")
        vae_device.reshape(ttnn.MeshShape(1, 4))
    vae_parallel_config = create_vae_parallel_config(vae_device, parallel_manager)

    parameters = TtVaeDecoderParameters.from_torch(
        torch_vae_decoder=torch_model, dtype=ttnn_dtype, parallel_config=vae_parallel_config
    )

    # inp = torch.randn(batch, in_channels, height, width)
    inp = torch.normal(1, 2, (batch, in_channels, height, width))
    # inp = torch.load("torch_latent.pt")  # .permute(0, 3, 1, 2)
    logger.info(f"data shape :{inp.shape}")

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=vae_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(vae_device),
    )

    # ttnn.visualize_mesh_device(mesh_device, tensor=tt_inp)
    # breakpoint(0)

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=vae_device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tracy.signpost("Compilation/Cache pass")
    tt_out = sd_vae_decode(tt_inp, parameters)

    tracy.signpost("Performance pass")
    for i in range(10):
        start_time = time.time()
        tt_out = sd_vae_decode(tt_inp, parameters)
        ttnn.synchronize_device(vae_device)
        end_time = time.time()
        logger.info(f"vae_decode {i} time: {end_time-start_time}")

        tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

        logger.info(print_stats("torch", out))
        logger.info(print_stats("tt", tt_out_torch, device=vae_device))
        assert_quality(out, tt_out_torch, pcc=0.99, ccc=0.99)
        print(comp_allclose(out, tt_out_torch))
        result, output = comp_pcc(out, tt_out_torch)
        logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
