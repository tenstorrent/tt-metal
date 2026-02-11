# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import diffusers
import diffusers.models.transformers.transformer_flux2
import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2Transformer,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache, tensor
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig


class ModelLocationGenerator(Protocol):
    def __call__(
        self,
        model_version: str,
        *,
        model_subdir: str = "",
        download_if_ci_v2: bool = False,
        ci_v2_timeout_in_s: int = 300,
        endpoint_prefix: str = "",
        download_dir_suffix: str = "",
    ) -> str: ...


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 8), id="1x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_modulation(mesh_device: ttnn.MeshDevice) -> None:
    tp_axis = 1

    batch_size = 2
    sequence_length = 4096
    size = 6144
    mod_param_sets = 2

    torch_model = diffusers.models.transformers.transformer_flux2.Flux2Modulation(size, mod_param_sets)
    torch_model.eval()

    tt_model = Flux2Modulation(size, mod_param_sets=mod_param_sets, device=mesh_device, tp_axis=tp_axis)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    inp = torch.randn([batch_size, sequence_length, size])
    tt_inp = tensor.from_torch(inp, device=mesh_device)

    logger.info("running Torch model...")
    with torch.no_grad():
        out = torch_model.forward(inp)
    out = [o2 for o1 in out for o2 in o1]  # flatten

    logger.info("running TT model...")
    tt_out = tt_model.forward(tt_inp)

    for out1, tt_out1 in zip(out, tt_out, strict=True):
        tt_out_torch1 = tensor.to_torch(tt_out1, mesh_axes=[..., tp_axis])
        assert_quality(out1, tt_out_torch1, pcc=0.99998, relative_rmse=0.009)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "skip_layers", "skip_single_layers"),
    [
        pytest.param((1, 8), 0, 1, 1, 0, 0, id="1x8sp0tp1"),
        pytest.param((2, 4), 0, 1, 1, 4, 24, id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width", "prompt_seq_len"),
    [
        (1, 1024, 1024, 512),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_transformer(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    height: int,
    width: int,
    prompt_seq_len: int,
    skip_layers: int,
    skip_single_layers: int,
    model_location_generator: ModelLocationGenerator,
) -> None:
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    model_name = model_location_generator("black-forest-labs/FLUX.2-dev", model_subdir="transformer")
    torch_model = diffusers.Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
    assert isinstance(torch_model, diffusers.Flux2Transformer2DModel)
    torch_model.eval()

    del torch_model.transformer_blocks[len(torch_model.transformer_blocks) - skip_layers :]
    del torch_model.single_transformer_blocks[len(torch_model.single_transformer_blocks) - skip_single_layers :]

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    if num_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
    else:
        padding_config = None

    tt_model = Flux2Transformer(
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers - skip_layers,
        num_single_layers=torch_model.config.num_single_layers - skip_single_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        out_channels=torch_model.out_channels,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    if not cache.initialize_from_cache(
        tt_model,
        torch_model.state_dict(),
        "FLUX.2-dev",
        "transformer",
        parallel_config,
        tuple(mesh_device.shape),
        "bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())

    spatial_seq_len = height * width // 16**2

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3)

    # prepare for ROPE
    text_ids = _prepare_ids(text_sequence_length=prompt_seq_len)
    image_ids = _prepare_ids(height=height // 16, width=width // 16)
    prompt_rope_cos, prompt_rope_sin = torch_model.pos_embed.forward(text_ids)
    spatial_rope_cos, spatial_rope_sin = torch_model.pos_embed.forward(image_ids)

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)
    tt_guidance = tensor.from_torch(guidance.unsqueeze(-1), device=mesh_device)

    tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=mesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=mesh_device)

    logger.info("running Torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,
            guidance=guidance,
            img_ids=image_ids,
            txt_ids=text_ids,
        ).sample

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        guidance=tt_guidance,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.996, relative_rmse=0.09)


def _prepare_ids(*, height: int = 1, width: int = 1, text_sequence_length: int = 1) -> torch.Tensor:
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    s = torch.arange(text_sequence_length)

    return torch.cartesian_prod(t, h, w, s)
