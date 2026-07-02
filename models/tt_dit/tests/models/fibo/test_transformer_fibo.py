# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import diffusers as reference
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_fibo import FiboCheckpoint
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tracing import Tracer

_PATCH_SIZE = 2  # pipeline-level 2x2 packing; FIBO's transformer itself uses patch_size=1


def _build_ids(*, prompt_seq_len: int, latents_height: int, latents_width: int) -> torch.Tensor:
    """Construct the [text_ids; img_ids] tensor that FIBO feeds to its 3-axis RoPE.

    Mirrors the diffusers FIBO pipeline: text positions are all-zero (no rotation); image
    positions encode (0, h, w) where ``h`` and ``w`` are post-packing latent grid coordinates.
    """
    h = latents_height // _PATCH_SIZE
    w = latents_width // _PATCH_SIZE

    img_ids = torch.zeros(h, w, 3)
    img_ids[..., 1] = torch.arange(h)[:, None]
    img_ids[..., 2] = torch.arange(w)[None, :]
    img_ids = img_ids.reshape(h * w, 3)

    text_ids = torch.zeros(prompt_seq_len, 3)
    return torch.cat([text_ids, img_ids], dim=0)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), 0, 1, 4, id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "latents_height", "latents_width", "prompt_seq_len"),
    [
        (1, 64, 64, 3008),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 34000000}],
    indirect=True,
)
def test_transformer(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    latents_height: int,
    latents_width: int,
    prompt_seq_len: int,
) -> None:
    torch.manual_seed(0)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    checkpoint_name = "briaai/FIBO"
    torch_model = reference.BriaFiboTransformer2DModel.from_pretrained(checkpoint_name, subfolder="transformer")
    assert isinstance(torch_model, reference.BriaFiboTransformer2DModel)
    torch_model.eval()

    config = torch_model.config
    in_channels = config.in_channels
    joint_attention_dim = config.joint_attention_dim
    text_encoder_dim = config.text_encoder_dim
    total_num_blocks = config.num_layers + config.num_single_layers

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    checkpoint = FiboCheckpoint(checkpoint_name)
    tt_model = checkpoint.build(ccl_manager=ccl_manager, parallel_config=parallel_config)

    spatial_seq_len = (latents_height // _PATCH_SIZE) * (latents_width // _PATCH_SIZE)

    tracer = Tracer(tt_model.forward, device=mesh_device)

    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    text_encoder_layers = [torch.randn([batch_size, prompt_seq_len, text_encoder_dim]) for _ in range(total_num_blocks)]
    timestep = torch.full([batch_size], fill_value=500.0)

    ids = _build_ids(prompt_seq_len=prompt_seq_len, latents_height=latents_height, latents_width=latents_width)
    torch_rope_cos, torch_rope_sin = torch_model.pos_embed(ids)
    prompt_rope_cos = torch_rope_cos[:prompt_seq_len]
    prompt_rope_sin = torch_rope_sin[:prompt_seq_len]
    spatial_rope_cos = torch_rope_cos[prompt_seq_len:]
    spatial_rope_sin = torch_rope_sin[prompt_seq_len:]

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_text_encoder_layers = [tensor.from_torch(layer, device=mesh_device) for layer in text_encoder_layers]
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)

    tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=mesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=mesh_device)

    logger.info("running TT model...")
    tt_output = tracer(
        spatial=tt_spatial,
        prompt=tt_prompt,
        text_encoder_layers=tt_text_encoder_layers,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    logger.info("running torch reference...")
    with torch.no_grad():
        # FIBO's diffusers transformer takes img_ids/txt_ids separately and computes RoPE
        # internally; pad/trim of text_encoder_layers is the pipeline's responsibility, so the
        # transformer receives exactly ``num_layers + num_single_layers`` entries here.
        h = latents_height // _PATCH_SIZE
        w = latents_width // _PATCH_SIZE
        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] = torch.arange(h)[:, None]
        img_ids[..., 2] = torch.arange(w)[None, :]
        img_ids = img_ids.reshape(h * w, 3)
        txt_ids = torch.zeros(prompt_seq_len, 3)

        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            text_encoder_layers=text_encoder_layers,
            pooled_projections=None,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.998, relative_rmse=0.06)
