# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import diffusers as reference
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0
from models.tt_dit.models.transformers.transformer_fibo import FiboCheckpoint, image_ids
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tracing import Tracer


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), 0, 1, 4, id="4x8sp0tp1_wh", marks=run_for_wormhole_b0()),
        pytest.param((4, 8), 0, 1, 2, id="4x8sp0tp1_bh", marks=run_for_blackhole()),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "latents_height", "latents_width", "prompt_seq_len"),
    [
        (1, 32, 32, 3008),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 34000000}],
    indirect=True,
)
@pytest.mark.parametrize("with_reference", [False, True], ids=["without_reference", "with_reference"])
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
    with_reference: bool,
) -> None:
    torch.manual_seed(0)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    checkpoint_name = "briaai/FIBO"
    torch_model = reference.BriaFiboTransformer2DModel.from_pretrained(
        checkpoint_name, subfolder="transformer", torch_dtype=torch.float32
    )
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

    spatial_seq_len = latents_height * latents_width

    tracer = Tracer(tt_model.forward, device=mesh_device)

    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    # SmolLM3-3B emits 37 hidden states (36 layers plus the embedding), fewer than the transformer
    # has blocks; the remaining blocks reuse the last one.
    text_encoder_layers = [torch.randn([batch_size, prompt_seq_len, text_encoder_dim]) for _ in range(37)]
    timestep = torch.full([batch_size], fill_value=500.0)

    # FIBO Edit appends the reference image's tokens to the target's, at the same resolution.
    reference_tokens = torch.randn([batch_size, spatial_seq_len, in_channels])

    tt_spatial_rope, tt_prompt_rope = checkpoint.rope_tables(
        latents_height=latents_height,
        latents_width=latents_width,
        prompt_sequence_length=prompt_seq_len,
        with_reference=with_reference,
        device=mesh_device,
        sp_axis=sp_axis,
    )

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_reference = (
        tensor.from_torch(reference_tokens, device=mesh_device, mesh_axes=[None, sp_axis, None])
        if with_reference
        else None
    )
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_text_encoder_layers = [tensor.from_torch(layer, device=mesh_device) for layer in text_encoder_layers]
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)

    logger.info("running TT model...")
    tt_output = tracer(
        spatial=tt_spatial,
        prompt=tt_prompt,
        text_encoder_layers=tt_text_encoder_layers,
        timestep=tt_timestep,
        spatial_rope=tt_spatial_rope,
        prompt_rope=tt_prompt_rope,
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
        reference=tt_reference,
    )

    logger.info("running torch reference...")
    with torch.no_grad():
        # FIBO's diffusers transformer takes img_ids/txt_ids separately and computes RoPE
        # internally, and expects exactly one text encoder layer per block; the diffusers pipeline
        # pads with copies of the last one, which the TT transformer does itself.
        padded_text_encoder_layers = text_encoder_layers + [text_encoder_layers[-1]] * (
            total_num_blocks - len(text_encoder_layers)
        )

        torch_output = torch_model.forward(
            hidden_states=torch.cat([spatial, reference_tokens], dim=1) if with_reference else spatial,
            encoder_hidden_states=prompt,
            text_encoder_layers=padded_text_encoder_layers,
            pooled_projections=None,
            timestep=timestep,
            img_ids=image_ids(
                latents_height=latents_height, latents_width=latents_width, with_reference=with_reference
            ),
            txt_ids=torch.zeros(prompt_seq_len, 3),
            guidance=None,
            return_dict=False,
        )[0][:, :spatial_seq_len]

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.998, relative_rmse=0.06)
