# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""PCC tests for Flux2 image-to-image pipeline components."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux2.pipeline_flux2 import _extract_noise_latents_from_combined
from ....utils import tensor
from ....utils.check import assert_quality
from .test_pipeline_flux2 import line_params_flux2


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params",
    [
        [(4, 8), 0, 1, 4, line_params_flux2],
        [(1, 8), 0, 1, 1, line_params_flux2],
    ],
    ids=["wh_4x8", "wh_1x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("noise_seq_len", "image_seq_len", "channels"),
    [
        (4096, 4096, 128),
        (1024, 256, 128),
    ],
    ids=["symmetric_1024", "asymmetric_512_256"],
)
def test_combine_img2img_spatial_input(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    noise_seq_len: int,
    image_seq_len: int,
    channels: int,
) -> None:
    """PCC: torch.cat([noise, image]) then SP shard matches global [noise | image] layout."""
    batch_size = 1

    torch.manual_seed(0)
    noise = torch.randn(batch_size, noise_seq_len, channels)
    image = torch.randn(batch_size, image_seq_len, channels)
    ref_combined = torch.cat([noise, image], dim=1)
    ref_combined_torch = tensor.to_torch(
        tensor.from_torch(ref_combined, device=mesh_device, mesh_axes=[None, sp_axis, None]),
        mesh_axes=[None, sp_axis, None],
    )
    ttnn.synchronize_device(mesh_device)

    assert_quality(ref_combined, ref_combined_torch, pcc=0.99999, relative_rmse=0.003)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params",
    [
        [(4, 8), 0, 1, 4, line_params_flux2],
        [(1, 8), 0, 1, 1, line_params_flux2],
    ],
    ids=["wh_4x8", "wh_1x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("noise_seq_len", "image_seq_len", "channels"),
    [
        (4096, 4096, 128),
        (1024, 256, 128),
    ],
    ids=["symmetric_1024", "asymmetric_512_256"],
)
def test_extract_noise_latents_from_combined(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    noise_seq_len: int,
    image_seq_len: int,
    channels: int,
) -> None:
    """PCC: noise-prefix extraction from combined [noise | image] tensor matches diffusers slice."""
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    batch_size = 1
    total_seq_len = noise_seq_len + image_seq_len

    torch.manual_seed(1)
    ref_pred = torch.randn(batch_size, total_seq_len, channels)
    ref_noise_pred = ref_pred[:, :noise_seq_len]

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    tt_pred = tensor.from_torch(ref_pred, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_noise_pred = _extract_noise_latents_from_combined(
        ccl_manager,
        parallel_config,
        tt_pred,
        noise_sequence_length=noise_seq_len,
    )
    ttnn.synchronize_device(mesh_device)

    tt_noise_pred_torch = tensor.to_torch(tt_noise_pred, mesh_axes=[None, sp_axis, None])
    assert_quality(ref_noise_pred, tt_noise_pred_torch, pcc=0.99999, relative_rmse=0.003)


def test_img2img_spatial_rope_ids(tt_dit_cache_dir) -> None:
    """PCC: combined noise + image position IDs match diffusers Flux2Pipeline."""
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    from ....pipelines.flux2.pipeline_flux2 import _prepare_ids, _prepare_image_ids

    torch.manual_seed(0)
    noise_h, noise_w = 64, 64
    cond_latent = torch.randn(1, 128, 32, 32)

    noise_ids = Flux2Pipeline._prepare_latent_ids(torch.randn(1, 128, noise_h, noise_w))[0]
    ours_noise_ids = _prepare_ids(height=noise_h, width=noise_w, text_sequence_length=1)
    assert torch.equal(noise_ids, ours_noise_ids)

    ref_image_ids = Flux2Pipeline._prepare_image_ids([cond_latent[0]]).squeeze(0)
    ours_image_ids = _prepare_image_ids([cond_latent[0]]).squeeze(0)
    assert torch.equal(ref_image_ids, ours_image_ids)

    ref_combined = torch.cat([noise_ids, ref_image_ids], dim=0)
    ours_combined = torch.cat([ours_noise_ids, ours_image_ids], dim=0)
    assert torch.equal(ref_combined, ours_combined)


def test_img2img_euler_step_matches_torch(tt_dit_cache_dir) -> None:
    """PCC: manual flow-match Euler update on noise latents matches torch reference."""
    torch.manual_seed(2)
    batch_size = 1
    noise_seq_len = 256
    channels = 128

    spatial = torch.randn(batch_size, noise_seq_len, channels)
    noise_pred = torch.randn(batch_size, noise_seq_len, channels)
    sigma_difference = -0.05

    ref_spatial = spatial + noise_pred * sigma_difference

    spatial_out = spatial.clone()
    spatial_out.add_(noise_pred, alpha=sigma_difference)

    assert torch.allclose(ref_spatial, spatial_out)
    assert_quality(ref_spatial, spatial_out, pcc=0.99999, relative_rmse=0.001)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, topology, num_links, device_params",
    [
        [(4, 8), 0, 1, ttnn.Topology.Linear, 4, line_params_flux2],
    ],
    ids=["wh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_transformer_img2img_separate_latents(
    tt_dit_cache_dir,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    model_location_generator,
) -> None:
    """PCC: transformer img2img via separate noise/image tensors (production path)."""
    import diffusers

    from ....models.transformers.transformer_flux2 import Flux2Transformer
    from ....pipelines.flux2.pipeline_flux2 import _prepare_image_ids
    from ....utils import cache
    from ....utils.padding import PaddingConfig
    from .test_transformer_flux2 import _prepare_ids

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    batch_size = 1
    height = 1024
    width = 1024
    cond_height = 512
    cond_width = 512
    prompt_seq_len = 512

    skip_layers = 7
    skip_single_layers = 47

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

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
        if num_heads % tp_factor != 0
        else None
    )

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
    cache.load_model(
        tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="FLUX.2-dev",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
    )

    noise_seq_len = height * width // 16**2
    cond_seq_len = cond_height * cond_width // 16**2
    total_seq_len = noise_seq_len + cond_seq_len

    torch.manual_seed(0)
    noise_latents = torch.randn([batch_size, noise_seq_len, in_channels])
    cond_latents = torch.randn([batch_size, cond_seq_len, in_channels])
    concat_latents = torch.cat([noise_latents, cond_latents], dim=1)

    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3)

    text_ids = _prepare_ids(text_sequence_length=prompt_seq_len)
    noise_ids = _prepare_ids(height=height // 16, width=width // 16, text_sequence_length=1)
    cond_patchified = torch.randn(1, in_channels, cond_height // 16, cond_width // 16)
    cond_ids = _prepare_image_ids([cond_patchified[0]]).squeeze(0)
    combined_ids = torch.cat([noise_ids, cond_ids], dim=0)

    prompt_rope_cos, prompt_rope_sin = torch_model.pos_embed.forward(text_ids)
    spatial_rope_cos, spatial_rope_sin = torch_model.pos_embed.forward(combined_ids)

    ref_concat = torch.cat([noise_latents, cond_latents], dim=1)
    tt_concat = tensor.from_torch(ref_concat, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)
    tt_guidance = tensor.from_torch(guidance.unsqueeze(-1), device=mesh_device)
    tt_spatial_rope_cos = tensor.from_torch(
        spatial_rope_cos.unsqueeze(0).unsqueeze(0), device=mesh_device, mesh_axes=[None, None, sp_axis, None]
    )
    tt_spatial_rope_sin = tensor.from_torch(
        spatial_rope_sin.unsqueeze(0).unsqueeze(0), device=mesh_device, mesh_axes=[None, None, sp_axis, None]
    )
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos.unsqueeze(0).unsqueeze(0), device=mesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin.unsqueeze(0).unsqueeze(0), device=mesh_device)

    logger.info("running Torch model (img2img concat)...")
    with torch.no_grad():
        torch_output_full = torch_model.forward(
            hidden_states=concat_latents,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,
            guidance=guidance,
            img_ids=combined_ids,
            txt_ids=text_ids,
        ).sample
    torch_output = torch_output_full[:, :noise_seq_len]

    logger.info("running TT model (img2img separate latents + combine)...")
    tt_output_full = tt_model.forward(
        spatial=tt_concat,
        prompt=tt_prompt,
        timestep=tt_timestep,
        guidance=tt_guidance,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=total_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )
    tt_output_sliced = _extract_noise_latents_from_combined(
        ccl_manager, parallel_config, tt_output_full, noise_sequence_length=noise_seq_len
    )
    tt_output_torch = tensor.to_torch(tt_output_sliced, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.996, relative_rmse=0.09)
