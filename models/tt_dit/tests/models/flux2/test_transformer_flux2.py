# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock as HFFlux2SingleTransformerBlock
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel, Flux2TransformerBlock
from loguru import logger

import ttnn

from ....models.transformers.transformer_flux2 import (
    Flux2DoubleTransformerBlock,
    Flux2SingleTransformerBlock,
    Flux2Transformer,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils import matmul as matmul_utils
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard

# ---------------------------------------------------------------------------
# Block-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((1, 2), (1, 2), 0, 1, 2, "1x2sp0tp1", id="1x2sp0tp1"),
        pytest.param((2, 2), (2, 2), 0, 1, 2, "2x2sp0tp1", id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_flux2_double_transformer_block(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    model_name = model_location_generator("black-forest-labs/FLUX.2-dev", model_subdir="transformer")
    parent_torch_model = Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
    parent_torch_model.eval()

    torch_block = parent_torch_model.transformer_blocks[0]
    assert isinstance(torch_block, Flux2TransformerBlock)

    cfg = parent_torch_model.config
    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    num_heads = cfg.num_attention_heads
    head_dim = cfg.attention_head_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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

    tt_block = Flux2DoubleTransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=cfg.mlp_ratio,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, inner_dim])
    prompt = torch.randn([batch_size, prompt_seq_len, inner_dim])
    temb = torch.randn([batch_size, inner_dim])

    with torch.no_grad():
        mod_img = parent_torch_model.double_stream_modulation_img(temb)
        mod_txt = parent_torch_model.double_stream_modulation_txt(temb)

    rope_cos = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])
    rope_sin = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])
    concat_rotary_emb = (rope_cos, rope_sin)

    with torch.no_grad():
        torch_enc_out, torch_hidden_out = torch_block.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            temb_mod_img=mod_img,
            temb_mod_txt=mod_txt,
            image_rotary_emb=concat_rotary_emb,
        )

    tt_spatial = bf16_tensor_2dshard(
        spatial,
        device=submesh_device,
        shard_mapping={sp_axis: 1, tp_axis: 2},
    )
    tt_prompt = bf16_tensor(prompt, device=submesh_device, mesh_axis=tp_axis, shard_dim=2)

    tt_mod_img = bf16_tensor(mod_img, device=submesh_device, mesh_axis=tp_axis, shard_dim=-1)
    tt_mod_txt = bf16_tensor(mod_txt, device=submesh_device, mesh_axis=tp_axis, shard_dim=-1)

    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=submesh_device)

    logger.info("running TT double block...")
    tt_spatial_out, tt_prompt_out = tt_block.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        mod_img=tt_mod_img,
        mod_txt=tt_mod_txt,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
    )
    ttnn.synchronize_device(submesh_device)

    shard_dims_spatial = [None, None]
    shard_dims_spatial[sp_axis], shard_dims_spatial[tp_axis] = 1, 2
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims_spatial)),
    )[:batch_size]

    shard_dims_prompt = [None, None]
    shard_dims_prompt[sp_axis], shard_dims_prompt[tp_axis] = 0, 2
    tt_prompt_torch = ttnn.to_torch(
        tt_prompt_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims_prompt)),
    )[:batch_size]

    torch_combined = torch.cat([torch_enc_out, torch_hidden_out], dim=1)
    tt_combined = torch.cat([tt_prompt_torch, tt_spatial_torch], dim=1)

    assert_quality(torch_combined, tt_combined, pcc=0.999, relative_rmse=6.3)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((1, 2), (1, 2), 0, 1, 2, "1x2sp0tp1", id="1x2sp0tp1"),
        pytest.param((2, 2), (2, 2), 0, 1, 2, "2x2sp0tp1", id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_flux2_single_transformer_block(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    model_name = model_location_generator("black-forest-labs/FLUX.2-dev", model_subdir="transformer")
    parent_torch_model = Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
    parent_torch_model.eval()

    torch_block = parent_torch_model.single_transformer_blocks[0]
    assert isinstance(torch_block, HFFlux2SingleTransformerBlock)

    cfg = parent_torch_model.config
    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    num_heads = cfg.num_attention_heads
    head_dim = cfg.attention_head_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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

    tt_block = Flux2SingleTransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=cfg.mlp_ratio,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, inner_dim])
    prompt = torch.randn([batch_size, prompt_seq_len, inner_dim])
    temb = torch.randn([batch_size, inner_dim])

    with torch.no_grad():
        mod = parent_torch_model.single_stream_modulation(temb)

    rope_cos = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])
    rope_sin = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])
    concat_rotary_emb = (rope_cos, rope_sin)

    with torch.no_grad():
        torch_enc_out, torch_hidden_out = torch_block.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            temb_mod=mod,
            image_rotary_emb=concat_rotary_emb,
            split_hidden_states=True,
        )

    tt_spatial = bf16_tensor_2dshard(
        spatial,
        device=submesh_device,
        shard_mapping={sp_axis: 1, tp_axis: 2},
    )
    tt_prompt = bf16_tensor(prompt, device=submesh_device, mesh_axis=tp_axis, shard_dim=2)

    tt_mod = bf16_tensor(mod, device=submesh_device, mesh_axis=tp_axis, shard_dim=-1)

    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=submesh_device)

    logger.info("running TT single block...")
    tt_spatial_out, tt_prompt_out = tt_block.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        mod=tt_mod,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
    )
    ttnn.synchronize_device(submesh_device)

    shard_dims_spatial = [None, None]
    shard_dims_spatial[sp_axis], shard_dims_spatial[tp_axis] = 1, 2
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims_spatial)),
    )[:batch_size]

    shard_dims_prompt = [None, None]
    shard_dims_prompt[sp_axis], shard_dims_prompt[tp_axis] = 0, 2
    tt_prompt_torch = ttnn.to_torch(
        tt_prompt_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims_prompt)),
    )[:batch_size]

    torch_combined = torch.cat([torch_enc_out, torch_hidden_out], dim=1)
    tt_combined = torch.cat([tt_prompt_torch, tt_spatial_torch], dim=1)

    assert_quality(torch_combined, tt_combined, pcc=0.999, relative_rmse=6.3)


# ---------------------------------------------------------------------------
# Full transformer test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dit_unit_test",
    [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
)
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        # BH Only
        pytest.param((1, 2), (1, 2), 0, 1, 2, "1x2sp0tp1", id="1x2sp0tp1"),
        pytest.param((2, 2), (2, 2), 0, 1, 2, "2x2sp0tp1", id="2x2sp0tp1"),
        # WH Only
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, "2x4sp1tp0", id="2x4sp1tp0"),
        # BH and WH
        pytest.param((4, 8), (4, 4), 0, 1, 4, "4x4sp0tp1", id="4x4sp0tp1"),
        pytest.param((4, 8), (4, 8), 1, 0, 4, "4x8sp1tp0", id="4x8sp1tp0"),
        pytest.param((4, 8), (4, 8), 0, 1, 4, "4x8sp0tp1", id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_flux2_transformer(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
    model_location_generator,
    dit_unit_test: bool,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    if dit_unit_test:
        torch_model = Flux2Transformer2DModel(num_layers=1, num_single_layers=1)
    else:
        model_name = model_location_generator("black-forest-labs/FLUX.2-dev", model_subdir="transformer")
        torch_model = Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
    assert isinstance(torch_model, Flux2Transformer2DModel)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim
    with_guidance_embeds = torch_model.config.guidance_embeds

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers,
        num_single_layers=torch_model.config.num_single_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=torch_model.config.timestep_guidance_channels,
        out_channels=torch_model.out_channels,
        mlp_ratio=torch_model.config.mlp_ratio,
        guidance_embeds=with_guidance_embeds,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    cache.load_model(
        tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="Flux.2-dev",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(submesh_device.shape),
    )

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3) if with_guidance_embeds else None

    pos_embed = Flux2PosEmbed(
        theta=torch_model.config.rope_theta,
        axes_dim=list(torch_model.config.axes_dims_rope),
    )
    latent_height = latent_width = int(spatial_seq_len**0.5)
    t_ids = torch.arange(1)
    h_ids = torch.arange(latent_height)
    w_ids = torch.arange(latent_width)
    l_ids = torch.arange(1)
    latent_ids = torch.cartesian_prod(t_ids, h_ids, w_ids, l_ids)

    t_ids_t = torch.arange(1)
    h_ids_t = torch.arange(1)
    w_ids_t = torch.arange(1)
    l_ids_t = torch.arange(prompt_seq_len)
    text_ids = torch.cartesian_prod(t_ids_t, h_ids_t, w_ids_t, l_ids_t)

    img_rope_cos, img_rope_sin = pos_embed.forward(latent_ids)
    txt_rope_cos, txt_rope_sin = pos_embed.forward(text_ids)

    tt_spatial = bf16_tensor(spatial, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
    tt_prompt = bf16_tensor(prompt, device=submesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=submesh_device,
    )
    tt_guidance = bf16_tensor(guidance.unsqueeze(-1), device=submesh_device) if guidance is not None else None

    tt_spatial_rope_cos = bf16_tensor(img_rope_cos, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(img_rope_sin, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(txt_rope_cos, device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(txt_rope_sin, device=submesh_device)

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,
            guidance=guidance,
            img_ids=latent_ids,
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

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 0
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size]

    assert_quality(torch_output, tt_output_torch, pcc=0.997, relative_rmse=8.1)


# ---------------------------------------------------------------------------
# Matmul fallback coverage test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_flux2_no_matmul_fallback(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
    model_location_generator,
) -> None:
    """Verify that every matmul shape hit during a full Flux2 forward pass has an explicit config entry."""
    matmul_utils._warned_matmul_signatures.clear()

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    torch_model = Flux2Transformer2DModel(num_layers=1, num_single_layers=1)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=torch_model.config.timestep_guidance_channels,
        out_channels=torch_model.out_channels,
        mlp_ratio=torch_model.config.mlp_ratio,
        guidance_embeds=torch_model.config.guidance_embeds,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3)

    pos_embed = Flux2PosEmbed(
        theta=torch_model.config.rope_theta,
        axes_dim=list(torch_model.config.axes_dims_rope),
    )
    latent_height = latent_width = int(spatial_seq_len**0.5)
    t_ids = torch.arange(1)
    h_ids = torch.arange(latent_height)
    w_ids = torch.arange(latent_width)
    l_ids = torch.arange(1)
    latent_ids = torch.cartesian_prod(t_ids, h_ids, w_ids, l_ids)

    t_ids_t = torch.arange(1)
    h_ids_t = torch.arange(1)
    w_ids_t = torch.arange(1)
    l_ids_t = torch.arange(prompt_seq_len)
    text_ids = torch.cartesian_prod(t_ids_t, h_ids_t, w_ids_t, l_ids_t)

    img_rope_cos, img_rope_sin = pos_embed.forward(latent_ids)
    txt_rope_cos, txt_rope_sin = pos_embed.forward(text_ids)

    tt_spatial = bf16_tensor(spatial, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
    tt_prompt = bf16_tensor(prompt, device=submesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=submesh_device,
    )
    tt_guidance = bf16_tensor(guidance.unsqueeze(-1), device=submesh_device)

    tt_spatial_rope_cos = bf16_tensor(img_rope_cos, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(img_rope_sin, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(txt_rope_cos, device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(txt_rope_sin, device=submesh_device)

    matmul_utils._warned_matmul_signatures.clear()

    tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        guidance=tt_guidance,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    fallback_signatures = matmul_utils._warned_matmul_signatures.copy()
    assert (
        len(fallback_signatures) == 0
    ), f"Matmul fallback warnings fired for {len(fallback_signatures)} shapes: {fallback_signatures}"


# ---------------------------------------------------------------------------
# Full transformer test -- no guidance variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_flux2_transformer_no_guidance(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
    model_location_generator,
) -> None:
    """Exercises the guidance_embeds=False branch of Flux2TimestepGuidanceEmbeddings."""
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    torch_model = Flux2Transformer2DModel(
        num_layers=1,
        num_single_layers=1,
        guidance_embeds=False,
    )
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=torch_model.config.timestep_guidance_channels,
        out_channels=torch_model.out_channels,
        mlp_ratio=torch_model.config.mlp_ratio,
        guidance_embeds=False,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)

    pos_embed = Flux2PosEmbed(
        theta=torch_model.config.rope_theta,
        axes_dim=list(torch_model.config.axes_dims_rope),
    )
    latent_height = latent_width = int(spatial_seq_len**0.5)
    t_ids = torch.arange(1)
    h_ids = torch.arange(latent_height)
    w_ids = torch.arange(latent_width)
    l_ids = torch.arange(1)
    latent_ids = torch.cartesian_prod(t_ids, h_ids, w_ids, l_ids)

    t_ids_t = torch.arange(1)
    h_ids_t = torch.arange(1)
    w_ids_t = torch.arange(1)
    l_ids_t = torch.arange(prompt_seq_len)
    text_ids = torch.cartesian_prod(t_ids_t, h_ids_t, w_ids_t, l_ids_t)

    img_rope_cos, img_rope_sin = pos_embed.forward(latent_ids)
    txt_rope_cos, txt_rope_sin = pos_embed.forward(text_ids)

    tt_spatial = bf16_tensor(spatial, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
    tt_prompt = bf16_tensor(prompt, device=submesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=submesh_device,
    )

    tt_spatial_rope_cos = bf16_tensor(img_rope_cos, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(img_rope_sin, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(txt_rope_cos, device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(txt_rope_sin, device=submesh_device)

    logger.info("running torch model (no guidance)...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,
            guidance=None,
            img_ids=latent_ids,
            txt_ids=text_ids,
        ).sample

    logger.info("running TT model (no guidance)...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        guidance=None,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 0
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size]

    assert_quality(torch_output, tt_output_torch, pcc=0.997, relative_rmse=8.1)


# ---------------------------------------------------------------------------
# Traced vs untraced parity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_flux2_traced_parity(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    id: str,
) -> None:
    """Verify that traced execution produces the same output as untraced."""
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    torch_model = Flux2Transformer2DModel(num_layers=1, num_single_layers=1)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=torch_model.config.timestep_guidance_channels,
        out_channels=torch_model.out_channels,
        mlp_ratio=torch_model.config.mlp_ratio,
        guidance_embeds=torch_model.config.guidance_embeds,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3)

    pos_embed = Flux2PosEmbed(
        theta=torch_model.config.rope_theta,
        axes_dim=list(torch_model.config.axes_dims_rope),
    )
    latent_height = latent_width = int(spatial_seq_len**0.5)
    t_ids = torch.arange(1)
    h_ids = torch.arange(latent_height)
    w_ids = torch.arange(latent_width)
    l_ids = torch.arange(1)
    latent_ids = torch.cartesian_prod(t_ids, h_ids, w_ids, l_ids)

    t_ids_t = torch.arange(1)
    h_ids_t = torch.arange(1)
    w_ids_t = torch.arange(1)
    l_ids_t = torch.arange(prompt_seq_len)
    text_ids = torch.cartesian_prod(t_ids_t, h_ids_t, w_ids_t, l_ids_t)

    img_rope_cos, img_rope_sin = pos_embed.forward(latent_ids)
    txt_rope_cos, txt_rope_sin = pos_embed.forward(text_ids)

    def _make_inputs():
        tt_spatial = bf16_tensor(spatial, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
        tt_prompt_t = bf16_tensor(prompt, device=submesh_device)
        tt_timestep = ttnn.from_torch(
            timestep.unsqueeze(-1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=submesh_device,
        )
        tt_guidance = bf16_tensor(guidance.unsqueeze(-1), device=submesh_device)
        tt_spatial_rope_cos = bf16_tensor(img_rope_cos, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
        tt_spatial_rope_sin = bf16_tensor(img_rope_sin, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
        tt_prompt_rope_cos = bf16_tensor(txt_rope_cos, device=submesh_device)
        tt_prompt_rope_sin = bf16_tensor(txt_rope_sin, device=submesh_device)
        return (
            tt_spatial,
            tt_prompt_t,
            tt_timestep,
            tt_guidance,
            tt_spatial_rope_cos,
            tt_spatial_rope_sin,
            tt_prompt_rope_cos,
            tt_prompt_rope_sin,
        )

    def _run_forward(inputs):
        (s, p, ts, g, src, srs, prc, prs) = inputs
        return tt_model.forward(
            spatial=s,
            prompt=p,
            timestep=ts,
            guidance=g,
            spatial_rope=(src, srs),
            prompt_rope=(prc, prs),
            spatial_sequence_length=spatial_seq_len,
            prompt_sequence_length=prompt_seq_len,
        )

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 0

    logger.info("running untraced forward...")
    untraced_output = _run_forward(_make_inputs())
    untraced_torch = ttnn.to_torch(
        untraced_output,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size]

    logger.info("capturing trace...")
    trace_inputs = _make_inputs()
    tid = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    trace_output = _run_forward(trace_inputs)
    ttnn.end_trace_capture(submesh_device, tid, cq_id=0)

    logger.info("running traced forward...")
    fresh_inputs = _make_inputs()
    for i in range(8):
        ttnn.copy_host_to_device_tensor(fresh_inputs[i], trace_inputs[i])

    ttnn.execute_trace(submesh_device, tid, cq_id=0, blocking=True)

    traced_torch = ttnn.to_torch(
        trace_output,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size]

    ttnn.release_trace(submesh_device, tid)

    assert_quality(untraced_torch, traced_torch, pcc=0.99999, relative_rmse=0.1)


# ---------------------------------------------------------------------------
# Memory footprint test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_flux2_memory_footprint(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    id: str,
    model_location_generator,
) -> None:
    """Load the full Flux2 transformer with bf8 weights and report peak DRAM usage."""
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    model_name = model_location_generator("black-forest-labs/FLUX.2-dev", model_subdir="transformer")
    torch_model = Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer")
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    ccl_manager = CCLManager(
        mesh_device=submesh_device,
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
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers,
        num_single_layers=torch_model.config.num_single_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=torch_model.config.timestep_guidance_channels,
        out_channels=torch_model.out_channels,
        mlp_ratio=torch_model.config.mlp_ratio,
        guidance_embeds=torch_model.config.guidance_embeds,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
        weights_dtype=ttnn.bfloat8_b,
    )

    cache.load_model(
        tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="Flux.2-dev",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(submesh_device.shape),
        dtype="bf8",
    )

    for i, device in enumerate(submesh_device.get_devices()):
        mem_alloc = device.get_memory_allocation_statistics(ttnn.BufferType.DRAM)
        total_bytes = mem_alloc.total_allocatable_size_bytes
        allocated_bytes = mem_alloc.total_allocated_bytes
        usage_pct = allocated_bytes / total_bytes * 100 if total_bytes > 0 else 0
        logger.info(f"Device {i}: DRAM {allocated_bytes / 1e9:.2f} GB / {total_bytes / 1e9:.2f} GB ({usage_pct:.1f}%)")
        assert usage_pct < 80, f"Device {i} DRAM usage {usage_pct:.1f}% exceeds 80% threshold"
