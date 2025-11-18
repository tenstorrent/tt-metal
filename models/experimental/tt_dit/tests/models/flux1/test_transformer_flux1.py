# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# import diffusers.models.transformers.transformer_flux as reference
import diffusers as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....models.transformers.transformer_flux1 import (
    Flux1SingleTransformerBlock,
    Flux1Transformer,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from time import time


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((1, 4), (1, 4), 0, 1, 1, "1x4sp0tp1", id="1x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, "2x4sp1tp0", id="2x4sp1tp0"),
        pytest.param((4, 8), (4, 4), 0, 1, 4, "4x4sp0tp1", id="4x4sp0tp1"),
        pytest.param((4, 8), (4, 8), 0, 1, 4, "4x8sp0tp1", id="4x8sp0tp1"),
        pytest.param((4, 8), (4, 8), 1, 0, 4, "4x8sp1tp0", id="4x8sp1tp0"),
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
def test_single_transformer_block(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    prompt_seq_len: int,
    spatial_seq_len: int,
    id: str,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    model_name = model_location_generator(f"black-forest-labs/FLUX.1-dev", model_subdir="transformer")
    parent_torch_model = reference.FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer")
    torch_model = parent_torch_model.single_transformer_blocks[0]
    assert isinstance(torch_model, reference.models.transformers.transformer_flux.FluxSingleTransformerBlock)
    torch_model.eval()

    inner_dim = torch_model.attn.inner_dim
    num_heads = torch_model.attn.heads
    head_dim = inner_dim // num_heads

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

    tt_model = Flux1SingleTransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    combined = torch.randn([batch_size, prompt_seq_len + spatial_seq_len, inner_dim])
    time_embed = torch.randn([batch_size, inner_dim])
    rope_cos = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])
    rope_sin = torch.randn([prompt_seq_len + spatial_seq_len, head_dim])

    tt_spatial = bf16_tensor_2dshard(
        combined[:, prompt_seq_len:], device=submesh_device, shard_mapping={sp_axis: 1, tp_axis: 2}
    )
    tt_prompt = bf16_tensor(combined[:, :prompt_seq_len], device=submesh_device, mesh_axis=tp_axis, shard_dim=2)
    tt_time_embed = bf16_tensor(time_embed.unsqueeze(1), device=submesh_device)
    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=submesh_device)

    with torch.no_grad():
        torch_combined = torch_model.forward(
            hidden_states=combined[:, prompt_seq_len:],
            encoder_hidden_states=combined[:, :prompt_seq_len],
            temb=time_embed,
            image_rotary_emb=(rope_cos, rope_sin),
        )

    if not is_ci_env:
        try:
            from tracy import signpost

            signpost("caching")
            tt_spatial_out, tt_prompt_out = tt_model.forward(
                spatial=tt_spatial,
                prompt=tt_prompt,
                time_embed=tt_time_embed,
                spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
                prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
                spatial_sequence_length=spatial_seq_len,
            )

            signpost("performance")
        except ImportError:
            logger.info("Tracy profiler not available, continuing without profiling")

    itr = 1
    start = time()
    for _ in range(itr):
        tt_spatial_out, tt_prompt_out = tt_model.forward(
            spatial=tt_spatial,
            prompt=tt_prompt,
            time_embed=tt_time_embed,
            spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
            prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
            spatial_sequence_length=spatial_seq_len,
        )
        # ttnn.synchronize_device(submesh_device)
    ttnn.synchronize_device(submesh_device)
    logger.info(f"Time taken for {id}: {(time() - start)*1000/itr} ms")

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 2
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 0, 2
    tt_prompt_torch = ttnn.to_torch(
        tt_prompt_out,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size]

    tt_combined_torch = torch.concat([tt_prompt_torch, tt_spatial_torch], dim=1)
    torch_combined = torch.concat(torch_combined, dim=1)

    assert_quality(torch_combined, tt_combined_torch, pcc=0.99959, relative_rmse=6.3)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "id"),
    [
        pytest.param((1, 4), (1, 4), 0, 1, 1, "1x4sp0tp1", id="1x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, "2x4sp1tp0", id="2x4sp1tp0"),
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
def test_transformer(
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
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    # Flux.1 variant "dev" is like "schnell" but with additional guidance parameter.
    model_name = model_location_generator(f"black-forest-labs/FLUX.1-dev", model_subdir="transformer")
    torch_model = reference.FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer")
    assert isinstance(torch_model, reference.FluxTransformer2DModel)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim
    pooled_projection_dim = torch_model.config.pooled_projection_dim
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

    tt_model = Flux1Transformer(
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers,
        num_single_layers=torch_model.config.num_single_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        pooled_projection_dim=pooled_projection_dim,
        out_channels=torch_model.out_channels,
        axes_dims_rope=torch_model.config.axes_dims_rope,
        with_guidance_embeds=with_guidance_embeds,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    if not cache.initialize_from_cache(
        tt_model, torch_model, "Flux.1-dev", "transformer", parallel_config, tuple(submesh_device.shape), "bf16"
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    pooled = torch.randn([batch_size, pooled_projection_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3) if with_guidance_embeds else None

    # prepare for ROPE
    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = torch.randint(1024 * 1024, [spatial_seq_len, 3])
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = torch_model.pos_embed.forward(ids)

    tt_spatial = bf16_tensor(spatial, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
    tt_prompt = bf16_tensor(prompt, device=submesh_device)
    tt_pooled = bf16_tensor(pooled, device=submesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=submesh_device
    )
    tt_guidance = bf16_tensor(guidance.unsqueeze(-1), device=submesh_device) if guidance is not None else None

    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=submesh_device)

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            timestep=timestep / 1000,
            guidance=guidance,
            img_ids=image_ids,
            txt_ids=text_ids,
        ).sample

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled=tt_pooled,
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
