# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...models.transformers.transformer_motif import MotifTransformer
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.motif_image import configuration_motifimage, modeling_dit
from ...utils.check import assert_quality
from ...utils.padding import PaddingConfig
from ...utils.substate import substate
from ...utils.tensor import bf16_tensor


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), (1, 2), 0, 1, 1, id="1x2sp0tp1"),
        pytest.param((2, 4), (2, 1), 1, 0, 1, id="2x1sp1tp0"),
        pytest.param((2, 4), (2, 2), 0, 1, 1, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 2), 1, 0, 1, id="2x2sp1tp0"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, id="2x4sp1tp0"),
        pytest.param((4, 8), (4, 4), 0, 1, 4, id="4x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "prompt_seq_len"),
    [
        (2, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_transformer_motif(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    prompt_seq_len: int,
) -> None:
    model_path = "motif_image_preview.bin"

    # checkpoint config
    modulation_dim = 4096  # new parameter (is hidden_dim in SD3 large in some places)
    time_embed_dim = 4096  # new parameter (related constant in SD3 large is 256)
    register_token_num = 4
    num_layers = 30
    num_heads = 30
    hidden_dim = 1920
    pos_emb_size = 64
    patch_size = 2
    pooled_text_dim = 2048

    # model config
    in_channels = 16
    vae_scale_factor = 8

    height = 1024
    width = 1024
    head_dim = hidden_dim // num_heads

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    torch_model = modeling_dit.MotifDiT(
        configuration_motifimage.MotifImageConfig(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            in_channel=4,
            out_channel=4,
            time_embed_dim=time_embed_dim,
            attn_embed_dim=4096,
            num_attention_heads=num_heads,
            num_key_value_heads=30,
            use_scaled_dot_product_attention=True,
            dropout=0.0,
            mlp_hidden_dim=7680,
            use_modulation=True,
            modulation_type="film",
            register_token_num=register_token_num,
            additional_register_token_num=0,
            skip_register_token_num=0,
            attn_mode="flash",
            use_final_layer_norm=False,
            pos_emb_size=pos_emb_size,
            conv_header=False,
            use_time_token_in_attn=True,
            modulation_dim=modulation_dim,
            pooled_text_dim=pooled_text_dim,
        )
    )
    torch_model.eval()

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

    tt_model = MotifTransformer(
        patch_size=patch_size,
        num_layers=num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        pooled_projection_dim=pooled_text_dim,
        pos_embed_max_size=pos_emb_size,
        modulation_dim=modulation_dim,
        time_embed_dim=time_embed_dim,
        register_token_num=register_token_num,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    logger.info("loading state dict...")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), mmap=True)

    logger.info("loading state dict into Torch model...")
    torch_model.load_state_dict(substate(state_dict, "dit"))

    logger.info("loading state dict into TT-NN model...")
    tt_model.load_torch_state_dict(state_dict)

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, in_channels, height // vae_scale_factor, width // vae_scale_factor])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    pooled = torch.randn([batch_size, pooled_projection_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3) if with_guidance_embeds else None

    # prepare for ROPE
    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = torch.randint(1024 * 1024, [spatial_seq_len, 3])
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = torch_model.pos_embed.forward(ids)

    tt_spatial = bf16_tensor(spatial.permute(0, 2, 3, 1), device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
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
    # tt_rope_cos = bf16_tensor(rope_cos, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    # tt_rope_sin = bf16_tensor(rope_sin, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)

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
        # combined_rope=(tt_rope_cos, tt_rope_sin),
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
