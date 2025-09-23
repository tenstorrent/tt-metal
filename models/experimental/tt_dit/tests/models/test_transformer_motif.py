# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...models.transformers.transformer_motif import MotifTransformer, convert_motif_transformer_state
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
        # pytest.param((2, 4), (2, 1), 1, 0, 1, id="2x1sp1tp0"),
        # pytest.param((2, 4), (2, 2), 0, 1, 1, id="2x2sp0tp1"),
        # pytest.param((2, 4), (2, 2), 1, 0, 1, id="2x2sp1tp0"),
        # pytest.param((2, 4), (2, 4), 0, 1, 1, id="2x4sp0tp1"),
        # pytest.param((2, 4), (2, 4), 1, 0, 1, id="2x4sp1tp0"),
        # pytest.param((4, 8), (4, 4), 0, 1, 4, id="4x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_transformer_motif(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
) -> None:
    model_path = "motif_image_preview.bin"

    # checkpoint config
    modulation_dim = 4096  # new parameter (corresponds to hidden_dim in SD3 large in some places)
    time_embed_dim = 4096  # new parameter (a related constant in SD3 large is 256)
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
    state_dict = substate(state_dict, "dit")

    logger.info("loading state dict into Torch model...")
    missing, unexpected = torch_model.load_state_dict(state_dict, strict=False)
    assert unexpected == ["pos_embed"]
    assert not missing

    logger.info("loading state dict into TT-NN model...")
    converted_state_dict = dict(state_dict)
    convert_motif_transformer_state(converted_state_dict, num_layers=num_layers)
    tt_model.load_torch_state_dict(converted_state_dict, strict=False)

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, in_channels, height // vae_scale_factor, width // vae_scale_factor])
    pooled = torch.randn([batch_size, 2048])
    timestep = torch.full([batch_size], fill_value=500)

    prompt_embeddings = [
        torch.randn([2, 256, 4096]),
        torch.randn([2, 77, 768]),
        torch.randn([2, 77, 1280]),
    ]
    prompt = _combine_prompt_embeddings(*prompt_embeddings)

    tt_spatial = bf16_tensor(spatial.permute(0, 2, 3, 1), device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
    tt_prompt = bf16_tensor(prompt, device=submesh_device)
    tt_pooled = bf16_tensor(pooled, device=submesh_device)
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=submesh_device
    )

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(spatial, timestep, prompt_embeddings, pooled)

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled=tt_pooled,
        timestep=tt_timestep,
    )

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 0
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.create_mesh_composer(submesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )[:batch_size].permute(0, 3, 1, 2)

    assert_quality(torch_output, tt_output_torch, pcc=0.9986, relative_rmse=5.6)


def _combine_prompt_embeddings(t5: torch.Tensor, clip_a: torch.Tensor, clip_b: torch.Tensor) -> torch.Tensor:
    clip_emb = torch.cat([clip_a, clip_b], dim=-1)
    clip_emb = torch.nn.functional.pad(clip_emb, (0, t5.shape[-1] - clip_emb.shape[-1]))
    return torch.cat([clip_emb, t5], dim=-2)
