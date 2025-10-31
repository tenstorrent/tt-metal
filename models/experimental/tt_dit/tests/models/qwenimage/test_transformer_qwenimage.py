# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....models.transformers.transformer_qwenimage import QwenImageTransformer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache, tensor
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), (2, 4), 0, 1, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), (4, 8), 0, 1, 4, id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "latents_height", "latents_width", "prompt_seq_len"),
    [
        (1, 128, 128, 512),  # TODO: set correct values
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
    latents_height: int,
    latents_width: int,
    prompt_seq_len: int,
) -> None:
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    torch_model = reference.QwenImageTransformer2DModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer")
    assert isinstance(torch_model, reference.QwenImageTransformer2DModel)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim
    patch_size = 2

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

    tt_model = QwenImageTransformer(
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        out_channels=torch_model.out_channels,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    if not cache.initialize_from_cache(
        tt_model, torch_model, "Qwen-Image", "transformer", parallel_config, tuple(submesh_device.shape), "bf16"
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())

    spatial_seq_len = (latents_height // patch_size) * (latents_width // patch_size)

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)

    # prepare ROPE
    img_shapes = [[(1, latents_height // patch_size, latents_width // patch_size)]] * batch_size
    txt_seq_lens = [prompt_seq_len] * batch_size
    spatial_rope, prompt_rope = torch_model.pos_embed.forward(img_shapes, txt_seq_lens, "cpu")

    spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

    tt_spatial = tensor.from_torch(spatial, device=submesh_device, mesh_axes=[None, sp_axis, None])
    tt_prompt = tensor.from_torch(prompt, device=submesh_device)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=submesh_device)

    tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=submesh_device, mesh_axes=[sp_axis, None])
    tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=submesh_device, mesh_axes=[sp_axis, None])
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=submesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=submesh_device)

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=torch.tensor([]),  # an empty tensor to mark that this value is never used
            timestep=timestep / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
        ).sample

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.99935, relative_rmse=0.037)
