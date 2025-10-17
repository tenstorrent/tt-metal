# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import time

import diffusers.models.transformers.transformer_flux
import pytest
import torch
import ttnn
from loguru import logger

from ...blocks.transformer_block import TransformerBlock
from ...models.transformers.transformer_motif import convert_motif_transformer_block_state
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.motif.modeling_dit import MotifDiTBlock as MotifDiTBlockReference
from ...utils import tensor
from ...utils.check import assert_quality
from ...utils.padding import PaddingConfig
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "mesh_id"),
    [
        pytest.param((1, 8), (1, 8), 0, 1, 1, "1x8sp0tp1", id="1x8sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, "2x4sp1tp0", id="2x4sp1tp0"),
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
def test_transformer_block_flux(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    mesh_id: str,
    is_ci_env: bool,
) -> None:
    torch.manual_seed(0)

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    inner_dim = 3072
    head_dim = 128
    num_heads = 24

    torch_model = diffusers.models.transformers.transformer_flux.FluxTransformerBlock(
        dim=inner_dim, num_attention_heads=num_heads, attention_head_dim=head_dim
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

    tt_model = TransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_pre_only=False,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    spatial = torch.randn([batch_size, spatial_seq_len, inner_dim])
    prompt = torch.randn([batch_size, prompt_seq_len, inner_dim])
    time_embed = torch.randn([batch_size, inner_dim])
    rope_cos = torch.randn([spatial_seq_len + prompt_seq_len, head_dim])
    rope_sin = torch.randn([spatial_seq_len + prompt_seq_len, head_dim])

    spatial_padded = tt_model.pad_spatial_sequence(spatial, sp_factor=sp_factor)
    spatial_rope_cos_padded = tt_model.pad_spatial_sequence(rope_cos[prompt_seq_len:], sp_factor=sp_factor)
    spatial_rope_sin_padded = tt_model.pad_spatial_sequence(rope_sin[prompt_seq_len:], sp_factor=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=submesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_prompt = bf16_tensor(prompt, device=submesh_device, mesh_axis=tp_axis, shard_dim=2)
    tt_time_embed = bf16_tensor(time_embed.unsqueeze(1), device=submesh_device)
    tt_spatial_rope_cos = bf16_tensor(spatial_rope_cos_padded, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(spatial_rope_sin_padded, device=submesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=submesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=submesh_device)

    with torch.no_grad():
        torch_prompt, torch_spatial = torch_model.forward(
            spatial, prompt, temb=time_embed, image_rotary_emb=(rope_cos, rope_sin)
        )

    if not is_ci_env:
        try:
            from tracy import signpost

            signpost("caching")
            tt_spatial_out, tt_prompt_out = tt_model.forward(
                tt_spatial,
                tt_prompt,
                tt_time_embed,
                spatial_sequence_length=spatial_seq_len,
                spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
                prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
            )

            signpost("performance")
        except ImportError:
            logger.info("Tracy profiler not available, continuing without profiling")

    itr = 1
    start = time()
    for _ in range(itr):
        tt_spatial_out, tt_prompt_out = tt_model.forward(
            tt_spatial,
            tt_prompt,
            tt_time_embed,
            spatial_sequence_length=spatial_seq_len,
            spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
            prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        )

    ttnn.synchronize_device(submesh_device)
    logger.info(f"Time taken for {mesh_id}: {(time() - start) * 1000 / itr} ms")

    tt_spatial_torch = tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.99998, relative_rmse=0.006)

    tt_prompt_torch = tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])
    assert_quality(torch_prompt, tt_prompt_torch, pcc=0.99998, relative_rmse=0.006)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), (1, 2), 0, 1, 1, id="1x2sp0tp1"),
        pytest.param((2, 4), (2, 1), 1, 0, 1, id="2x1sp1tp0"),
        pytest.param((2, 4), (2, 2), 0, 1, 1, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 2), 1, 0, 1, id="2x2sp1tp0"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, id="2x4sp0tp1"),
        # pytest.param((2, 4), (2, 4), 1, 0, 1, id="2x4sp1tp0"),  # hangs
        pytest.param((4, 8), (4, 4), 0, 1, 4, id="4x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (2, 4100, 334),  # Motif Image 6B
    ],
)
@pytest.mark.parametrize("is_last_block", [False, True])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_motif(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    is_last_block: bool,
) -> None:
    torch.manual_seed(0)

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    num_heads = 30
    inner_dim = 1920
    modulation_dim = 4096
    head_dim = inner_dim // num_heads

    class ReferenceAttnConfig:
        def __init__(self) -> None:
            self.hidden_dim = inner_dim
            self.num_attention_heads = num_heads
            self.attn_mode = "flash"

    torch_model = MotifDiTBlockReference(
        emb_dim=inner_dim,
        t_emb_dim=modulation_dim,
        attn_emb_dim=inner_dim,
        mlp_dim=4 * inner_dim,
        attn_config=ReferenceAttnConfig(),
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

    tt_model = TransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        modulation_dim=modulation_dim,
        context_pre_only=is_last_block,
        context_head_scaling=True,
        add_attention_to_output=False,
        ff_activation_fn="silu",
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    converted_state_dict = dict(torch_model.state_dict())
    convert_motif_transformer_block_state(converted_state_dict, is_last_block=is_last_block)
    tt_model.load_torch_state_dict(converted_state_dict)

    spatial = torch.randn([batch_size, spatial_seq_len, inner_dim])
    prompt = torch.randn([batch_size, prompt_seq_len, inner_dim])
    time_embed = torch.randn([batch_size, modulation_dim])

    spatial_padded = tt_model.pad_spatial_sequence(spatial, sp_factor=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=submesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_prompt = bf16_tensor(prompt, device=submesh_device, mesh_axis=tp_axis, shard_dim=2)
    tt_time_embed = bf16_tensor(time_embed.unsqueeze(1), device=submesh_device)

    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model.forward(spatial, prompt, time_embed)

    tt_spatial_out, tt_prompt_out = tt_model.forward(
        tt_spatial,
        tt_prompt,
        tt_time_embed,
        spatial_sequence_length=spatial_seq_len,
    )

    tt_spatial_torch = tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])
    tt_spatial_torch = tt_spatial_torch[:, :spatial_seq_len]
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.999996, relative_rmse=0.003)

    if is_last_block:
        assert tt_prompt_out is None
        return

    tt_prompt_torch = tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])
    assert_quality(torch_prompt, tt_prompt_torch, pcc=0.999996, relative_rmse=0.003)
