# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger

from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ....utils.check import assert_quality
from ....models.transformers.attention_mochi import MochiAttention
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.padding import pad_vision_seq_parallel
from diffusers import MochiTransformer3DModel
from models.tt_transformers.tt.common import get_rot_transformation_mat


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(1, 1), 0, 1, 1],
        [(1, 2), 0, 1, 1],
        [(1, 2), 1, 0, 1],
        [(2, 1), 0, 1, 1],
        [(2, 1), 1, 0, 1],
        [(2, 2), 0, 1, 1],
        [(2, 2), 1, 0, 1],
        [(2, 4), 0, 1, 1],
        [(2, 4), 1, 0, 1],
        [(4, 8), 0, 1, 4],
        [(4, 8), 1, 0, 4],
    ],
    ids=[
        "1x1sp0tp1",
        "1x2sp0tp1",
        "1x2sp1tp0",
        "2x1sp0tp1",
        "2x1sp1tp0",
        "2x2sp0tp1",
        "2x2sp1tp0",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "4x8sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, spatial_seq_len, prompt_seq_len"),
    [
        (1, 4000, 118),  # Similar to SD3.5 config
        (1, 44520, 118),  # Similar to SD3.5 config
    ],
    ids=["short_seq", "long_seq"],
)
@pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize("context_pre_only", [True, False], ids=["yes_context_pre", "no_context_pre"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mochi_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    context_pre_only: bool,
    is_fsdp: bool,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    query_dim = 3072
    added_kv_proj_dim = 1536  # Different from query_dim
    heads = 24
    head_dim = 128
    bias = False
    added_proj_bias = False
    out_dim = query_dim
    out_context_dim = added_kv_proj_dim  # Different output dim for context
    out_bias = True
    eps = 1e-5
    layer_id = 0 if not context_pre_only else -1

    MIN_PCC = 0.997 if not context_pre_only else 0.98

    parent_torch_model = MochiTransformer3DModel.from_pretrained(
        f"genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.transformer_blocks[layer_id].attn1
    torch_model.eval()

    assert torch_model.out_dim == out_dim
    assert torch_model.out_context_dim == out_context_dim
    assert torch_model.context_pre_only == context_pre_only
    assert torch_model.heads == heads

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    # Create TT model
    tt_model = MochiAttention(
        query_dim=query_dim,
        added_kv_proj_dim=added_kv_proj_dim,
        heads=heads,
        head_dim=head_dim,
        bias=bias,
        added_proj_bias=added_proj_bias,
        out_bias=out_bias,
        out_context_dim=out_context_dim,
        context_pre_only=context_pre_only,
        eps=eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Initialize weights randomly for testing
    torch.manual_seed(0)
    # Create input tensors
    spatial_input = torch.randn((B, spatial_seq_len, query_dim), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq_len, added_kv_proj_dim), dtype=torch_dtype)

    # TODO: Use real ROPE embeddings
    rope_cos = torch.randn(spatial_seq_len, heads, head_dim // 2)
    rope_sin = torch.randn(spatial_seq_len, heads, head_dim // 2)

    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    # Sequence fractured spatial
    tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    # Replicated prompt
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)

    # Rope cos and sin sequence fractured and head fractured
    tt_rope_cos = bf16_tensor_2dshard(rope_cos_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_rope_sin = bf16_tensor_2dshard(rope_sin_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)
    tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
    attention_mask = torch.ones((B, prompt_seq_len), dtype=torch_dtype)
    torch_spatial_out, torch_prompt_out = torch_model(
        spatial_input,
        prompt_input,
        attention_mask=attention_mask,
        image_rotary_emb=[rope_cos, rope_sin],
    )

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )
    tt_spatial_out, tt_prompt_out = tt_model(
        tt_spatial,
        tt_prompt,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 0
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]

    logger.info(f"Checking spatial outputs")
    for i in range(tt_spatial_out.shape[0]):
        assert_quality(torch_spatial_out, tt_spatial_out[i], pcc=MIN_PCC)

    if not context_pre_only:
        prompt_concat_dims = [None, None]
        prompt_concat_dims[sp_axis] = 0
        prompt_concat_dims[tp_axis] = 1
        tt_prompt_out = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=prompt_concat_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )
        tt_prompt_out = tt_prompt_out[:, :, :prompt_seq_len, :]
        # Get all replicas into the first dimension for checking
        tt_prompt_out = tt_prompt_out.reshape(-1, prompt_seq_len, added_kv_proj_dim)

        logger.info(f"Checking prompt outputs")
        for i in range(tt_prompt_out.shape[0]):
            assert_quality(torch_prompt_out, tt_prompt_out[i], pcc=0.99)
