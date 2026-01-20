# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger

from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ....utils.check import assert_quality
from ....models.transformers.wan2_2.attention_wan import WanAttention
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.padding import pad_vision_seq_parallel
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.test import ring_params, line_params
from diffusers import WanTransformer3DModel


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology",
    [
        [(2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear],
        [(2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear],
        [(4, 8), 0, 1, 4, ring_params, ttnn.Topology.Ring],
        [(4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring],
        [(4, 8), 0, 1, 2, line_params, ttnn.Topology.Linear],
        [(4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear],
    ],
    ids=[
        "2x4sp0tp1",
        "2x4sp1tp0",
        "wh_4x8sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp0tp1",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "T, H, W",
    [
        (31, 40, 80),
        (21, 60, 104),
        (21, 90, 160),
    ],
    ids=["5b-720p", "14b-480p", "14b-720p"],
)
@pytest.mark.parametrize("prompt_seq_len", [None, 26, 126], ids=["no_prompt", "short_prompt", "long_prompt"])
@pytest.mark.parametrize("is_fsdp", [True], ids=["yes_fsdp"])
def test_wan_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1

    # Wan2.2 Model configuration - typical transformer dimensions
    dim = 5120
    num_heads = 40
    head_dim = dim // num_heads
    qk_norm = True
    eps = 1e-6
    layer_id = 0
    patch_size = (1, 2, 2)
    in_channels = 16
    p_t, p_h, p_w = patch_size
    patch_F, patch_H, patch_W = T // p_t, H // p_h, W // p_w
    spatial_seq_len = patch_F * patch_H * patch_W

    attn_type = "self" if prompt_seq_len is None else "cross"
    logger.info(f"attn_type: {attn_type}")

    MIN_PCC = 0.988

    # Load Wan2.2-T2V-14B model from HuggingFace
    parent_torch_model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", torch_dtype=torch_dtype, trust_remote_code=True
    )
    parent_torch_model.eval()

    # Access the first layer's attention modules
    first_layer = parent_torch_model.blocks[layer_id]
    torch_model = first_layer.attn1 if attn_type == "self" else first_layer.attn2
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    # Create TT model based on attention type
    tt_model = WanAttention(
        dim=dim,
        num_heads=num_heads,
        qk_norm=qk_norm,
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
    spatial_input = torch.randn((B, spatial_seq_len, dim), dtype=torch_dtype)
    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    logger.info(f"spatial_input shape: {spatial_input.shape}. tt_spatial shape: {tt_spatial.shape}")

    if attn_type == "cross":
        prompt_input = torch.randn((B, prompt_seq_len, dim), dtype=torch_dtype)
        tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
        logger.info(f"prompt_input shape: {prompt_input.shape}. tt_prompt shape: {tt_prompt.shape}")
        rotary_emb = None
        tt_rope_cos = None
        tt_rope_sin = None
        tt_trans_mat = None
    else:
        prompt_input = None
        tt_prompt = None
        # TODO: Use real ROPE embeddings
        rope_cos = torch.randn(B, spatial_seq_len, 1, head_dim // 2)
        rope_sin = torch.randn(B, spatial_seq_len, 1, head_dim // 2)

        torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)
        rotary_emb = [torch_rope_cos, torch_rope_sin]

        rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
        rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)
        rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
        rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)
        # Rope cos and sin sequence fractured and head fractured
        tt_rope_cos = bf16_tensor(rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        tt_rope_sin = bf16_tensor(rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)

        # Create transformation matrix for RoPE
        trans_mat = get_rot_transformation_mat()
        tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

        logger.info(f"torch_rope_cos shape: {torch_rope_cos.shape}. tt_rope_cos shape: {tt_rope_cos.shape}")
        logger.info(f"torch_rope_sin shape: {torch_rope_sin.shape}. tt_rope_sin shape: {tt_rope_sin.shape}")
        logger.info(f"trans_mat shape: {trans_mat.shape}. tt_trans_mat shape: {tt_trans_mat.shape}")

    # Run torch model
    logger.info(f"Running torch model")

    torch_spatial_out = torch_model(
        hidden_states=spatial_input,
        encoder_hidden_states=prompt_input,
        rotary_emb=rotary_emb,
    )

    # Run TT model
    logger.info(f"Running TT model")
    tt_spatial_out = tt_model(
        tt_spatial,
        N=spatial_seq_len,
        prompt_1BLP=tt_prompt,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 3
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]

    logger.info(f"Checking spatial outputs for {attn_type} attention")
    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC)
