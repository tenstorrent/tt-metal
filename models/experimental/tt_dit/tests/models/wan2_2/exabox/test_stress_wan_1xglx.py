# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from loguru import logger

from .....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from .....models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from .....parallel.manager import CCLManager
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....utils.padding import pad_vision_seq_parallel
from .....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from .....utils.test import ring_params, line_params
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp",
    [
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False],
        [(4, 32), (4, 32), 1, 0, 2, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "bh_4x8sp1tp0",
        "bh_quad_4x32_sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq_len"),
    [
        (1, 21, 90, 160, 118),  # 14B-720p
    ],
    ids=["14b-720p"],
)
def test_wan_transformer_block(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    torch_dtype = torch.float32
    parent_mesh_device = mesh_device
    mesh_device = parent_mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Wan2.2 Model configuration
    dim = 5120
    ffn_dim = 13824
    num_attention_heads = 40
    attention_head_dim = dim // num_attention_heads
    cross_attention_norm = True
    eps = 1e-6
    patch_size = (1, 2, 2)
    in_channels = 16
    p_t, p_h, p_w = patch_size
    patch_F, patch_H, patch_W = T // p_t, H // p_h, W // p_w
    spatial_seq_len = patch_F * patch_H * patch_W
    layer_id = 0

    # Tight error bounds based on test config
    MIN_PCC = 0.999_500
    MAX_RMSE = 0.032

    # Load Wan2.2-T2V-14B model from HuggingFace
    parent_torch_model = TorchWanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", torch_dtype=torch_dtype, trust_remote_code=True
    )

    parent_torch_model = TorchWanTransformer3DModel(num_layers=1)
    torch_model = parent_torch_model.blocks[layer_id]
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

    # Create TT model
    tt_model = WanTransformerBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_attention_heads,
        cross_attention_norm=cross_attention_norm,
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
    prompt_input = torch.randn((B, prompt_seq_len, dim), dtype=torch_dtype)
    temb_input = torch.randn((B, 6, dim), dtype=torch_dtype)

    # Create ROPE embeddings
    rope_cos = torch.randn(B, spatial_seq_len, 1, attention_head_dim // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, attention_head_dim // 2)

    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    # Sequence fractured spatial
    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    # Replicated prompt
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    # Replicated time embedding
    tt_temb = bf16_tensor(temb_input.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)

    # Rope cos and sin sequence fractured and head fractured
    tt_rope_cos = bf16_tensor(rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_rope_sin = bf16_tensor(rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat()
    tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )
    tt_spatial_out = tt_model(
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        temb_1BTD=tt_temb,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    assert os.environ.get("TT_MM_THROTTLE_PERF") == "1", "TT_MM_THROTTLE_PERF must be set to 1 for stress test"

    logger.info(f"running stress test")
    for i in range(1_000_000_000):
        for j in range(1000):
            tt_spatial_out = tt_model(
                spatial_1BND=tt_spatial,
                prompt_1BLP=tt_prompt,
                temb_1BTD=tt_temb,
                N=spatial_seq_len,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
                trans_mat=tt_trans_mat,
            )

        ttnn.synchronize_device(mesh_device)
        logger.info(f"Iteration {i}")

    logger.info(f"stress test done")
