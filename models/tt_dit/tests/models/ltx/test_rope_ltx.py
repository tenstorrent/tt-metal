# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, apply_rotary_emb, precompute_freqs_cis
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard


def make_indices_grid(F: int, H: int, W: int) -> torch.Tensor:
    """
    Create a 3D position indices grid (seq_len, 3) for temporal + spatial dims.
    Each row is (t_idx, h_idx, w_idx) for one token in the flattened sequence.
    """
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float()
    return indices_grid.unsqueeze(0)  # (1, seq_len, 3)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(1, 2), 0, 1],
        [(2, 1), 0, 1],
        [(2, 2), 0, 1],
        [(2, 4), 0, 1],
        [(2, 4), 1, 0],
    ],
    ids=[
        "1x1sp0tp1",
        "1x2sp0tp1",
        "2x1sp0tp1",
        "2x2sp0tp1",
        "2x4sp0tp1",
        "2x4sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "F, H, W",
    [
        (5, 16, 28),  # Small: 2240 tokens
        (5, 30, 52),  # 480p-ish: 7800 tokens
    ],
    ids=["small", "480p"],
)
def test_ltx_rope_interleaved(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, F: int, H: int, W: int):
    """
    Test LTX-2 interleaved RoPE: compute cos/sin on CPU, apply via ttnn.experimental.rotary_embedding_llama,
    compare against PyTorch reference.
    """
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads  # 128
    theta = 10000.0
    max_pos = [20, 2048, 2048]
    B = 1

    # Build position indices grid
    indices_grid = make_indices_grid(F, H, W)  # (1, seq_len, 3)
    seq_len = F * H * W
    logger.info(f"indices_grid shape: {indices_grid.shape}, seq_len: {seq_len}")

    # Precompute cos/sin on CPU
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        theta=theta,
        max_pos=max_pos,
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.INTERLEAVED,
    )
    logger.info(f"cos_freq shape: {cos_freq.shape}, sin_freq shape: {sin_freq.shape}")

    # Create random input: (B, seq_len, num_heads, head_dim)
    torch.manual_seed(42)
    input_tensor = torch.randn(B, seq_len, num_heads, head_dim, dtype=torch.float32)

    # PyTorch reference: apply RoPE
    # cos/sin from interleaved mode have shape (B, seq_len, dim) — need to match input shape
    # Reshape to (B, seq_len, num_heads, head_dim) for broadcasting
    cos_for_apply = cos_freq.reshape(B, seq_len, num_heads, head_dim)
    sin_for_apply = sin_freq.reshape(B, seq_len, num_heads, head_dim)
    output_ref = apply_rotary_emb(input_tensor, (cos_for_apply, sin_for_apply), LTXRopeType.INTERLEAVED)
    logger.info(f"Reference output shape: {output_ref.shape}")

    # Prepare ttnn tensors
    # ttnn.experimental.rotary_embedding_llama expects (B, num_heads, seq_len, head_dim)
    input_bhnd = input_tensor.permute(0, 2, 1, 3)  # (B, H, N, D)
    cos_bhnd = cos_for_apply.permute(0, 2, 1, 3)  # (B, H, N, D)
    sin_bhnd = sin_for_apply.permute(0, 2, 1, 3)  # (B, H, N, D)

    tt_input = bf16_tensor_2dshard(input_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    # cos/sin sharded same as input: SP axis on sequence, TP axis on heads
    tt_cos = bf16_tensor_2dshard(cos_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Apply RoPE on device
    tt_output = ttnn.experimental.rotary_embedding_llama(tt_input, tt_cos, tt_sin, tt_trans_mat)

    # Gather output back to host
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    tt_output_torch = tt_output_torch.permute(0, 2, 1, 3)  # Back to (B, N, H, D)

    # Compare
    assert_quality(output_ref, tt_output_torch, pcc=0.99)
    logger.info("PASSED: LTX interleaved RoPE matches PyTorch reference")
