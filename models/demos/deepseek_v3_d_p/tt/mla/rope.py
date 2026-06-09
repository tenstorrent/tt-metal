# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3YarnRotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks


def get_rot_transformation_mat():
    """Generate the rotation transformation matrix for RoPE. Uses a single tile."""
    dhead = ttnn.TILE_SIZE
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_cos_sin_matrix(hf_config):
    """Get cos and sin matrices for rotary positional embeddings.

    HuggingFace returns cos/sin in [max_seq_len, dim] with dim = [t1,..,td//2, t1,..,td//2].
    We convert to Meta-style [t1,t1,..,td//2,td//2] format and return [1, 1, max_seq_len, dim].
    """
    args = {
        "dim": hf_config.qk_rope_head_dim,
        "max_position_embeddings": hf_config.max_seq_len,
        "base": hf_config.rope_theta * 1.0,
        "device": "cpu",
        "scaling_factor": hf_config.rope_scaling["factor"],
        "original_max_position_embeddings": hf_config.rope_scaling["original_max_position_embeddings"],
        "beta_fast": hf_config.rope_scaling["beta_fast"],
        "beta_slow": hf_config.rope_scaling["beta_slow"],
        "mscale": hf_config.rope_scaling["mscale"],
        "mscale_all_dim": hf_config.rope_scaling["mscale_all_dim"],
    }

    reference_rope = DeepseekV3YarnRotaryEmbedding(**args)

    cos = reference_rope.cos_cached
    sin = reference_rope.sin_cached

    # Undo the HF permute to Meta-style
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]

    return cos, sin


def block_cyclic_reorder(matrix: torch.Tensor, chunk_local: int, sp_factor: int, seq_dim: int = 2) -> torch.Tensor:
    """Reorder a [.., seq, ..] matrix into block-cyclic order keyed by `chunk_local`.

    Splits the sequence into blocks of `chunk_local` rows and concatenates them so that device c's
    contiguous shard (after a plain SP shard over `seq_dim`) holds blocks c, c+sp, c+2sp, ... — the
    same block-cyclic layout the per-chip KV cache writes into. This makes the indexed-RoPE op's
    contiguous, `update_idxt`-offset read of each device's cos/sin shard land on the right global
    positions, including the boundary chip's older-then-wrap rows.
    """
    seq_len = matrix.shape[seq_dim]
    assert seq_len % chunk_local == 0, f"seq_len {seq_len} must be a multiple of chunk_local {chunk_local}"
    num_blocks = seq_len // chunk_local
    assert num_blocks % sp_factor == 0, f"num_blocks {num_blocks} must be a multiple of sp_factor {sp_factor}"
    blocks = list(torch.split(matrix, chunk_local, dim=seq_dim))
    order = [b for c in range(sp_factor) for b in range(c, num_blocks, sp_factor)]
    return torch.cat([blocks[b] for b in order], dim=seq_dim)


class RotarySetup:
    """Rotary positional embedding setup for MLA prefill with SP sharding and balanced reordering."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int = 0,
        is_balanced: bool = False,
    ):
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.is_balanced = is_balanced
        self.sp_factor = mesh_device.shape[sp_axis]

    def get_rope_tensors(self, seq_len: int) -> dict[str, ttnn.Tensor]:
        """Get cos, sin, and transformation matrices sharded over SP axis.

        If is_balanced, cos/sin are reordered according to balanced chunk order
        before sharding so each SP device gets rope values matching its chunk positions.
        """
        cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(self.hf_config)

        assert (
            seq_len <= self.hf_config.max_seq_len
        ), f"seq_len {seq_len} must be less than or equal to max_seq_len {self.hf_config.max_seq_len}"
        cos_matrix_torch = cos_matrix_torch[..., :seq_len, :]
        sin_matrix_torch = sin_matrix_torch[..., :seq_len, :]

        if self.is_balanced:
            chunk_order = create_balanced_chunk_order(self.sp_factor)
            cos_matrix_torch = reorder_tensor_chunks(cos_matrix_torch, chunk_order, seq_dim=2)
            sin_matrix_torch = reorder_tensor_chunks(sin_matrix_torch, chunk_order, seq_dim=2)

        shard_dims = [None, None]
        shard_dims[self.sp_axis] = 2

        cos_matrix = ttnn.from_torch(
            cos_matrix_torch,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.mesh_device.shape),
        )
        sin_matrix = ttnn.from_torch(
            sin_matrix_torch,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.mesh_device.shape),
        )

        trans_mat_torch = get_rot_transformation_mat()
        trans_matrix = ttnn.from_torch(
            trans_mat_torch,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return {"cos_matrix": cos_matrix, "sin_matrix": sin_matrix, "trans_matrix": trans_matrix}
