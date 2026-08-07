# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3YarnRotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    block_cyclic_reorder,
    create_balanced_chunk_order,
    reorder_tensor_chunks,
)


def get_rot_transformation_mat():
    """Generate the rotation transformation matrix for RoPE. Uses a single tile."""
    dhead = ttnn.TILE_SIZE
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_cos_sin_matrix(hf_config, interleave: bool = True):
    """Get cos and sin matrices for rotary positional embeddings.

    HuggingFace returns cos/sin in [max_seq_len, dim] with dim = [t1,..,td//2, t1,..,td//2].
    Returns cos, sin as [1, 1, max_seq_len, dim].

    interleave (which physical columns form a rotated pair):
        True  -> Meta-style duplicated pairs [t1,t1,..,td//2,td//2]; pairs (0,1),(2,3),..
                 consumed by ``rotary_embedding_llama`` + ``get_rot_transformation_mat``
                 (MLA q_pe/k_pe, and the GLM indexer).
        False -> rotate_half / half-split [t1,..,td//2, t1,..,td//2]; pairs (i, i+dim/2)
                 consumed by ``rotary_embedding_hf`` (no trans_mat) — the DeepSeek indexer.

    Tables are always pure rotations (unit modulus): ``_mscale = m(mscale)/m(mscale_all_dim)``
    is forced to 1. The YaRN mscale amplitude is applied elsewhere (the attention softmax
    scale); for every shipped config the baked factor is 1, so this is exact.
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
    # Force _mscale = m(mscale)/m(mscale_all_dim) = 1 → pure rotations (no amplitude in
    # cos/sin). The mscale amplitude is applied in the attention softmax scale instead.
    args["mscale_all_dim"] = args["mscale"]

    reference_rope = DeepseekV3YarnRotaryEmbedding(**args)

    cos = reference_rope.cos_cached
    sin = reference_rope.sin_cached

    # HF stores [t1,..,td//2, t1,..,td//2]; take one half, then lay out per `interleave`.
    cos = cos[:, : cos.shape[1] // 2]
    sin = sin[:, : sin.shape[1] // 2]
    if interleave:
        cos = torch.stack((cos, cos), dim=-1).flatten(-2)  # [t1,t1,t2,t2,..]
        sin = torch.stack((sin, sin), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)  # [t1,..,td//2, t1,..,td//2]
        sin = torch.cat((sin, sin), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]

    return cos, sin


def interleaved_to_halfsplit_perm(rope_dim: int = 64) -> torch.Tensor:
    """Dim permutation that maps this MLA's RoPE layout to vLLM's (and back — it is
    its own structural counterpart applied to the other side).

    RoPE CONVENTION NOTE (verified vs DeepSeek-V3.2-Exp reference, layers 0/30/60).
    This MLA carries q_pe and stores k_pe in the **interleaved** layout of the
    official ``inference/model.py`` (``apply_rotary_emb(interleaved=True)``: the
    2-D rotated pairs are dims (0,1),(2,3),...). vLLM's ``DeepseekV32`` uses the HF
    **rotate_half / half-split** layout (pairs (i, i+rope_dim/2)) with projection
    weights pre-permuted to suit. The two are exactly one fixed dim permutation
    apart, so:
      * q·k — and therefore the entire MLA output — is IDENTICAL in both layouts
        (the dot product sums over the rope dims, which the permutation only
        reorders). Measured: output PCC unchanged at 0.99983 either way.
      * the stored k_pe *values* differ element-wise: a direct comparison of our
        k_pe against a vLLM-written cache reads ~0.43 PCC, while the SAME tensor
        reindexed by this permutation matches at 0.99997.

    So within our self-consistent stack the layout is irrelevant. It matters ONLY
    when interoperating with a vLLM-written KV cache (e.g. cross-stack disaggregated
    prefill/decode, or validating against vLLM's recorded k_pe): reindex the rope
    half of the cache row with ``kpe[..., interleaved_to_halfsplit_perm()]`` to put
    it in vLLM's layout (or the reverse to ingest a vLLM cache).

    Returns the index tensor p such that ``interleaved_kpe[..., p] == halfsplit_kpe``:
    p = [0, 2, 4, ..., 1, 3, 5, ...] (even dims = cos halves, then odd dims = sin).
    """
    return torch.cat([torch.arange(0, rope_dim, 2), torch.arange(1, rope_dim, 2)])


def interleaved_perm_matrix(rope_dim: int = 64) -> torch.Tensor:
    """``[rope_dim, rope_dim]`` permutation matrix that reorders a half-split rope layout into the
    interleaved one (``out = in @ P``). Purpose: run a half-split (DeepSeek) q/k through the
    interleaved-only ``rotary_embedding_indexed`` op — applied to BOTH q and k the permutation cancels
    in ``q·k`` (score/top-k unchanged), while letting the interleaved op pair the right dims with each
    frequency. Built from ``interleaved_to_halfsplit_perm`` (its inverse), the canonical convention:
    ``interleaved = halfsplit[argsort(p)]``, so ``P[argsort(p)[j], j] = 1``."""
    src = torch.argsort(interleaved_to_halfsplit_perm(rope_dim))  # out[j] = in[src[j]]
    perm = torch.zeros(rope_dim, rope_dim)
    perm[src, torch.arange(rope_dim)] = 1.0
    return perm


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

        Always Meta-style interleaved cos/sin + a trans_matrix (rotary_embedding_llama) — the
        MLA's own RoPE layout.
        """
        cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(self.hf_config, interleave=True)

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

    def get_rope_tensors_indexed(self, cache_seq_len_global: int, chunk_size_global: int) -> dict[str, ttnn.Tensor]:
        """Get whole-cache cos/sin/trans matrices for the KV-pad-aware *indexed* rotated path.

        This builds the rope values for the *entire* cache once. The cos/sin are
        block-cyclic-reordered keyed by the per-chip chunk size and SP-sharded, so device ``c``'s
        contiguous shard holds -- in local-cache-row order -- the rope values for every global
        position it will ever carry. ``rotary_embedding_indexed`` then derives each chunk's start
        row in that shard on-device from a single ``kv_actual_global`` runtime arg (the same
        ``update_idxt`` math the per-chip KV-cache writer uses), so the same tensors are reused
        across all chunks and only ``kv_actual_global`` varies.

        Args:
            cache_seq_len_global: total cache length in tokens across all SP devices (the cos/sin
                cover every position the cache can hold). Per-chip shard = cache_seq_len_global // sp.
            chunk_size_global: global chunk size (one chunk across all SP devices). Per-chip chunk =
                chunk_size_global // sp_factor, which keys the block-cyclic reorder.

        Constraints (mirroring the block-cyclic / cache layout):
            * ``cache_seq_len_global <= max_seq_len``
            * ``chunk_size_global % (TILE_SIZE * sp_factor) == 0``
            * ``cache_seq_len_global % chunk_size_global == 0``
        """
        assert not self.is_balanced, "indexed rotated rope is incompatible with is_balanced"
        sp = self.sp_factor
        assert (
            cache_seq_len_global <= self.hf_config.max_seq_len
        ), f"cache_seq_len_global ({cache_seq_len_global}) must be <= max_seq_len {self.hf_config.max_seq_len}"
        assert (
            chunk_size_global % (ttnn.TILE_SIZE * sp) == 0
        ), f"chunk_size_global ({chunk_size_global}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
        assert cache_seq_len_global % chunk_size_global == 0, (
            f"cache_seq_len_global ({cache_seq_len_global}) must be a multiple of "
            f"chunk_size_global ({chunk_size_global})"
        )
        chunk_local = chunk_size_global // sp

        cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(self.hf_config)
        cos_matrix_torch = cos_matrix_torch[..., :cache_seq_len_global, :]
        sin_matrix_torch = sin_matrix_torch[..., :cache_seq_len_global, :]

        # Block-cyclic reorder keyed by the per-chip chunk so a plain SP shard hands device c the
        # rope values for every global position it carries, in local-cache-row order.
        cos_matrix_torch = block_cyclic_reorder(cos_matrix_torch, chunk_local, sp, seq_dim=2)
        sin_matrix_torch = block_cyclic_reorder(sin_matrix_torch, chunk_local, sp, seq_dim=2)

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
