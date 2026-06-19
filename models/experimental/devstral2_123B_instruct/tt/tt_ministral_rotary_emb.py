# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RoPE for Devstral-2 / Ministral3.

The HF reference (``Ministral3RotaryEmbedding``) computes per-token (cos, sin) with optional YaRN
scaling, then ``apply_rotary_pos_emb`` does::

    q_emb = q * cos + rotate_half(q) * sin
    k_emb = k * cos + rotate_half(k) * sin

After RoPE, the query is additionally multiplied by a position-dependent Llama-4 scale::

    s = 1 + beta * log1p(floor(pos / original_max_position_embeddings))
    q_emb = q_emb * s

This module:
  1) Pre-computes ``cos`` / ``sin`` and the Llama-4 ``s`` table on the host with YaRN scaling.
  2) Builds the transformation matrix used by ``ttnn.experimental.rotary_embedding_llama`` so we
     can apply RoPE on-device with the same numerics.
  3) Exposes slicing helpers for prefill (``[start_pos:start_pos+seq_len]``) and decode
     (``[current_pos]`` via on-device ``ttnn.embedding`` — no host gather / ``from_torch``).

Llama-4 query scaling is **baked into cos/sin** for the query path so the on-device RoPE op
multiplies through with no extra ops. See the ``cos_q`` / ``sin_q`` upload below.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args, RopeParameters
from models.experimental.devstral2_123B_instruct.tt.weight_loading import (
    resolve_weight_cache_path,
    rope_table_mem_config,
    upload_replicated_tile,
    weight_cache_file,
)

__all__ = ["TtRotaryEmbedding", "precompute_cos_sin", "compute_llama4_scale", "get_rot_transformation_mat"]


def _yarn_inv_freq(head_dim: int, rope: RopeParameters) -> tuple[torch.Tensor, float]:
    """Compute YaRN-scaled inverse frequencies and the attention-scaling multiplier.

    Replicates HF's ``ROPE_INIT_FUNCTIONS["yarn"]``. For ``rope_type == "default"`` only the base
    geometric inv_freq is returned and ``attention_factor == 1.0``.
    """
    base = rope.rope_theta
    dim = head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if rope.rope_type == "default":
        return inv_freq, 1.0
    if rope.rope_type != "yarn":
        raise ValueError(f"Unsupported rope_type {rope.rope_type!r}; expected 'yarn' or 'default'.")

    factor = rope.factor
    beta_fast = rope.beta_fast
    beta_slow = rope.beta_slow
    orig = rope.original_max_position_embeddings
    mscale = rope.mscale
    mscale_all_dim = rope.mscale_all_dim

    def _find_correction_dim(num_rotations, dim_, base_, max_pos_):
        return (dim_ * math.log(max_pos_ / (num_rotations * 2 * math.pi))) / (2 * math.log(base_))

    def _find_correction_range(low_rot, high_rot, dim_, base_, max_pos_):
        low = math.floor(_find_correction_dim(low_rot, dim_, base_, max_pos_))
        high = math.ceil(_find_correction_dim(high_rot, dim_, base_, max_pos_))
        return max(low, 0), min(high, dim_ - 1)

    def _linear_ramp_mask(low, high, dim_):
        if low == high:
            high = high + 0.001
        linear = (torch.arange(dim_, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear, 0.0, 1.0)

    def _get_mscale(scale, m_scale):
        if scale <= 1:
            return 1.0
        return 0.1 * m_scale * math.log(scale) + 1.0

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = _find_correction_range(beta_fast, beta_slow, dim, base, orig)
    inv_freq_mask = 1.0 - _linear_ramp_mask(low, high, dim // 2).to(dtype=torch.float32)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    attention_factor = float(_get_mscale(factor, mscale) / _get_mscale(factor, mscale_all_dim))
    return inv_freq, attention_factor


def precompute_cos_sin(
    head_dim: int,
    max_position_embeddings: int,
    rope: RopeParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``cos``/``sin`` tables of shape ``[max_position_embeddings, head_dim]`` (float32).

    Layout is **pairwise-interleaved** — each frequency ``f_k`` is duplicated at positions
    ``2k`` and ``2k+1``. This is the layout ``ttnn.experimental.rotary_embedding_llama`` expects
    (cf. ``models/tt_transformers/tt/common.py::gather_cos_sin``). The math result is identical to
    HF's split-half RoPE when the Q/K tensors are likewise pre-permuted into interleaved layout
    (see ``permute_split_half_to_interleaved`` below; the attention module bakes that into the
    Q/K projection weights at load time).
    """
    inv_freq, attention_factor = _yarn_inv_freq(head_dim, rope)
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [pos, dim/2]
    # Interleave: cos'[pos, 2k] = cos'[pos, 2k+1] = cos(freqs[pos, k]).
    emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)
    cos = emb.cos() * attention_factor
    sin = emb.sin() * attention_factor
    return cos.to(torch.float32), sin.to(torch.float32)


def permute_split_half_to_interleaved(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute a (Hq*head_dim, in) weight so each head's ``head_dim`` rows go from HF's
    split-half ``[a_0, ..., a_{D/2-1}, b_0, ..., b_{D/2-1}]`` to pairwise-interleaved
    ``[a_0, b_0, a_1, b_1, ..., a_{D/2-1}, b_{D/2-1}]``.

    Applied to ``q_proj`` and ``k_proj`` weights so the device-side RoPE op
    (``ttnn.experimental.rotary_embedding_llama``) — which expects interleaved layout — gets
    matching activations without a per-forward permute. V is **not** permuted: it never enters
    RoPE, and the SDPA output stays in V's layout to match what ``o_proj`` expects.
    """
    out, in_ = weight.shape
    if out % head_dim != 0:
        raise ValueError(f"weight out={out} not divisible by head_dim={head_dim}")
    n_heads = out // head_dim
    # (Hq, head_dim, in) -> reorder head_dim from split-half to interleaved -> flatten.
    w = weight.reshape(n_heads, head_dim, in_)
    half = head_dim // 2
    idx = torch.empty(head_dim, dtype=torch.long)
    idx[0::2] = torch.arange(half)
    idx[1::2] = torch.arange(half, head_dim)
    return w[:, idx, :].reshape(out, in_).contiguous()


def compute_llama4_scale(
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    beta: float,
) -> torch.Tensor:
    """``s[p] = 1 + beta * log1p(floor(p / orig))``; shape ``[max_position_embeddings]``."""
    if beta == 0.0 or original_max_position_embeddings <= 0:
        return torch.ones(max_position_embeddings, dtype=torch.float32)
    p = torch.arange(max_position_embeddings, dtype=torch.float32)
    floored = torch.floor(p / float(original_max_position_embeddings))
    return 1.0 + float(beta) * torch.log1p(floored)


def get_rot_transformation_mat() -> torch.Tensor:
    """Canonical 32x32 trans-mat for ``ttnn.experimental.rotary_embedding_llama``.

    Pairwise/interleaved RoPE: each adjacent pair ``(2k, 2k+1)`` swaps with signs ``(+1, -1)``,
    so ``(x @ M)[2k] = -x[2k+1]`` and ``(x @ M)[2k+1] = x[2k]``. Combined with cos/sin tables in
    pairwise layout, this yields the standard 2D rotation per pair.

    Matches ``models/common/tensor_utils.py::get_rot_transformation_mat``.
    """
    mat = torch.zeros((32, 32), dtype=torch.bfloat16)
    mat[torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1.0
    mat[torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1.0
    return mat.reshape(1, 1, 32, 32)


class TtRotaryEmbedding:
    """Owns the device-resident cos/sin/scale tables and exposes prefill/decode getters.

    Tables are replicated across the mesh (RoPE is a per-head pointwise op, so each TP shard sees
    the same cos/sin values for its slice of heads).
    """

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        *,
        max_position_embeddings: Optional[int] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        max_pos = int(max_position_embeddings or args.max_seq_len)
        self._max_pos = max_pos

        cos, sin = precompute_cos_sin(args.head_dim, max_pos, args.rope)
        scale = compute_llama4_scale(
            max_pos,
            args.rope.original_max_position_embeddings,
            args.rope.llama_4_scaling_beta,
        )

        # Bake Llama-4 query scaling into the Q tables so device RoPE produces a scaled Q directly.
        cos_q = cos * scale.unsqueeze(-1)
        sin_q = sin * scale.unsqueeze(-1)

        self._cos_host = cos
        self._sin_host = sin
        self._cos_q_host = cos_q
        self._sin_q_host = sin_q
        cache_path = resolve_weight_cache_path(weight_cache_path, args)
        prefill_rope_mem = rope_table_mem_config(max_pos, args.head_dim)

        def _upload(t: torch.Tensor, name: str) -> ttnn.Tensor:
            tt = t.to(torch.bfloat16).reshape(1, 1, max_pos, args.head_dim)
            return upload_replicated_tile(
                tt,
                mesh_device,
                dtype=dtype,
                memory_config=prefill_rope_mem,
                weight_cache_path=cache_path,
                cache_key=f"rotary_{name}_pos{max_pos}",
            )

        self.cos_q = _upload(cos_q, "cos_q")
        self.sin_q = _upload(sin_q, "sin_q")
        self.cos_k = _upload(cos, "cos_k")
        self.sin_k = _upload(sin, "sin_k")

        self.trans_mat = upload_replicated_tile(
            get_rot_transformation_mat(),
            mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            weight_cache_path=cache_path,
            cache_key="rotary_trans_mat",
        )

        # Decode: ROW_MAJOR DRAM tables for ``ttnn.embedding`` row gather (cf. tt_transformers HfRotarySetup).
        def _upload_decode_lookup(host_2d: torch.Tensor, name: str) -> ttnn.Tensor:
            return ttnn.as_tensor(
                host_2d.to(torch.bfloat16).contiguous(),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=weight_cache_file(cache_path, f"rotary_decode_{name}_pos{max_pos}"),
            )

        self.cos_q_decode = _upload_decode_lookup(cos_q, "cos_q")
        self.sin_q_decode = _upload_decode_lookup(sin_q, "sin_q")
        self.cos_k_decode = _upload_decode_lookup(cos, "cos_k")
        self.sin_k_decode = _upload_decode_lookup(sin, "sin_k")

        self._sharded_trans_mat: dict[int, ttnn.Tensor] = {}

    def get_sharded_trans_mat(self, batch_size: int) -> ttnn.Tensor:
        """Height-sharded trans_mat for decode RoPE (cached per batch size)."""
        if batch_size not in self._sharded_trans_mat:
            grid_size = ttnn.CoreCoord(8, 8)
            batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
            mem_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._sharded_trans_mat[batch_size] = ttnn.interleaved_to_sharded(self.trans_mat, mem_config)
        return self._sharded_trans_mat[batch_size]

    def _decode_rot_idxs(self, current_pos: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        rot_idxs = ttnn.reshape(current_pos, (1, batch_size))
        return ttnn.typecast(rot_idxs, ttnn.uint32)

    def _lookup_decode_table(self, rot_idxs: ttnn.Tensor, table: ttnn.Tensor, *, batch_size: int) -> ttnn.Tensor:
        """``ttnn.embedding`` gather → ``[1, batch, TILE, head_dim]`` interleaved L1 for decode RoPE."""
        emb = ttnn.embedding(
            rot_idxs,
            table,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb = ttnn.unsqueeze_to_4D(emb)
        emb = ttnn.transpose(emb, 1, 2)
        if batch_size % ttnn.TILE_SIZE != 0:
            emb = emb[:, :batch_size, :, :]
        return ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)

    def get_prefill_tables(
        self, start_pos: int, seq_len: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        end = start_pos + seq_len
        if end > self._max_pos:
            raise ValueError(f"Prefill end {end} exceeds precomputed max_position {self._max_pos}")

        def _slice(t: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.slice(t, [0, 0, start_pos, 0], [1, 1, end, self.args.head_dim])

        return _slice(self.cos_q), _slice(self.sin_q), _slice(self.cos_k), _slice(self.sin_k)

    def get_decode_tables(
        self,
        current_pos: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Gather cos/sin rows for ``current_pos`` via ``ttnn.embedding``.

        ``current_pos`` must be a device tensor of shape ``[batch]`` (int32). Returns four
        interleaved L1 tensors shaped ``[1, batch, TILE, head_dim]`` for sharding in attention.
        """
        if not isinstance(current_pos, ttnn.Tensor):
            raise TypeError(f"current_pos must be ttnn.Tensor on device, got {type(current_pos)!r}")
        batch_size = int(current_pos.shape[0])
        rot_idxs = self._decode_rot_idxs(current_pos, batch_size)
        return (
            self._lookup_decode_table(rot_idxs, self.cos_q_decode, batch_size=batch_size),
            self._lookup_decode_table(rot_idxs, self.sin_q_decode, batch_size=batch_size),
            self._lookup_decode_table(rot_idxs, self.cos_k_decode, batch_size=batch_size),
            self._lookup_decode_table(rot_idxs, self.sin_k_decode, batch_size=batch_size),
        )

    def apply(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos_q: ttnn.Tensor,
        sin_q: ttnn.Tensor,
        cos_k: ttnn.Tensor,
        sin_k: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        q_out = ttnn.experimental.rotary_embedding_llama(q, cos_q, sin_q, self.trans_mat)
        k_out = ttnn.experimental.rotary_embedding_llama(k, cos_k, sin_k, self.trans_mat)
        return q_out, k_out
