# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
     (``[current_pos]``).

Llama-4 query scaling is **baked into cos/sin** for the query path so the on-device RoPE op
multiplies through with no extra ops. See the ``cos_q`` / ``sin_q`` upload below.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_large.tt.model_args import Devstral2Args, RopeParameters

__all__ = ["TtRotaryEmbedding", "precompute_cos_sin", "compute_llama4_scale", "get_rot_transformation_mat"]


# --- Pure-host RoPE table computation (mirrors HF ``Ministral3RotaryEmbedding``) ---


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


# --- TT-side wrapper ---


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
        self._scale_host = scale

        def _upload(t: torch.Tensor) -> ttnn.Tensor:
            tt = t.to(torch.bfloat16).reshape(1, 1, max_pos, args.head_dim)
            return ttnn.from_torch(
                tt,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.cos_q = _upload(cos_q)
        self.sin_q = _upload(sin_q)
        self.cos_k = _upload(cos)
        self.sin_k = _upload(sin)

        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # --- Prefill: returns (cos_q, sin_q, cos_k, sin_k) sliced to ``[start:start+seq_len]`` ---

    def get_prefill_tables(
        self, start_pos: int, seq_len: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        end = start_pos + seq_len
        if end > self._max_pos:
            raise ValueError(f"Prefill end {end} exceeds precomputed max_position {self._max_pos}")

        def _slice(t: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.slice(t, [0, 0, start_pos, 0], [1, 1, end, self.args.head_dim])

        return _slice(self.cos_q), _slice(self.sin_q), _slice(self.cos_k), _slice(self.sin_k)

    # --- Decode: returns per-position rows uploaded fresh each step ---

    def get_decode_tables(
        self,
        current_pos_host: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """``current_pos_host`` is a host int tensor of shape ``[batch]``.

        Indexed on the host and uploaded fresh; mirrors HF semantics (``forward(x, position_ids)``
        rebuilds the table for given positions) and avoids on-device gather.
        """
        positions = current_pos_host.long()
        batch = int(positions.shape[0])
        head_dim = self.args.head_dim

        def _gather(t: torch.Tensor) -> ttnn.Tensor:
            rows = t[positions].to(torch.bfloat16).reshape(1, batch, 1, head_dim)
            # Pad seq dim to a tile so ``ttnn.experimental.rotary_embedding_llama`` is happy.
            pad_to = ttnn.TILE_SIZE
            padded = torch.zeros((1, batch, pad_to, head_dim), dtype=torch.bfloat16)
            padded[:, :, 0:1, :] = rows
            return ttnn.from_torch(
                padded,
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            _gather(self._cos_q_host),
            _gather(self._sin_q_host),
            _gather(self._cos_host),
            _gather(self._sin_host),
        )

    # --- Convenience: apply RoPE on device. ---

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
