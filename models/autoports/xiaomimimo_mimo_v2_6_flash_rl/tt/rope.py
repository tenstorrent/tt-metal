# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE for MiMo on top of ``ttnn.experimental.rotary_embedding_llama``.

HF rotates only the first ``rope_dim`` (64) of the 192 head dims, pairing dim i with dim i + 32
(``rotate_half``). ``rotary_embedding_llama`` rotates adjacent pairs (2i, 2i + 1) inside each tile.
We permute the first 64 rows of every Q/K head in the weights from HF order to interleaved order
(``meta_rope_permutation``), and build 192-wide cos/sin tables with cos = 1 and sin = 0 on the
128 no-RoPE dims, so the op is the identity there. Q.K is unchanged by a permutation applied to
both, so attention scores match HF exactly.
"""

import torch

import ttnn
from models.tt_transformers.tt.rope import RotarySetup


def meta_rope_permutation(head_dim, rope_dim):
    """Index list: new dim order for one head (HF rotate_half pairs -> adjacent pairs)."""
    half = rope_dim // 2
    rope = torch.stack((torch.arange(half), torch.arange(half) + half), dim=-1).flatten()
    return torch.cat((rope, torch.arange(rope_dim, head_dim)))


def permute_heads_for_rope(w, num_heads, head_dim, rope_dim):
    """w [num_heads * head_dim, in] in HF order -> rows permuted per head for interleaved RoPE."""
    perm = meta_rope_permutation(head_dim, rope_dim)
    return w.view(num_heads, head_dim, -1)[:, perm].reshape(num_heads * head_dim, -1)


def partial_rope_tables(max_seq_len, head_dim, rope_dim, theta):
    """cos/sin [1, 1, max_seq_len, head_dim] in interleaved order; identity on dims >= rope_dim."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)  # [S, rope_dim / 2]
    cos = torch.ones(max_seq_len, head_dim)
    sin = torch.zeros(max_seq_len, head_dim)
    cos[:, :rope_dim] = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    sin[:, :rope_dim] = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    return cos[None, None], sin[None, None]


class MiMoRotarySetup(RotarySetup):
    """``RotarySetup`` whose cos/sin tables implement MiMo's partial RoPE for one theta."""

    def __init__(self, device, batch_size, head_dim, rope_dim, max_seq_len, rope_theta, datatype=ttnn.bfloat16):
        super().__init__(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=None,
            datatype=datatype,
        )
        cos, sin = partial_rope_tables(max_seq_len, head_dim, rope_dim, rope_theta)
        mapper = ttnn.ReplicateTensorToMesh(device)
        for attr in ("cos_matrix", "sin_matrix", "cos_matrix_prefill", "sin_matrix_prefill"):
            getattr(self, attr).deallocate(True)

        def upload(t, layout):
            return ttnn.from_torch(t, device=device, layout=layout, dtype=datatype, mesh_mapper=mapper)

        self.cos_matrix = upload(cos, ttnn.ROW_MAJOR_LAYOUT)
        self.sin_matrix = upload(sin, ttnn.ROW_MAJOR_LAYOUT)
        self.cos_matrix_prefill = upload(cos, ttnn.TILE_LAYOUT)
        self.sin_matrix_prefill = upload(sin, ttnn.TILE_LAYOUT)
        self.max_seq_len = max_seq_len

    def get_prefill_rot_mats(self, start_pos, seq_len):
        end = start_pos + seq_len
        assert end <= self.max_seq_len, f"prefill end {end} > max_seq_len {self.max_seq_len}"
        return [
            self.cos_matrix_prefill[:, :, start_pos:end, :],
            self.sin_matrix_prefill[:, :, start_pos:end, :],
        ]
