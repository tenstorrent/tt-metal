# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE for MiMo on top of ``ttnn.experimental.rotary_embedding_llama``.

HF rotates only the first ``rope_dim`` (64) of the 192 head dims, pairing dim i with dim i + 32
(``rotate_half``). ``rotary_embedding_llama`` rotates adjacent pairs (2i, 2i + 1) inside each tile.
We permute the first 64 rows of every Q/K head in the weights from HF order to interleaved order
(``meta_rope_permutation``), and use 192-wide cos/sin with cos = 1 and sin = 0 on the 128 no-RoPE
dims, so the op is the identity there. Q.K is unchanged by a permutation applied to both, so
attention scores match HF exactly.

cos/sin are computed on the host for the positions in use, like HF (fp32, then bf16), and uploaded:
per decode step that is 32 rows; per prefill, the prompt's rows. There are no ``max_seq_len``-sized
device tables, which at 1M context would cost ~3 GB per chip for the two thetas.
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


def partial_rope_cos_sin(positions, head_dim, rope_dim, theta):
    """positions [n] -> cos/sin [n, head_dim] (fp32) in interleaved order; identity on dims >= rope_dim."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [n, rope_dim / 2]
    cos = torch.ones(positions.numel(), head_dim)
    sin = torch.zeros(positions.numel(), head_dim)
    cos[:, :rope_dim] = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    sin[:, :rope_dim] = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    return cos, sin


class MiMoRotarySetup(RotarySetup):
    """Transformation matrices and batch sharding from ``RotarySetup``; cos/sin come from the host."""

    def __init__(self, device, batch_size, head_dim, rope_dim, rope_theta, datatype=ttnn.bfloat16):
        # The parent builds max_seq_len-sized tables we never read; keep them one tile long.
        super().__init__(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=ttnn.TILE_SIZE,
            rope_theta=rope_theta,
            rope_scaling=None,
            datatype=datatype,
        )
        self.rope_dim = rope_dim
        self.theta = rope_theta
        self.datatype = datatype

    def _host(self, t, device):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            device=self.device if device else None,
            dtype=self.datatype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def get_prefill_rot_mats(self, start_pos, seq_len):
        """Device cos/sin [1, 1, seq_len, head_dim] for positions start_pos .. start_pos + seq_len - 1."""
        pos = torch.arange(start_pos, start_pos + seq_len)
        cos, sin = partial_rope_cos_sin(pos, self.head_dim, self.rope_dim, self.theta)
        return [self._host(cos, True), self._host(sin, True)]

    def decode_host_inputs(self, positions):
        """Host cos/sin [1, 1, B, head_dim] for one decode step (copied into trace inputs)."""
        cos, sin = partial_rope_cos_sin(positions.clamp(min=0), self.head_dim, self.rope_dim, self.theta)
        return self._host(cos, False), self._host(sin, False)

    def decode_rot_mats(self, cos, sin):
        """Device-only: [1, 1, B, D] cos/sin -> height-sharded [1, B, 32, D], as ``rotary_embedding_llama`` wants."""
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        out = []
        for t in (cos, sin):
            t = ttnn.transpose(t, 1, 2)  # [1, B, 1, D]
            if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
                t = t[:, : self.batch_size_per_device_group, :, :]
            out.append(ttnn.interleaved_to_sharded(t, mem_config))
        return out

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        """Eager path for tests: positions (torch [B]) -> sharded decode cos/sin."""
        assert not return_rot_idxs
        cos, sin = (ttnn.to_device(t, self.device) for t in self.decode_host_inputs(position_idxs))
        mats = self.decode_rot_mats(cos, sin)
        cos.deallocate(True)
        sin.deallocate(True)
        return mats
