# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE for MiMo on top of ``ttnn.experimental.rotary_embedding_llama``.

HF rotates only the first ``rope_dim`` (64) of the 192 head dims, pairing dim i with dim i + 32
(``rotate_half``). ``rotary_embedding_llama`` rotates adjacent pairs (2i, 2i + 1) inside each tile.
We permute the first 64 rows of every Q/K head in the weights from HF order to interleaved order
(``meta_rope_permutation``), and use 192-wide cos/sin with cos = 1 and sin = 0 on the 128 no-RoPE
dims, so the op is the identity there. Q.K is unchanged by a permutation applied to both, so
attention scores match HF exactly.

cos/sin are computed on the host for the positions in use, like HF (fp32, then bf16): per prefill, the prompt's
rows, uploaded. Decode, a speculative verify and an MTP draft instead read rows of device tables of every position
(TtMiMoModel.setup_decode_rope_tables, ``partial_rope_table``) at their own position inputs, so a step uploads no
cos/sin; with MIMO_DEVICE_ROPE=0 they copy ``decode_host_inputs`` (32 rows) every step. Past 128K context the tables
keep only the ``rope_dim`` rotated dims of a row (``compact``) and the step appends the constant 1 / 0 of the other
dims: 4 x 64 x 2 bytes per position, 537 MB per chip at 1M instead of 1.61 GB for whole 192-wide rows.
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


def partial_rope_table(num_positions, head_dim, rope_dim, theta, width=None, batch=32):
    """cos/sin [num_positions, width] (fp32) of positions 0 .. num_positions - 1: the first ``width`` dims (default
    ``head_dim``: whole rows; ``rope_dim``: the rotated dims, the rest being the constant 1 / 0) of
    ``partial_rope_cos_sin``, called on ``batch`` positions at a time: the call ``decode_host_inputs`` makes for one
    decode step of ``batch`` slots. torch's CPU sin / cos of a value can differ in the last fp32 bit with the size of
    the tensor it is computed in (vector grouping, thread split): one call over all 32K or 128K positions, as these
    tables were made before, left 1-9% of the rows a bit off the per-step values on an arm64 host (equal after bf16
    rounding there, but not by construction). In the host's own call shape each row is computed exactly as in a decode
    step, whatever the step's other positions (tests/test_rope_tables_host.py). ~1.3 s per 1M positions."""
    width = head_dim if width is None else width
    assert rope_dim <= width <= head_dim
    cos = torch.empty(num_positions, width)
    sin = torch.empty(num_positions, width)
    for start in range(0, num_positions, batch):
        c, s = partial_rope_cos_sin(torch.arange(start, start + batch), head_dim, rope_dim, theta)
        n = min(batch, num_positions - start)
        cos[start : start + n] = c[:n, :width]
        sin[start : start + n] = s[:n, :width]
    return cos, sin


def rope_table_rows(cos_table, sin_table, positions, head_dim):
    """Torch mirror of TtMiMoModel.rope_table_rows: rows ``positions`` of the tables, a compact table's (narrower than
    ``head_dim``) followed by the 1 (cos) / 0 (sin) of the dims it leaves out. [n, head_dim] each."""
    out = []
    for table, tail in ((cos_table, 1.0), (sin_table, 0.0)):
        rows = table[positions]
        pad = torch.full((rows.shape[0], head_dim - rows.shape[1]), tail, dtype=rows.dtype)
        out.append(torch.cat((rows, pad), dim=-1))
    return out


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
        self.prefill_cache = None  # {(start, seq_len): [cos, sin]} while a prefill chunk is enqueued
        self.prefill_pinned = set()  # its keys the caller owns

    def _host(self, t, device):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            device=self.device if device else None,
            dtype=self.datatype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def get_prefill_rot_mats(self, start_pos, seq_len):
        """Device cos/sin [1, 1, seq_len, head_dim] for positions start_pos .. start_pos + seq_len - 1. Inside
        ``begin_prefill_chunk`` / ``end_prefill_chunk`` they are made once and shared by every caller (who must not
        free them: ``owns_prefill_rot_mats`` is False); outside, every call makes its own pair for the caller."""
        key = (start_pos, seq_len)
        if self.prefill_cache is not None and key in self.prefill_cache:
            return self.prefill_cache[key]
        pos = torch.arange(start_pos, start_pos + seq_len)
        cos, sin = partial_rope_cos_sin(pos, self.head_dim, self.rope_dim, self.theta)
        mats = [self._host(cos, True), self._host(sin, True)]
        if self.prefill_cache is not None:
            self.prefill_cache[key] = mats
        return mats

    @property
    def owns_prefill_rot_mats(self):
        """Whether a caller of ``get_prefill_rot_mats`` frees what it got."""
        return self.prefill_cache is None

    def begin_prefill_chunk(self, pinned=None):
        """Share each (start, length)'s prefill cos/sin among the layers of one chunk (one upload, not one a layer).
        ``pinned``: {(start, length): [cos, sin]} the caller owns (a traced prefill's persistent inputs)."""
        assert self.prefill_cache is None, "prefill chunks do not nest"
        self.prefill_cache = dict(pinned or {})
        self.prefill_pinned = set(self.prefill_cache)

    def end_prefill_chunk(self):
        """Free the chunk's shared cos/sin (enqueued: freeing is ordered with the work that reads them). Nothing
        made for a prefill outlives it, so no buffer allocated after a decode capture stays alive over a replay."""
        for key, mats in self.prefill_cache.items():
            if key not in self.prefill_pinned:
                for t in mats:
                    t.deallocate(True)
        self.prefill_cache = None
        self.prefill_pinned = set()

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
