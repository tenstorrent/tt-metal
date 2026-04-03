# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 Partial RoPE — applies rotary embeddings to first 64 of 256 head dims.
"""

import torch

import ttnn
from models.demos.qwen35_27b.tt.model_config import ROPE_DIM
from models.tt_transformers.tt.rope import RotarySetup


class Qwen35PartialRopeSetup(RotarySetup):
    """RoPE setup for Qwen3.5 partial rotary embeddings (64 of 256 dims).

    Precomputes cos/sin in HuggingFace split-halves format for use with
    apply_partial_rope_decode.
    """

    def __init__(
        self,
        device,
        batch_size,
        head_dim,
        max_seq_len,
        rope_theta=10_000_000.0,
        rope_scaling=None,
        use_qk_fused=False,
        prefetcher=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            head_dim=ROPE_DIM,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_qk_fused=False,
            prefetcher=prefetcher,
        )
        self.full_head_dim = head_dim

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self._cos_table = ttnn.from_torch(
            emb.cos().unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self._sin_table = ttnn.from_torch(
            emb.sin().unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self._batch_size = batch_size

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        """Return HF-format [cos, sin] for given positions. Shape: [1, B, 1, ROPE_DIM]."""
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs, on_host=True)
        else:
            rot_idxs = position_idxs

        if rot_idxs.device != self.device:
            rot_idxs = ttnn.to_device(rot_idxs, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos = ttnn.embedding(rot_idxs, self._cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self._sin_table, layout=ttnn.TILE_LAYOUT)

        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]


def get_prefill_rot_mats(rope_setup, seq_len, device):
    """Get cos/sin tables for prefill: [1, 1, seq_len, rope_dim].

    Uses the precomputed tables directly (sliced to seq_len).
    """
    cos = ttnn.slice(rope_setup._cos_table, (0, 0, 0), (1, seq_len, ROPE_DIM))
    sin = ttnn.slice(rope_setup._sin_table, (0, 0, 0), (1, seq_len, ROPE_DIM))
    cos = ttnn.reshape(cos, (1, 1, seq_len, ROPE_DIM))
    sin = ttnn.reshape(sin, (1, 1, seq_len, ROPE_DIM))
    return cos, sin


def apply_partial_rope_prefill(x, cos_tt, sin_tt, n_heads, rope_dim=ROPE_DIM):
    """Apply partial RoPE on device for prefill.

    Args:
        x: [1, n_heads, seq_len, HD] where HD=256
        cos_tt, sin_tt: [1, 1, seq_len, rope_dim] where rope_dim=64
    Returns:
        [1, n_heads, seq_len, HD] with first rope_dim dims rotated
    """
    hd = x.shape[-1]
    seq_len = x.shape[-2]

    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd))

    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, n_heads, seq_len, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)

    roped = ttnn.add(
        ttnn.multiply(x_rope, cos_tt),
        ttnn.multiply(x_rot, sin_tt),
    )
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)

    target_mem = ttnn.DRAM_MEMORY_CONFIG
    roped = ttnn.to_memory_config(roped, target_mem)
    x_pass = ttnn.to_memory_config(x_pass, target_mem)

    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result


def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim=ROPE_DIM):
    """Apply partial RoPE on device for decode.

    Args:
        x: [1, B, n_heads, HD] where HD=256
        cos_tt, sin_tt: [1, B, 1, rope_dim] where rope_dim=64
    Returns:
        [1, B, n_heads, HD] with first rope_dim dims rotated
    """
    hd = x.shape[-1]
    B = batch_size

    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd))

    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, B, n_heads, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, B, n_heads, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)

    roped = ttnn.add(
        ttnn.multiply(x_rope, cos_tt),
        ttnn.multiply(x_rot, sin_tt),
    )
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)

    target_mem = ttnn.DRAM_MEMORY_CONFIG
    roped = ttnn.to_memory_config(roped, target_mem)
    x_pass = ttnn.to_memory_config(x_pass, target_mem)

    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result
