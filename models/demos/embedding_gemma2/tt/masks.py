# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Additive attention masks for the bidirectional encoder (+ SDPA block maps for long buckets).

HF semantics (`masking_utils`):
    full layers     keep (q, k)  iff  key k is a valid (non-padding) token
    sliding layers  keep (q, k)  iff  key k is valid  and  |q - k| <= sliding_window (512, inclusive)

TT SDPA cannot combine its `sliding_window_size` argument with an `attn_mask`
(`sdpa_device_operation.cpp:141`), so both rules are baked into one dense additive mask, built per
forward in two device ops (E9.2 topology):

    band     [1, 1, S, S] = 0 if |q - k| <= W else -1e9   constant per bucket, built at setup
    full     [B, 1, S, S] = zeros[1, 1, S, 1] + (keep - 1) * 1e9
                            (row + column broadcast add; the key bias (keep - 1) * 1e9 is an SFPU
                            pre-activation of the add's second input, not two separate ops)
    sliding  [B, 1, S, S] = band + full                   (dense + dense, no broadcast)

`sliding` reads the already built `full` instead of row-broadcasting the key bias: the same
values bit for bit, and 4.3x faster per build at 8192 / 16384 than `band + key_bias` (E9.2
probe_mask, doc/prefill/02-topology/).

Both masks are bfloat4_b (E5a): SDPA reads them at the same speed as bf16 (sweep at 512-8192) with
identical outputs, and they take 3.6x less DRAM, which is what bounds the longest bucket. In
bfp4, 0 stays exactly 0 and -1e9 / -2e9 become about -0.8e9 / -1.9e9 (still fully masking).
Valid query rows always keep themselves, so their softmax is well defined; a fully masked padded
query row averages padded values and is never read back.

Block map (E5a): for buckets >= BLOCK_MAP_MIN_SEQ the band leaves most (q chunk, k chunk) blocks
fully masked. `block_map(S)` is a constant [1, 1, S/q_chunk, S/k_chunk] int32 map (1 where the
band reaches the block) that lets SDPA skip the other blocks: 2.2x faster per sliding layer at
4096, 5x at 8192 (sdpa_variants). Padding keys stay masked by the dense mask inside visited blocks.
The map has the mask's batch dim, so it is used for batch 1 only (the dense mask alone is
correct for any batch).

Contract
    prepare(S)        setup-time host -> device upload of the band (+ block map) for bucket S
    forward(keep)     keep [B, 1, 1, S] bf16 TILE (1 = valid token, 0 = padding), device-only ops
                      returns {"sliding_attention": [B,1,S,S], "full_attention": [B,1,S,S]} bfp4 TILE DRAM,
                      + "sliding_attention_block_map" (int32 ROW_MAJOR) where the grid's policy builds the
                      padding-aware map per forward (E9.4, `attention.sliding_map_policy`, batch 1)
    block_map(S, B)   the prepared (band-only) sliding-layer block map for batch B, or None

Which buckets get a block map and whether it is padding-aware is per compute grid
(`attention.sliding_map_policy`; default from BLOCK_MAP_MIN_SEQ, no padding-aware map).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import FULL, MASK_NEG, SLIDING, compute_grid

# E5a default; the text model passes its precision policy's `mask_dtype`
MASK_DTYPE = ttnn.bfloat4_b
BLOCK_MAP_MIN_SEQ = int(os.environ.get("EG2_BLOCK_MAP_MIN_SEQ", "4096"))
# E9.2 topology switches (default on; 0 restores the stage-1 ops for same-session A/B controls):
# key bias as pre-activations of the full-mask add (2 ops fewer), sliding mask = band + full.
MASK_BIAS_FUSED = os.environ.get("EG2_MASK_BIAS_FUSED", "1") == "1"
MASK_SLIDING_FROM_FULL = os.environ.get("EG2_MASK_SLIDING_FROM_FULL", "1") == "1"
# forward() key of the per-forward sliding block map (padded_block_map), next to the masks of each kind
BLOCK_MAP_SUFFIX = "_block_map"


def band_mask(seq_len: int, radius: int) -> torch.Tensor:
    """[1, 1, S, S] bf16: 0 where |q - k| <= radius else MASK_NEG. Built in row blocks so the host
    peak stays at the bf16 output (8 GiB at S = 65536) instead of an S x S int64 distance matrix."""
    band = torch.full((seq_len, seq_len), MASK_NEG, dtype=torch.bfloat16)
    rows = 1024
    k = torch.arange(seq_len)
    for r0 in range(0, seq_len, rows):
        q = torch.arange(r0, min(r0 + rows, seq_len))
        keep = (q[:, None] - k[None, :]).abs() <= radius
        band[r0 : r0 + len(q)][keep] = 0.0
    return band.reshape(1, 1, seq_len, seq_len)


def band_block_map(seq_len: int, radius: int, q_chunk: int, k_chunk: int) -> torch.Tensor:
    """[1, 1, ceil(S/q_chunk), ceil(S/k_chunk)] int32: 1 where some (q, k) in the block has |q - k| <= radius.
    A chunk that does not divide S leaves a partial last block (rows / keys clipped to S)."""
    nq, nk = -(-seq_len // q_chunk), -(-seq_len // k_chunk)
    q0 = torch.arange(nq)[:, None] * q_chunk
    k0 = torch.arange(nk)[None, :] * k_chunk
    q1 = torch.clamp(q0 + q_chunk - 1, max=seq_len - 1)
    k1 = torch.clamp(k0 + k_chunk - 1, max=seq_len - 1)
    # closest distance between [q0, q1] and [k0, k1]
    gap = torch.clamp(torch.maximum(k0 - q1, q0 - k1), min=0)
    return (gap <= radius).to(torch.int32).reshape(1, 1, nq, nk)


class AttentionMasks(LightweightModule):
    def __init__(self, device, sliding_window: int, sliding_head_dim: int = 256, num_heads: int = 4, dtype=MASK_DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.sliding_window = sliding_window
        self.sliding_head_dim = sliding_head_dim
        self.num_heads = num_heads
        self.bands: dict[int, ttnn.Tensor | None] = {}
        self.zero_cols: dict[int, ttnn.Tensor] = {}
        self.block_maps: dict[int, ttnn.Tensor] = {}
        self.band_map_tiles: dict[int, ttnn.Tensor] = {}  # padding-aware map (sliding_map_policy)
        self.map_k_chunk: dict[int, int] = {}

    def prepare(self, seq_len: int) -> None:
        if seq_len in self.bands:
            return
        self.zero_cols[seq_len] = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if seq_len <= self.sliding_window + 1:
            # every |q - k| <= S - 1 <= W: the band masks nothing
            self.bands[seq_len] = None
            return
        self.bands[seq_len] = ttnn.from_torch(
            band_mask(seq_len, self.sliding_window),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        from models.demos.embedding_gemma2.tt.attention import sdpa_program_config, sliding_map_policy

        grid = compute_grid(self.device)
        min_seq, pad_map = sliding_map_policy(grid.x, grid.y, BLOCK_MAP_MIN_SEQ)
        if seq_len >= min_seq:
            pc = sdpa_program_config(self.sliding_head_dim, seq_len, 1, self.num_heads, grid)
            bm = band_block_map(seq_len, self.sliding_window, pc.q_chunk_size, pc.k_chunk_size)
            self.block_maps[seq_len] = ttnn.from_torch(
                bm,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if pad_map and seq_len % pc.k_chunk_size == 0:
                # bf16 TILE copy of the band map: forward() multiplies it by the per-key-chunk validity
                self.band_map_tiles[seq_len] = ttnn.from_torch(
                    bm.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                self.map_k_chunk[seq_len] = pc.k_chunk_size

    def block_map(self, seq_len: int, batch: int) -> ttnn.Tensor | None:
        """The constant (band-only) sliding block map of the bucket, batch 1."""
        return self.block_maps.get(seq_len) if batch == 1 else None

    def padded_block_map(self, keep: ttnn.Tensor) -> ttnn.Tensor | None:
        """Per-forward sliding block map, batch 1: the band map with every key chunk that holds only padding
        cleared, so SDPA skips those blocks (E9.4). The dense mask still masks padding keys inside the
        visited blocks; a skipped block only held keys the mask sets to MASK_NEG. Five small device ops:
        [1,1,1,S] keep -> [1,1,S/k,k] -> max over each chunk -> [1,1,1,S/k] -> x band [1,1,S/q,S/k] -> int32
        ROW_MAJOR (the layout SDPA reads)."""
        B, S = keep.shape[0], keep.shape[-1]
        if B != 1 or S not in self.band_map_tiles:
            return None
        k = self.map_k_chunk[S]
        dram = ttnn.DRAM_MEMORY_CONFIG
        chunks = ttnn.reshape(keep, (1, 1, S // k, k))
        valid = ttnn.max(chunks, dim=-1, keepdim=True, memory_config=dram)  # [1,1,S/k,1]: 1 if any key valid
        ttnn.deallocate(chunks)
        row = ttnn.transpose(valid, -2, -1, memory_config=dram)  # [1,1,1,S/k]
        ttnn.deallocate(valid)
        m = ttnn.multiply(self.band_map_tiles[S], row, memory_config=dram)
        ttnn.deallocate(row)
        mi = ttnn.typecast(m, ttnn.int32, memory_config=dram)
        ttnn.deallocate(m)
        out = ttnn.to_layout(mi, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.deallocate(mi)
        return out

    @staticmethod
    def key_bias_activations() -> list:
        """(keep - 1) * 1e9 as SFPU pre-activations of a binary op's keep input: 0 valid, -1e9 padding."""
        return [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SUB_UNARY_SFPU, 1.0),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, -MASK_NEG),
        ]

    def forward(self, keep: ttnn.Tensor) -> dict[str, ttnn.Tensor]:
        B, _, _, S = keep.shape
        if S not in self.bands:
            raise RuntimeError(f"AttentionMasks.prepare({S}) must run before forward (setup-time upload)")
        dram = ttnn.DRAM_MEMORY_CONFIG
        band = self.bands[S]
        bias = None
        if MASK_BIAS_FUSED:
            full = ttnn.add(
                self.zero_cols[S],
                keep,
                dtype=self.dtype,
                memory_config=dram,
                input_tensor_b_activations=self.key_bias_activations(),
            )
        else:  # stage-1 topology (A/B control): two ops for the key bias
            bias = ttnn.multiply(ttnn.subtract(keep, 1.0), -MASK_NEG, memory_config=dram)
            full = ttnn.add(self.zero_cols[S], bias, dtype=self.dtype, memory_config=dram)
        if band is None:
            sliding = full
        elif MASK_SLIDING_FROM_FULL:
            sliding = ttnn.add(band, full, dtype=self.dtype, memory_config=dram)
        else:  # stage-1 topology (A/B control): row-broadcast add of the key bias
            if bias is None:
                bias = ttnn.multiply(ttnn.subtract(keep, 1.0), -MASK_NEG, memory_config=dram)
            sliding = ttnn.add(band, bias, dtype=self.dtype, memory_config=dram)
        if bias is not None:
            ttnn.deallocate(bias)
        out = {SLIDING: sliding, FULL: full}
        pad_map = self.padded_block_map(keep)
        if pad_map is not None:
            out[SLIDING + BLOCK_MAP_SUFFIX] = pad_map
        return out
