# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionPooler (HF `Gemma4VisionPooler`, modeling_gemma4.py:630-694) as one batched matmul.

HF: zero the padding patches, average each k x k (3 x 3) block of patches by position
(`kernel_idx = x // k + (max_x // k) * (y // k)`), scale by sqrt(hidden) in fp32, then the caller
strips output rows whose block has no valid patch.

Device form:
    pooled = (P @ h) * (sqrt(hidden) / k^2)
P [B, 1, S_out, S] is the host-built one-hot block-membership matrix (`pooling_matrix`): exactly
1.0 at (kernel_idx(patch), patch) for valid patches, 0 elsewhere. Padding columns are zero, which
is the HF masked_fill. The 1.0 entries are exact in bf16; the 1/k^2 average and the sqrt(hidden)
scale are applied once in fp32 after the fp32-accumulated matmul. Rows >= n_valid / k^2 stay zero
and are stripped on the host (`VisionInputs.num_soft_tokens`).

Contract
    input   h [B, 1, S, 768] bf16 TILE DRAM (encoder output); P [B, 1, S_out, S] bf16 TILE DRAM
    output  [B, 1, S_out, 768] float32 TILE DRAM
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.embedding_gemma2.tt.linear import spec_program_config
from models.demos.embedding_gemma2.tt.vision.config import VisionConfig
from models.demos.embedding_gemma2.tt.vision.linear import vision_spec


def pooling_matrix(position_ids: torch.Tensor, k: int, out_rows: int, seq_len: int) -> tuple[torch.Tensor, list[int]]:
    """Host: position_ids [B, N, 2] (-1 = padding) -> (P [B, 1, out_rows, seq_len] fp32 one-hot, n_out per image).

    Mirrors `Gemma4VisionPooler._avg_pool_by_positions` (max_x from the clamped positions) with the
    HF output length N // k^2; columns N..seq_len-1 (bucket padding) are zero.
    """
    B, N, _ = position_ids.shape
    length = N // (k * k)
    if length * k * k != N:
        raise ValueError(f"{N} patches is not a multiple of k^2 = {k * k}")
    if length > out_rows or N > seq_len:
        raise ValueError(f"pooled length {length} / patches {N} exceed the bucket ({out_rows}, {seq_len})")
    valid = (position_ids != -1).all(dim=-1)  # [B, N]
    clamped = position_ids.clamp(min=0)
    max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kidx = torch.div(clamped, k, rounding_mode="floor")
    kidx = kidx[..., 0] + (max_x // k) * kidx[..., 1]  # [B, N]
    if (kidx[valid] >= length).any():
        raise ValueError("a valid patch maps outside the pooled length")
    P = torch.zeros(B, 1, out_rows, seq_len)
    b_idx, n_idx = valid.nonzero(as_tuple=True)
    P[b_idx, 0, kidx[b_idx, n_idx], n_idx] = 1.0
    counts = P[:, 0].sum(-1)  # [B, out_rows]
    n_out = []
    for b in range(B):
        nz = (counts[b] > 0).nonzero().flatten()
        n = int(nz.numel())
        # HF strips empty blocks with a boolean mask; on processor output the non-empty blocks are rows 0..n-1
        if n and int(nz.max()) != n - 1:
            raise ValueError(f"image {b}: non-empty pooled rows are not a prefix (rows {nz.tolist()[:8]}...)")
        n_out.append(n)
    return P, n_out


class VisionPooler(LightweightModule):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.scale = config.root_hidden_size / config.k_squared
        self.compute_kernel_config = compute_kernel_config_hifi4()

    def forward(self, h: ttnn.Tensor, pool: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        # role "pool" (M = S_out) in the per-grid tuned table (tt/vision/linear.py VISION_TUNED); without an
        # entry ttnn picks the config (6 cores at 1 x 2560: 279 us at 12 x 10, E10.1 Tracy)
        B, S_out, S, D = pool.shape[0], pool.shape[-2], pool.shape[-1], h.shape[-1]
        grid = h.device().compute_with_storage_grid_size()
        spec = vision_spec("pool", S_out, grid, batch=B)
        pc = None if spec is None else spec_program_config(spec, S_out, S, D, grid.x, grid.y, None, True)
        summed = ttnn.matmul(
            pool,
            h,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=dram,
            program_config=pc,
        )
        out = ttnn.multiply(summed, self.scale, memory_config=dram)
        ttnn.deallocate(summed)
        return out
