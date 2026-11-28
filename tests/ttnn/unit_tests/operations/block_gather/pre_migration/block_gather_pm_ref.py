# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent torch reference and tensor helpers for the block_gather pre-migration suite.

The reference is written from the operation's contract (README / docstring), not from its planner:
    out[..., r, j * B + o] = scores[..., r, ids[..., r, j] * B + o]   when ids[..., r, j] < W / B
                           = -inf                                     otherwise (0xFFFFFFFF included)
Exact data selection: every comparison is on bf16 bit patterns.
"""

from __future__ import annotations

import torch
import ttnn

SENT = 0xFFFFFFFF


def u32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int64) & 0xFFFFFFFF


def ref_block_gather(scores: torch.Tensor, ids: torch.Tensor, block: int) -> torch.Tensor:
    """scores [.., R, W] bf16, ids [.., R, K] (int64 holding uint32 values) -> [.., R, K block] bf16."""
    W = scores.shape[-1]
    nb = W // block
    ids = u32(ids)
    lead = tuple(ids.shape[:-1])
    K = ids.shape[-1]
    flat_s = scores.reshape(-1, W)
    flat_i = ids.reshape(-1, K)
    out = torch.full((flat_s.shape[0], K, block), float("-inf"), dtype=torch.bfloat16)
    for r in range(flat_s.shape[0]):
        row = flat_s[r].view(nb, block)
        valid = flat_i[r] < nb
        out[r][valid] = row[flat_i[r][valid]]
    return out.reshape(*lead, K * block)


def make_scores(shape, gen: torch.Generator) -> torch.Tensor:
    """Random bf16 scores with ties, a -inf (causal) tail on one row, negative and zero values."""
    s = torch.randn(*shape, generator=gen).to(torch.bfloat16)
    W = shape[-1]
    flat = s.view(-1, W)
    flat[:, ::7] = 0.5  # ties
    if flat.shape[0] > 3:
        flat[3, W // 2 :] = float("-inf")  # a causal tail
    flat[0, 1::5] = -0.0
    return s


def make_ids(lead, K: int, nb: int, gen: torch.Generator) -> torch.Tensor:
    """Block ids [.., K] int64 (uint32 values): random in range, plus none (0xFFFFFFFF), just out of range (nb),
    far out of range, a row of one repeated block and a row of only none."""
    ids = torch.randint(0, nb, (*lead, K), generator=gen, dtype=torch.int64)
    flat = ids.view(-1, K)
    R = flat.shape[0]
    flat[0, -min(5, K) :] = SENT
    if R > 1:
        flat[1, min(3, K - 1)] = nb
        flat[1, 0] = nb + 12345
    if R > 2:
        flat[2, :] = nb - 1
    if R > 4:
        flat[4, :] = SENT
    return ids


def to_dev(t: torch.Tensor, device, dtype, *, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=None, mesh=False):
    if dtype == ttnn.uint32:
        t = u32(t).to(torch.int64)
        t = torch.where(t >= 2**31, t - 2**32, t).to(torch.int32)
    kwargs = {}
    if mesh:
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    )


def to_host(t: ttnn.Tensor, *, mesh=False) -> list[torch.Tensor]:
    """Every device's copy (one for a single device)."""
    if mesh:
        return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
    return [ttnn.to_torch(t)]


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def mismatch(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    bad = (a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).nonzero()
    return f"{len(bad)} mismatched values, first {bad[:6].tolist()}"
