# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic inputs for the ``sparse_sdpa_tiled`` tests and microtests (fixtures, not a served path): the packed ``[V | K]`` cache, the query
rows, the block-id rows under the producer contract (the first ``min(IDS, complete_j)`` slots valid, the rest bait)
and the positions, in the selection patterns the numerics plan names."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sparse_sdpa_tiled as sst

PATTERNS = ("shared", "disjoint", "clustered", "random", "ranges")


@dataclass
class Inputs:
    q: torch.Tensor  # [1, H, S, Dq] bf16 values (float32 storage of bf16-rounded numbers)
    kv: torch.Tensor  # [1, 1, T, W] bf16 values
    block_ids: torch.Tensor  # [1, 1, S, IDS] int64 (uint32 values)
    positions: torch.Tensor  # [S] int64
    P: int
    T: int

    @property
    def S(self) -> int:
        return int(self.q.shape[2])


def _round_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def select_rows(
    S: int, T: int, IDS: int, P: int, pattern: str, gen: torch.Generator, *, block_tokens: int = 4, bait: str = "valid"
) -> torch.Tensor:
    """``[S, IDS]`` int64 block ids: row j's first ``min(IDS, complete_j)`` slots are distinct blocks below
    ``complete_j`` (its valid selection, in a descending-score-like arbitrary order), the rest bait: ``valid`` =
    distinct blocks at or past ``complete_j`` when the cache has them (a leak shows), ``sentinel`` = 0xFFFFFFFF,
    ``range`` = ids at or past ``T / block_tokens`` (an addressed one faults the watcher)."""

    blocks = T // block_tokens
    bait_gen = torch.Generator().manual_seed(0x5EED)  # the bait never moves the valid selections or their order
    ids = torch.full((S, IDS), sst.MASKED_INDEX, dtype=torch.int64)
    positions = P + torch.arange(S)
    complete = (positions + 1) // block_tokens
    max_complete = int(complete.max())
    perm = torch.randperm(max(max_complete, 1), generator=gen)  # the shared pattern's order
    for j in range(S):
        c = int(complete[j])
        n = min(IDS, c)
        if n:
            if pattern == "shared":
                chosen = perm[perm < c][:n]
                chosen = chosen[torch.randperm(chosen.numel(), generator=gen)]  # the same set, another order per row
            elif pattern == "disjoint":
                # rows of one 16-query tile take disjoint slices of the past when it is wide enough
                lane = j % 16
                pool = torch.arange(c)
                pool = pool[pool % 16 == lane]
                chosen = pool[torch.randperm(pool.numel(), generator=gen)][:n]
                if chosen.numel() < n:  # not enough disjoint blocks: fall back to random distinct
                    rest = torch.tensor(
                        [b for b in torch.randperm(c, generator=gen).tolist() if b not in set(chosen.tolist())]
                    )
                    chosen = torch.cat([chosen, rest[: n - chosen.numel()]])
            elif pattern == "clustered":
                near = max(1, min(64, c))
                k_near = min(n, max(1, (n * 3) // 5))
                near_ids = torch.arange(c - near, c)[torch.randperm(near, generator=gen)][:k_near]
                far_pool = torch.arange(0, max(c - near, 0))
                far_ids = (
                    far_pool[torch.randperm(far_pool.numel(), generator=gen)][: n - k_near]
                    if far_pool.numel()
                    else far_pool
                )
                chosen = torch.cat([near_ids, far_ids])
                if chosen.numel() < n:
                    rest = torch.tensor(
                        [b for b in torch.randperm(c, generator=gen).tolist() if b not in set(chosen.tolist())]
                    )
                    chosen = torch.cat([chosen, rest[: n - chosen.numel()]])
                chosen = chosen[torch.randperm(chosen.numel(), generator=gen)]
            elif pattern == "random":
                chosen = torch.randperm(c, generator=gen)[:n]
            elif pattern == "ranges":
                # row q of a 16-query tile takes the contiguous run [c - (lane + 1) n, c - lane n): in the ascending
                # union every chunk of n blocks belongs to one row, so the others are memberless there (T9)
                lane = j % 16
                if c >= 16 * n:
                    chosen = torch.arange(c - (lane + 1) * n, c - lane * n)
                else:
                    chosen = torch.randperm(c, generator=gen)[:n]
            else:
                raise ValueError(f"unknown pattern {pattern!r}; one of {PATTERNS}")
            ids[j, :n] = chosen
        if n < IDS:
            if bait == "sentinel":
                ids[j, n:] = sst.MASKED_INDEX
            elif bait == "range":
                ids[j, n:] = blocks + torch.arange(IDS - n)
            else:
                future = torch.arange(c, blocks)
                if future.numel():
                    take = future[torch.randperm(future.numel(), generator=bait_gen)]
                    fill = take[: IDS - n]
                    if fill.numel() < IDS - n:
                        fill = torch.cat(
                            [fill, torch.full((IDS - n - fill.numel(),), sst.MASKED_INDEX, dtype=torch.int64)]
                        )
                    ids[j, n:] = fill
                else:
                    ids[j, n:] = sst.MASKED_INDEX
    return ids


def make_inputs(
    *,
    S: int,
    T: int,
    IDS: int,
    H: int = 6,
    P: int = 0,
    pattern: str = "random",
    seed: int = 0,
    Dq: int = 256,
    W: int = 512,
    bait: str = "valid",
) -> Inputs:
    """Random q / kv (rounded to bf16 once), the block ids of ``select_rows`` and the positions ``P + j``."""

    if P % 4:
        raise ValueError("P must be a multiple of 4 (the slab offsets are)")
    if P + S > T:
        raise ValueError(f"the rows P + S = {P + S} must lie inside the cache T = {T}")
    gen = torch.Generator().manual_seed(seed)
    q = _round_bf16(torch.randn(1, H, S, Dq, generator=gen))
    kv = _round_bf16(torch.randn(1, 1, T, W, generator=gen))
    ids = select_rows(S, T, IDS, P, pattern, gen, bait=bait)
    return Inputs(q=q, kv=kv, block_ids=ids.reshape(1, 1, S, IDS), positions=P + torch.arange(S), P=P, T=T)


def upload(inputs: Inputs, mesh):
    """The four device tensors the program takes (ROW_MAJOR, DRAM)."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    q = ttnn.from_torch(
        inputs.q.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram
    )
    kv = ttnn.from_torch(
        inputs.kv.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram
    )
    ids = ttnn.from_torch(
        inputs.block_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    pos = ttnn.from_torch(
        inputs.positions.reshape(1, 1, 1, -1).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    return q, kv, ids, pos


def probe_row(inputs: Inputs, head: int, row: int, block: int, token_in_block: int, beta: float = 4.0) -> torch.Tensor:
    """Point head ``head``'s query of ``row`` at the K of token ``block * 4 + token_in_block`` (q = beta * k, rounded to
    bf16): with beta = 4 the attended score margin is ~60 nats, so the softmax puts its whole mass on that token and
    the output is its V row (if the row may attend the token) or its own reference (if not)."""

    t = block * 4 + token_in_block
    k = inputs.kv[0, 0, t, 256:512]
    q = inputs.q.clone()
    q[0, head, row] = _round_bf16(beta * k)
    return q
