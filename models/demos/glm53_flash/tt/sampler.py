# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device sampling for the decode step (plan workstream D, 2026-10-31).

The head's logits are the complete vocab on every chip (``TtModelHead`` all-gathers the TP column blocks), so the
sampler is model-local and shape-static: ``ttnn.topk`` keeps the ``CANDIDATES`` = 64 largest logits of every user
(two tiles: the op's unit tests cover widths 64 and 256; a one-tile row hangs the chip, 2026-10-31 09:00),
``ttnn.sampling`` draws one id per user from those candidates under the user's top-k / top-p / temperature
(``k`` uint32, ``p`` and ``1/T`` bf16, one value per user) after ``ttnn.manual_seed`` seeded every user's
generator, and a blend on the greedy mask hands greedy users the head's own argmax bit-identically (the greedy path
never sees the sampler's tie-break). Every parameter lives in a persistent device tensor written by
``copy_host_to_device_tensor`` before the step, so the whole thing captures into the decode trace; the seeds must
change between steps (``set_seeds``), or a replay repeats the draw.

Measured on one Blackhole chip (probe v2): topk(64) over ``[1, 1, 32, 154880]`` 1.0 ms warm, sampling 0.15 ms,
output ``[1, 1, 1, users]`` uint32 ROW_MAJOR; same seeds -> identical draws, new seeds -> varied draws, a peaked row
always its argmax, every draw inside the torch top-32.

Routing (the vLLM plugin, 2026-10-31): a batch goes to the device only when every sampled row has
``1 <= top_k <= max_device_top_k``; ``MAX_TOP_K`` = 32 is what the reference models declare. Rows with an unbounded
top-k (vLLM's default) need the plugin's opt-in to approximate it by ``MAX_TOP_K`` (``clamp_k``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn

CANDIDATES = 64  # top-k candidates per user handed to ttnn.sampling: two tiles (never one: a one-tile row hangs)
MAX_TOP_K = 32  # ttnn.sampling walks at most 32 candidates (the reference models' max_device_top_k)
MIN_TOP_K = 2  # k = 1 is untested by the op's unit tests; a greedy row takes the argmax through the mask instead
MAX_USERS = 32  # ttnn.sampling: 1 to 32 users per call (one core each)


def clamp_k(k: int) -> int:
    """vLLM's ``top_k`` to the device's: ``< 1`` (no top-k) and ``> MAX_TOP_K`` become ``MAX_TOP_K`` (sampling from
    the top-32 under top-p: the reference models' approximation), ``1`` becomes ``MIN_TOP_K`` (its row is greedy
    through the mask)."""
    k = int(k)
    if k < 1 or k > MAX_TOP_K:
        return MAX_TOP_K
    return max(k, MIN_TOP_K)


class TtSampler:
    """One instance per mesh (every chip samples its own row's users): ``users`` = the rows of the head's decode
    logits ``[1, 1, users, vocab]`` on a chip (the row's user slots, <= 32)."""

    def __init__(self, device, users: int):
        if not 1 <= users <= MAX_USERS:
            raise ValueError(f"ttnn.sampling serves 1..{MAX_USERS} users per call, not {users}")
        self.device, self.users = device, users
        rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
        u32 = lambda t: ttnn.from_torch(t, dtype=ttnn.uint32, layout=rm, device=device, memory_config=dram)
        b16 = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=rm, device=device, memory_config=dram)
        self.k = u32(torch.full((users,), MAX_TOP_K, dtype=torch.int32))
        self.p = b16(torch.ones(users))
        self.temp = b16(torch.ones(users))  # 1 / T
        self.seeds = u32(torch.arange(users, dtype=torch.int32))
        self.user_ids = u32(torch.arange(users, dtype=torch.int32))
        # the greedy mask as fp32 TILE [1, 1, users, 1]: 1.0 keeps the head's argmax, 0.0 takes the draw
        self.greedy = ttnn.from_torch(
            torch.ones(1, 1, users, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        self.tok = u32(torch.zeros(1, 1, 1, users, dtype=torch.int32))  # ttnn.sampling's output, persistent

    # ---- parameters (host -> the persistent device tensors; trace-safe) ----------------------------------
    def set_params(
        self, k: Sequence[int], p: Sequence[float], inv_temp: Sequence[float], greedy: Sequence[bool]
    ) -> None:
        """Per-user top-k (vLLM's, clamped), top-p in [0, 1], ``1 / T`` and the greedy flag (``temperature == 0`` or
        ``top_k == 1``: the row keeps the head's argmax)."""
        n = self.users
        if not (len(k) == len(p) == len(inv_temp) == len(greedy) == n):
            raise ValueError(f"{n} users: k {len(k)} p {len(p)} 1/T {len(inv_temp)} greedy {len(greedy)}")
        rm = ttnn.ROW_MAJOR_LAYOUT
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([clamp_k(v) for v in k], dtype=torch.int32), dtype=ttnn.uint32, layout=rm),
            self.k,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([min(max(float(v), 0.0), 1.0) for v in p]), dtype=ttnn.bfloat16, layout=rm),
            self.p,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([float(v) for v in inv_temp]), dtype=ttnn.bfloat16, layout=rm), self.temp
        )
        mask = torch.tensor([1.0 if g else 0.0 for g in greedy]).reshape(1, 1, n, 1)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.greedy)

    def set_seeds(self, seeds: Sequence[int]) -> None:
        """One seed per user for the next draw (a request's seed stepped per token, or the host RNG's)."""
        if len(seeds) != self.users:
            raise ValueError(f"{len(seeds)} seeds for {self.users} users")
        host = torch.tensor([int(s) & 0x7FFFFFFF for s in seeds], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.seeds
        )

    # ---- the step ----------------------------------------------------------------------------------------
    def __call__(self, logits: ttnn.Tensor, greedy_ids: ttnn.Tensor) -> ttnn.Tensor:
        """``logits`` ``[1, 1, users, vocab]`` bf16 TILE (the head's), ``greedy_ids`` ``[1, 1, users, 1]`` uint32
        ROW_MAJOR (the head's argmax) -> ids ``[1, 1, users, 1]`` uint32 ROW_MAJOR: the argmax on greedy users, the
        seeded draw elsewhere. Trace-capturable (no host values, persistent parameter tensors)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        n = self.users
        vals, idx = ttnn.topk(logits, k=CANDIDATES, dim=-1, largest=True, sorted=True)
        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(idx)
        ttnn.manual_seed(seeds=self.seeds, device=self.device, user_ids=self.user_ids)
        ttnn.sampling(vals, idx_rm, k=self.k, p=self.p, temp=self.temp, output_tensor=self.tok)  # [1, 1, 1, n]
        ttnn.deallocate(vals)
        ttnn.deallocate(idx_rm)
        # the draw [1, 1, 1, n] -> [1, 1, n, 1] (the same element order), both ids to fp32 (ids < 2^24) for the blend
        draw = ttnn.reshape(self.tok, [1, 1, n, 1])
        draw_f = ttnn.typecast(
            ttnn.to_layout(draw, ttnn.TILE_LAYOUT, memory_config=dram), ttnn.float32, memory_config=dram
        )
        greedy_f = ttnn.typecast(
            ttnn.to_layout(greedy_ids, ttnn.TILE_LAYOUT, memory_config=dram), ttnn.float32, memory_config=dram
        )
        keep = ttnn.multiply(greedy_f, self.greedy, memory_config=dram)
        take = ttnn.multiply(draw_f, ttnn.rsub(self.greedy, 1.0, memory_config=dram), memory_config=dram)
        out_f = ttnn.add(keep, take, memory_config=dram)
        out = ttnn.to_layout(ttnn.typecast(out_f, ttnn.uint32, memory_config=dram), ttnn.ROW_MAJOR_LAYOUT)
        for t in (draw_f, greedy_f, keep, take, out_f):
            ttnn.deallocate(t)
        return out
