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

On a mesh every row's chips sample that row's own users: the parameter tensors are sharded over the mesh rows
(``GLMMesh.shard_2d(rows=0)``) and replicated over the TP columns, and the host-side lists are flat in row-major
order (row ``r``'s users at ``[r * users, (r + 1) * users)``: the slot table's ``read_index``).

Measured on one Blackhole chip (probe v2 + test_head_sampler): topk(64) over ``[1, 1, 32, 154880]`` 1.0 ms warm,
sampling 0.15 ms, the whole step 0.81 ms eager at 8 users; output ``[1, 1, 1, users]`` uint32 ROW_MAJOR; same seeds
-> identical draws, new seeds -> varied draws, a peaked row always its argmax, every draw inside the torch top-32.

Routing (the vLLM plugin, 2026-10-31): a batch goes to the device only when every sampled row has
``1 <= top_k <= max_device_top_k``; ``MAX_TOP_K`` = 32 is what the reference models declare. Rows with an unbounded
top-k (vLLM's default) need the plugin's opt-in to approximate it by ``MAX_TOP_K`` (``clamp_k``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn
from models.demos.glm53_flash.tt.sampling_config import (  # noqa: E402  (re-exported names)
    CANDIDATES,
    DEVICE_SAMPLING_ENV,
    MAX_TOP_K,
    MAX_USERS,
    MIN_TOP_K,
    SAMPLING_SEED_ENV,
    clamp_k,
    device_sampling_enabled,
)

__all__ = [
    "CANDIDATES",
    "DEVICE_SAMPLING_ENV",
    "MAX_TOP_K",
    "MAX_USERS",
    "MIN_TOP_K",
    "SAMPLING_SEED_ENV",
    "TtSampler",
    "clamp_k",
    "device_sampling_enabled",
]


class TtSampler:
    """``users`` = the rows of the head's decode logits ``[1, 1, users, vocab]`` on a chip (the row's user slots,
    <= 32); ``mesh`` (a ``GLMMesh``) shards the parameters over its rows (``None``: one chip, ``rows = 1``)."""

    def __init__(self, device, users: int, mesh=None):
        if not 1 <= users <= MAX_USERS:
            raise ValueError(f"ttnn.sampling serves 1..{MAX_USERS} users per call, not {users}")
        self.device, self.users, self.mesh = device, users, mesh
        self.rows = int(mesh.rows) if mesh is not None else 1
        n = self.rows * users
        self.k = self._dev(torch.full((n,), MAX_TOP_K, dtype=torch.int32), ttnn.uint32, dim=0)
        self.p = self._dev(torch.ones(n), ttnn.bfloat16, dim=0)
        self.temp = self._dev(torch.ones(n), ttnn.bfloat16, dim=0)  # 1 / T
        self.seeds = self._dev(torch.arange(n, dtype=torch.int32), ttnn.uint32, dim=0)
        self.user_ids = self._dev(torch.arange(users, dtype=torch.int32).repeat(self.rows), ttnn.uint32, dim=0)
        # the greedy mask as fp32 TILE [1, 1, users, 1] per chip: 1.0 keeps the head's argmax, 0.0 takes the draw
        self.greedy = self._dev(torch.ones(1, 1, n, 1), ttnn.float32, dim=2, layout=ttnn.TILE_LAYOUT)
        # ttnn.sampling's output [1, 1, 1, users] per chip, persistent (replicated: every chip writes its own)
        self.tok = self._dev(torch.zeros(1, 1, 1, users, dtype=torch.int32), ttnn.uint32, dim=None)

    # ---- host <-> device plumbing -------------------------------------------------------------------------
    def _host(self, t: torch.Tensor, dtype, dim, layout=ttnn.ROW_MAJOR_LAYOUT) -> ttnn.Tensor:
        """A host tensor in the device tensor's placement: dim ``dim`` split over the mesh rows (``None``:
        replicated); the single-chip form takes it whole."""
        if self.mesh is None or dim is None:
            return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout)
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.mesh.shard_2d(dim, None))

    def _dev(self, t: torch.Tensor, dtype, dim, layout=ttnn.ROW_MAJOR_LAYOUT) -> ttnn.Tensor:
        return ttnn.to_device(self._host(t, dtype, dim, layout), self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _put(self, t: torch.Tensor, dst: ttnn.Tensor, dtype, dim, layout=ttnn.ROW_MAJOR_LAYOUT) -> None:
        ttnn.copy_host_to_device_tensor(self._host(t, dtype, dim, layout), dst)

    # ---- parameters (host -> the persistent device tensors; trace-safe) ----------------------------------
    def set_params(
        self, k: Sequence[int], p: Sequence[float], inv_temp: Sequence[float], greedy: Sequence[bool]
    ) -> None:
        """Per-user (flat, row-major over the mesh rows) top-k (vLLM's, clamped), top-p in [0, 1], ``1 / T`` and the
        greedy flag (``temperature == 0`` or ``top_k == 1``: the user keeps the head's argmax)."""
        n = self.rows * self.users
        if not (len(k) == len(p) == len(inv_temp) == len(greedy) == n):
            raise ValueError(f"{n} users: k {len(k)} p {len(p)} 1/T {len(inv_temp)} greedy {len(greedy)}")
        self._put(torch.tensor([clamp_k(v) for v in k], dtype=torch.int32), self.k, ttnn.uint32, 0)
        self._put(torch.tensor([min(max(float(v), 0.0), 1.0) for v in p]), self.p, ttnn.bfloat16, 0)
        self._put(torch.tensor([float(v) for v in inv_temp]), self.temp, ttnn.bfloat16, 0)
        mask = torch.tensor([1.0 if g else 0.0 for g in greedy]).reshape(1, 1, n, 1)
        self._put(mask, self.greedy, ttnn.float32, 2, ttnn.TILE_LAYOUT)

    def set_greedy(self) -> None:
        """Every user keeps the head's argmax (the plugin samples on the host, or nobody samples)."""
        n = self.rows * self.users
        self.set_params([MAX_TOP_K] * n, [1.0] * n, [1.0] * n, [True] * n)

    def set_seeds(self, seeds: Sequence[int]) -> None:
        """One seed per user (flat, row-major) for the next draw: a request's seed stepped per token, or the
        host RNG's."""
        n = self.rows * self.users
        if len(seeds) != n:
            raise ValueError(f"{len(seeds)} seeds for {n} users")
        self._put(torch.tensor([int(s) & 0x7FFFFFFF for s in seeds], dtype=torch.int32), self.seeds, ttnn.uint32, 0)

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
