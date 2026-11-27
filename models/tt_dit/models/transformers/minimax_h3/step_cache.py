# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""First Block Cache for the H3 denoise loop: a step whose first block changes the hidden state almost as it did
at the last fully computed step skips blocks 1.. and adds what they added then. The decisions are `DBCacheContext`'s
(one branch: the loop runs one forward per step); this module holds the device side.

MINIMAX_H3_STEP_CACHE="<threshold>[,<warmup steps>[,<max reused steps in a row>]]" turns it on; unset or a zero
threshold leaves every step computed. Untraced denoise only: the decision is a host readback.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import ttnn

from ....utils.dbcache import DBCacheConfig, DBCacheContext

STEP_CACHE_ENV = "MINIMAX_H3_STEP_CACHE"


def step_cache_config_from_env() -> DBCacheConfig | None:
    """The cache's config from MINIMAX_H3_STEP_CACHE, or None when it is off. The threshold is the relative L1
    change of the first block's residual below which a step is reused. Warmup steps (default 0) always compute;
    the last field (default -1, no cap) bounds the reused steps in a row."""
    raw = os.environ.get(STEP_CACHE_ENV, "").strip()
    if not raw:
        return None
    fields = raw.split(",")
    try:
        if len(fields) > 3:
            raise ValueError
        threshold = float(fields[0])
        warmup = int(fields[1]) if len(fields) > 1 else 0
        in_a_row = int(fields[2]) if len(fields) > 2 else -1
        if threshold < 0.0 or warmup < 0:
            raise ValueError
    except ValueError:
        msg = f"{STEP_CACHE_ENV}={raw!r}: expected '<threshold>[,<warmup steps>[,<max reused steps in a row>]]'"
        raise ValueError(msg) from None
    config = DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        residual_diff_threshold=threshold,
        max_warmup_steps=warmup,
        max_continuous_cached_steps=in_a_row,
    )
    return config if config.enabled else None


class FirstBlockCache:
    """One denoise's cache. `valid_rows` is `[1, 1, N, 1]`, sharded as the hidden state: 1 on the sequence's rows
    and 0 on its padding, which stays out of the change test."""

    def __init__(
        self, config: DBCacheConfig, *, mesh_device: ttnn.MeshDevice, valid_rows: ttnn.Tensor | None = None
    ) -> None:
        if ttnn.using_distributed_env():
            msg = "the H3 step cache's change readback is not implemented for multi-host meshes"
            raise NotImplementedError(msg)
        self.context = DBCacheContext(config, name="minimax_h3", num_branches=1)
        self.mesh_device = mesh_device
        self.valid_rows = valid_rows
        # The last computed step's: block 0's change, its L1 norm, and what blocks 1.. added.
        self._first_change: ttnn.Tensor | None = None
        self._first_norm = 0.0
        self._rest_change: ttnn.Tensor | None = None

    def run_stack(
        self, hidden: ttnn.Tensor, run_block: Callable[[int, ttnn.Tensor], ttnn.Tensor], num_blocks: int
    ) -> ttnn.Tensor:
        """One step's pass over the blocks: `run_block(index, hidden)` runs one block."""
        context = self.context
        context.mark_step_begin()
        first = run_block(0, hidden)
        change = ttnn.subtract(first, hidden)
        if self.valid_rows is not None:
            masked = ttnn.multiply(change, self.valid_rows)
            ttnn.deallocate(change)
            change = masked
        gate = context.gate(0)
        reuse = gate == "cache"
        if gate == "dynamic":
            apart = ttnn.subtract(change, self._first_change)
            reuse = context.decide(0, self._l1(apart) / self._first_norm if self._first_norm > 0.0 else float("inf"))
            ttnn.deallocate(apart)
        if reuse:
            context.add_cached_step(0)
            ttnn.deallocate(change)
            return ttnn.add(first, self._rest_change)

        # A block's output may sit in a CCL ping-pong buffer that the later blocks write again.
        first = ttnn.clone(first)
        out = first
        for index in range(1, num_blocks):
            out = run_block(index, out)
        rest_change = ttnn.subtract(out, first)
        self.release()
        self._first_change, self._rest_change = change, rest_change
        self._first_norm = self._l1(change)
        context.add_computed_step(0)
        return out

    def _l1(self, value: ttnn.Tensor) -> float:
        """sum|value| over the mesh: one small readback. The per-device sums stay bf16; their rounding is far below
        any useful threshold."""
        magnitude = ttnn.abs(value)
        local = ttnn.sum(magnitude, dim=[2, 3], keepdim=True)
        ttnn.deallocate(magnitude)
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        total = float(ttnn.to_torch(local, mesh_composer=composer).double().sum())
        ttnn.deallocate(local)
        return total

    def release(self) -> None:
        """Free the device tensors. The step counts stay for `summary`."""
        for tensor in (self._first_change, self._rest_change):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self._first_change = self._rest_change = None

    def summary(self) -> str:
        """One log line: how many steps were reused, which, and each tested step's relative change."""
        report = self.context.summary()
        reused, changes = report["cached_steps"][0], report["residual_diffs"][0]
        tested = " ".join(f"{step}:{change:.4f}" for step, change in sorted(changes.items()))
        return (
            f"step cache ({report['config']}): reused {len(reused)} of {report['executed_steps']} steps "
            f"{reused}; relative change by step: {tested}"
        )
