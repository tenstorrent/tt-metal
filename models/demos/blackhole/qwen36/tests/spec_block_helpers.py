# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test helper: the ``mamba_state`` that vllm-tt-plugin hands a speculative MambaSpec model (Qwen36MTPContractForCausalLM),
built without vLLM or the plugin. No ttnn import.

The plugin's rule (vllm_tt_plugin.mamba_state, stage C06): a step of n query tokens at start_pos writes the state after
query token t into ``windows[i, t]``, which is block-table column t in mamba_cache_mode "none" and column W + t with
W = (start_pos + n - 1) // block_size in mode "align"; its start state is in ``start_blocks[i]``, the previous step's
``windows[i, a - 1]`` for the count a that step committed, or the block a prefill wrote. A prefill writes its end state
into ``windows[i, 0]``."""

from dataclasses import dataclass

import torch

from models.demos.blackhole.qwen36.tests.gdn_block_helpers import GroupState


@dataclass(frozen=True)
class SpecStepState:
    """Same fields as vllm_tt_plugin.mamba_state.MambaStepState."""

    groups: tuple
    checkpoint_positions: torch.Tensor | None = None
    intermediate_prefill_mask: torch.Tensor | None = None


class SpecBlocks:
    """The MambaSpec block ids of requests in fixed rows of a speculative launch with K drafts (see the module
    docstring). Blocks come from ``free`` (default: 1 .. num_blocks - 1, in order)."""

    def __init__(self, layout, K, mode, free=None):
        assert mode in ("none", "align")
        self.layout, self.K, self.mode = layout, K, mode
        self.bs = layout.block_size
        self.groups = range(layout.num_groups)
        self.free = list(free) if free is not None else list(range(1, layout.num_blocks))
        self.cols = {}  # row -> per group {block-table column: block}
        self.last = {}  # row -> per group, the blocks of the last step's windows
        self.count = {}  # row -> count its last step committed

    def _block(self, row, g, col):
        cols = self.cols[row][g]
        if col not in cols:
            cols[col] = self.free.pop(0)
        return cols[col]

    def admit(self, row):
        self.cols[row] = [{} for _ in self.groups]
        self.last.pop(row, None)
        self.count[row] = 1

    def release(self, row):
        """The row's blocks go to the front of the free list, so the next request reuses them."""
        freed = sorted({b for cols in self.cols.pop(row, []) for b in cols.values()})
        self.free[:0] = freed
        self.last.pop(row, None)
        self.count.pop(row, None)

    def prefill(self, row, start, end, hit=None):
        """(destination per group, start block per group or None at start 0) of a prefill of tokens [start, end).
        ``hit``: the blocks per group of a prefix-cache hit of ``start`` tokens (mode "align")."""
        if self.mode == "none":
            dest = [self._block(row, g, 0) for g in self.groups]
            starts = dest if start else None
        else:
            if hit is not None:
                for g in self.groups:
                    self.cols[row][g][(start - 1) // self.bs] = hit[g]
            dest = [self._block(row, g, (end - 1) // self.bs) for g in self.groups]
            starts = [self.cols[row][g][(start - 1) // self.bs] for g in self.groups] if start else None
        self.last[row] = [[d] for d in dest]
        self.count[row] = 1
        return dest, starts

    def decode(self, row, start_pos, n):
        """(windows per group [1 + K blocks], start block per group) of a step of n query tokens at start_pos."""
        W = 0 if self.mode == "none" else (start_pos + n - 1) // self.bs
        starts = [self.last[row][g][self.count[row] - 1] for g in self.groups]
        windows = [[self._block(row, g, W + t) for t in range(self.K + 1)] for g in self.groups]
        self.last[row] = windows
        self.count[row] = 1
        return windows, starts

    def commit(self, row, a):
        """The accept walk committed a tokens of the row's last step."""
        assert 1 <= a <= len(self.last[row][0]), a
        self.count[row] = a

    def state(self, rows, bmax, intermediate=None):
        """mamba_state for ``rows``: {row: (windows per group, starts per group or None)}; other rows are padding
        (block 0). ``intermediate``: rows whose prefill chunk does not end their prompt (prefill steps only)."""
        groups = []
        for g, names in enumerate(self.layout.group_names):
            windows = torch.zeros(bmax, self.K + 1, dtype=torch.int32)
            starts = torch.zeros(bmax, dtype=torch.int32)
            for r, (win, start) in rows.items():
                w = win[g] if isinstance(win[g], list) else [win[g]]
                windows[r, : len(w)] = torch.tensor(w, dtype=torch.int32)
                if start is not None:
                    starts[r] = start[g]
            groups.append(GroupState(layer_names=names, windows=windows, start_blocks=starts))
        mask = None
        if intermediate is not None:
            mask = torch.ones(bmax, dtype=torch.bool)
            for r in rows:
                mask[r] = r in intermediate
        return SpecStepState(groups=tuple(groups), intermediate_prefill_mask=mask)
