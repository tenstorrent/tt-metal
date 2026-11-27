# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contracts of a long-context accuracy case: the Block under test on the device, how it is scored, and the case data.

One case runs one Block (KDA, MoE, DSA, ...) of one model over the rows ``[0, context_len)`` of an FP64 golden,
chunk by chunk in order, teacher-fed from the golden, and scores every chunk against it (``run.run_case``). Shared
code here names no model or block; each block type adds a ``BlockScorer``, each model a ``DeviceBlock`` per block and
its ``LongContextCase``s.

Metric tiers: every ``BlockScorer.score_run`` returns its metrics grouped by ``metric_tier`` (``METRIC_TIERS``):
0 the headline (for example output NRMSE over the run), 1 the tails (per-window and per-row percentiles, worst
windows), 2 diagnostics (gain, histograms, cache errors). Each tier is a list of measurements, each a dict with
``start``, ``end`` and ``rows`` (``WindowedErrors.whole`` / ``windows``) plus metrics; a metric name lives in one
tier only. Bars (``acceptance_bars``) name metrics, wherever their tier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, GrowthBar
from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden

METRIC_TIERS = (0, 1, 2)  # metric_tier: 0 headline, 1 tails, 2 diagnostics

MetricTiers = dict[int, list[dict]]  # metric_tier -> measurements (dicts with start, end, rows and metrics)


@dataclass(frozen=True)
class BlockOutputs:
    """One device call's results on rows ``[lo, hi)``, on the host, in global row order, without padding rows.

    ``output``: the Block's output ``[hi - lo, ...]``. ``extras``: what else its scorer reads per call, by name
    (each ``[hi - lo, ...]`` or per call), for example MoE routing ids and weights, DSA top-k indices, a KDA state
    snapshot."""

    output: torch.Tensor
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)


class DeviceBlock(Protocol):
    """One Block of one model on the device, fed host rows and returning host results.

    It owns the mesh layout, the sequence-parallel permutation, the padding of a partial last chunk, the readbacks
    and its device resources. ``run_case`` calls ``build`` once, ``forward`` for consecutive chunks from row 0,
    ``carried_state`` once after the last chunk, and ``release`` always (also after a failure or an early stop)."""

    def build(self, mesh: Any, context_len: int, chunk: int) -> None:
        """Allocate weights, caches and state for rows ``[0, context_len)`` in calls of ``chunk`` rows."""

    def forward(self, x: torch.Tensor, lo: int, hi: int) -> BlockOutputs:
        """Run rows ``[lo, hi)`` given their host input ``x`` ``[hi - lo, hidden]``; ``hi - lo`` <= ``chunk``."""

    def carried_state(self) -> Mapping[str, torch.Tensor]:
        """The state the Block carries after the last chunk, on the host (KDA recurrent state, DSA caches; empty
        for a stateless Block)."""

    def release(self) -> None:
        """Free the device resources ``build`` and ``forward`` took."""


class BlockScorer(Protocol):
    """Scores one Block type's device results against its golden products, chunk by chunk, then over the run.

    ``golden_tensors``: the golden products (``Fp64Golden.read_rows`` names) ``score_chunk`` / ``check_chunk`` read
    for each chunk; ``run_case`` prefetches them."""

    golden_tensors: Sequence[str]

    def score_chunk(self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor]) -> None:
        """Accumulate the chunk ``[lo, lo + n)``; ``rows`` holds the golden products of ``golden_tensors`` and of the
        case's ``input_tensors`` for these rows."""

    def check_chunk(
        self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor], bars: Sequence[AcceptanceBar]
    ) -> list[str]:
        """Every violation of the chunk just scored: the block's contracts (finite output, distinct routing, ...) and
        ``bars``, the context length's per-chunk bars. ``run_case`` stops at the first chunk with any."""

    def score_run(self, carried_state: Mapping[str, torch.Tensor], golden: Fp64Golden) -> MetricTiers:
        """The run's metrics by ``metric_tier``, after every chunk is scored; ``carried_state`` from the device."""


@dataclass(frozen=True)
class GoldenIdentity:
    """What the case's golden must have been built from (``Fp64Golden.require``): manifest ``identity`` fields
    compared by equality, and per-tensor-group ``weights_sha256`` digests of the weights the case loaded."""

    identity: Mapping[str, Any] = field(default_factory=dict)
    weights_sha256: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBars:
    """The bars of one context length: ``chunk`` checked on every chunk (``BlockScorer.check_chunk``), ``run`` on
    the run's metric tiers."""

    chunk: Sequence[AcceptanceBar] = ()
    run: Sequence[AcceptanceBar | GrowthBar] = ()


@dataclass(frozen=True)
class PreparedCase:
    """What a case loads from the checkpoint and the golden before device time (``LongContextCase.prepare``).

    ``block_input(lo, hi, rows)`` returns the device input ``[hi - lo, hidden]`` of rows ``[lo, hi)`` from the
    golden products ``input_tensors`` (in ``rows``); it runs on the prefetch thread, beside the device call of the
    previous chunk. ``device_block``: the case's Block, not yet built."""

    identity: GoldenIdentity
    scorer: BlockScorer
    input_tensors: Sequence[str]
    block_input: Callable[[int, int, Mapping[str, torch.Tensor]], torch.Tensor]
    device_block: DeviceBlock


@dataclass(frozen=True)
class LongContextCase:
    """One Block of one model against one golden, at its context lengths.

    * ``id``: the case's name in results and test ids;
    * ``golden_id`` and ``golden_reader``: the registered golden and the ``Fp64Golden`` reader of its layout;
    * ``placement``: the ``pytest.param`` of ``(mesh_shape, device_params)`` (with marks) the device test opens;
    * ``chunk``: rows per device call; a context length that is not a multiple ends in a partial chunk;
    * ``bars``: the ``ContextBars`` of every context length the case runs (its keys are the context lengths);
    * ``checkpoint()``: the checkpoint directory ``prepare`` reads (skips the test when there is none);
    * ``prepare(golden, checkpoint, context_len)``: loads what needs the checkpoint (weights, digests, embeddings)
      and returns the ``PreparedCase``."""

    id: str
    golden_id: str
    golden_reader: type[Fp64Golden]
    placement: Any
    chunk: int
    bars: Mapping[int, ContextBars]
    checkpoint: Callable[[], Path]
    prepare: Callable[[Fp64Golden, Path, int], PreparedCase]

    @property
    def context_lengths(self) -> tuple[int, ...]:
        return tuple(self.bars)

    @property
    def mesh_shape(self) -> tuple[int, ...]:
        return tuple(self.placement.values[0])
