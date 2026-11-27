# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run one long-context accuracy case (``block.LongContextCase``) at one context length: ``run_case``.

Order, so that a wrong golden or weights fail before any device time: open the golden through the case's reader,
``case.prepare`` (weights, digests, scorer), require the golden's identity and every product the case reads, then
build the device Block. The chunks run in order from row 0. One prefetch thread reads chunk ``c + 1``'s golden rows
and builds its device input while the device runs chunk ``c`` and the host scores it. Each chunk is scored, then
checked (``BlockScorer.check_chunk``); the first chunk with a violation stops the run. After the last chunk the
carried state is read, the run is scored into metric tiers and checked against the context length's run bars.

First the host heap is set to keep freed memory for reuse (``_retain_freed_heap``, one ``LC_HEAP`` line).

Progress (a 1M run takes minutes; a stage silent beyond its expected time is stuck): a timestamped ``LC_STAGE`` line
when each stage outside the chunk loop starts and ends (golden open, weight load, identity, device build, then
carried state and score run), and one ``LC_CHUNK`` line per chunk with the case, chunk i/N, rows [lo, hi), elapsed
time, ETA (from the chunks after the first, whose one-time cost it leaves out; ``?`` after the first) and the chunk's
time split: ``golden`` (read
and input build on the prefetch thread, overlapped), ``wait`` (the part the loop waited for it), ``device``
(forward including readback) and ``score`` (``score_chunk`` + ``check_chunk``). Then one ``LONG_CONTEXT_RESULT=``
JSON line.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import check
from models.demos.common.prefill.long_context_accuracy.block import METRIC_TIERS, LongContextCase, MetricTiers
from models.demos.common.prefill.long_context_accuracy.fp64_golden import open_golden


@dataclass
class Result:
    """One run of a case: its metric tiers (empty after an early stop), every violation (chunk contracts and bars,
    or run bars), the first row of the chunk that stopped the run (``None`` when every chunk ran) and the time split
    in seconds."""

    case_id: str
    golden_id: str
    context_len: int
    chunk: int
    metric_tiers: MetricTiers = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    stopped_at: int | None = None
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


_SPAN_KEYS = {"start", "end", "rows"}


def _check_tiers(tiers: MetricTiers) -> None:
    """Raise unless ``tiers`` is keyed by ``METRIC_TIERS``, every measurement has its span (``start``, ``end``,
    ``rows``: a bar failure names it) and no metric name is in two tiers (a bar names one quantity)."""
    unknown = sorted(set(tiers) - set(METRIC_TIERS))
    if unknown:
        raise ValueError(f"metric tiers {unknown} are not among {METRIC_TIERS}")
    owner: dict[str, int] = {}
    for tier, measurements in sorted(tiers.items()):
        unspanned = [m for m in measurements if not _SPAN_KEYS <= set(m)]
        if unspanned:
            raise ValueError(f"metric tier {tier} has measurements without {sorted(_SPAN_KEYS)}: {unspanned[:1]}")
        for name in {k for m in measurements for k in m} - _SPAN_KEYS:
            if owner.setdefault(name, tier) != tier:
                raise ValueError(f"metric {name!r} is in metric tiers {owner[name]} and {tier}")


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s" if math.isfinite(seconds) else "?"


# glibc mallopt parameters (malloc.h) and the values that keep freed heap in the process (int range: mallopt takes
# an int, so the MALLOC_*_ environment values of m06n.12 do not fit).
_HEAP_REUSE = {"M_MMAP_THRESHOLD": (-3, 2**31 - 1), "M_TRIM_THRESHOLD": (-1, 2**31 - 1), "M_TOP_PAD": (-2, 2**30)}


def _retain_freed_heap() -> dict[str, int]:
    """Make glibc keep freed host heap for reuse (no mmap per large block, no trim, 1 GiB top pad) instead of
    returning it to the OS: scoring allocates the same large temporaries every chunk, and zero-faulting fresh pages
    for each dominated it (m06n.12: DSA scoring 2.2x and KDA 5.5x faster, results bit-identical). Process-wide and
    lasting: freed heap up to the scoring peak stays resident, for tt-metal's host allocations too. In code, not in
    ``MALLOC_*`` environment variables, so every launcher times the same. Returns each ``mallopt`` result (1 set,
    0 refused)."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    return {name: libc.mallopt(param, value) for name, (param, value) in _HEAP_REUSE.items()}


def run_case(case: LongContextCase, context_len: int, mesh: Any, checkpoint: Path) -> Result:
    """Run ``case`` over rows ``[0, context_len)`` on ``mesh`` and return its ``Result``; the caller asserts
    ``result.passed``. Raises the registry's ``GoldenError`` on a missing or mismatched golden, before any device
    work."""
    if context_len not in case.bars:
        raise ValueError(f"case {case.id} has no bars for context length {context_len} (has {case.context_lengths})")
    mesh_shape = tuple(mesh.shape)
    if mesh_shape != case.mesh_shape:
        raise ValueError(f"case {case.id} runs on a {case.mesh_shape} mesh, not {mesh_shape}")
    bars = case.bars[context_len]
    tag = f"{case.id} {context_len}"
    began = time.perf_counter()
    heap = _retain_freed_heap()
    logger.info(
        f"LC_HEAP {tag}: glibc mallopt "
        + " ".join(f"{name}={_HEAP_REUSE[name][1]} (set {ok})" for name, ok in heap.items())
    )
    seconds: dict[str, float] = {}

    @contextmanager
    def stage(name: str) -> Iterator[None]:
        logger.info(f"LC_STAGE {tag}: {name} ...")
        t = time.perf_counter()
        yield
        seconds[name] = time.perf_counter() - t
        logger.info(f"LC_STAGE {tag}: {name} done in {_format_seconds(seconds[name])}")

    with stage("golden open"):
        golden = open_golden(case.golden_id, reader=case.golden_reader)
        if context_len > golden.rows:
            raise ValueError(f"golden {golden.golden_id} has {golden.rows} rows, fewer than {context_len}")
    with stage("weight load"):
        prepared = case.prepare(golden, checkpoint, context_len)
    scorer = prepared.scorer
    names = list(dict.fromkeys([*prepared.input_tensors, *scorer.golden_tensors]))
    with stage("identity"):
        golden.require(
            identity=dict(prepared.identity.identity),
            weights_sha256=dict(prepared.identity.weights_sha256),
            products=names,
        )
    result = Result(case.id, golden.golden_id, context_len, case.chunk, seconds=seconds)
    block = prepared.device_block
    num_chunks = -(-context_len // case.chunk)

    def span(c: int) -> tuple[int, int]:
        return c * case.chunk, min((c + 1) * case.chunk, context_len)

    def fetch(c: int) -> tuple[Mapping[str, torch.Tensor], torch.Tensor, float]:
        t = time.perf_counter()
        lo, hi = span(c)
        rows = golden.read_rows(names, lo, hi)
        return rows, prepared.block_input(lo, hi, rows), time.perf_counter() - t

    totals = {"golden": 0.0, "wait": 0.0, "device": 0.0, "score": 0.0}
    try:
        with stage("device build"):
            block.build(mesh, context_len, case.chunk)
        first_chunk_end = None  # (the ETA's rate leaves out the first chunk's one-time cost: weight load, compile)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lc-golden") as prefetch:
            pending = prefetch.submit(fetch, 0)
            try:
                for c in range(num_chunks):
                    lo, hi = span(c)
                    t = time.perf_counter()
                    rows, x, golden_s = pending.result()
                    wait_s = time.perf_counter() - t
                    if c + 1 < num_chunks:
                        pending = prefetch.submit(fetch, c + 1)
                    t = time.perf_counter()
                    outputs = block.forward(x, lo, hi)
                    device_s = time.perf_counter() - t
                    t = time.perf_counter()
                    scorer.score_chunk(lo, outputs, rows)
                    violations = scorer.check_chunk(lo, outputs, rows, bars.chunk)
                    score_s = time.perf_counter() - t
                    for key, value in zip(totals, (golden_s, wait_s, device_s, score_s)):
                        totals[key] += value
                    now = time.perf_counter()
                    if first_chunk_end is None:
                        first_chunk_end, eta = now, math.nan
                    else:
                        eta = (now - first_chunk_end) / c * (num_chunks - c - 1)
                    logger.info(
                        f"LC_CHUNK {case.id} chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) "
                        f"elapsed {_format_seconds(now - began)} eta {_format_seconds(eta)} | "
                        f"golden {golden_s:.3f}s wait {wait_s:.3f}s device {device_s:.3f}s score {score_s:.3f}s"
                    )
                    if violations:
                        result.violations = [f"rows [{lo}, {hi}): {v}" for v in violations]
                        result.stopped_at = lo
                        logger.error(f"LC_STOP {tag}: chunk {c + 1}/{num_chunks} rows [{lo}, {hi}): {violations}")
                        break
            finally:
                pending.cancel()
        carried = {}
        if result.stopped_at is None:
            with stage("carried state"):
                carried = block.carried_state()
    finally:
        block.release()
    seconds.update({f"chunk_{key}": value for key, value in totals.items()})

    if result.stopped_at is None:
        with stage("score run"):
            tiers = scorer.score_run(carried, golden)
            _check_tiers(tiers)
            result.metric_tiers = tiers
            result.violations = check([m for tier in METRIC_TIERS for m in tiers.get(tier, [])], bars.run)
    seconds["total"] = time.perf_counter() - began
    result.seconds = {key: round(value, 3) for key, value in seconds.items()}
    logger.info("LONG_CONTEXT_RESULT=" + result.to_json())
    return result
