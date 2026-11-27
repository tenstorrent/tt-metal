# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader of the long FP64 KDA goldens (``KimiK3KdaGolden``), written by ``golden_producer.py`` next to it.

The golden ``K3_LAYER0_GOLDEN`` (``k3-l0-kda-fp64-stored-fp32-1m``) is Kimi-K3 ``layers.0`` over 1,044,480 code
tokens of a vLLM trace, computed in FP64 and stored FP32. Every segment file (``golden_producer.segment_path``) holds
the segment's ``output`` rows and the layer's full state after it; the manifest names them as products ``output``
(``row``) and ``state.<name>`` (``segment_end``: ``state.recurrent`` ``[1, heads, k, v]`` and
``state.{q,k,v}_convolution``). Products are returned in their stored dtype; the scorers widen them to FP64.

Registered goldens are read through ``Fp64Golden.read_rows``. ``load_window`` and ``state_after`` read a golden
directory by path, also one still being written (the dev analysis reads goldens in progress).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import MANIFEST, Fp64Golden, MappedFile, read_slices
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer import (
    PRODUCER,
    PRODUCER_VERSION,
    segment_path,
)

STATE_PREFIX = "state."  # segment-end products are the state's fields, "state.<name>"


def _rows(run_dir: Path, segment: int, start: int, end: int) -> torch.Tensor:
    """Output rows ``[start, end)`` from the segment files (the caller checks the bounds)."""
    parts = []
    for index in range(start // segment, -(-end // segment)):
        file = MappedFile(segment_path(run_dir, index))
        lo, hi = file.contents["start"], file.contents["end"]
        parts.append((file, file.contents["output"][max(start, lo) - lo : min(end, hi) - lo]))
    return read_slices({"output": parts})["output"]


def _state(run_dir: Path, index: int) -> dict[str, torch.Tensor]:
    """The state after segment ``index``, read out of its file (the segment's output rows are not read)."""
    file = MappedFile(segment_path(run_dir, index))
    return read_slices({name: [(file, t)] for name, t in file.contents["state"].items()})


class KimiK3KdaGolden(Fp64Golden):
    """A registered long KDA golden of ``golden_producer`` version ``PRODUCER_VERSION``: products ``output``
    (rows) and ``state.<name>`` (after a segment)."""

    producer = PRODUCER
    producer_version = PRODUCER_VERSION

    def _read_rows(self, names: list[str], start: int, end: int) -> dict[str, torch.Tensor]:
        if names != ["output"]:
            raise ValueError(f"golden {self.golden_id}'s row product is output, not {names}")
        return {"output": _rows(self.path, self.manifest["segment"], start, end)}

    def _segment_end(self, names: list[str], index: int) -> dict[str, torch.Tensor]:
        state = _state(self.path, index)
        return {n: state[n.removeprefix(STATE_PREFIX)] for n in names}


def load_window(run_dir: Path, start: int, end: int) -> tuple[torch.Tensor, KDAReferenceState | None]:
    """Golden output rows ``[start, end)`` and the state before ``start`` (``None`` at token 0), from the golden
    directory ``run_dir`` (complete or not).

    ``start`` must fall on a segment boundary, since states are saved only there.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    segment = manifest["segment"]
    if start % segment:
        raise ValueError(f"start {start} is not a multiple of the segment size {segment}")
    available = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    if not start < end <= available:
        raise ValueError(f"window [{start}, {end}) is not inside the {available} golden tokens of {run_dir}")
    first = start // segment
    state = KDAReferenceState(**_state(run_dir, first - 1)) if first else None
    return _rows(run_dir, segment, start, end), state


def state_after(run_dir: Path, end: int) -> KDAReferenceState:
    """The golden state after token ``end``, which must be the end of a complete segment of ``run_dir``."""
    run_dir = Path(run_dir)
    ends = [segment["end"] for segment in json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))["segments"]]
    if end not in ends:
        raise ValueError(f"token {end} is not the end of a complete segment of the golden in {run_dir}")
    return KDAReferenceState(**_state(run_dir, ends.index(end)))
