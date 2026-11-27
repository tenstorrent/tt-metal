# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader of the long FP64 KDA goldens (``KimiK3KdaGolden``), written by ``tests/kda/long_context_golden.py``.

The golden ``K3_LAYER0_GOLDEN`` (``k3-l0-kda-fp64-1m``) is Kimi-K3 ``layers.0`` over 1,044,480 code tokens of a vLLM
trace. Every segment file (``long_context_golden.segment_path``) holds the segment's FP64 ``output`` rows and the
layer's full state after it; the manifest names them as products ``output`` (``row``) and ``state.<name>``
(``segment_end``: ``state.recurrent`` ``[1, heads, k, v]`` and ``state.{q,k,v}_convolution``).

Registered goldens are read through ``Fp64Golden.read_rows``. ``load_window`` and ``state_after`` read a golden
directory by path, also one still being written (the dev analysis reads goldens in progress).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import MANIFEST, Fp64Golden, open_golden
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    K3_LAYER0_GOLDEN,
    PRODUCER,
    PRODUCER_VERSION,
    segment_path,
)

STATE_PREFIX = "state."  # segment-end products are the state's fields, "state.<name>"


def _rows(run_dir: Path, segment: int, start: int, end: int) -> torch.Tensor:
    """Output rows ``[start, end)`` from the segment files (the caller checks the bounds)."""
    rows = []
    for index in range(start // segment, -(-end // segment)):
        payload = torch.load(segment_path(run_dir, index), weights_only=True, mmap=True)
        lo = payload["start"]
        rows.append(payload["output"][max(start, lo) - lo : min(end, payload["end"]) - lo])
    return torch.cat(rows)  # (the cat copies out of the mapped files)


def _state(run_dir: Path, index: int) -> dict[str, torch.Tensor]:
    return torch.load(segment_path(run_dir, index), weights_only=True)["state"]


class KimiK3KdaGolden(Fp64Golden):
    """A registered long KDA golden of ``long_context_golden`` version ``PRODUCER_VERSION``: products ``output``
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


def open_k3_layer0_golden(root: Path | None = None) -> KimiK3KdaGolden:
    """The registered Kimi-K3 ``layers.0`` golden, checked to be this producer's, on the first-layer input."""
    from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT

    golden = open_golden(K3_LAYER0_GOLDEN, root, reader=KimiK3KdaGolden)
    golden.require(
        identity={"model": "Kimi-K3", "layer": "model.layers.0", "input": FIRST_LAYER_INPUT},
        products=["output", "state.recurrent"],
    )
    return golden


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
