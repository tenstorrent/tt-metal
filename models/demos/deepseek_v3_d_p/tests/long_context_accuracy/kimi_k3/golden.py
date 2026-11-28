# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader of the long FP64 KDA goldens (``KimiK3KdaGolden``), written by ``golden_producer.py`` next to it.

The golden ``K3_LAYER0_GOLDEN`` (``kimi_k3_code_debug_1m_cpu_fp64_l0_kda``) is Kimi-K3 ``layers.0`` over the first
1,044,480 tokens of the vLLM trace ``k3_vllm_code_debug_1M``, computed in FP64 and stored FP32. Every segment file
(``golden_producer.segment_path``) holds the segment's ``output`` rows and the layer's full state after it; the manifest
names them as products ``output`` (``row``) and ``state.<name>`` (``segment_end``: ``state.recurrent`` ``[1, heads,
k, v]`` and ``state.{q,k,v}_convolution``). Products are returned in their stored dtype; the scorers widen them to FP64.

Registered goldens are read through ``Fp64Golden.read_rows``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden, MappedFile, read_slices
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
