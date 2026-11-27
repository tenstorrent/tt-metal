# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``run_case`` with a fake DeviceBlock and a tiny golden: stage order (identity before device work), chunks in
order with the next chunk's golden read overlapping the device call, early stop on a chunk violation, run bars over
metric tiers, release on every path, and the progress lines. CPU only, no device."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, check
from models.demos.common.prefill.long_context_accuracy.block import (
    BlockOutputs,
    ContextBars,
    GoldenIdentity,
    LongContextCase,
    PreparedCase,
)
from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    ROOT_ENV,
    Fp64Golden,
    GoldenMismatch,
    GoldenNotFound,
    product_schema,
    write_manifest,
    write_token_ids,
)
from models.demos.common.prefill.long_context_accuracy.run import run_case
from models.demos.common.prefill.long_context_accuracy.windowed_errors import WindowedErrors, errors, nrmse, row_sums

ROWS, SEGMENT, CHUNK, HIDDEN = 10, 2, 4, 3  # chunks [0, 4) [4, 8) [8, 10): the last one partial
GOLDEN_ID = "tiny-block-fp64-10"


class TinyGolden(Fp64Golden):
    """Row r's ``input`` is ``[r + 1] * HIDDEN`` and its ``target`` twice that; the ``state`` after row e is
    ``[e]``. Served from memory; the manifest is on disk."""

    producer = "tests.tiny_block"
    producer_version = 1
    reads: list[tuple[int, int]] = []  # (rows read, in order: the test resets it)

    def _read_rows(self, names, start, end):
        TinyGolden.reads.append((start, end))
        x = torch.arange(start + 1, end + 1, dtype=torch.float64).unsqueeze(1).expand(-1, HIDDEN)
        return {"input": x.clone(), "target": 2 * x}

    def _segment_end(self, names, index):
        return {"state": torch.tensor([float(self.manifest["segments"][index]["end"])])}


def write_golden(root: Path) -> None:
    directory = root / GOLDEN_ID
    directory.mkdir(parents=True)
    write_manifest(
        directory,
        {
            "golden_id": GOLDEN_ID,
            "producer": TinyGolden.producer,
            "producer_version": TinyGolden.producer_version,
            "identity": {"model": "tiny", "token_source": "test", "weights_sha256": {"layers.0": "w0"}},
            "token_ids_sha256": write_token_ids(directory, torch.arange(ROWS)),
            "rows": ROWS,
            "segment": SEGMENT,
            "segments": [{"start": s, "end": min(s + SEGMENT, ROWS)} for s in range(0, ROWS, SEGMENT)],
            "products": {
                "input": product_schema("row", torch.zeros(1, HIDDEN, dtype=torch.float64)),
                "target": product_schema("row", torch.zeros(1, HIDDEN, dtype=torch.float64)),
                "state": product_schema("segment_end", torch.zeros(1)),
            },
        },
    )


@dataclass
class FakeBlock:
    """Doubles its input, times ``1 + error[lo]`` on the chunk starting at ``lo``; records every call. With
    ``wait_for_prefetch``, each call but the last waits until the next chunk's golden has been read."""

    error: dict[int, float] = field(default_factory=dict)
    fail_at: int | None = None
    wait_for_prefetch: threading.Event | None = None
    calls: list = field(default_factory=list)

    def build(self, mesh, context_len, chunk):
        self.calls.append(("build", context_len, chunk))

    def forward(self, x, lo, hi):
        self.calls.append(("forward", lo, hi))
        assert x.shape == (hi - lo, HIDDEN)
        if self.fail_at == lo:
            raise RuntimeError(f"device fault at row {lo}")
        if self.wait_for_prefetch is not None and hi < ROWS:
            self.calls.append(("prefetched", self.wait_for_prefetch.wait(timeout=10)))
            self.wait_for_prefetch.clear()
        return BlockOutputs(2 * x * (1 + self.error.get(lo, 0.0)), {"rows": torch.arange(lo, hi)})

    def carried_state(self):
        self.calls.append(("carried_state",))
        return {"state": torch.tensor([float(ROWS)])}

    def release(self):
        self.calls.append(("release",))


class FakeScorer:
    """Scores ``target`` per row; checks each chunk's NRMSE against ``chunk_nrmse`` bars; tier 0 is the run's NRMSE,
    tier 1 the per-chunk-window NRMSE, tier 2 the carried state's error."""

    golden_tensors = ("target",)

    def __init__(self):
        self.series = WindowedErrors()
        self.scored: list[int] = []

    def score_chunk(self, lo, outputs, rows):
        assert torch.equal(outputs.extras["rows"], torch.arange(lo, lo + rows["target"].shape[0]))
        self.series.append(lo, sums=row_sums(rows["target"], outputs.output))
        self.scored.append(lo)

    def check_chunk(self, lo, outputs, rows, bars):
        return check(
            [
                {
                    "start": lo,
                    "end": lo + len(outputs.output),
                    "chunk_nrmse": nrmse(row_sums(rows["target"], outputs.output)),
                }
            ],
            bars,
        )

    def score_run(self, carried_state, golden):
        state = golden.read_rows(["state"], 0, ROWS)["state"]
        return {
            0: [self.series.whole(lambda w: errors(w["sums"], "output_"))],
            1: self.series.windows(CHUNK, lambda w: {"window_nrmse": nrmse(w["sums"])}),
            2: [{"start": 0, "end": ROWS, "rows": ROWS, "state_error": float((carried_state["state"] - state).abs())}],
        }


@dataclass
class Mesh:
    shape: tuple = (1, 1)


def make_case(block: FakeBlock, bars: ContextBars = ContextBars(), weights: str = "w0", scorer=None, prefetched=None):
    scorer = scorer or FakeScorer()

    def block_input(lo, hi, rows):
        if prefetched is not None and lo > 0:
            prefetched.set()
        return rows["input"]

    def prepare(golden, checkpoint, context_len):
        block.calls.append(("prepare", context_len))
        return PreparedCase(
            identity=GoldenIdentity({"model": "tiny"}, {"layers.0": weights}),
            scorer=scorer,
            input_tensors=("input",),
            block_input=block_input,
            device_block=block,
        )

    return LongContextCase(
        id="tiny-block",
        golden_id=GOLDEN_ID,
        golden_reader=TinyGolden,
        placement=pytest.param((1, 1), {}),
        chunk=CHUNK,
        bars={ROWS: bars, 8: bars},
        checkpoint=lambda: Path("/nonexistent"),
        prepare=prepare,
    )


@pytest.fixture
def golden_root(tmp_path, monkeypatch):
    """The tiny golden in a registry root of its own (``ROOT_ENV``)."""
    write_golden(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    TinyGolden.reads = []
    return tmp_path


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    yield lines
    logger.remove(sink)


def test_run_scores_every_chunk_in_order_and_logs_progress(golden_root, log_lines) -> None:
    block = FakeBlock(error={4: 0.01})
    case = make_case(block, ContextBars(run=[AcceptanceBar("output_nrmse", "max", 0.01)]))
    result = run_case(case, ROWS, Mesh(), Path("/ckpt"))
    assert result.passed and result.stopped_at is None, result.violations
    assert block.calls == [
        ("prepare", ROWS),
        ("build", ROWS, CHUNK),
        ("forward", 0, 4),
        ("forward", 4, 8),
        ("forward", 8, 10),
        ("carried_state",),
        ("release",),
    ]
    assert TinyGolden.reads == [(0, 4), (4, 8), (8, 10)]  # (each chunk's rows read once; score_run reads none)
    tiers = result.metric_tiers
    # only rows [4, 8) are off, by 1%: 0.01 * sqrt((5² + 6² + 7² + 8²) / (1² + ... + 10²))
    assert tiers[0][0]["output_nrmse"] == pytest.approx(0.01 * (174 / 385) ** 0.5)
    assert [w["window_nrmse"] for w in tiers[1]] == [0.0, pytest.approx(0.01), 0.0]
    assert tiers[2][0]["state_error"] == 0.0
    stage_names = ["golden open", "weight load", "identity", "device build", "carried state", "score run"]
    assert set(result.seconds) == {*stage_names, "chunk_golden", "chunk_wait", "chunk_device", "chunk_score", "total"}
    heap = [line for line in log_lines if line.startswith("LC_HEAP")]
    assert heap == [
        "LC_HEAP tiny-block 10: glibc mallopt "
        "M_MMAP_THRESHOLD=2147483647 (set 1) M_TRIM_THRESHOLD=2147483647 (set 1) M_TOP_PAD=1073741824 (set 1)"
    ], heap
    assert log_lines.index(heap[0]) < min(i for i, line in enumerate(log_lines) if line.startswith("LC_STAGE"))
    stages = [line for line in log_lines if line.startswith("LC_STAGE")]
    assert [line.removeprefix("LC_STAGE tiny-block 10: ").split(" done in ")[0] for line in stages] == [
        text for name in stage_names for text in (f"{name} ...", name)
    ]
    assert all(re.fullmatch(r".* done in \d+\.\ds", line) for line in stages[1::2]), stages
    # the tail stages (carried state, score run) come after the last chunk and before the result line
    order = [line.split(" ")[0] for line in log_lines if line.startswith(("LC_", "LONG_CONTEXT_RESULT="))]
    assert order[-6:-1] == ["LC_CHUNK"] + ["LC_STAGE"] * 4 and order[-1].startswith("LONG_CONTEXT_RESULT="), order
    chunks = [line for line in log_lines if line.startswith("LC_CHUNK")]
    assert len(chunks) == 3
    for i, (lo, hi) in enumerate([(0, 4), (4, 8), (8, 10)]):
        assert chunks[i].startswith(f"LC_CHUNK tiny-block chunk {i + 1}/3 rows [{lo}, {hi}) elapsed ")
        for part in (" eta ", " golden ", " wait ", " device ", " score "):
            assert part in chunks[i], chunks[i]
    # no ETA from the first chunk alone (its one-time cost would dominate); later chunks give one in seconds
    assert " eta ? |" in chunks[0]
    assert all(re.search(r" eta \d+\.\ds \|", line) for line in chunks[1:]), chunks
    assert sum(line.startswith("LONG_CONTEXT_RESULT={") for line in log_lines) == 1


def test_a_chunk_violation_stops_the_run(golden_root, log_lines) -> None:
    block = FakeBlock(error={4: 0.5})
    scorer = FakeScorer()
    case = make_case(block, ContextBars(chunk=[AcceptanceBar("chunk_nrmse", "max", 0.1)]), scorer=scorer)
    result = run_case(case, ROWS, Mesh(), Path("/ckpt"))
    assert not result.passed and result.stopped_at == 4
    assert result.violations == ["rows [4, 8): chunk_nrmse 0.5 > 0.1 in [4, 8)"]
    assert [c for c in block.calls if c[0] in ("forward", "carried_state", "release")] == [
        ("forward", 0, 4),
        ("forward", 4, 8),
        ("release",),
    ]
    assert scorer.scored == [0, 4] and result.metric_tiers == {}
    assert sum(line.startswith("LC_CHUNK") for line in log_lines) == 2


def test_run_bars_report_every_violation(golden_root) -> None:
    block = FakeBlock(error={0: 0.02, 8: 0.03})
    bars = ContextBars(run=[AcceptanceBar("window_nrmse", "max", 0.01), AcceptanceBar("missing_metric", "max", 1)])
    result = run_case(make_case(block, bars), ROWS, Mesh(), Path("/ckpt"))
    assert result.stopped_at is None and len([c for c in block.calls if c[0] == "forward"]) == 3
    assert result.violations == [
        "window_nrmse 0.02 > 0.01 in [0, 4)",
        "window_nrmse 0.03 > 0.01 in [8, 10)",
        "missing_metric: not measured by any window",
    ]


def test_golden_identity_is_checked_before_device_work(golden_root, monkeypatch, expect_error) -> None:
    block = FakeBlock()
    with expect_error(GoldenMismatch, "weights layers.0"):
        run_case(make_case(block, weights="other"), ROWS, Mesh(), Path("/ckpt"))
    assert block.calls == [("prepare", ROWS)] and TinyGolden.reads == []
    with expect_error(GoldenNotFound, "no golden"):
        monkeypatch.setenv(ROOT_ENV, str(golden_root / "elsewhere"))
        run_case(make_case(block), ROWS, Mesh(), Path("/ckpt"))
    monkeypatch.setenv(ROOT_ENV, str(golden_root))
    with expect_error(ValueError, "no bars for context length 6"):
        run_case(make_case(block), 6, Mesh(), Path("/ckpt"))
    with expect_error(ValueError, "mesh"):
        run_case(make_case(block), ROWS, Mesh((2, 4)), Path("/ckpt"))
    assert block.calls == [("prepare", ROWS)]


def test_next_chunk_golden_is_read_while_the_device_runs(golden_root) -> None:
    """Each device call but the last waits for the prefetch thread to build the next chunk's input: a serial loop
    would never get there and the wait would time out."""
    prefetched = threading.Event()
    block = FakeBlock(wait_for_prefetch=prefetched)
    result = run_case(make_case(block, prefetched=prefetched), ROWS, Mesh(), Path("/ckpt"))
    assert result.passed
    assert [c for c in block.calls if c[0] == "prefetched"] == [("prefetched", True), ("prefetched", True)]


def test_the_block_is_released_when_the_device_fails(golden_root, expect_error) -> None:
    block = FakeBlock(fail_at=4)
    with expect_error(RuntimeError, "device fault at row 4"):
        run_case(make_case(block), ROWS, Mesh(), Path("/ckpt"))
    assert block.calls[-1] == ("release",) and ("carried_state",) not in block.calls


def test_metric_tiers_need_spans_and_one_tier_per_metric(golden_root, expect_error) -> None:
    span = {"start": 0, "end": 8, "rows": 8}

    class Tiers(FakeScorer):
        def __init__(self, tiers):
            super().__init__()
            self.tiers = tiers

        def score_run(self, carried_state, golden):
            return self.tiers

    with expect_error(ValueError, "metric 'nrmse' is in metric tiers 0 and 1"):
        run_case(
            make_case(FakeBlock(), scorer=Tiers({0: [{**span, "nrmse": 0.0}], 1: [{**span, "nrmse": 0.0}]})),
            8,
            Mesh(),
            Path("/ckpt"),
        )
    with expect_error(ValueError, "metric tier 1 has measurements without"):
        run_case(
            make_case(FakeBlock(), scorer=Tiers({0: [{**span, "a": 0.0}], 1: [{"start": 0, "b": 0.0}]})),
            8,
            Mesh(),
            Path("/ckpt"),
        )
