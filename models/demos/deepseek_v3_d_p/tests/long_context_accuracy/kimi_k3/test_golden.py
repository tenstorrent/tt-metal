# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the long KDA golden reader (``KimiK3KdaGolden``) on a small registered golden: output rows across
segments, the state after a segment end, refusal of a state between segment ends, of products it does not hold and of
another producer version; the path readers agree with it. CPU only."""

from __future__ import annotations

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    GoldenMismatch,
    open_golden,
    product_schema,
    write_manifest,
    write_token_ids,
)
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden import (
    KimiK3KdaGolden,
    load_window,
    state_after,
)
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer import (
    PRODUCER,
    PRODUCER_VERSION,
    segment_path,
)

ROWS, SEGMENT = 10, 4  # segments [0, 4) [4, 8) [8, 10)
GOLDEN_ID = "k3-test-kda"


def _write_golden(root, producer_version: int = PRODUCER_VERSION):
    """A registered KDA golden: row r's ``output`` is ``[r, -r]``; the state after the segment ending at row e has
    ``recurrent`` filled with e and the q, k, v convolutions with -e."""
    directory = root / GOLDEN_ID
    directory.mkdir(parents=True)
    segments = []
    for i, start in enumerate(range(0, ROWS, SEGMENT)):
        end = min(start + SEGMENT, ROWS)
        r = torch.arange(start, end, dtype=torch.float64)
        state = {
            "recurrent": torch.full((1, 2, 2, 2), float(end)),
            **{f"{c}_convolution": torch.full((1, 3, 4), -end) for c in "qkv"},
        }
        payload = {"start": start, "end": end, "output": torch.stack([r, -r], 1), "state": state}
        torch.save(payload, segment_path(directory, i))
        segments.append({"start": start, "end": end})
    write_manifest(
        directory,
        {
            "golden_id": GOLDEN_ID,
            "producer": PRODUCER,
            "producer_version": producer_version,
            "identity": {"model": "test", "token_source": "test", "weights_sha256": "w"},
            "token_ids_sha256": write_token_ids(directory, torch.arange(ROWS)),
            "rows": ROWS,
            "segment": SEGMENT,
            "segments": segments,
            "products": {
                "output": product_schema("row", payload["output"]),
                **{f"state.{n}": product_schema("segment_end", t) for n, t in state.items()},
            },
        },
    )
    return directory


def test_read_rows_spans_segments_and_reads_the_state_at_a_segment_end(tmp_path) -> None:
    directory = _write_golden(tmp_path)
    golden = open_golden(GOLDEN_ID, tmp_path, reader=KimiK3KdaGolden)
    got = golden.read_rows(["output", "state.recurrent", "state.q_convolution"], 3, 8)
    assert torch.equal(got["output"][:, 0], torch.arange(3, 8, dtype=torch.float64))
    assert torch.equal(got["output"][:, 1], -torch.arange(3, 8, dtype=torch.float64))
    assert torch.equal(got["state.recurrent"], torch.full((1, 2, 2, 2), 8.0))  # (the state after row 8)
    assert torch.equal(got["state.q_convolution"], torch.full((1, 3, 4), -8))
    last = golden.read_rows(["output", "state.recurrent"], 8, 10)
    assert last["output"][:, 0].tolist() == [8.0, 9.0] and float(last["state.recurrent"].max()) == 10.0
    # The path readers serve the same rows and states.
    rows, before = load_window(directory, 4, 10)
    assert torch.equal(rows, golden.read_rows(["output"], 4, 10)["output"])
    assert torch.equal(before.recurrent, torch.full((1, 2, 2, 2), 4.0))
    assert torch.equal(state_after(directory, 8).recurrent, got["state.recurrent"])


def test_read_rows_refuses_what_the_golden_does_not_hold(tmp_path, expect_error) -> None:
    _write_golden(tmp_path)
    golden = open_golden(GOLDEN_ID, tmp_path, reader=KimiK3KdaGolden)
    with expect_error(ValueError, "does not end a segment"):
        golden.read_rows(["output", "state.recurrent"], 0, 6)
    with expect_error(ValueError, "has no products"):
        golden.read_rows(["state.conv"], 0, 4)
    with expect_error(ValueError, "not inside"):
        golden.read_rows(["output"], 8, 11)


def test_open_refuses_another_producer_version(tmp_path, expect_error) -> None:
    _write_golden(tmp_path, producer_version=PRODUCER_VERSION + 1)
    with expect_error(GoldenMismatch, "producer_version"):
        open_golden(GOLDEN_ID, tmp_path, reader=KimiK3KdaGolden)
