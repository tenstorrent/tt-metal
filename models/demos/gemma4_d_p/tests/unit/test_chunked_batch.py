# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedBatchPlan, ChunkedRequest
from models.demos.gemma4_d_p.tt.runners.chunked_batch_runtime import ChunkedBatchRuntime


def requests():
    return tuple(ChunkedRequest(i, i, i * 1024, tuple(range(i * 1024, (i + 1) * 1024))) for i in range(4))


def test_each_rank_owns_128_rows_of_every_request():
    plan = ChunkedBatchPlan()
    batch = requests()
    packed = torch.tensor(plan.pack(batch)).reshape(8, 4, 128)
    for rank in range(8):
        for lane in range(4):
            torch.testing.assert_close(
                packed[rank, lane], torch.arange(lane * 1024 + rank * 128, lane * 1024 + (rank + 1) * 128)
            )
    assert plan.unpack(packed.reshape(-1, 1)).squeeze(-1).tolist() == [list(r.token_ids) for r in batch]


@pytest.mark.parametrize("lengths", [(1, 31, 32, 33), (127, 128, 129, 1024)])
def test_partial_padding_and_absolute_positions(lengths):
    plan = ChunkedBatchPlan()
    batch = tuple(replace(req, token_ids=(17,) * n) for req, n in zip(requests(), lengths))
    tokens = plan.unpack(torch.tensor(plan.pack(batch)).reshape(-1, 1)).squeeze(-1)
    positions = plan.unpack(torch.tensor(plan.pack(batch, positions=True)).reshape(-1, 1)).squeeze(-1)
    for lane, req in enumerate(batch):
        n = len(req.token_ids)
        assert tokens[lane, :n].tolist() == [17] * n
        assert positions[lane, :n].tolist() == list(range(req.actual_start, req.actual_end))
        assert not tokens[lane, n:].count_nonzero() and not positions[lane, n:].count_nonzero()


@pytest.mark.parametrize(
    "change",
    [
        dict(slot_id=4),
        dict(slot_id=1),
        dict(request_id=1),
        dict(actual_start=1),
        dict(actual_start=-1024),
        dict(token_ids=()),
        dict(token_ids=(1,) * 1025),
        dict(token_ids=(-1,)),
        dict(actual_start=8192),
    ],
)
def test_invalid_batches(change, expect_error):
    batch = list(requests())
    batch[0] = replace(batch[0], **change)
    with expect_error(ValueError, "slot|request|Request|tokens|Token"):
        ChunkedBatchPlan().validate(batch, num_slots=4, max_seq_len=8192, vocab_size=10000)


def test_continuation_identity_and_final_chunks(expect_error):
    runtime = object.__new__(ChunkedBatchRuntime)
    runtime.plan = ChunkedBatchPlan()
    runtime.num_slots = 4
    runtime.model = SimpleNamespace(max_seq_len=16384, vocab_size=10000)
    runtime.slot_ends = {i: i * 1024 for i in range(4)}
    runtime.slot_owners = {i: i for i in range(4)}
    runtime.validate(requests())
    runtime.slot_ends[1] = 33
    with expect_error(ValueError, "continuation"):
        runtime.validate(requests())
    runtime.slot_ends[1] = 1024
    runtime.slot_owners[1] = 100
    with expect_error(ValueError, "continuation"):
        runtime.validate(requests())
    runtime.validate(tuple(replace(req, actual_start=0) for req in requests()))


def test_requires_four_requests(expect_error):
    with expect_error(ValueError, "exactly"):
        ChunkedBatchPlan().validate(requests()[:3], num_slots=4, max_seq_len=8192, vocab_size=10000)
