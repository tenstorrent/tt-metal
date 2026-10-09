# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedBatchPlan, ChunkedRequest
from models.demos.gemma4_d_p.tt.runners.chunked_batch_runtime import ChunkedBatchRuntime


@pytest.fixture(params=[(2, 4096), (4, 1024)], ids=["2x4k", "4x1k"])
def plan(request):
    return ChunkedBatchPlan(batch_size=request.param[0], chunk_size=request.param[1])


def requests(plan):
    return tuple(
        ChunkedRequest(i, i, i * plan.chunk_size, tuple(range(i * plan.chunk_size, (i + 1) * plan.chunk_size)))
        for i in range(plan.batch_size)
    )


def test_each_rank_owns_local_rows_of_every_request(plan):
    batch = requests(plan)
    packed = torch.tensor(plan.pack(batch)).reshape(plan.cp, plan.batch_size, plan.local_rows)
    for rank in range(plan.cp):
        for lane in range(plan.batch_size):
            torch.testing.assert_close(
                packed[rank, lane],
                torch.arange(
                    lane * plan.chunk_size + rank * plan.local_rows,
                    lane * plan.chunk_size + (rank + 1) * plan.local_rows,
                ),
            )
    assert plan.unpack(packed.reshape(-1, 1)).squeeze(-1).tolist() == [list(r.token_ids) for r in batch]


@pytest.mark.parametrize("lengths", [(1, 31), (32, 33), (511, 512), (513, 4096)])
def test_partial_padding_and_absolute_positions(plan, lengths):
    lengths = [min(n, plan.chunk_size) for n in lengths] * (plan.batch_size // 2)
    batch = tuple(replace(req, token_ids=(17,) * n) for req, n in zip(requests(plan), lengths))
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
        dict(slot_id=0.5),
        dict(slot_id=1),
        dict(request_id=1),
        dict(actual_start=1),
        dict(actual_start=0.0),
        dict(actual_start=-1024),
        dict(token_ids=()),
        "oversize",
        dict(token_ids=(-1,)),
        dict(token_ids=(1.5,)),
        dict(actual_start=32768),
    ],
)
def test_invalid_batches(plan, change, expect_error):
    if change == "oversize":
        change = dict(token_ids=(1,) * (plan.chunk_size + 1))
    batch = list(requests(plan))
    batch[0] = replace(batch[0], **change)
    with expect_error(ValueError, "slot|request|Request|tokens|Token"):
        plan.validate(batch, num_slots=plan.batch_size, max_seq_len=32768, vocab_size=100000)


def test_continuation_identity_and_final_chunks(plan, expect_error):
    runtime = object.__new__(ChunkedBatchRuntime)
    runtime.plan = plan
    runtime.num_slots = plan.batch_size
    runtime.model = SimpleNamespace(max_seq_len=16384, vocab_size=10000)
    runtime.slot_ends = {i: i * plan.chunk_size for i in range(plan.batch_size)}
    runtime.slot_owners = {i: i for i in range(plan.batch_size)}
    runtime.validate(requests(plan))
    runtime.slot_ends[1] = 33
    with expect_error(ValueError, "continuation"):
        runtime.validate(requests(plan))
    runtime.slot_ends[1] = plan.chunk_size
    runtime.slot_owners[1] = 100
    with expect_error(ValueError, "continuation"):
        runtime.validate(requests(plan))
    runtime.validate(tuple(replace(req, actual_start=0) for req in requests(plan)))


def test_requires_full_batch(plan, expect_error):
    with expect_error(ValueError, "exactly"):
        plan.validate(requests(plan)[:-1], num_slots=plan.batch_size, max_seq_len=32768, vocab_size=100000)
