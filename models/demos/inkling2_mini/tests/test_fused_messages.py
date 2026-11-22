# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""E10d FusedMessageSplit: exact row copies + dense twins of the 48 loop messages, vs the selector-matmul path of
model._messages (tf32-rounded copies, PCC), traced twice bit-equal, and timed against 24 selector matmuls."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.messages import FusedMessageSplit

C = 2304
RESULTS = os.environ.get("INKLING_E10_RESULTS")


def host(t):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])


def traced_us(device, fn, n=100):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / n * 1e6
    return tid, out, us


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 * 1024 * 1024}], indirect=True)
def test_message_split(device):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(48, C, generator=g)
    msgs = ttnn.from_torch(x[None, None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    split = FusedMessageSplit(device)
    rows = R.RowOps(device)
    sels = [
        (rows.selector(3, 48, [(j, 4 * l + j) for j in range(3)]), rows.selector(1, 48, [(0, 4 * l + 3)]))
        for l in range(12)
    ]

    tid, out, us_new = traced_us(device, lambda: split(msgs))
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    r1 = [host(t) for pair in out for t in pair]
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    r2 = [host(t) for pair in out for t in pair]
    ttnn.release_trace(device, tid)
    assert all(torch.equal(a, b) for a, b in zip(r1, r2)), "trace replays differ"
    for l, (q, m) in enumerate(out):
        assert torch.equal(host(q)[:3], x[4 * l : 4 * l + 3]), l
        assert torch.equal(host(m)[:1], x[4 * l + 3 : 4 * l + 4]), l
        for t, n in ((q, 3), (m, 1)):
            dense, nvec = F.twin(t)
            assert nvec == n
            got = F.from_dense_matrix(ttnn.to_torch(dense).float(), C, n)
            assert torch.equal(got, host(t)[:n]), l

    def old():
        return [(rows.mm(a, msgs), rows.mm(b, msgs)) for a, b in sels]

    tid, ref, us_old = traced_us(device, old)
    ttnn.release_trace(device, tid)
    for (q, m), (rq, rm) in zip(out, ref):
        for a, b in ((host(q)[:3], host(rq)[:3]), (host(m)[:1], host(rm)[:1])):
            p = torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
            assert p > 0.99999, p
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(name="message_split", us_split=us_new, us_24_matmuls=us_old)) + "\n")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 * 1024 * 1024}], indirect=True)
def test_all_messages_cost(device):
    """E10d: device time of model._messages' input side today (2 seals + attend over them, E2 RowOps ops) plus
    the split, per loop pass. Timing only."""
    g = torch.Generator().manual_seed(6)
    rows = R.RowOps(device)
    w = {"msg_q": torch.randn(1, 1, 12 * 32, C, generator=g)}
    msg = R.MessageAttnRes(rows, w, "msg_q")
    split = FusedMessageSplit(device)

    def row(t, dt):
        return ttnn.from_torch(t[None, None], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    h12, h17 = row(torch.randn(1, C, generator=g), ttnn.bfloat16), row(torch.randn(1, C, generator=g), ttnn.bfloat16)
    tid, _, us_attend = traced_us(device, lambda: msg.all_messages(h12, h17))
    ttnn.release_trace(device, tid)
    tid, _, us_all = traced_us(device, lambda: split(msg.all_messages(h12, h17)))
    ttnn.release_trace(device, tid)
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(name="all_messages", us_attend=us_attend, us_attend_split=us_all)) + "\n")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 * 1024 * 1024}], indirect=True)
def test_fused_messages(device):
    """E10d FusedMessages (seal + one 12-core attend, straight into the per-layer message tensors) vs the model's
    path (MessageAttnRes.all_messages, E2 ops) + split: PCC per message, traced twice bit-equal, timed."""
    from models.demos.inkling2_mini.tt.fused.messages import FusedMessages

    g = torch.Generator().manual_seed(9)
    rows = R.RowOps(device)
    w = {"msg_q": torch.randn(1, 1, 12 * 32, C, generator=g)}
    msg, split, fused = R.MessageAttnRes(rows, w, "msg_q"), FusedMessageSplit(device), FusedMessages(device, w)

    def row(t, dt):
        return ttnn.from_torch(t[None, None], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    h12 = row(torch.randn(1, C, generator=g) * 2, ttnn.bfloat16)
    h17 = row(torch.randn(1, C, generator=g), ttnn.bfloat16)
    ref = [(host(q)[:3], host(m)[:1]) for q, m in split(msg.all_messages(h12, h17))]
    tid, out, us = traced_us(device, lambda: fused(h12, h17))
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    r1 = [host(t) for pair in out for t in pair]
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    r2 = [host(t) for pair in out for t in pair]
    ttnn.release_trace(device, tid)
    assert all(torch.equal(a, b) for a, b in zip(r1, r2)), "trace replays differ"
    worst = 1.0
    for l, (q, m) in enumerate(out):
        for got, want, n in ((host(q)[:3], ref[l][0], 3), (host(m)[:1], ref[l][1], 1)):
            for i in range(n):
                p = torch.corrcoef(torch.stack([got[i].double(), want[i].double()]))[0, 1].item()
                assert p == p, (l, i)  # finite
                worst = min(worst, p)
        for t, n in ((q, 3), (m, 1)):
            dense, nvec = F.twin(t)
            assert nvec == n and torch.equal(F.from_dense_matrix(ttnn.to_torch(dense).float(), C, n), host(t)[:n])
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(name="fused_messages", us=us, pcc_min=worst)) + "\n")
    assert worst >= 0.9999, worst
