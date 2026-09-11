# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host checks for native GDN batch capacity and slot ordering."""

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.prefill_recurrence import NativeGatedDeltaRule


@pytest.mark.parametrize("batch,expected_batches", [(1, [1]), (9, [9]), (10, [9, 1]), (32, [9, 9, 9, 5])])
def test_native_batch_partition_preserves_slot_and_head_order(monkeypatch, batch, expected_batches):
    recurrence = object.__new__(NativeGatedDeltaRule)
    recurrence.compute_cores = 110
    calls = []
    tensors = [torch.arange(batch).reshape(batch, 1, 1).expand(batch, 2, 12) for _ in range(5)]
    state = torch.arange(batch).reshape(batch, 1, 1, 1).expand(batch, 12, 2, 2)

    def forward(q, k, v, g, beta, initial, scale):
        calls.append(q.shape[0])
        slots = initial[:, 0, 0, 0]
        for tensor in (q, k, v, g, beta):
            assert torch.equal(tensor[:, 0, 0], slots)
        heads = slots[:, None] * 12 + torch.arange(12)
        return heads.reshape(-1, 1, 1), initial + 100

    recurrence._forward = forward
    monkeypatch.setattr(ttnn, "concat", lambda values, dim: torch.cat(values, dim=dim))
    output, final = recurrence._forward_batches(*tensors, state, 1.0)
    assert calls == expected_batches
    assert torch.equal(output.flatten(), torch.arange(batch * 12))
    assert torch.equal(final, state + 100)


def test_single_batch_forwards_original_tensor_objects():
    recurrence = object.__new__(NativeGatedDeltaRule)
    recurrence.compute_cores = 110
    tensors = [torch.zeros(1, 2, 12) for _ in range(6)]
    sentinel = object()

    def forward(*args):
        assert all(actual is original for actual, original in zip(args[:-1], tensors))
        return sentinel

    recurrence._forward = forward
    assert recurrence._forward_batches(*tensors, 1.0) is sentinel
