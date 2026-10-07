# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode selection and scratch lifetime; numerical validation requires hardware."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt import decoder
from models.demos.qwen38_27b_qb2.tt.gdn_step.workspace import DecodeWorkspace
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def test_workspace_keeps_existing_trace_buffers_across_batch_changes(monkeypatch, expect_error):
    allocations = []

    def allocate(shape, *args):
        result = SimpleNamespace(shape=tuple(shape), serial=len(allocations))
        allocations.append(result)
        return result

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    workspace = DecodeWorkspace(object(), 12)
    workspace.prepare(16)
    original = dict(workspace.outputs)
    assert set(original) == {1, 8, 16}
    workspace.prepare(8)
    workspace.prepare(32)
    workspace.prepare(16)
    assert len(allocations) == 4
    assert workspace.output(32).shape == (384, 128)
    assert all(workspace.output(batch) is tensor for batch, tensor in original.items())
    with expect_error(RuntimeError, "cache allocation"):
        workspace.output(4)
    assert len(allocations) == 4, "Decode lookup must not allocate while tracing"
    other = DecodeWorkspace(object(), 12)
    other.prepare(16)
    assert all(other.output(batch) is not tensor for batch, tensor in original.items())


@pytest.mark.parametrize("batch", [0, 65, True, 1.5])
def test_workspace_rejects_unqualified_batches(batch, expect_error):
    with expect_error(ValueError, "batches"):
        DecodeWorkspace(object(), 12).prepare(batch)


def test_single_token_prefill_keeps_chunked_scan_and_decode_uses_in_place_step(monkeypatch):
    calls = []
    scratch = object()

    def single_step(q, k, v, g, beta, state, output):
        assert output is scratch
        state.add_(2)
        calls.append("single_step")
        return torch.zeros(q.shape[0] * 12, 32, 128)

    def scan(q, k, v, g, beta, **kwargs):
        calls.append(("scan", q.shape[0]))
        return torch.zeros(q.shape[0] * 12, 32, 128), kwargs["initial_state"] + 1

    ops = SimpleNamespace(
        transformer=SimpleNamespace(chunk_gated_delta_rule=scan),
        concat=lambda tensors, dim: torch.cat(tensors, dim=dim),
        copy=lambda source, target: target.copy_(source),
    )
    monkeypatch.setattr(decoder, "ttnn", ops)
    monkeypatch.setattr(decoder, "step_from_flat", single_step)
    layer = SimpleNamespace(
        config=SimpleNamespace(linear_num_value_heads=12),
        policy={"decode_recurrence": "single_step"},
        device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=12, y=10)),
        delta_constants={},
        gdn_decode_workspace=SimpleNamespace(output=lambda batch: scratch),
    )
    inputs = [torch.zeros(16, 32, width) for width in (512, 512, 1536, 12, 12)]
    state = SimpleNamespace(recurrent=torch.zeros(16, 12, 128, 128))
    # Even a single live prefill token uses the padded 32-row chunked path.
    decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, state)
    assert calls == [("scan", 10), ("scan", 6)]
    assert torch.all(state.recurrent == 1)
    decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, state, decode=True)
    assert calls[-1] == "single_step"
    assert torch.all(state.recurrent == 3)
    layer.policy["decode_recurrence"] = "native"
    decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, state, decode=True)
    assert calls[-2:] == [("scan", 10), ("scan", 6)]
    assert torch.all(state.recurrent == 4)


def test_recurrence_policy_is_explicit_and_backward_compatible(expect_error):
    policy = load_precision("baseline")
    del policy["decode_recurrence"]
    assert load_precision(policy)["decode_recurrence"] == "native"
    candidate = load_precision(dict(policy, decode_recurrence="single_step"))
    assert decoder_policy(candidate, 0)["decode_recurrence"] == "single_step"
    with expect_error(ValueError, "Unsupported decode recurrence"):
        load_precision(dict(policy, decode_recurrence="unknown"))
