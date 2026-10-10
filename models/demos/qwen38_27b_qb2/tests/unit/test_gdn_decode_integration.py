# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode selection and scratch lifetime; numerical validation requires hardware."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt import decoder
from models.demos.qwen38_27b_qb2.tt.gdn_step.workspace import (
    COMBINED_GDN_POLICY,
    COMPACT_GDN_POLICIES,
    CompactScratch,
    DecodeWorkspace,
)
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


def test_compact_workspace_keeps_owned_l1_buffers_across_batch_switches(monkeypatch, expect_error):
    allocations = []

    def allocate(shape, dtype, layout, mesh, memory):
        tensor = SimpleNamespace(shape=tuple(shape), serial=len(allocations), memory=memory)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    workspace = DecodeWorkspace(
        object(), 12, shared_qk_heads=4, fused_epilogue=True, flat_prepare=True, compact_frontend=True
    )
    workspace.prepare(16)
    buffers = workspace.compact_buffers(16)
    assert [t.shape for t in buffers] == [(1, 16, w) for w in (512, 512, 1536, 1536)]
    assert all(t.memory == ttnn.L1_MEMORY_CONFIG for t in buffers)
    for batch in (32, 8, 16):
        workspace.prepare(batch)
    assert workspace.compact_buffers(16) is buffers
    assert set(workspace.compact_outputs) == {16, 32}
    count = len(allocations)
    with expect_error(RuntimeError, "cache allocation"):
        workspace.compact_buffers(8)
    assert len(allocations) == count
    with expect_error(ValueError, "Compact GDN needs"):
        DecodeWorkspace(object(), 12, compact_frontend=True)


def test_shared_workspace_is_persistent_disjoint_and_skips_small_batches(monkeypatch, expect_error):
    allocations = []

    def allocate(shape, *args):
        result = SimpleNamespace(shape=tuple(shape), serial=len(allocations))
        allocations.append(result)
        return result

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    workspace = DecodeWorkspace(object(), 12, shared_qk_heads=4)
    workspace.prepare(16)
    original = dict(workspace.shared_outputs)
    assert set(original) == {8, 16}
    assert len(allocations) == 7
    assert all(tensor.shape == (batch * 4, 128) for batch, pair in original.items() for tensor in pair)
    assert all(workspace.shared_qk(batch) is None for batch in (1, 2, 4))
    with expect_error(RuntimeError, "cache allocation"):
        workspace.shared_qk(32)
    assert len(allocations) == 7
    workspace.prepare(32)
    workspace.prepare(16)
    assert len(allocations) == 10
    assert len({id(tensor) for tensor in allocations}) == 10
    assert all(workspace.shared_qk(batch) is pair for batch, pair in original.items())
    assert DecodeWorkspace(object(), 12).shared_qk(16) is None


def test_compact_l1_pool_is_shared_only_within_one_serialized_stack(monkeypatch, expect_error):
    allocated = []

    def allocate(shape, dtype, layout, mesh, memory):
        tensor = SimpleNamespace(shape=tuple(shape), serial=len(allocated), mesh=mesh, memory=memory)
        allocated.append(tensor)
        return tensor

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    mesh = object()
    pool = CompactScratch(mesh)
    options = dict(shared_qk_heads=4, fused_epilogue=True, flat_prepare=True, compact_frontend=True)
    layers = [DecodeWorkspace(mesh, 12, compact_pool=pool, **options) for _ in range(48)]
    for layer in layers:
        layer.prepare(16)
    original = layers[0].compact_buffers(16)
    assert all(layer.compact_buffers(16) is original for layer in layers)
    assert len([t for t in allocated if t.memory == ttnn.L1_MEMORY_CONFIG]) == 4
    assert len({id(layer.output(16)) for layer in layers}) == 48
    assert len({id(layer.flat_outputs(16)[0]) for layer in layers}) == 48
    for layer in layers:
        layer.prepare(32)
    assert len([t for t in allocated if t.memory == ttnn.L1_MEMORY_CONFIG]) == 8
    assert all(layer.compact_buffers(16) is original for layer in layers)
    independent = DecodeWorkspace(mesh, 12, compact_pool=CompactScratch(mesh), **options)
    independent.prepare(16)
    assert all(a is not b for a, b in zip(original, independent.compact_buffers(16)))
    with expect_error(ValueError, "same mesh"):
        DecodeWorkspace(object(), 12, compact_pool=pool, **options)
    with expect_error(ValueError, "compact policy"):
        DecodeWorkspace(mesh, 12, compact_pool=pool)


@pytest.mark.parametrize(
    "recurrence",
    ["single_step", "single_step_shared_qk_epilogue", "single_step_flat_prepare_epilogue", *COMPACT_GDN_POLICIES],
)
def test_single_token_prefill_keeps_chunked_scan_and_decode_uses_in_place_step(monkeypatch, recurrence):
    calls = []
    scratch = object()

    def single_step(q, k, v, g, beta, state, output, **options):
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
        policy={"decode_recurrence": recurrence},
        device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=12, y=10)),
        delta_constants={},
        gdn_decode_workspace=SimpleNamespace(
            output=lambda batch: scratch, shared_qk=lambda batch: None, flat_outputs=lambda batch: None
        ),
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
    shared = load_precision(dict(policy, decode_recurrence="single_step_shared_qk"))
    assert decoder_policy(shared, 0)["decode_recurrence"] == "single_step_shared_qk"
    fused = load_precision(dict(policy, decode_recurrence="single_step_shared_qk_epilogue"))
    assert decoder_policy(fused, 0)["decode_recurrence"] == "single_step_shared_qk_epilogue"
    compact = load_precision(dict(policy, decode_recurrence="single_step_compact_gdn"))
    assert decoder_policy(compact, 0)["decode_recurrence"] == "single_step_compact_gdn"
    with expect_error(ValueError, "Unsupported decode recurrence"):
        load_precision(dict(policy, decode_recurrence="unknown"))


@pytest.mark.parametrize(
    "batch,decode,selected", [(16, True, True), (32, True, True), (8, True, False), (16, False, False)]
)
@pytest.mark.parametrize("recurrence", COMPACT_GDN_POLICIES)
def test_compact_gdn_only_selected_for_explicit_decode_buckets(batch, decode, selected, recurrence):
    class OrdinaryPath(Exception):
        pass

    def ordinary(*args, **kwargs):
        raise OrdinaryPath()

    layer = SimpleNamespace(
        policy={"decode_recurrence": recurrence},
        config=SimpleNamespace(linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=128),
        _delta_compact=lambda x, state, b: (state, b),
        _linear=ordinary,
    )
    x = SimpleNamespace(shape=(1, 1, batch, 5120))
    state = object()
    try:
        result = decoder.Qwen38Decoder._delta(layer, x, state, decode=decode)
    except OrdinaryPath:
        assert not selected
    else:
        assert selected and result == (state, batch)


@pytest.mark.parametrize("batch", [1, 16, 32])
def test_shared_policy_passes_only_preallocated_scratch_to_decode(monkeypatch, batch):
    output, pair = object(), (object(), object())
    calls = []

    def single_step(*args, shared_qk_outputs):
        assert args[-1] is output
        assert shared_qk_outputs is (None if batch == 1 else pair)
        calls.append(batch)
        return output

    monkeypatch.setattr(decoder, "step_from_flat", single_step)
    layer = SimpleNamespace(
        config=SimpleNamespace(linear_num_value_heads=12),
        policy={"decode_recurrence": "single_step_shared_qk"},
        gdn_decode_workspace=SimpleNamespace(
            output=lambda b: output,
            shared_qk=lambda b: None if b == 1 else pair,
        ),
    )
    inputs = [torch.zeros(batch, 32, width) for width in (512, 512, 1536, 12, 12)]
    state = SimpleNamespace(recurrent=object())
    assert decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, state, decode=True) is output
    assert calls == [batch]


def test_epilogue_workspace_preserves_trace_addresses_and_small_batch_fallback(monkeypatch, expect_error):
    allocations = []

    def allocate(shape, *args):
        result = SimpleNamespace(shape=tuple(shape), serial=len(allocations))
        allocations.append(result)
        return result

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    workspace = DecodeWorkspace(object(), 12, shared_qk_heads=4, fused_epilogue=True)
    workspace.prepare(16)
    original = workspace.epilogue_output(16)
    assert original.shape == (16, 1, 1536)
    workspace.prepare(32)
    workspace.prepare(8)
    workspace.prepare(16)
    assert workspace.epilogue_output(16) is original
    assert workspace.epilogue_output(32).shape == (32, 1, 1536)
    count = len(allocations)
    for batch in (1, 8, 64):
        with expect_error(RuntimeError, "cache allocation"):
            workspace.epilogue_output(batch)
    assert len(allocations) == count
    assert len({id(tensor) for tensor in allocations}) == count


@pytest.mark.parametrize("batch", [1, 8, 16, 32, 64])
def test_epilogue_policy_only_skips_output_layout_for_qualified_batches(monkeypatch, batch):
    scratch, pair = object(), (object(), object())
    seen = []

    def step(*args, **kwargs):
        seen.append(kwargs)
        return args[-1]

    monkeypatch.setattr(decoder, "step_from_flat", step)
    layer = SimpleNamespace(
        config=SimpleNamespace(linear_num_value_heads=12),
        policy={"decode_recurrence": "single_step_shared_qk_epilogue"},
        gdn_decode_workspace=SimpleNamespace(output=lambda b: scratch, shared_qk=lambda b: pair),
    )
    inputs = [torch.zeros(batch, 32, width) for width in (512, 512, 1536, 12, 12)]
    state = SimpleNamespace(recurrent=object())
    assert decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, state, decode=True) is scratch
    assert seen == [dict(shared_qk_outputs=pair, **({"raw_output": True} if batch in (16, 32) else {}))]


def test_flat_prepare_scratch_survives_bucket_changes_without_allocating_during_lookup(monkeypatch, expect_error):
    allocations = []

    def allocate(shape, *args):
        tensor = SimpleNamespace(shape=tuple(shape), serial=len(allocations))
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    workspace = DecodeWorkspace(object(), 12, shared_qk_heads=4, fused_epilogue=True, flat_prepare=True)
    workspace.prepare(16)
    original = workspace.flat_outputs(16)
    assert [t.shape for t in original] == [(192, 128), (192, 8)]
    workspace.prepare(32)
    workspace.prepare(8)
    workspace.prepare(16)
    assert workspace.flat_outputs(16) is original
    assert [t.shape for t in workspace.flat_outputs(32)] == [(384, 128), (384, 8)]
    assert all(workspace.flat_outputs(b) is None for b in (1, 8, 64))
    count = len(allocations)
    for _ in range(3):
        workspace.flat_outputs(16)
        workspace.flat_outputs(32)
    assert len(allocations) == count == len({id(t) for t in allocations})
    other = DecodeWorkspace(object(), 12, flat_prepare=True)
    with expect_error(RuntimeError, "cache allocation"):
        other.flat_outputs(16)


@pytest.mark.parametrize("batch", [1, 8, 16, 32])
def test_flat_prepare_policy_selects_preallocated_buffers_and_retains_fallback(monkeypatch, batch):
    scratch, qk, flat = object(), (object(), object()), (object(), object())
    seen = []

    def step(*args, **options):
        seen.append(options)
        return args[-1]

    monkeypatch.setattr(decoder, "step_from_flat", step)
    layer = SimpleNamespace(
        config=SimpleNamespace(linear_num_value_heads=12),
        policy={"decode_recurrence": "single_step_flat_prepare_epilogue"},
        gdn_decode_workspace=SimpleNamespace(
            output=lambda b: scratch,
            shared_qk=lambda b: qk,
            flat_outputs=lambda b: flat if b in (16, 32) else None,
        ),
    )
    inputs = [torch.zeros(batch, 32, width) for width in (512, 512, 1536, 12, 12)]
    assert (
        decoder.Qwen38Decoder._delta_recurrence(layer, *inputs, SimpleNamespace(recurrent=object()), decode=True)
        is scratch
    )
    assert seen == [
        dict(
            shared_qk_outputs=qk,
            flat_prepare_outputs=flat if batch in (16, 32) else None,
            **({"raw_output": True} if batch in (16, 32) else {}),
        )
    ]


def test_combined_artifact_keeps_qualified_precision_and_changes_only_decode_policy():
    config = Path(__file__).resolve().parents[2] / "config"
    before = load_precision(config / "precision_single_step_compact_gdn_bfp8_all.json")
    after = load_precision(config / f"precision_{COMBINED_GDN_POLICY}_bfp8_all.json")
    assert {k: v for k, v in before.items() if k not in ("config_id", "decode_recurrence")} == {
        k: v for k, v in after.items() if k not in ("config_id", "decode_recurrence")
    }
    assert decoder_policy(after, 0)["decode_recurrence"] == COMBINED_GDN_POLICY


@pytest.mark.parametrize("batch", [1, 8, 16, 32])
def test_combined_resident_kernel_only_receives_prepared_compact_buckets(monkeypatch, batch):
    captured = []
    scratch = object()
    monkeypatch.setattr(decoder, "step_from_flat", lambda *a, **k: captured.append(k) or scratch)
    compact = batch in (16, 32)
    layer = SimpleNamespace(
        config=SimpleNamespace(linear_num_value_heads=12),
        policy={"decode_recurrence": COMBINED_GDN_POLICY},
        gdn_decode_workspace=SimpleNamespace(
            output=lambda b: scratch,
            shared_qk=lambda b: None,
            flat_outputs=lambda b: None,
        ),
    )
    inputs = [torch.zeros(1, batch, w) if compact else torch.zeros(batch, 32, w) for w in (512, 512, 1536, 12, 12)]
    result = decoder.Qwen38Decoder._delta_recurrence(
        layer,
        *inputs,
        SimpleNamespace(recurrent=object()),
        decode=True,
        compact_qkv=compact,
        compact_gates=compact,
    )
    assert result is scratch
    assert captured[0].get("resident_state", False) is compact
    assert captured[0].get("compact_gates", False) is compact
