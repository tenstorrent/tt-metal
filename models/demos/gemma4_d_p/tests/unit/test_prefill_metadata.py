# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.tt import model as model_module
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


@pytest.fixture
def metadata_model(monkeypatch):
    hidden = SimpleNamespace(shape=(1, 1, 128, 128), is_sharded=lambda: False)
    mesh = SimpleNamespace(device=object(), cp_degree=8, tp_degree=1)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda _: None)
    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **_: tensor.clone())
    monkeypatch.setattr(ttnn, "to_device", lambda tensor, *args, **kwargs: tensor.clone())
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, target: target.copy_(source))
    monkeypatch.setattr(
        model_module,
        "Gemma4DecoderLayer",
        lambda **kwargs: Mock(return_value=hidden, self_attn=SimpleNamespace(ring_kv_cache=object())),
    )
    model = model_module.Gemma4Model(
        mesh,
        SimpleNamespace(
            hidden_size=128,
            vocab_size=256,
            num_hidden_layers=2,
            layer_types=("sliding_attention", "full_attention"),
        ),
        {"model.language_model.placeholder": None},
        ccl_manager=None,
        prefill_chunk_size=4096,
        precision=SimpleNamespace(get=lambda _, default: default),
        max_seq_len=32768,
        stripe_size=32,
        max_prefill_requests=3,
    )
    model.lookup_packed_rope = lambda layer_type: layer_type
    return model, hidden


def test_staging_and_layout_changes_reuse_model_metadata(metadata_model):
    model, hidden = metadata_model
    metadata = model.request_metadata
    buffers = tuple((m.slot_idx, m.kv_actual_global, m.valid_global) for m in metadata)
    assert model.prefill_metadata is metadata[0]
    assert len({id(tensor) for entry in buffers for tensor in entry}) == 9

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.model = model
    runtime.mesh_config = model.mesh_config
    runtime.config = SimpleNamespace(chunk_size=4096)
    runtime.positions = torch.zeros((1, 4096), dtype=torch.int64)
    runtime._host_rows = lambda rows: torch.tensor(rows).reshape(1, -1)
    schedules = (
        (PrefillSegment(4, 12896, 13152), PrefillSegment(1, 25248, 25760)),
        (PrefillSegment(2, 544, 1312),),
        (PrefillSegment(0, 0, 256), PrefillSegment(5, 29408, 30432), PrefillSegment(3, 13888, 15424)),
    )
    for segments in schedules:
        runtime._stage_batch(segments)
        assert model(hidden, prefill_lengths=runtime._layout) is hidden
        for layer in model.layers:
            requests = layer.call_args.kwargs["prefill_batch"]
            assert tuple(length for length, _ in requests) == runtime._layout
            for index, ((_, entry), segment) in enumerate(zip(requests, segments)):
                assert entry is metadata[index]
                assert (entry.slot_idx.item(), entry.kv_actual_global.item(), entry.valid_global.item()) == (
                    segment.slot_id,
                    segment.actual_start,
                    segment.actual_end,
                )
        assert model.request_metadata is metadata
        assert all(
            actual is original
            for entry, original_buffers in zip(metadata, buffers)
            for actual, original in zip((entry.slot_idx, entry.kv_actual_global, entry.valid_global), original_buffers)
        )


def test_legacy_prefill_uses_first_model_metadata_entry(metadata_model):
    model, hidden = metadata_model
    model(hidden, user_id=2, chunk_start_idx=8192)
    assert model.request_metadata[0].slot_idx.item() == 2
    assert model.request_metadata[0].kv_actual_global.item() == 8192
    for layer in model.layers:
        assert layer.call_args.kwargs["prefill_batch"] is None
        assert layer.call_args.kwargs["prefill_metadata"] is model.request_metadata[0]


@pytest.mark.parametrize(
    "lengths, error",
    [
        ((), "metadata capacity"),
        ((256, 256, 256, 256), "metadata capacity"),
        ((257,), "multiples of 256"),
        ((0,), "positive multiples"),
        ((2048, 2048, 256), "chunk size"),
    ],
)
def test_invalid_layouts_fail_before_attention(metadata_model, expect_error, lengths, error):
    model, hidden = metadata_model
    with expect_error(ValueError, error):
        model(hidden, prefill_lengths=lengths)
    for layer in model.layers:
        layer.assert_not_called()
