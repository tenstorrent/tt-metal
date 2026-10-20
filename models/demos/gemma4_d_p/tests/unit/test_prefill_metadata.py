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
    buffer = model.packed_prefill_metadata.tensor
    assert all(entry.descriptors is buffer for entry in metadata)

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.model = model
    runtime.mesh_config = model.mesh_config
    runtime.config = SimpleNamespace(chunk_size=4096)
    runtime.positions = torch.zeros((1, 4096), dtype=torch.int64)
    runtime._host_rows = lambda rows: torch.tensor(rows).reshape(1, -1)
    schedules = (
        (PrefillSegment(4, 12896, 13152), PrefillSegment(1, 25248, 25760)),
        (PrefillSegment(2, 544, 801),),
        (PrefillSegment(0, 0, 256), PrefillSegment(5, 29408, 30432), PrefillSegment(3, 13888, 15424)),
    )
    for segments in schedules:
        runtime._stage_batch(segments)
        assert model(hidden, prefill_lengths=runtime._layout) is hidden
        for layer in model.layers:
            requests = layer.call_args.kwargs["prefill_batch"]
            assert tuple(length for length, _ in requests) == runtime._layout
            offset = 0
            for index, ((_, entry), segment) in enumerate(zip(requests, segments)):
                assert entry is metadata[index]
                span = segment.padded_length(8)
                assert entry.index == index
                assert entry.descriptors[index].tolist() == [
                    1,
                    segment.slot_id,
                    segment.actual_start,
                    segment.actual_end - segment.actual_start,
                    offset // 256,
                    span // 256,
                    0,
                    0,
                ]
                offset += span
            assert not torch.any(buffer[len(segments) :])
        assert model.request_metadata is metadata
        assert all(entry.descriptors is buffer for entry in metadata)


def test_legacy_prefill_uses_first_model_metadata_entry(metadata_model):
    model, hidden = metadata_model
    model(hidden, user_id=2, chunk_start_idx=8192)
    assert model.prefill_metadata.slot_idx.item() == 2
    assert model.prefill_metadata.kv_actual_global.item() == 8192
    for layer in model.layers:
        assert layer.call_args.kwargs["prefill_batch"] is None
        assert layer.call_args.kwargs["prefill_metadata"] is model.prefill_metadata


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


@pytest.mark.parametrize(
    "requests,error",
    [
        ((), "metadata capacity"),
        (((0, 1, 256, 256),), "Invalid packed request"),
        (((0, 0, 257, 256),), "Invalid packed request"),
        (((0, 0, 0, 256),), "Invalid packed request"),
        (((0, 0, 257, 257),), "Invalid packed request"),
        (((-1, 0, 256, 256),), "Invalid packed request"),
        (((0, 0, 4097, 4352),), "chunk size"),
        (((0, 0x100000000, 256, 256),), "uint32"),
    ],
)
def test_invalid_descriptors_do_not_change_staged_table(metadata_model, expect_error, requests, error):
    model, _ = metadata_model
    metadata = model.packed_prefill_metadata
    metadata.update(((2, 544, 257, 512),))
    before = metadata.tensor.clone()
    with expect_error(ValueError, error):
        metadata.update(requests)
    torch.testing.assert_close(metadata.tensor, before)
    assert metadata.layout == (512,)


def test_layout_must_match_staged_descriptor_spans(metadata_model, expect_error):
    model, hidden = metadata_model
    model.packed_prefill_metadata.update(((0, 32, 257, 512),))
    with expect_error(ValueError, "staged descriptor spans"):
        model(hidden, prefill_lengths=(256,))
    for layer in model.layers:
        layer.assert_not_called()
