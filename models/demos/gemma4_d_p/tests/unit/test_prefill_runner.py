# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, validate_params
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment
from models.demos.gemma4_d_p.tt.runners.prepare_prefill_inputs import write_producer_manifest
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def service_params():
    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=262144,
        chunk_size=8192,
        num_users=6,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
    )


def test_adapter_uses_canonical_weight_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    (tmp_path / "tensor_cache_bf16_mesh8x4").mkdir()
    adapter = get_adapter("gemma4_d_p")
    assert isinstance(adapter, Gemma4PrefillAdapter)
    assert adapter.weight_cache_path((8, 4)) == tmp_path / "tensor_cache_bf16_mesh8x4"
    validate_params(service_params())


@pytest.mark.parametrize(
    "override", [{"num_users": 7}, {"mesh_shape": (4, 8)}, {"num_layers": 30}, {"chunk_size": 768}]
)
def test_unsupported_service_shapes(override, expect_error):
    with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot|Request|request"):
        validate_params(replace(service_params(), **override))


def test_slot_offsets_and_partial_final_chunk(expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.slot_ends = [8192, 16384, 0, 0, 0, 253952]
    runtime.validate_chunk(0, 8192, 16384)
    runtime.validate_chunk(1, 16384, 16385)
    runtime.validate_chunk(5, 253952, 262144)
    runtime.validate_chunk(5, 0, 8192)
    for chunk in [(6, 0, 8192), (0, 16384, 24576), (0, 1, 8193), (5, 253952, 262145)]:
        with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot|Request|request"):
            runtime.validate_chunk(*chunk)


@pytest.mark.parametrize("use_trace", [False, True])
def test_batch_completions_identify_chunks_after_device_sync(monkeypatch, use_trace):
    import ttnn

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = replace(service_params(), use_trace=use_trace)
    runtime.mesh_config = SimpleNamespace(cp_degree=8)
    runtime.mesh_device = object()
    runtime.input_tokens = object()
    runtime.slot_ends = [0, 512, 0, 0, 2048, 0]
    runtime._ready = True
    runtime._next_request_id = 0
    runtime._check_cache = Mock()
    runtime._stage_batch = Mock()
    runtime._forward = Mock()
    segments = (PrefillSegment(4, 2048, 3648), PrefillSegment(1, 512, 769))
    events = []
    runtime.set_layer_completion_sink(SimpleNamespace(layers_completed=lambda *span: events.append(span)))
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, shape: tensor)
    monkeypatch.setattr(ttnn, "copy", lambda *args: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda device: events.append("synchronized"))
    if use_trace:
        runtime._layout = tuple(s.padded_length() for s in segments)
        runtime.traces = OrderedDict([(runtime._layout, 123)])
        monkeypatch.setattr(ttnn, "execute_trace", lambda *args, **kwargs: events.append("replayed"))

    runtime.prefill_batch(object(), object(), segments=segments)

    assert events == (["replayed"] if use_trace else []) + [
        "synchronized",
        (0, 60, 4, 2048, 3648),
        (0, 60, 1, 512, 769),
    ]
    assert runtime.slot_ends == [0, 769, 0, 0, 3648, 0]


def test_prepared_prompts_use_shared_producer_format(tmp_path):
    from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids

    prompts = [[1, 2, 3], [4, 5]]
    manifest = json.loads(write_producer_manifest(tmp_path, prompts).read_text())
    assert manifest["env"]["PREFILL_MODEL"] == "gemma4_d_p"
    assert manifest["env"]["PREFILL_NUM_USERS"] == "2"
    assert manifest["workload"]["max_requests"] == 2
    assert manifest["workload"]["interleave"] == "round_robin"
    assert not manifest["workload"]["check_pcc"]
    assert [load_trace_token_ids(path) for path in manifest["workload"]["slot_prompts"]] == prompts


@pytest.mark.parametrize("prefill_override", [False, True])
def test_tt_cache_resolution(monkeypatch, tmp_path, prefill_override):
    import ttnn
    from models.demos.gemma4_d_p.tt.model_config import resolve_cache_dir_from_tt_cache_path

    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "regular"))
    expected = tmp_path / "regular/tensor_cache_bf16_mesh8x4"
    expected.mkdir(parents=True)
    assert (
        resolve_cache_dir_from_tt_cache_path(tmp_path / "regular", dtype=ttnn.bfloat16, mesh_shape=(8, 4)) == expected
    )
    if prefill_override:
        monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path / "service"))
        expected = tmp_path / "service/tensor_cache_bf16_mesh8x4"
        expected.mkdir(parents=True)
    assert Gemma4PrefillAdapter().weight_cache_path((8, 4)) == expected


def test_tt_cache_requires_configured_root(monkeypatch, tmp_path, expect_error):
    import ttnn
    from models.demos.gemma4_d_p.tt.model_config import resolve_cache_dir_from_tt_cache_path

    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_MODEL", str(tmp_path))
    with expect_error(ValueError, "tt_cache_path must be provided"):
        resolve_cache_dir_from_tt_cache_path(None, dtype=ttnn.bfloat16, mesh_shape=(8, 4))
    with expect_error(ValueError, "tt_cache_path must be provided"):
        Gemma4PrefillAdapter().weight_cache_path((8, 4))
