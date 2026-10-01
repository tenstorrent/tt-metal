# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, validate_params
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
    "override", [{"num_users": 7}, {"mesh_shape": (4, 8)}, {"num_layers": 30}, {"chunk_size": 4096}]
)
def test_unsupported_service_shapes(override, expect_error):
    with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
        validate_params(replace(service_params(), **override))


def test_slot_offsets_and_partial_final_chunk(expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.slot_ends = [8192, 16384, 0, 0, 0, 253952]
    runtime.validate_chunk(0, 8192, 16384)
    runtime.validate_chunk(1, 16384, 16385)
    runtime.validate_chunk(5, 253952, 262144)
    runtime.validate_chunk(5, 0, 8192)
    runtime.validate_chunk(0, 7008, 9000)
    runtime.validate_chunk(1, 16352, 17001)
    runtime.validate_chunk(5, 253920, 262112)
    for chunk in [(6, 0, 8192), (0, 16384, 24576), (0, 8224, 12000), (0, 1, 8193), (5, 253952, 262145)]:
        with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
            runtime.validate_chunk(*chunk)


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


def test_runtime_compile_and_replay_stage_all_slot_bounds(monkeypatch):
    import torch

    from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata
    from models.demos.gemma4_d_p.tt.runners import runtime as runtime_module

    config = service_params()
    runtime = Gemma4PrefillRuntime(
        mesh_device=SimpleNamespace(shape=config.mesh_shape),
        hf_model_id="google/gemma-4-31B-it",
        tt_cache_path="/tmp/weights",
        config=config,
    )
    created = []

    def create_model(**kwargs):
        created.append(kwargs)
        metadata = object.__new__(PrefillMetadata)
        metadata.mesh_config = runtime.mesh_config
        metadata.chunk_size = config.chunk_size
        metadata.max_seq_len = config.max_seq_len
        metadata.num_users = kwargs["max_batch_size"]
        metadata._stage = lambda name, values, seq_dim=None: values
        return None, SimpleNamespace(prefill_metadata=metadata), None, None

    monkeypatch.setattr(runtime_module, "create_tt_model", create_model)
    monkeypatch.setattr(runtime, "make_chunk_input", lambda token_ids: object())
    monkeypatch.setattr(runtime, "_forward", lambda: SimpleNamespace(deallocate=lambda force: None))
    monkeypatch.setattr(runtime, "_check_cache", lambda cache: None)
    for name in ("from_torch", "ReplicateTensorToMesh", "copy", "synchronize_device", "deallocate"):
        monkeypatch.setattr(runtime_module.ttnn, name, Mock())
    monkeypatch.setattr(runtime_module.ttnn, "reshape", lambda tensor, shape: tensor)
    replay = Mock()
    monkeypatch.setattr(runtime_module.ttnn, "execute_trace", replay)
    cache = object()
    runtime.compile(cache)
    assert created[0]["max_batch_size"] == config.num_users
    assert runtime.model._prefill_metadata_external
    runtime.trace_id = 1
    runtime.slot_ends[5] = 8192

    for slot, start, end in [(5, 7008, 9000), (5, 8992, 12001), (5, 4096, 8000), (0, 0, 8192)]:
        runtime.prefill_chunk(object(), cache, slot_id=slot, actual_start=start, actual_end=end)
        metadata = runtime.model.prefill_metadata
        assert (metadata.slot_idx.item(), metadata.kv_actual_global.item(), metadata.actual_end.item()) == (
            slot,
            start,
            end,
        )
        positions = torch.arange(start, start + config.chunk_size)
        expected = torch.cat([positions[(positions // 1024) % 8 == rank] for rank in range(8)])
        torch.testing.assert_close(metadata.positions.flatten(), expected)
        assert runtime.slot_ends[slot] == end
    assert replay.call_count == 4
