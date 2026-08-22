# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for trace-stable resumed-prefill runtime inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as generator_vllm_module
from models.autoports.poolside_laguna_xs_2_1.tt import multichip_decoder as multichip_decoder_module
from models.autoports.poolside_laguna_xs_2_1.tt import optimized_decoder as optimized_decoder_module
from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.prefill_runtime import (
    PrefillRuntimeOffsets,
    prefill_chunk_plan,
)


def test_exact_production_bucket_and_chunk_slot_geometry(monkeypatch):
    monkeypatch.delenv("TT_LAGUNA_PREFILL_WARM_CAP", raising=False)
    bridge = object.__new__(LagunaForCausalLM)
    bridge.max_model_len = 131072
    buckets = bridge._prefill_bucket_lens()

    assert buckets == [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    keys = set()
    for L in buckets:
        plan = prefill_chunk_plan(L, pipe_threshold=2048, outer_chunk=8192, block_size=64)
        for ordinal, (_offset, length) in enumerate(plan):
            keys.add(("pipe" if L > 2048 else "single", ordinal, length))

    assert len(keys) == 24
    assert sum(key[2] for key in keys) == 139232
    assert ("pipe", 0, 4096) in keys
    assert {key for key in keys if key[0] == "pipe" and key[2] == 8192} == {
        ("pipe", ordinal, 8192) for ordinal in range(16)
    }


def test_prefill_chunk_plan_validates_alignment_and_coverage():
    assert prefill_chunk_plan(2048, pipe_threshold=2048, outer_chunk=8192, block_size=64) == ((0, 2048),)
    assert prefill_chunk_plan(20000, pipe_threshold=2048, outer_chunk=8192, block_size=64) == (
        (0, 8192),
        (8192, 8192),
        (16384, 3616),
    )
    with pytest.raises(ValueError, match="multiple of block size"):
        prefill_chunk_plan(4096, pipe_threshold=2048, outer_chunk=2000, block_size=64)


def test_runtime_path_is_d2_only_and_preserves_cold_single_shot():
    bridge = object.__new__(LagunaForCausalLM)
    bridge.model = SimpleNamespace(layers=[SimpleNamespace(PIPE_CHUNK=2048)])
    calls = []
    bridge._refresh_prefill_runtime_offsets = lambda L, start, bs: calls.append((L, start, bs)) or object()

    bridge.D = 2
    assert bridge._runtime_offsets_for_prefill(2048, 0, 64) is None
    assert calls == []
    assert bridge._runtime_offsets_for_prefill(2048, 64, 64) is not None
    assert bridge._runtime_offsets_for_prefill(4096, 0, 64) is not None
    assert calls == [(2048, 64, 64), (4096, 0, 64)]

    bridge.D = 1
    assert bridge._runtime_offsets_for_prefill(4096, 64, 64) is None
    assert calls == [(2048, 64, 64), (4096, 0, 64)]


def test_prefix_cache_requires_d2_and_cannot_disable_program_freeze(monkeypatch):
    assert LagunaForCausalLM._validate_prefix_cache_topology(2, True) is None
    with pytest.raises(RuntimeError, match="only on the p150x2"):
        LagunaForCausalLM._validate_prefix_cache_topology(1, True)

    fake_model = SimpleNamespace(precision_policy=SimpleNamespace(kv_cache=object()))
    fake_gen = SimpleNamespace(model=fake_model, tokenizer=object(), vocab=8, hidden=4)
    fake_mesh = SimpleNamespace(get_num_devices=lambda: 2)
    monkeypatch.setattr(LagunaForCausalLM, "_PREFIX_CACHE_ENABLED", True)
    monkeypatch.setenv("TT_LAGUNA_FREEZE_PROGRAM_CACHE", "0")
    monkeypatch.setattr(LagunaForCausalLM, "_report_dram", lambda *_args, **_kwargs: None)

    bridge = LagunaForCausalLM(fake_gen, fake_mesh, max_batch_size=1, max_model_len=128)

    assert bridge._freeze_program_cache is True


class _FakeTensor:
    def __init__(self, shape, dtype, layout):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout


class _FakeGenerator:
    def _rep(self, tensor, dtype, layout=None):
        return _FakeTensor(tensor.shape, dtype, layout)


def _runtime_bridge():
    bridge = object.__new__(LagunaForCausalLM)
    bridge.gen = _FakeGenerator()
    bridge.max_model_len = 16384
    full = SimpleNamespace(
        PIPE_CHUNK=2048,
        _prefill_pipe_chunk=8192,
        cfg=SimpleNamespace(attention_type="full_attention", rotary_dim=64),
    )
    sliding = SimpleNamespace(
        PIPE_CHUNK=2048,
        _prefill_pipe_chunk=8192,
        cfg=SimpleNamespace(attention_type="sliding_attention", rotary_dim=128),
    )
    bridge.model = SimpleNamespace(layers=[full, sliding], meta={"max_seq_len": 32768})
    bridge._prefill_bucket_lens = lambda: [32, 64, 4096, 8192, 16384]
    return bridge


def test_runtime_allocation_reuses_leading_pipeline_slots_across_buckets():
    bridge = _runtime_bridge()
    state = {}

    bridge._allocate_prefill_runtime_offsets(state, 64)

    assert state["runtime_block_size"] == 64
    assert state["runtime_slot_count"] == 5
    r8 = state["runtime_offsets"][8192]
    r16 = state["runtime_offsets"][16384]
    assert r8.position_ids[0] is r16.position_ids[0]
    assert r8.chunk_start_idxs[0] is r16.chunk_start_idxs[0]
    assert r8.rope_outputs["full_attention"][0] == r16.rope_outputs["full_attention"][0]
    assert r16.chunk_offsets == (0, 8192)
    assert r16.chunk_lengths == (8192, 8192)
    assert r16.rope_outputs["sliding_attention"][1][0].shape == (1, 1, 8192, 128)


def test_runtime_refresh_writes_absolute_positions_and_distinct_chunk_starts(monkeypatch):
    bridge = _runtime_bridge()
    state = {}
    bridge._allocate_prefill_runtime_offsets(state, 64)
    bridge._pf = state
    bridge._prefill_state = lambda _block_size=None: state
    uploads = []

    def host(tensor, dtype):
        return tensor.clone(), dtype

    bridge.gen._host = host

    def record_copy(source, target):
        uploads.append((source, target))

    monkeypatch.setattr(generator_vllm_module.ttnn, "copy_host_to_device_tensor", record_copy)

    runtime = bridge._refresh_prefill_runtime_offsets(16384, 320, 64)

    assert runtime is state["runtime_offsets"][16384]
    pos0, dtype0 = uploads[0][0]
    start0, start_dtype0 = uploads[1][0]
    pos1, dtype1 = uploads[2][0]
    start1, start_dtype1 = uploads[3][0]
    assert pos0.shape == pos1.shape == (1, 8192)
    assert (pos0[0, 0].item(), pos0[0, -1].item()) == (320, 8511)
    assert (pos1[0, 0].item(), pos1[0, -1].item()) == (8512, 16703)
    assert start0.tolist() == [320]
    assert start1.tolist() == [8512]
    assert dtype0 == dtype1 == generator_vllm_module.ttnn.uint32
    assert start_dtype0 == start_dtype1 == generator_vllm_module.ttnn.int32


def test_indexed_rope_is_built_once_per_kind_chunk_and_reused_by_layers(monkeypatch):
    indexed_calls = []
    layer_calls = []

    class _Layer:
        PIPE_CHUNK = 4

        def __init__(self, name, kind):
            self.name = name
            self.cfg = SimpleNamespace(attention_type=kind)

        def _rope_prefill_indexed(self, position_ids, sin=False, output_tensor=None):
            result = (self.name, position_ids, sin, output_tensor)
            indexed_calls.append(result)
            return result

        def prefill_forward(self, hidden, _kv, _pt, **kwargs):
            layer_calls.append((self.name, kwargs["rope_mats"], kwargs["runtime_offsets"]))
            return hidden

    full0 = _Layer("full0", "full_attention")
    sliding = _Layer("sliding", "sliding_attention")
    full1 = _Layer("full1", "full_attention")
    model = object.__new__(LagunaModel)
    model.layers = [full0, sliding, full1]
    p0, p1 = object(), object()
    s0, s1 = object(), object()
    full_outputs = ((object(), object()), (object(), object()))
    sliding_outputs = ((object(), object()), (object(), object()))
    runtime = PrefillRuntimeOffsets(
        bucket_len=8,
        chunk_offsets=(0, 4),
        chunk_lengths=(4, 4),
        position_ids=(p0, p1),
        chunk_start_idxs=(s0, s1),
        rope_outputs={"full_attention": full_outputs, "sliding_attention": sliding_outputs},
    )
    hidden = torch.zeros((1, 8, 1))

    out = model.prefill_layers(
        hidden,
        [object(), object(), object()],
        object(),
        fill_page_table=object(),
        fill_page_table_base_pos=64,
        start_pos=64,
        runtime_offsets=runtime,
    )

    assert out is hidden
    assert len(indexed_calls) == 8  # two chunks x cos/sin x two distinct attention kinds
    assert all(call[0] != "full1" for call in indexed_calls)
    assert layer_calls[0][1] is layer_calls[2][1]
    assert layer_calls[0][2] is layer_calls[1][2] is layer_calls[2][2] is runtime


def test_small_runtime_rope_stays_a_one_chunk_tuple_for_pipeline():
    captured = []

    class _Layer:
        PIPE_CHUNK = 2048
        cfg = SimpleNamespace(attention_type="full_attention")

        @staticmethod
        def _rope_prefill_indexed(position_ids, sin=False, output_tensor=None):
            return position_ids, sin, output_tensor

        @staticmethod
        def prefill_forward(hidden, _kv, _pt, **kwargs):
            captured.append(kwargs["rope_mats"])
            return hidden

    model = object.__new__(LagunaModel)
    model.layers = [_Layer()]
    runtime = PrefillRuntimeOffsets(
        bucket_len=64,
        chunk_offsets=(0,),
        chunk_lengths=(64,),
        position_ids=(object(),),
        chunk_start_idxs=(object(),),
        rope_outputs={"full_attention": ((object(), object()),)},
    )

    model.prefill_layers(
        torch.zeros((1, 64, 1)),
        [object()],
        object(),
        fill_page_table=object(),
        fill_page_table_base_pos=32704,
        start_pos=32704,
        runtime_offsets=runtime,
    )

    assert isinstance(captured[0], tuple)
    assert len(captured[0]) == 1


def test_indexed_rope_embedding_writes_the_preallocated_output(monkeypatch):
    dec = object.__new__(OptimizedDecoder)
    dec.cfg = SimpleNamespace(rotary_dim=64)
    dec.cos_2d = object()
    dec.sin_2d = object()
    position_ids = SimpleNamespace(shape=(1, 32))
    output = object()
    gathered = object()
    calls = []

    def embedding(indices, table, **kwargs):
        calls.append((indices, table, kwargs))
        return gathered

    monkeypatch.setattr(optimized_decoder_module.ttnn, "embedding", embedding)
    monkeypatch.setattr(optimized_decoder_module.ttnn, "reshape", lambda value, shape: (value, shape))

    result = OptimizedDecoder._rope_prefill_indexed(dec, position_ids, output_tensor=output)

    assert calls == [
        (
            position_ids,
            dec.cos_2d,
            {"layout": optimized_decoder_module.ttnn.TILE_LAYOUT, "output_tensor": output},
        )
    ]
    assert result == (gathered, (1, 1, 32, 64))


@pytest.mark.parametrize("sliding", [False, True])
def test_chunked_attention_accepts_flexible_chunk_start_tensor(monkeypatch, sliding):
    dec = object.__new__(MultichipDecoder)
    dec.cfg = SimpleNamespace(is_sliding=sliding, sliding_window=512, scaling=1.0)
    dec.PREFILL_SDPA_CHUNK = 8192
    dec._sdpa_pc_chunked = object()
    dec._sdpa_compute = object()
    start_tensor = object()
    calls = []

    monkeypatch.setattr(multichip_decoder_module.ttnn, "slice", lambda *_args, **_kwargs: object())

    def chunked(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        multichip_decoder_module.ttnn.transformer,
        "chunked_scaled_dot_product_attention",
        chunked,
    )

    out = MultichipDecoder._prefill_attention(
        dec,
        object(),
        object(),
        object(),
        {"k": object(), "v": object()},
        torch.zeros((1, 8), dtype=torch.int32),
        0,
        64,
        32,
        chunk_start_idx_tensor=start_tensor,
    )

    assert out is not None
    assert calls[0][1]["chunk_start_idx_tensor"] is start_tensor
    assert "chunk_start_idx" not in calls[0][1]
    assert calls[0][1].get("sliding_window_size") == (512 if sliding else None)


def test_small_runtime_prefill_routes_to_one_chunk_pipeline():
    dec = object.__new__(MultichipDecoder)
    dec.PIPE_CHUNK = 2048
    runtime = object()
    rope = ((object(), object()),)
    calls = []
    dec._prefill_pipelined = lambda *args, **kwargs: calls.append((args, kwargs)) or "pipeline"

    result = MultichipDecoder.prefill_forward(
        dec,
        torch.zeros((1, 64, 1)),
        object(),
        object(),
        fill_page_table=object(),
        fill_page_table_base_pos=32704,
        user_id=0,
        start_pos=32704,
        rope_mats=rope,
        runtime_offsets=runtime,
    )

    assert result == "pipeline"
    assert calls[0][1]["fill_page_table_base_pos"] == 32704
    assert calls[0][1]["rope_mats"] is rope
    assert calls[0][1]["runtime_offsets"] is runtime


def test_pipelined_fill_uses_relative_columns_and_flexible_chunk_starts(monkeypatch):
    dec = object.__new__(MultichipDecoder)
    dec.PIPE_CHUNK = 64
    dec.PREFILL_FAST = False
    dec.PREFILL_FAST_CHUNK = 8192
    dec.cfg = SimpleNamespace(
        is_sliding=False,
        sliding_window=512,
        hidden=1,
        num_heads=1,
        head_dim=1,
    )
    dec.w = {"input_ln": object(), "wo": object(), "post_ln": object()}
    dec._ck_o = object()
    dec._sdpa_compute = object()
    dec._rms = lambda value, _weight: value
    dec._cast_fill = lambda value, _dtype: value
    dec._gate = lambda value, _ln: value
    dec._reduce = lambda value: value
    dec._mlp = lambda _ln, length, sharded: torch.zeros((1, 1, length, 1))
    rope_calls = []

    def qkv(_ln, length, absolute, rope=None):
        rope_calls.append((length, absolute, rope))
        value = torch.zeros((1, 1, length, 1))
        return value, value, value

    dec._qkv_roped = qkv
    fill_calls = []
    sdpa_calls = []

    def sliced(value, starts, ends):
        return value[tuple(slice(start, end) for start, end in zip(starts, ends))]

    monkeypatch.setattr(multichip_decoder_module.ttnn, "slice", sliced)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "linear", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "add", torch.add)
    monkeypatch.setattr(multichip_decoder_module.ttnn, "concat", lambda values, dim: torch.cat(values, dim=dim))
    monkeypatch.setattr(
        multichip_decoder_module.ttnn.experimental,
        "paged_fill_cache",
        lambda _cache, _value, table, **_kwargs: fill_calls.append(table.clone()),
    )
    monkeypatch.setattr(
        multichip_decoder_module.ttnn.experimental,
        "nlp_concat_heads",
        lambda value, **_kwargs: value,
    )

    def chunked(q, *_args, **kwargs):
        sdpa_calls.append(kwargs)
        return q

    monkeypatch.setattr(
        multichip_decoder_module.ttnn.transformer,
        "chunked_scaled_dot_product_attention",
        chunked,
    )

    starts = (object(), object())
    runtime = PrefillRuntimeOffsets(
        bucket_len=128,
        chunk_offsets=(0, 64),
        chunk_lengths=(64, 64),
        position_ids=(object(), object()),
        chunk_start_idxs=starts,
        rope_outputs={"full_attention": ((object(), object()), (object(), object()))},
    )
    rope = (("cos0", "sin0"), ("cos1", "sin1"))
    fill = torch.tensor([[12, 13, -1, -1]], dtype=torch.int32)

    out = MultichipDecoder._prefill_pipelined(
        dec,
        torch.zeros((1, 128, 1)),
        {"k": object(), "v": object(), "dtype": torch.float32, "block_size": 64},
        torch.tensor([[1, 2, 12, 13, 99, 99]], dtype=torch.int32),
        fill,
        0,
        128,
        fill_page_table_base_pos=128,
        rope_mats=rope,
        runtime_offsets=runtime,
    )

    assert out.shape == (1, 128, 1)
    assert [table.tolist() for table in fill_calls] == [[[12]], [[12]], [[13]], [[13]]]
    assert rope_calls == [(64, 128, rope[0]), (64, 192, rope[1])]
    assert [call["chunk_start_idx_tensor"] for call in sdpa_calls] == list(starts)
    assert all("chunk_start_idx" not in call for call in sdpa_calls)


def test_program_cache_freeze_records_post_trace_count(monkeypatch):
    class _Mesh:
        def __init__(self):
            self.allowed = True

        @staticmethod
        def num_program_cache_entries():
            return 123

        def set_program_cache_misses_allowed(self, allowed):
            self.allowed = allowed

    bridge = object.__new__(LagunaForCausalLM)
    bridge.mesh_device = _Mesh()
    bridge._freeze_program_cache = True
    bridge._program_cache_entries_after_trace = None
    monkeypatch.setattr(generator_vllm_module.ttnn, "synchronize_device", lambda _device: None)

    bridge._freeze_program_cache_after_trace()

    assert bridge._program_cache_entries_after_trace == 123
    assert bridge.mesh_device.allowed is False
