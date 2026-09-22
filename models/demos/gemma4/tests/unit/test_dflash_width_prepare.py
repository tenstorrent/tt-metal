# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for dFlash preparation before ordinary trace capture.

Production constructors, width allocation, context seeding, and lifecycle
methods run with stub TT tensor operations and a stub fused computation.
These checks establish host ordering, not captured device-buffer correctness.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _unused(*args, **kwargs):
    raise AssertionError("A host lifecycle test reached an unstubbed device operation")


def _module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


class Tensor:
    def __init__(self, data):
        self.data = data.clone()
        self.shape = self.data.shape
        self.freed = False

    def deallocate(self, force=False):
        self.freed = True


class Runtime:
    def __init__(self):
        self.events = []
        self.active_trace = None
        self.next_trace = 1
        self.fail = None

    def from_torch(self, data, **kwargs):
        tensor = Tensor(data)
        if kwargs.get("device") is not None:
            self.events.append(("allocate", tensor))
        return tensor

    def copy(self, source, destination):
        destination.data = source.data.clone()

    def embedding(self, positions, table, **kwargs):
        return Tensor(table[positions.data.to(torch.long)])

    def reshape(self, tensor, shape):
        return Tensor(tensor.data.reshape(shape))

    def synchronize(self, mesh):
        self.events.append(("sync",))
        if self.fail == "sync":
            raise RuntimeError("injected sync failure")

    def begin(self, mesh, cq_id):
        assert self.active_trace is None
        self.active_trace = self.next_trace
        self.next_trace += 1
        self.events.append(("begin", self.active_trace))
        return self.active_trace

    def end(self, mesh, trace, cq_id):
        assert self.active_trace == trace
        self.events.append(("end", trace))
        self.active_trace = None

    def release(self, mesh, trace):
        self.events.append(("release", trace))


class Target:
    def __init__(self, runtime):
        self.runtime = runtime
        self._dflash_sharded_logits = False
        self.tap_layers = None
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace()))]
        self.hf_config = SimpleNamespace(layer_types=["full_attention"])

    def dflash_capture_taps(self, layers, buffers=None):
        self.tap_layers = layers
        self.runtime.events.append(("taps", None if layers is None else tuple(layers)))


@pytest.fixture
def production(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", "5")
    monkeypatch.setenv("GEMMA4_DFLASH_PACKED", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_CTX_CACHE", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_WARMUP_DECODE", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_WIDTH_RUNGS", "0")
    runtime = Runtime()
    tt = _module(
        "ttnn",
        bfloat16="bfloat16",
        float32="float32",
        uint32="uint32",
        int32="int32",
        TILE_LAYOUT="tile",
        ROW_MAJOR_LAYOUT="row_major",
        from_torch=runtime.from_torch,
        copy_host_to_device_tensor=runtime.copy,
        embedding=runtime.embedding,
        reshape=runtime.reshape,
        assign=runtime.copy,
        synchronize_device=runtime.synchronize,
        begin_trace_capture=runtime.begin,
        end_trace_capture=runtime.end,
        release_trace=runtime.release,
    )
    monkeypatch.setitem(sys.modules, "ttnn", tt)
    monkeypatch.setitem(
        sys.modules,
        "models.demos.gemma4.tt.ccl",
        _module("models.demos.gemma4.tt.ccl", ccl_allgather=_unused, ccl_allreduce=_unused),
    )
    parent = importlib.import_module("models.demos.gemma4.tt")
    name = "models.demos.gemma4.tt.dflash_drafter"
    monkeypatch.setattr(parent, "dflash_drafter", None, raising=False)
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)

    def body(decoder):
        runtime.events.append(("body", decoder.pv_sk, runtime.active_trace, decoder))
        decoder.target._dflash_sharded_logits = True
        if runtime.fail == "body":
            raise RuntimeError("injected body failure")
        if decoder.out_ids is None:
            decoder.out_ids = runtime.from_torch(torch.zeros(1, 21), device=runtime)
        for index, tensor in enumerate(decoder.tap_bufs):
            if tensor is None:
                decoder.tap_bufs[index] = runtime.from_torch(torch.zeros(1, 1, 6, 8), device=runtime)
        return decoder.out_ids, decoder.fc_prev

    real_pv_upload = module.DFlashFusedDecoder._pv_upload
    real_upload_iter_inputs = module.DFlashFusedDecoder._upload_iter_inputs
    monkeypatch.setattr(module.DFlashFusedDecoder, "_body", body)
    monkeypatch.setattr(
        module.DFlashFusedDecoder,
        "_pv_upload",
        lambda decoder, start: runtime.events.append(("pv_upload", decoder.pv_sk, start)),
    )
    monkeypatch.setattr(
        module.DFlashFusedDecoder,
        "_upload_iter_inputs",
        lambda decoder, anchor, start: runtime.events.append(("inputs", decoder.pv_sk, anchor, start)),
    )
    yield SimpleNamespace(
        module=module,
        runtime=runtime,
        tt=tt,
        real_pv_upload=real_pv_upload,
        real_upload_iter_inputs=real_upload_iter_inputs,
    )
    sys.modules.pop(name, None)


@pytest.fixture
def expect_error():
    return pytest.raises  # allow-pytest.raises: root conftest requires hardware runtime


def _drafter(runtime):
    def project(raw, cos, sin):
        runtime.events.append(("project_ctx_kv", raw.shape[2]))
        if runtime.fail == "seed":
            raise RuntimeError("injected seed failure")
        return [(Tensor(torch.zeros(1, 1, raw.shape[2], 2)), Tensor(torch.zeros(1, 1, raw.shape[2], 2)))]

    return SimpleNamespace(
        mesh_device=runtime,
        _replicate=object(),
        tp=1,
        block_size=16,
        hidden=8,
        local_kv=1,
        head_dim=2,
        layers=[object()],
        _use_sdpa=True,
        _mask_rows=None,
        _embed_w_host=torch.zeros(128, 8),
        mask_token_id=1,
        target_layer_ids=[0],
        _cos_2d=torch.ones(4096, 2),
        _sin_2d=torch.zeros(4096, 2),
        project_ctx_kv=project,
    )


def _decoder(production):
    runtime = production.runtime
    decoder = production.module.DFlashFusedDecoder(
        Target(runtime), _drafter(runtime), object(), torch.zeros(1, 64, dtype=torch.int32), ctx_cap=32
    )
    runtime.events.clear()
    return decoder


def _clean(decoder):
    assert decoder.target.tap_layers is None
    assert decoder.target._dflash_sharded_logits is False


def test_prepare_allocates_all_widths_and_seeds_full_context_before_bodies(production):
    decoder = _decoder(production)
    decoder.win_first = 3
    decoder.prepare_widths([4096, 3072, 3072], anchor_id=17, start=9)
    events = production.runtime.events
    first_body = next(index for index, event in enumerate(events) if event[0] == "body")
    assert all(event[0] != "begin" for event in events)
    assert all(record["trace"] is None for record in decoder._pv_widths.values())
    assert decoder._prepared_widths == {3072, 4096}
    assert decoder.width_for(0) is None
    assert ("project_ctx_kv", 32) in events[:first_body]
    assert decoder.ctx_pos.data.tolist() == [list(range(3, 35))]
    allocated = [event[1] for event in events[:first_body] if event[0] == "allocate"]
    for record in decoder._pv_widths.values():
        assert record["pv_iota"] in allocated
        assert all(table in allocated for table in record["pv_tables"])
    assert [event[1] for event in events if event[0] == "body"] == [3072, 4096]
    assert ("inputs", 3072, 17, 9) in events
    assert ("inputs", 4096, 17, 9) in events
    assert decoder.start is None and decoder.anchor is None
    _clean(decoder)


def test_direct_capture_prepares_every_width_before_first_trace(production):
    decoder = _decoder(production)
    costs = decoder.capture_widths([4096, 3072])
    events = production.runtime.events
    first_begin = next(index for index, event in enumerate(events) if event[0] == "begin")
    assert [event[1] for event in events[:first_begin] if event[0] == "body"] == [3072, 4096]
    assert not any(event[0] == "allocate" for event in events[first_begin:])
    assert set(costs) == {3072, 4096}
    assert all(value >= 0 for value in costs.values())
    assert decoder.width_for(0) == 3072
    assert decoder.width_for(3072) == 4096
    assert decoder._replay_on_first is True
    assert production.runtime.active_trace is None
    _clean(decoder)


def test_capture_retains_prepared_buffers_without_another_eager_body(production):
    decoder = _decoder(production)
    decoder.prepare_widths([3072, 4096])
    records = dict(decoder._pv_widths)
    outputs = (decoder.out_ids, decoder.fc_prev, tuple(decoder.tap_bufs))
    production.runtime.events.clear()
    decoder.capture_widths([3072, 4096])
    assert all(decoder._pv_widths[width] is record for width, record in records.items())
    assert outputs == (decoder.out_ids, decoder.fc_prev, tuple(decoder.tap_bufs))
    assert not any(event[0] in ("allocate", "project_ctx_kv") for event in production.runtime.events)
    assert all(event[2] is not None for event in production.runtime.events if event[0] == "body")
    _clean(decoder)


def test_prepare_and_capture_are_idempotent(production):
    decoder = _decoder(production)
    decoder.prepare_widths([3072, 4096])
    production.runtime.events.clear()
    decoder.prepare_widths([4096, 3072, 3072])
    assert production.runtime.events == []
    decoder.capture_widths([3072, 4096])
    traces = {width: record["trace"] for width, record in decoder._pv_widths.items()}
    production.runtime.events.clear()
    assert decoder.capture_widths([4096, 3072]) == {}
    assert traces == {width: record["trace"] for width, record in decoder._pv_widths.items()}
    assert not any(event[0] in ("begin", "body", "allocate") for event in production.runtime.events)
    _clean(decoder)


@pytest.mark.parametrize("operation", ["prepare_widths", "capture_widths"])
def test_new_width_is_rejected_before_device_work_after_capture(production, expect_error, operation):
    decoder = _decoder(production)
    decoder.capture_widths([3072])
    production.runtime.events.clear()
    with expect_error(RuntimeError, match="after trace capture"):
        getattr(decoder, operation)([3072, 4096])
    assert production.runtime.events == []
    assert set(decoder._pv_widths) == {3072}


@pytest.mark.parametrize("failure", ["seed", "body", "sync"])
def test_prepare_failure_disarms_target_and_does_not_mark_width_ready(production, expect_error, failure):
    decoder = _decoder(production)
    production.runtime.fail = failure
    with expect_error(RuntimeError, match=f"injected {failure} failure"):
        decoder.prepare_widths([3072, 4096])
    _clean(decoder)
    assert not getattr(decoder, "_prepared_widths", set())
    assert all(record["trace"] is None for record in decoder._pv_widths.values())
    assert production.runtime.active_trace is None
    production.runtime.fail = None
    decoder.prepare_widths([3072, 4096])
    assert decoder._prepared_widths == {3072, 4096}


def test_capture_failure_disarms_target_and_does_not_publish_failed_trace(production, expect_error):
    decoder = _decoder(production)
    decoder.prepare_widths([3072])
    production.runtime.events.clear()
    production.runtime.fail = "body"
    with expect_error(RuntimeError, match="injected body failure"):
        decoder.capture_widths([3072])
    _clean(decoder)
    assert decoder._pv_widths[3072]["trace"] is None
    assert decoder.width_for(0) is None
    assert production.runtime.active_trace is None
    trace_events = [event for event in production.runtime.events if event[0] in ("begin", "end", "release")]
    assert trace_events == [("begin", 1), ("end", 1), ("release", 1)]
    production.runtime.fail = None
    assert set(decoder.capture_widths([3072])) == {3072}
    assert decoder._pv_widths[3072]["trace"] == 2


def test_nonpacked_preparation_fails_without_allocating(production, expect_error):
    decoder = _decoder(production)
    decoder.use_packed = False
    with expect_error(NotImplementedError, match="packed-verify only"):
        decoder.prepare_widths([3072])
    assert production.runtime.events == []


@pytest.fixture
def generator_module(production, monkeypatch):
    lower_modules = {
        "models.demos.gemma4.tt.common": {"create_tt_model": _unused},
        "models.demos.gemma4.tt.generator": {
            "SDPA_CHUNK_ALIGN": 128,
            "ChunkedPrefillPageTableGuardMixin": type("ChunkedPrefillPageTableGuardMixin", (), {}),
            "align_num_cached_tokens_to_sdpa": _unused,
            "max_batched_prefill_users": _unused,
            "resolve_batched_prefill_chunk_users": _unused,
        },
        "models.demos.gemma4.tt.generator_trace": {
            name: _unused
            for name in (
                "maybe_disable_pli_prefill_trace",
                "patch_gemma4_trace_model_args",
                "resolve_gemma4_prefill_chunk_size",
                "resolve_gemma4_prefill_trace_enable",
                "should_auto_enable_bounded_sliding",
                "warmup_gemma4_model_prefill",
            )
        },
        "models.tt_transformers.tt.common": {"get_padded_prefill_len": _unused},
        "models.tt_transformers.tt.generator": {
            "SUPPORTED_PREFILL_BATCH_SIZES": (1, 2, 4, 8, 16, 32),
            "create_submeshes": _unused,
        },
        "models.tt_transformers.tt.generator_vllm": {
            "HybridAttentionForCausalLM": type("HybridAttentionForCausalLM", (), {}),
            "allocate_vllm_kv_cache": _unused,
        },
    }
    for name, members in lower_modules.items():
        monkeypatch.setitem(sys.modules, name, _module(name, **members))
    name = "models.demos.gemma4.tt.generator_vllm"
    parent = importlib.import_module("models.demos.gemma4.tt")
    monkeypatch.setattr(parent, "generator_vllm", None, raising=False)
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def test_generator_warmup_prepares_before_ordinary_capture_and_reuses_decoder(
    production, generator_module, monkeypatch, expect_error
):
    runtime = production.runtime
    model = generator_module.Gemma4DFlashContractForCausalLM.__new__(generator_module.Gemma4DFlashContractForCausalLM)
    model._contract_init()
    model._spec_width_set = True
    model._spec_decoder = None
    model._spec_width_ladder = None
    model._spec_horizon = 2048
    model.model_args = [SimpleNamespace(max_seq_len=4096)]
    model.model = [Target(runtime)]
    drafter = _drafter(runtime)

    def get_drafter():
        runtime.events.append(("get_drafter",))
        return drafter

    def ordinary_warmup(instance, **kwargs):
        assert instance._ct_warmup_depth == 1
        runtime.events.append(("ordinary", kwargs["enable_trace"]))
        if kwargs["enable_trace"]:
            assert instance._spec_decoder._prepared_widths == {3072, 4096}
        return "native-result"

    model._spec_get_drafter = get_drafter
    monkeypatch.setattr(generator_module.Gemma4ForCausalLM, "warmup_model_decode", ordinary_warmup)
    kv_cache = object()
    assert model.warmup_model_decode(enable_trace=False, kv_cache=kv_cache, num_blocks=64) == "native-result"
    decoder = model._spec_decoder
    first_body = next(index for index, event in enumerate(runtime.events) if event[0] == "body")
    assert runtime.events[0] == ("ordinary", False)
    assert ("get_drafter",) in runtime.events[1:first_body]
    assert decoder.kv_layers is kv_cache
    assert decoder.v_pt.shape == (16, 64)
    assert decoder.width_for(0) is None
    _clean(decoder)

    runtime.events.clear()
    assert model.warmup_model_decode(enable_trace=True, kv_cache=kv_cache, num_blocks=64) == "native-result"
    assert model._spec_decoder is decoder
    assert runtime.events[0] == ("ordinary", True)
    assert not any(event[0] in ("allocate", "project_ctx_kv", "get_drafter") for event in runtime.events)
    assert all(event[3] is decoder for event in runtime.events if event[0] == "body")
    assert model._ct_warmup_depth == 0
    assert not model._ct_requests and not model._ct_ordinary and model._ct_proposal is None
    _clean(decoder)

    runtime.events.clear()
    with expect_error(RuntimeError, match="widths changed after preparation"):
        model._spec_capture_width_set(kv_cache, 32)
    assert runtime.events == []
