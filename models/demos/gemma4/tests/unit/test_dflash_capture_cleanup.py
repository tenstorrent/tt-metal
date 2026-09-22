# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for releasing Gemma and ordinary traces before mesh closure.

Production cleanup runs against stub trace handles and lower model imports.
The checks cover ownership and ordering, not device teardown correctness.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _unused(*args, **kwargs):
    raise AssertionError("A host cleanup test reached an unstubbed device operation")


def _module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


class Runtime:
    def __init__(self):
        self.events = []
        self.closed = False

    def release(self, mesh, trace):
        self.events.append(("release", mesh, trace, self.closed))

    def close(self, mesh):
        self.events.append(("close", mesh))
        self.closed = True


class Buffer:
    def __init__(self):
        self.releases = 0

    def deallocate(self, force):
        self.releases += 1


@pytest.fixture
def production(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5]))
    runtime = Runtime()
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        _module("ttnn", release_trace=runtime.release, close_mesh_device=runtime.close, __path__=[]),
    )
    monkeypatch.setitem(sys.modules, "ttnn.tools", _module("ttnn.tools", trace_allocation_tracker=SimpleNamespace()))
    lower_modules = {
        "models.common.llama_models": {
            name: _unused
            for name in (
                "CompletionMessage",
                "StopReason",
                "TokenResult",
                "create_vision_mask",
                "encode_content",
                "extract_images_from_messages",
                "sample_top_p",
            )
        },
        "models.common.sampling": {
            "SamplingParams": type("SamplingParams", (), {}),
            **{
                name: _unused
                for name in (
                    "broadcast_sampling_params",
                    "chunk_sampling_params",
                    "format_sampling_params",
                    "scatter_sampling_params_to_slots",
                )
            },
        },
        "models.common.sampling.tt_log_probs": {
            "LogProbsResult": type("LogProbsResult", (), {}),
            "reformat_logprobs": _unused,
        },
        "models.common.warmup": {"WarmupForwardMixin": type("WarmupForwardMixin", (), {})},
        "models.tt_transformers.tt.common": {
            "Mode": SimpleNamespace(PREFILL="prefill", DECODE="decode"),
            **{
                name: _unused
                for name in (
                    "copy_host_to_device",
                    "get_all_padded_prefill_lengths",
                    "get_block_size",
                    "get_max_prefill_chunk_size",
                    "get_padded_prefill_len",
                    "num_blocks_in_seq",
                )
            },
        },
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
    }
    for name, members in lower_modules.items():
        monkeypatch.setitem(sys.modules, name, _module(name, **members))
    loaded = []

    def load(name):
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        monkeypatch.setattr(parent, attribute, None, raising=False)
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        loaded.append(name)
        return module

    common = load("models.tt_transformers.tt.generator")
    monkeypatch.setitem(
        sys.modules,
        "models.tt_transformers.tt.generator_vllm",
        _module(
            "models.tt_transformers.tt.generator_vllm",
            HybridAttentionForCausalLM=type("HybridAttentionForCausalLM", (common.Generator,), {}),
            allocate_vllm_kv_cache=_unused,
        ),
    )
    gemma = load("models.demos.gemma4.tt.generator_vllm")
    yield SimpleNamespace(common=common, gemma=gemma, runtime=runtime)
    for name in reversed(loaded):
        sys.modules.pop(name, None)


@pytest.fixture
def expect_error():
    return pytest.raises  # allow-pytest.raises: root conftest requires hardware runtime


def _model(production, *, gemma=False):
    cls = production.gemma.Gemma4DFlashContractForCausalLM if gemma else production.common.Generator
    model = cls.__new__(cls)
    runtime = production.runtime
    sampling = SimpleNamespace(reset_trace=lambda: runtime.release("mesh-0", 5))
    production.common.Generator.__init__(
        model,
        [SimpleNamespace(sampling=sampling, mesh_device="mesh-0"), SimpleNamespace(mesh_device="mesh-1")],
        [SimpleNamespace(mesh_device="mesh-0"), SimpleNamespace(mesh_device="mesh-1")],
        "root-mesh",
    )
    model.data_parallel = 1
    model.trace_id_prefill.update({"1024_0": 1, "2048_1": 2, "4096_0": None})
    model.trace_id_prefill_sampling.update({"sampling_1024_0": 3, "2048_1": 4})
    model.trace_ids_decode.update({False: {0: 6, 1: 7}})
    model._bucket_trace_store = {1: (model.trace_ids_decode,), 2: ({True: {0: 8}},)}
    model.trace_ids = {0: 9}
    if gemma:
        model._contract_init()
        model._spec_width_set = True
        model._spec_decoder_bucket = 3072
        model._spec_active = True
        model._spec_active_owner = object()
        model._spec_pending = object()
        model._spec_pending_owner = object()
        model._spec_carry = [17]
        model._spec_decoder = SimpleNamespace(
            trace=30,
            _pv_widths={3072: {"trace": 30, "pv_iota": Buffer()}, 4096: {"trace": 31, "pv_iota": Buffer()}},
            ctx_k=[Buffer()],
            ctx_v=[Buffer()],
            out_ids=Buffer(),
        )
    return model


def test_common_cleanup_releases_each_trace_once_before_mesh_close(production):
    model = _model(production)
    model.release_persistent_capture()
    releases = production.runtime.events.copy()
    assert sorted(event[2] for event in releases) == list(range(1, 10))
    assert all(event[0] == "release" and event[3] is False for event in releases)
    model.release_persistent_capture()
    assert production.runtime.events == releases
    production.runtime.close("root-mesh")
    events = production.runtime.events.copy()
    model.__del__()
    assert production.runtime.events == events


def test_session_release_retains_widths_but_shutdown_releases_fused_and_ordinary(production):
    model = _model(production, gemma=True)
    decoder = model._spec_decoder
    model._spec_release_decoder()
    assert model._spec_decoder is decoder
    assert not model._spec_active
    assert production.runtime.events == []
    assert decoder.out_ids.releases == 0

    model.release_persistent_capture()
    assert model._spec_decoder is None
    assert model._spec_pending is None and model._spec_pending_owner is None and model._spec_carry == []
    traces = [event[2] for event in production.runtime.events]
    assert traces[:2] == [30, 31]
    assert sorted(traces) == list(range(1, 10)) + [30, 31]
    assert decoder.out_ids.releases == 1
    assert all(tensor.releases == 1 for tensor in decoder.ctx_k + decoder.ctx_v)
    assert all(record["pv_iota"].releases == 1 for record in decoder._pv_widths.values())
    before_close = production.runtime.events.copy()
    model.release_persistent_capture()
    assert production.runtime.events == before_close
    production.runtime.close("root-mesh")
    events = production.runtime.events.copy()
    model.__del__()
    assert production.runtime.events == events


def test_ordinary_cleanup_runs_when_fused_cleanup_raises(production, monkeypatch, expect_error):
    model = _model(production, gemma=True)

    def fail(*, teardown):
        assert teardown is True
        raise RuntimeError("injected fused cleanup failure")

    monkeypatch.setattr(model, "_spec_release_decoder", fail)
    with expect_error(RuntimeError, match="injected fused cleanup failure"):
        model.release_persistent_capture()
    assert sorted(event[2] for event in production.runtime.events) == list(range(1, 10))
    assert all(event[3] is False for event in production.runtime.events)


def test_explicit_trace_cleanup_leaves_dp_submesh_close_to_destructor(production):
    model = _model(production)
    model.data_parallel = 2
    model.release_persistent_capture()
    assert all(event[0] == "release" for event in production.runtime.events)
    model.__del__()
    assert production.runtime.events[-2:] == [("close", "mesh-0"), ("close", "mesh-1")]
    model.data_parallel = 1
