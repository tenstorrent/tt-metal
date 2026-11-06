# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contracts for the Llama shared prefill adapter and runtime."""

from __future__ import annotations

import subprocess
import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

MODEL_PREFIX = "models.demos.llama_3p1_8b_d_p.tt."


def _params(**overrides):
    values = dict(
        mesh_shape=(4, 8),
        num_layers=32,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=2048,
        chunk_size=1024,
        num_users=2,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    values.update(overrides)
    return PrefillRunParams(**values)


class _Tensor:
    def __init__(self, address, events=None):
        self._address = address
        self._events = events

    def buffer_address(self):
        return self._address

    def deallocate(self, force):
        if self._events is not None:
            self._events.append(("free", force))


class _Input:
    def __init__(self, token_ids, bounds, events):
        self.token_ids = token_ids
        self.bounds = bounds
        self._events = events

    def deallocate(self, force):
        self._events.append(("free_input", force))


class _Cache(SimpleNamespace):
    def __init__(self, events=None, num_users=2):
        super().__init__(
            num_users=num_users,
            num_layers=32,
            max_seq_len=2048,
            sp=4,
            k=_Tensor(0x10000, events),
            v=_Tensor(0x20000, events),
        )
        self.populated_ends = {slot: 0 for slot in range(num_users)}

    def record_write(self, slot, actual_start, actual_end):
        if actual_start <= self.populated_ends[slot]:
            self.populated_ends[slot] = actual_end

    def truncate_prefix(self, slot, actual_start):
        self.populated_ends[slot] = min(self.populated_ends[slot], actual_start)


class _Model:
    num_layers = 32
    max_seq_len = 2048

    def __init__(self, events, failure=None, num_users=2):
        self.events = events
        self.failure = failure
        self.num_users = num_users

    def prefill_chunk(self, tokens, cache, **kwargs):
        self.events.append(("forward", tokens, cache, kwargs))
        if self.failure == "forward":
            raise RuntimeError("forward failed")
        cache.record_write(kwargs["slot_idx"], kwargs["actual_start"], kwargs["actual_end"])
        return _Tensor(0, self.events)


def _cache(events=None, num_users=2):
    return _Cache(events, num_users)


def _runtime(*, failure=None, num_users=2):
    events = []
    uploads = []

    def upload(token_ids, **bounds):
        tensor = _Input(token_ids, bounds, events)
        uploads.append(tensor)
        return tensor

    def synchronize(mesh):
        events.append(("sync", mesh))
        if failure == "sync":
            raise RuntimeError("sync failed")

    runtime = TtPrefillRuntime(
        "mesh",
        config=TtPrefillRuntimeConfig(max_seq_len=2048, chunk_size=1024, num_users=num_users),
        model=_Model(events, failure, num_users),
        synchronize=synchronize,
        upload=upload,
    )
    runtime.compiled = True
    return runtime, _cache(events, num_users), events, uploads


# Resolve the registered adapter in a clean interpreter and prove that producer-side imports stay device-free.
def test_adapter_registry_import_is_light():
    """Loading the registered adapter must not import model/device frameworks."""
    program = (
        "import sys; from types import SimpleNamespace; "
        "sys.modules['loguru']=SimpleNamespace(logger=SimpleNamespace(info=lambda *a: None)); "
        "from models.demos.common.prefill.adapter import get_adapter; "
        "assert get_adapter('llama_3p1_8b').name == 'llama_3p1_8b'; "
        "assert not set(sys.modules) & {'torch','ttnn','transformers','safetensors'}"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# Forward the engine-resolved SP4/TP8, two-slot, 2K geometry into the external cache allocator.
def test_adapter_allocates_the_runner_resolved_two_slot_cache():
    """Allocation must preserve the engine-owned SP4/TP8, two-slot, 2K geometry."""
    captured = {}
    allocated = object()
    dtype = object()

    def allocate(mesh, mesh_config, **kwargs):
        captured.update(mesh=mesh, mesh_config=mesh_config, **kwargs)
        return allocated

    stubs = {
        "ttnn": SimpleNamespace(bfloat8_b=dtype),
        MODEL_PREFIX + "config": SimpleNamespace(MeshConfig=lambda shape, tp: (tuple(shape), tp)),
        MODEL_PREFIX + "kv_cache": SimpleNamespace(allocate_kv_cache=allocate),
    }
    with patch.dict(sys.modules, stubs):
        result = get_adapter("llama_3p1_8b").allocate_kv_cache(mesh_device="mesh", hf_config=None, params=_params())

    assert result is allocated
    assert captured == {
        "mesh": "mesh",
        "mesh_config": ((4, 8), 8),
        "num_users": 2,
        "num_layers": 32,
        "max_seq_len": 2048,
        "cache_dtype": dtype,
        "bank_aligned": True,
    }


# A deployment's PREFILL_NUM_USERS reaches the allocator and the model; nothing pins the count to two.
@pytest.mark.parametrize("num_users", [1, 4, 16])
def test_runner_slot_count_reaches_the_allocator_and_the_model(num_users):
    captured = {}

    class Model:
        def __init__(self, mesh, checkpoint, **kwargs):
            captured["model"] = kwargs
            self.num_layers = kwargs["num_layers"]
            self.max_seq_len = kwargs["max_seq_len"]
            self.num_users = kwargs["num_users"]

    def allocate(mesh, mesh_config, **kwargs):
        captured["cache"] = kwargs

    stubs = {
        "ttnn": SimpleNamespace(bfloat8_b="bf8", synchronize_device=None),
        MODEL_PREFIX + "config": SimpleNamespace(MeshConfig=lambda shape, tp: (tuple(shape), tp)),
        MODEL_PREFIX + "kv_cache": SimpleNamespace(allocate_kv_cache=allocate),
        MODEL_PREFIX + "input": SimpleNamespace(upload_token_chunk=None),
        MODEL_PREFIX + "model": SimpleNamespace(PrefillModel=Model),
    }
    adapter = get_adapter("llama_3p1_8b")
    with patch.dict(sys.modules, stubs), patch(MODEL_PREFIX + "model_config.resolve_weights_path", lambda: "ckpt"):
        adapter.allocate_kv_cache(mesh_device="mesh", hf_config=None, params=_params(num_users=num_users))
        runtime = adapter.build_runtime(mesh_device="mesh", hf_config=None, params=_params(num_users=num_users))

    assert captured["cache"]["num_users"] == num_users
    assert captured["model"]["num_users"] == num_users
    assert runtime.config.num_users == runtime.geometry.num_users == num_users


# The runtime validates the cache against its configured count and serves the top slot of a wider cache.
def test_runtime_serves_the_top_slot_of_a_wider_cache_and_rejects_a_mismatched_one(expect_error):
    runtime, cache, events, _ = _runtime(num_users=4)
    runtime.set_layer_completion_sink(_Sink(events))

    runtime.prefill_chunk(object(), cache, slot_id=3, actual_start=0, actual_end=1024, request_id=1)
    assert events[-1] == ("ack", 0, 32, 3, 0, 1024)

    with expect_error(ValueError, "outside the allocated cache"):
        runtime.prefill_chunk(object(), cache, slot_id=4, actual_start=0, actual_end=1024, request_id=2)
    with expect_error(ValueError, "num_users=4"):
        runtime.prefill_chunk(object(), _cache(num_users=2), slot_id=0, actual_start=0, actual_end=1024, request_id=3)
    with expect_error(ValueError, "slot count differs"):
        TtPrefillRuntime(
            "mesh",
            config=TtPrefillRuntimeConfig(max_seq_len=2048, chunk_size=1024, num_users=4),
            model=_Model([], num_users=2),
            synchronize=None,
            upload=None,
        )


class _Sink:
    def __init__(self, events):
        self._events = events

    def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end):
        self._events.append(("ack", layer_start, layer_end, slot_id, pos_start, pos_end))


# Serve a two-chunk continuation and an independent slot while preserving the runner-owned tensors and cache.
def test_runtime_borrows_inputs_and_cache_for_two_distinct_slots_and_chunks():
    runtime, cache, events, uploads = _runtime()
    runtime.set_layer_completion_sink(_Sink(events))

    for request, slot, start in ((7, 0, 0), (8, 0, 1024), (9, 1, 0)):
        token_ids = [request] * 1024
        input_tensor = runtime.make_chunk_input(token_ids, actual_start=start)
        assert uploads[-1].token_ids is token_ids
        assert uploads[-1].bounds == {"actual_start": start, "actual_end": start + 1024}

        upload_count = len(uploads)
        begin = len(events)
        assert (
            runtime.prefill_chunk(
                input_tensor,
                cache,
                slot_id=slot,
                actual_start=start,
                actual_end=start + 1024,
                request_id=request,
                metadata_msg=object(),
            )
            is None
        )
        assert len(uploads) == upload_count

        chunk_events = events[begin:]
        assert chunk_events[0] == (
            "forward",
            input_tensor,
            cache,
            {"slot_idx": slot, "actual_start": start, "actual_end": start + 1024, "skip_lm_head": True},
        )
        assert chunk_events[1:3] == [("sync", "mesh"), ("free", True)]
        assert chunk_events[3:] == [("ack", 0, 32, slot, start, start + 1024)]


# A model or synchronization failure must emit no residency acknowledgements and poison the worker.
@pytest.mark.parametrize("failure", ["forward", "sync"])
def test_runtime_failure_emits_no_layer_acknowledgements(failure, expect_error):
    """A failed write or device wait must poison the runtime without certifying a layer."""
    runtime, cache, events, _ = _runtime(failure=failure)
    runtime.set_layer_completion_sink(_Sink(events))

    with expect_error(RuntimeError, f"{failure} failed"):
        runtime.prefill_chunk(object(), cache, slot_id=1, actual_start=32, actual_end=65, request_id=7)

    assert not [event for event in events if event[0] == "ack"]
    event_count = len(events)
    with expect_error(RuntimeError, "runtime failed"):
        runtime.prefill_chunk(object(), cache, slot_id=0, actual_start=0, actual_end=32, request_id=8)
    assert len(events) == event_count


# Export eight K then eight V stage anchors and forward the exact gathered layout to the table builder.
def test_runtime_exposes_all_sixteen_kv_migration_stages_and_table_hook():
    """The common runner must receive eight K then eight V stages and the production table builder."""
    runtime, cache, _, _ = _runtime()
    stage_type = namedtuple("KvCacheStage", "base_addr first_layer count")
    captured = {}

    def serialize(**kwargs):
        captured.update(kwargs)
        return kwargs["path"]

    with patch.dict(
        sys.modules,
        {
            "models.demos.common.prefill.runners.migration": SimpleNamespace(KvCacheStage=stage_type),
            MODEL_PREFIX + "runners.kv_chunk_table": SimpleNamespace(build_and_serialize_kv_chunk_table=serialize),
        },
    ):
        stages = runtime.kv_migration_stages(cache)
        assert [stage.base_addr for stage in stages] == [0x10000] * 8 + [0x20000] * 8
        assert [(stage.first_layer, stage.count) for stage in stages] == [(0, 32)] * 16

        gathered = [
            [{"base_addr": stage.base_addr, "first_layer": stage.first_layer, "count": stage.count}] for stage in stages
        ]
        assert runtime.build_kv_chunk_table(cache, "table.pb", stage_layouts=gathered) == "table.pb"

    assert captured == {
        "mesh_device": "mesh",
        "kv_cache": cache,
        "chunk_size": 1024,
        "path": "table.pb",
    }


# Compilation writes both program variants into slot zero, then clears only the cache's logical-prefix bookkeeping.
def test_runtime_compile_does_not_publish_warmup_tokens_as_an_external_cache_prefix():
    runtime, cache, events, _ = _runtime()
    runtime.compiled = False

    runtime.compile(cache)

    assert runtime.compiled
    assert cache.populated_ends == {0: 0, 1: 0}
    assert [(event[3]["actual_start"], event[3]["actual_end"]) for event in events if event[0] == "forward"] == [
        (0, 1024),
        (1024, 2048),
    ]
