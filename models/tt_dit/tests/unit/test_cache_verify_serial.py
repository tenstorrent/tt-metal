# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native CPU controls for the public, caller-thread verifier contract."""

import json
import threading
from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.utils import cache


def model(tensor):
    parameter = SimpleNamespace(data=tensor, on_host=True, device=None)
    return SimpleNamespace(named_children=lambda: (), named_parameters=lambda: [("weight", parameter)])


def test_operation_parameter_trace_keeps_both_caller_conversions(tmp_path, monkeypatch):
    ttnn = cache.ttnn
    tensor = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16)
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    tracer = ttnn.operation_tracer
    previous = tracer.is_tracing_enabled()
    original = tracer.serialize_operation_parameters
    caller, calls = threading.get_ident(), []
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("TTNN_OPERATION_TRACE_DIR", str(trace_dir))
    # A stale experiment setting must never bypass public instrumentation.
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY_WORKERS", "4")

    def serialize(operation, *args, **kwargs):
        if operation == "ttnn.to_torch":
            calls.append(threading.get_ident())
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(tracer, "serialize_operation_parameters", serialize)
    try:
        with ttnn.manage_config("enable_fast_runtime_mode", False):
            tracer.enable_tracing(True)
            assert cache.verify_saved_model(model(tensor), tmp_path)
    finally:
        tracer.enable_tracing(previous)
    assert tracer.is_tracing_enabled() == previous
    assert calls == [caller, caller]
    records = [json.loads(path.read_text()) for path in trace_dir.glob("*ttnn_to_torch*.json")]
    assert len(records) == 2
    assert all("return_value" in record for record in records)


def test_portable_native_verification_needs_no_binary_fingerprint(tmp_path, monkeypatch):
    tensor = cache.ttnn.from_torch(torch.ones(32, 32), dtype=cache.ttnn.bfloat16)
    cache.ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY_WORKERS", "4")

    def forbidden(*args, **kwargs):
        pytest.fail("native correctness must not inspect private binary hashes")

    monkeypatch.setattr(cache, "hashlib", SimpleNamespace(sha256=forbidden))
    stats = {}
    assert cache.verify_saved_model(model(tensor), tmp_path, stats=stats)
    assert stats["shards_completed"] == 1


@pytest.mark.parametrize(
    "dtype,layout",
    [
        ("bfloat16", "ROW_MAJOR_LAYOUT"),
        ("bfloat16", "TILE_LAYOUT"),
        ("bfloat8_b", "TILE_LAYOUT"),
        ("bfloat4_b", "TILE_LAYOUT"),
        ("float32", "TILE_LAYOUT"),
    ],
)
def test_native_padding_and_replicated_logical_coordinates(tmp_path, dtype, layout):
    ttnn = cache.ttnn
    values = torch.arange(35 * 67).float().reshape(35, 67) / 4096
    values[0, 0] = -0.0
    options = dict(dtype=getattr(ttnn, dtype), layout=getattr(ttnn, layout))
    a = ttnn.from_torch(values, pad_value=0.0, **options)
    b = ttnn.from_torch(values, pad_value=7.0, **options)
    resident = ttnn.from_host_shards([a] * 8, ttnn.MeshShape([1, 8]))
    fresh = ttnn.from_host_shards([b] * 8, ttnn.MeshShape([1, 8]))
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", fresh)
    stats = {}
    assert cache.verify_saved_model(model(resident), tmp_path, stats=stats)
    assert stats["shards_completed"] == 8


@pytest.mark.parametrize("case", ["value", "shape", "order", "nan", "inf", "negative_inf"])
def test_native_invalid_tensorbins(tmp_path, case):
    ttnn = cache.ttnn
    a = [torch.ones(32, 32), torch.full((32, 32), 2.0)]
    b = [x.clone() for x in a]
    if case == "value":
        b[1][0, 0] = 3
    elif case == "shape":
        b = [x.reshape(16, 64) for x in b]
    elif case == "order":
        b.reverse()
    else:
        a[1][0, 0] = b[1][0, 0] = float({"nan": "nan", "inf": "inf", "negative_inf": "-inf"}[case])

    def shards(values):
        return ttnn.from_host_shards([ttnn.from_torch(x, dtype=ttnn.bfloat16) for x in values], ttnn.MeshShape([1, 2]))

    ttnn.dump_tensor(tmp_path / "weight.tensorbin", shards(b))
    assert not cache.verify_saved_model(model(shards(a)), tmp_path)


def test_decorated_conversions_retain_actual_hooks(tmp_path):
    ttnn = cache.ttnn
    tensor = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16)
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    calls, caller = [], threading.get_ident()

    def hook(operation, *args):
        if operation.python_fully_qualified_name == "ttnn.to_torch":
            calls.append(threading.get_ident())

    with ttnn.manage_config("enable_fast_runtime_mode", False), ttnn.decorators.register_pre_operation_hook(hook):
        assert cache.verify_saved_model(model(tensor), tmp_path)
    assert calls == [caller, caller]
