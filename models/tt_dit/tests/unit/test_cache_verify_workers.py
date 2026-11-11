# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-selectable CPU controls for bounded host verification."""
from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.utils import cache


def model(tensor):
    parameter = SimpleNamespace(data=tensor, on_host=True, device=None)
    return SimpleNamespace(named_children=lambda: (), named_parameters=lambda: [("weight", parameter)])


@pytest.mark.parametrize("workers", [1, 2, 4, 99])
def test_native_worker_admission_and_default(tmp_path, workers):
    tensor = cache.ttnn.from_torch(torch.ones(35, 67), dtype=cache.ttnn.bfloat8_b, layout=cache.ttnn.TILE_LAYOUT)
    cache.ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    stats = {}
    assert cache.verify_saved_model(model(tensor), tmp_path, workers=workers, stats=stats)
    assert stats["workers"] == min(workers, 4)
    assert stats["parallel_pairs"] == (0 if workers == 1 else 1)
    assert stats["peak_workspace_bytes"] <= 512 << 20


@pytest.mark.parametrize("dtype", ["bfloat16", "bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("layout", ["TILE_LAYOUT", "ROW_MAJOR_LAYOUT"])
def test_native_conversion_parity_with_padding_and_replication(tmp_path, dtype, layout):
    if dtype != "bfloat16" and layout == "ROW_MAJOR_LAYOUT":
        # Block float storage requires tile layout; exercise padded tiles instead.
        layout = "TILE_LAYOUT"
    ttnn = cache.ttnn
    values = torch.arange(35 * 67).float().reshape(35, 67) / 4096
    values[0, 0] = -0.0
    shard = ttnn.from_torch(values, dtype=getattr(ttnn, dtype), layout=getattr(ttnn, layout))
    assert torch.equal(shard.to_torch(), ttnn.to_torch(shard))
    tensor = ttnn.from_host_shards([shard] * 8, ttnn.MeshShape([1, 8]))
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    for workers in (1, 2, 4):
        stats = {}
        assert cache.verify_saved_model(model(tensor), tmp_path, workers=workers, stats=stats)
        assert stats["shards_completed"] == 8
        assert stats["parallel_pairs"] == (0 if workers == 1 else 8)
        assert stats["peak_inflight_pairs"] <= workers


@pytest.mark.parametrize(
    "mode", ["logging", "comparison", "trace", "graph", "io", "pre_hook", "post_hook", "wrapper", "build"]
)
def test_instrumentation_and_build_fallback(monkeypatch, tmp_path, mode):
    ttnn = cache.ttnn
    tensor = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16)
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    original = ttnn.to_torch
    if mode in ("logging", "comparison"):
        monkeypatch.setattr(
            ttnn,
            "CONFIG",
            SimpleNamespace(enable_logging=mode == "logging", enable_comparison_mode=mode == "comparison"),
        )
    elif mode == "trace":
        monkeypatch.setattr(ttnn.tracer, "ENABLE_TRACER", True)
    elif mode in ("graph", "io"):
        monkeypatch.setattr(
            ttnn.graph, "is_graph_capture_active" if mode == "graph" else "is_python_io_recording_enabled", lambda: True
        )
    elif mode in ("pre_hook", "post_hook"):
        monkeypatch.setattr(
            ttnn.decorators, "PRE_OPERATION_HOOKS" if mode == "pre_hook" else "POST_OPERATION_HOOKS", [lambda *a: None]
        )
    elif mode == "wrapper":
        monkeypatch.setattr(ttnn, "to_torch", lambda t: original(t))
    else:
        monkeypatch.setattr(cache, "_verification_native_build", lambda: False)
    assert cache._verification_workspace(tensor, tensor) is None
    # Restore real instrumentation before serial invocation; assert the routing separately
    # so this test needs no trace recorder, logging database or device context.
    monkeypatch.undo()
    monkeypatch.setattr(cache, "_verification_native_build", lambda: False)
    stats = {}
    assert cache.verify_saved_model(model(tensor), tmp_path, workers=4, stats=stats)
    assert stats["serial_pairs"] == 1 and stats["parallel_pairs"] == 0


@pytest.fixture
def fake(monkeypatch, tmp_path):
    import threading
    import weakref

    state = SimpleNamespace(caller=threading.get_ident(), snapshots=[], decoded=[], live=[], active=0, loads=0, frees=0)

    class Tensor:
        def __init__(self, value=1.0, cost=64, failure=None):
            self.value, self.cost, self.failure = value, cost, failure

        def cpu(self, *, blocking):
            assert blocking and threading.get_ident() == state.caller
            state.snapshots.append(self)
            if self.failure == "snapshot":
                raise KeyboardInterrupt("snapshot interrupted")
            return self

        def to_torch(self):
            state.active += 1
            try:
                state.decoded.append((self, threading.get_ident()))
                if self.failure == "worker":
                    raise ValueError("worker conversion")
                result = torch.tensor(self.value).clone()
                state.live.append(weakref.ref(result))
                return result
            finally:
                state.active -= 1

    state.Tensor = Tensor
    state.fresh = []

    def load(path, *, device):
        assert threading.get_ident() == state.caller and not state.active
        assert not any(r() is not None for r in state.live)
        state.loads += 1
        return state.fresh

    def deallocate(tensor):
        assert not state.active and not any(r() is not None for r in state.live)
        state.frees += 1

    monkeypatch.setattr(cache.ttnn, "load_tensor", load)
    monkeypatch.setattr(cache.ttnn, "get_device_tensors", lambda t: t)
    monkeypatch.setattr(cache.ttnn, "deallocate", deallocate)
    monkeypatch.setattr(cache.ttnn, "to_torch", lambda t: t.to_torch())
    monkeypatch.setattr(cache, "_verification_workspace", lambda a, b: a.cost)

    def make(resident, fresh=None):
        state.fresh = fresh if fresh is not None else resident
        (tmp_path / "weight.tensorbin").write_bytes(b"owned")
        m = model(resident)
        m.named_parameters()[0][1].on_host = False
        return m

    state.model = make
    return state


@pytest.mark.parametrize("case", ["value", "shape", "count", "order", "nan", "inf", "negative_inf", "signed_zero"])
def test_parallel_semantic_parity(fake, tmp_path, case):
    T = fake.Tensor
    a, b = [T([1.0, 2.0]), T([3.0, 4.0])], [T([1.0, 2.0]), T([3.0, 4.0])]
    if case == "value":
        b[1].value = [4.0, 3.0]
    elif case == "shape":
        b[1].value = [[3.0, 4.0]]
    elif case == "count":
        b.append(T(1.0))
    elif case == "order":
        b.reverse()
    elif case == "signed_zero":
        a[0].value, b[0].value = 0.0, -0.0
    else:
        a[0].value = b[0].value = float({"nan": "nan", "inf": "inf", "negative_inf": "-inf"}[case])
    expected = case == "signed_zero"
    for workers in (1, 2, 4):
        assert cache.verify_saved_model(fake.model(a, b), tmp_path, workers=workers) is expected
    assert fake.frees == 3


@pytest.mark.parametrize("cost,expected_parallel,oversized", [(64, 8, 0), (65, 8, 0), (129, 0, 8), (None, 0, 0)])
def test_memory_admission_and_oversized_fallback(fake, tmp_path, monkeypatch, cost, expected_parallel, oversized):
    monkeypatch.setattr(cache, "_VERIFY_WORKSPACE_BYTES", 128)
    stats = {}
    assert cache.verify_saved_model(fake.model([fake.Tensor(cost=cost)] * 8), tmp_path, workers=4, stats=stats)
    assert stats["parallel_pairs"] == expected_parallel
    assert stats["oversized_pairs"] == oversized
    assert stats["peak_workspace_bytes"] <= 128
    assert stats["peak_inflight_pairs"] <= (2 if cost == 64 else 1)
    assert stats["largest_serial_workspace_bytes"] == (129 if oversized else 0)
    assert len(fake.snapshots) == 2 * expected_parallel
    if not expected_parallel:
        assert all(tid == fake.caller for _, tid in fake.decoded)


def test_worker_exception_is_false_and_drain_precedes_deallocation(fake, tmp_path):
    stats = {}
    shards = [fake.Tensor(), fake.Tensor(failure="worker"), fake.Tensor(), fake.Tensor()]
    assert not cache.verify_saved_model(fake.model(shards), tmp_path, workers=2, stats=stats)
    assert fake.frees == 1 and stats["shards_completed"] == 3
    assert stats["parameters_failed"] == 1 and stats["completed"]


def test_snapshot_interrupt_drains_all_work(fake, tmp_path):
    stats = {}
    shards = [fake.Tensor(), fake.Tensor(failure="snapshot")]
    with pytest.raises(KeyboardInterrupt):  # allow-pytest.raises: CPU-only control without device conftest
        cache.verify_saved_model(fake.model(shards), tmp_path, workers=4, stats=stats)
    assert fake.frees == 1 and stats["shards_completed"] == 1
    assert not stats["completed"]


def test_cancelled_future_is_false(fake, tmp_path, monkeypatch):
    from concurrent.futures import Future

    class CancelledPool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args):
            f = Future()
            assert f.cancel()
            return f

        def shutdown(self, **kwargs):
            assert kwargs == dict(wait=True, cancel_futures=True)

    monkeypatch.setattr(cache, "ThreadPoolExecutor", CancelledPool)
    stats = {}
    assert not cache.verify_saved_model(fake.model([fake.Tensor()] * 4), tmp_path, workers=2, stats=stats)
    assert fake.frees == 1 and stats["shards_completed"] == 0


def test_one_pool_across_recursion_and_actual_concurrency(fake, tmp_path, monkeypatch):
    import threading

    barrier = threading.Barrier(4, timeout=5)
    original = cache._verification_worker
    real_pool = cache.ThreadPoolExecutor
    pools = []

    def worker(a, b):
        barrier.wait()
        return original(a, b)

    def executor(*args, **kwargs):
        p = real_pool(*args, **kwargs)
        pools.append(p)
        return p

    monkeypatch.setattr(cache, "_verification_worker", worker)
    monkeypatch.setattr(cache, "ThreadPoolExecutor", executor)
    child = fake.model([fake.Tensor()] * 8)
    (tmp_path / "child.weight.tensorbin").write_bytes(b"owned")
    root = fake.model([fake.Tensor()] * 8)
    root.named_children = lambda: [("child", child)]
    stats = {}
    assert cache.verify_saved_model(root, tmp_path, workers=4, stats=stats)
    assert len(pools) == 1 and stats["worker_threads"] == 4
    assert fake.frees == 2 and fake.loads == 2 and stats["shards_completed"] == 16
    assert not any(t.name.startswith("cache-verify") for t in threading.enumerate())


@pytest.mark.parametrize("setting,workers", [("junk", 1), ("0", 1), ("2", 2), ("999", 4)])
def test_worker_environment_is_bounded(fake, tmp_path, monkeypatch, setting, workers):
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY_WORKERS", setting)
    stats = {}
    assert cache.verify_saved_model(fake.model([fake.Tensor()]), tmp_path, stats=stats)
    assert stats["workers"] == workers


def test_interrupt_while_waiting_keeps_removed_future_alive(fake, tmp_path, monkeypatch):
    import threading

    release, done = threading.Event(), threading.Event()
    original_worker = cache._verification_worker
    original_pool = cache.ThreadPoolExecutor
    original_deallocate = cache.ttnn.deallocate

    def worker(a, b):
        assert release.wait(5)
        # Keep the removed future in flight long enough to expose premature freeing.
        threading.Event().wait(0.05)
        try:
            return original_worker(a, b)
        finally:
            done.set()

    def pool(*args, **kwargs):
        executor = original_pool(*args, **kwargs)
        submit = executor.submit

        def interrupting_submit(*args):
            future = submit(*args)
            original_result = future.result

            def result(*args, **kwargs):
                future.result = original_result
                release.set()
                raise KeyboardInterrupt("caller interrupted while waiting")

            future.result = result
            return future

        executor.submit = interrupting_submit
        return executor

    def deallocate(tensor):
        assert done.is_set(), "temporary freed while removed future still running"
        original_deallocate(tensor)

    monkeypatch.setattr(cache, "_verification_worker", worker)
    monkeypatch.setattr(cache, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(cache.ttnn, "deallocate", deallocate)
    with pytest.raises(KeyboardInterrupt):  # allow-pytest.raises: CPU-only interrupt control
        cache.verify_saved_model(fake.model([fake.Tensor()]), tmp_path, workers=2)
    assert done.is_set() and fake.frees == 1


def test_padding_only_change_and_unsupported_native_dtype(tmp_path):
    ttnn = cache.ttnn
    values = torch.ones(35, 67)
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        a = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, pad_value=0.0)
        b = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, pad_value=7.0)
        ttnn.dump_tensor(tmp_path / "weight.tensorbin", b)
        stats = {}
        assert cache.verify_saved_model(model(a), tmp_path, workers=2, stats=stats)
        assert stats["parallel_pairs"] == (0 if dtype == ttnn.float32 else 1)


def test_decorated_serial_fallback_retains_actual_hooks(tmp_path):
    import threading

    ttnn = cache.ttnn
    tensor = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16)
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    calls, stats = [], {}
    caller = threading.get_ident()

    def hook(operation, *args):
        if operation.python_fully_qualified_name == "ttnn.to_torch":
            calls.append(threading.get_ident())

    with ttnn.decorators.register_pre_operation_hook(hook):
        assert cache.verify_saved_model(model(tensor), tmp_path, workers=4, stats=stats)
    assert stats["parallel_pairs"] == 0 and stats["serial_pairs"] == 1
    assert calls == [caller, caller]
