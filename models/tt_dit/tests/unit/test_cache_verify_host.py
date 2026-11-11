# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only verifier controls: no device fixture, broker, or cache reclamation."""

import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.utils import cache


class Model:
    def __init__(self, parameters=None, children=None):
        self.parameters = parameters or {}
        self.children = children or {}

    def named_children(self):
        return self.children.items()

    def named_parameters(self):
        return self.parameters.items()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = SimpleNamespace(files={}, events=[], refs=[], peak=0, caller=threading.get_ident())

    def event(kind, value):
        assert threading.get_ident() == state.caller
        state.events.append((kind, value))

    def load(path, *, device):
        event("load", device)
        item = state.files[path.name]
        if isinstance(item, BaseException):
            raise item
        return item

    def decode(tensor):
        event("decode", tensor)
        if isinstance(tensor, BaseException):
            raise tensor
        result = tensor.clone()
        state.refs.append(weakref.ref(result))
        state.peak = max(state.peak, sum(r() is not None for r in state.refs))
        return result

    def deallocate(tensor):
        assert all(r() is None for r in state.refs), "decode outlived temporary device tensor"
        event("deallocate", tensor)

    monkeypatch.setattr(cache.ttnn, "load_tensor", load)
    monkeypatch.setattr(cache.ttnn, "get_device_tensors", lambda tensors: tensors)
    monkeypatch.setattr(cache.ttnn, "to_torch", decode)
    monkeypatch.setattr(cache.ttnn, "deallocate", deallocate)

    def model(resident, fresh=None, *, on_host=True, name="weight"):
        state.files[name + ".tensorbin"] = resident if fresh is None else fresh
        (tmp_path / (name + ".tensorbin")).write_bytes(b"owned fixture")
        return Model({name: SimpleNamespace(data=resident, on_host=on_host, device="fake-device")})

    state.model = model
    return state


def test_decoded_storage_is_bounded_to_one_pair(fake, tmp_path):
    model = fake.model([torch.ones(32, 32) for _ in range(8)])
    assert cache.verify_saved_model(model, tmp_path)
    assert fake.peak <= 2
    assert all(r() is None for r in fake.refs)


@pytest.mark.parametrize("case", ["value", "shape", "count", "order", "nan", "inf", "negative_inf"])
def test_invalid_logical_shards_are_rejected(fake, tmp_path, case):
    a = [torch.ones(2, 3), torch.full((2, 3), 2.0)]
    b = [x.clone() for x in a]
    if case == "value":
        b[1][0, 1] = 3
    elif case == "shape":
        b[1] = b[1].reshape(3, 2)
    elif case == "count":
        b.append(b[-1])
    elif case == "order":
        b.reverse()
    else:
        a[1][0, 0] = b[1][0, 0] = float({"nan": "nan", "inf": "inf", "negative_inf": "-inf"}[case])
    assert not cache.verify_saved_model(fake.model(a, b), tmp_path)


def test_logical_equality_keeps_signed_zero_and_dtype_semantics(fake, tmp_path):
    a = [torch.tensor([0.0, 1.0], dtype=torch.float32)]
    b = [torch.tensor([-0.0, 1.0], dtype=torch.float64)]
    assert cache.verify_saved_model(fake.model(a, b), tmp_path)


@pytest.mark.parametrize("failure", ["load", "resident", "fresh"])
def test_errors_release_temporaries_and_leave_resident_alone(fake, tmp_path, failure):
    a, b = [torch.ones(2, 3)], [torch.ones(2, 3)]
    if failure == "resident":
        a = [RuntimeError("resident readback")]
    if failure == "fresh":
        b = [RuntimeError("fresh readback")]
    model = fake.model(a, b, on_host=False)
    if failure == "load":
        fake.files["weight.tensorbin"] = RuntimeError("load or upload")
    assert not cache.verify_saved_model(model, tmp_path)
    frees = [value for kind, value in fake.events if kind == "deallocate"]
    assert len(frees) == (0 if failure == "load" else 1)
    if frees:
        assert frees[0] is b
    assert model.parameters["weight"].data is a


def test_device_target_and_independent_readbacks_are_preserved(fake, tmp_path):
    a, b = [torch.ones(2, 3)], [torch.zeros(2, 3)]
    assert not cache.verify_saved_model(fake.model(a, b, on_host=False), tmp_path)
    assert fake.events[0] == ("load", "fake-device")
    decoded = [value for kind, value in fake.events if kind == "decode"]
    assert len(decoded) == 2 and decoded[0] is a[0] and decoded[1] is b[0]


def test_replication_visits_every_logical_coordinate(fake, tmp_path):
    a, b = torch.ones(2, 3), torch.ones(2, 3)
    assert cache.verify_saved_model(fake.model([a] * 8, [b] * 8), tmp_path)
    assert sum(kind == "decode" for kind, _ in fake.events) == 16


def test_missing_file_is_false_without_load(fake, tmp_path):
    model = fake.model([torch.ones(2, 3)])
    (tmp_path / "weight.tensorbin").unlink()
    assert not cache.verify_saved_model(model, tmp_path)
    assert not fake.events


def test_recursive_progress_is_one_aggregate_and_no_double_walltime(fake, tmp_path, monkeypatch):
    child = fake.model([torch.ones(2, 3)] * 3, name="child.weight")
    child.parameters["weight"] = child.parameters.pop("child.weight")
    root = fake.model([torch.ones(2, 3)])
    root.children["child"] = child
    records = []
    monkeypatch.setattr(cache.walltime, "record", lambda *a, **kw: records.append(a))
    stats = {}
    assert cache.verify_saved_model(root, tmp_path, stats=stats)
    assert stats["parameters_completed"] == 2
    assert stats["parameters_failed"] == 0
    assert stats["shards_completed"] == 4
    assert stats["tensorbin_bytes"] == 26
    assert stats["wall_s"] >= stats["load_s"] + stats["convert_s"] + stats["compare_s"]
    assert not records


def test_completed_progress_reports_at_bounded_cadence(fake, tmp_path, monkeypatch):
    messages = []
    now = [0.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
    real_decode = cache.ttnn.to_torch

    def slow_decode(t):
        now[0] += 3
        return real_decode(t)

    monkeypatch.setattr(cache.ttnn, "to_torch", slow_decode)
    monkeypatch.setattr(cache.logger, "info", lambda message: messages.append(message))
    assert cache.verify_saved_model(fake.model([torch.ones(2, 3)] * 3), tmp_path)
    progress = [m for m in messages if m.startswith("cache verify progress:")]
    assert len(progress) == 3
    assert '"shards_completed": 3' in progress[-1]
    assert len([m for m in messages if m.startswith("cache verify summary:")]) == 1


@pytest.mark.parametrize("dtype_name", ["bfloat16", "bfloat8_b", "bfloat4_b"])
def test_native_host_tensorbin_roundtrip(tmp_path, dtype_name):
    ttnn = cache.ttnn
    tensor = ttnn.from_torch(
        torch.arange(35 * 67).float().reshape(35, 67) / 4096, dtype=getattr(ttnn, dtype_name), layout=ttnn.TILE_LAYOUT
    )
    ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    model = Model({"weight": SimpleNamespace(data=tensor, on_host=True, device=None)})
    assert cache.verify_saved_model(model, tmp_path)
    assert ttnn.to_torch(tensor).shape == (35, 67)


def test_failure_does_not_skip_later_parameters(fake, tmp_path):
    model = fake.model([torch.ones(2, 3)], RuntimeError("bad first tensor"))
    other = fake.model([torch.ones(2, 3)], name="next")
    model.parameters.update(other.parameters)
    stats = {}
    assert not cache.verify_saved_model(model, tmp_path, stats=stats)
    assert stats["parameters_completed"] == 2 and stats["parameters_failed"] == 1
    assert stats["shards_completed"] == 1 and stats["completed"]


def test_interrupt_releases_fresh_tensor_and_marks_incomplete(fake, tmp_path):
    model = fake.model([torch.ones(2, 3)], [KeyboardInterrupt()], on_host=False)
    stats = {}
    try:
        cache.verify_saved_model(model, tmp_path, stats=stats)
    except KeyboardInterrupt:
        pass
    else:
        pytest.fail("interrupt was swallowed")
    assert not stats["completed"] and stats["shards_completed"] == 0
    assert sum(kind == "deallocate" for kind, _ in fake.events) == 1


def test_failed_publication_keeps_unrelated_leaf_and_cleans_staging(fake, tmp_path, monkeypatch):
    model = fake.model([torch.ones(2, 3)], [torch.zeros(2, 3)])
    sentinel = tmp_path / "unrelated" / "weight.tensorbin"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"unrelated model stays intact")
    model.save = lambda directory: (directory / "weight.tensorbin").write_bytes(b"owned fixture")
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY", "1")
    cache._publish_cache(model, tmp_path / "candidate", None)
    assert sentinel.read_bytes() == b"unrelated model stays intact"
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate.staging-*"))


@pytest.mark.parametrize("error_type", [OSError, ValueError, TypeError])
def test_unreadable_or_invalid_tensor_returns_false(fake, tmp_path, error_type):
    model = fake.model([torch.ones(2, 3)], error_type("bad tensor"))
    assert not cache.verify_saved_model(model, tmp_path)


@pytest.mark.parametrize("damage", ["truncated", "garbage"])
def test_native_bad_files_in_a_bounded_child(tmp_path, damage):
    # Native parser faults must never take down a publishing process in this test.
    import subprocess
    import sys

    source = cache.ttnn.from_torch(torch.ones(32, 64), dtype=cache.ttnn.bfloat16)
    path = tmp_path / "weight.tensorbin"
    cache.ttnn.dump_tensor(path, source)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2] if damage == "truncated" else b"not a tensorbin")
    script = """
import sys
from types import SimpleNamespace
import torch
from models.tt_dit.utils import cache
resident = cache.ttnn.from_torch(torch.ones(32,64), dtype=cache.ttnn.bfloat16)
p = SimpleNamespace(data=resident, on_host=True, device=None)
m = SimpleNamespace(named_children=lambda: (), named_parameters=lambda: [("weight", p)])
sys.exit(0 if cache.verify_saved_model(m, sys.argv[1]) is False else 1)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
