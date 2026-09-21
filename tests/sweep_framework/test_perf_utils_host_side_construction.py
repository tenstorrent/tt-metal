# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""host_side_tensor_construction() must route device-bound tensor setup through the host
and restore ttnn untouched, so device-perf measurement sees only the op under test."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import ttnn  # noqa: E402
from sweep_utils.perf_utils import host_side_tensor_construction  # noqa: E402


@pytest.fixture
def fake_ttnn(monkeypatch):
    calls = []

    def from_torch(tensor, dtype=None, **kwargs):
        calls.append(("from_torch", kwargs.get("device"), kwargs.get("memory_config")))
        return types.SimpleNamespace(kind="host", memory_config=kwargs.get("memory_config"))

    def to_device(tensor, device, memory_config=None, **kwargs):
        calls.append(("to_device", device, memory_config, kwargs.get("queue_id")))
        return types.SimpleNamespace(kind="device", device=lambda: device, memory_config=memory_config)

    def to_memory_config(tensor, memory_config, dtype=None, **kwargs):
        calls.append(("to_memory_config", memory_config, dtype))
        return types.SimpleNamespace(kind="device", memory_config=memory_config)

    def from_device(tensor, **kwargs):
        calls.append(("from_device",))
        return types.SimpleNamespace(kind="host")

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "to_device", to_device)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "from_device", from_device)
    monkeypatch.setattr(ttnn, "is_tensor_storage_on_device", lambda t: getattr(t, "kind", None) == "device")
    monkeypatch.delenv("SWEEPS_DEVICE_SIDE_SETUP", raising=False)
    return calls


def test_from_torch_with_device_builds_on_host_then_writes(fake_ttnn):
    dev, mc = object(), object()
    with host_side_tensor_construction():
        out = ttnn.from_torch("t", "bf16", layout="TILE", device=dev, memory_config=mc, cq_id=1)
    assert out.kind == "device" and out.memory_config is mc
    assert fake_ttnn == [("from_torch", None, None), ("to_device", dev, mc, 1)]


def test_from_torch_without_device_is_untouched(fake_ttnn):
    with host_side_tensor_construction():
        out = ttnn.from_torch("t", "bf16", layout="TILE")
    assert out.kind == "host"
    assert fake_ttnn == [("from_torch", None, None)]


def test_from_torch_spec_carries_memory_config(fake_ttnn):
    dev, mc = object(), object()
    spec = types.SimpleNamespace(memory_config=mc)
    with host_side_tensor_construction():
        ttnn.from_torch("t", spec=spec, device=dev)
    assert fake_ttnn[-1] == ("to_device", dev, mc, None)


def test_to_memory_config_on_device_tensor_round_trips_through_host(fake_ttnn):
    dev, mc = object(), object()
    device_tensor = types.SimpleNamespace(kind="device", device=lambda: dev)
    with host_side_tensor_construction():
        out = ttnn.to_memory_config(device_tensor, mc)
    assert out.memory_config is mc
    assert fake_ttnn == [("from_device",), ("to_device", dev, mc, None)]


def test_to_memory_config_with_dtype_or_host_tensor_falls_through(fake_ttnn):
    device_tensor = types.SimpleNamespace(kind="device", device=lambda: object())
    host_tensor = types.SimpleNamespace(kind="host")
    with host_side_tensor_construction():
        ttnn.to_memory_config(device_tensor, "mc", "bf8")
        ttnn.to_memory_config(host_tensor, "mc")
    assert fake_ttnn == [("to_memory_config", "mc", "bf8"), ("to_memory_config", "mc", None)]


def test_originals_restored_after_exit_and_on_error(fake_ttnn):
    orig = (ttnn.from_torch, ttnn.to_memory_config)
    with host_side_tensor_construction():
        assert ttnn.from_torch is not orig[0]
    assert (ttnn.from_torch, ttnn.to_memory_config) == orig
    with pytest.raises(RuntimeError):  # allow-pytest.raises: plain Python error, no device error text for CI triage
        with host_side_tensor_construction():
            raise RuntimeError("boom")
    assert (ttnn.from_torch, ttnn.to_memory_config) == orig


def test_opt_out_env_leaves_ttnn_untouched(fake_ttnn, monkeypatch):
    monkeypatch.setenv("SWEEPS_DEVICE_SIDE_SETUP", "1")
    orig = ttnn.from_torch
    with host_side_tensor_construction():
        assert ttnn.from_torch is orig
