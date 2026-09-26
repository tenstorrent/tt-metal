# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""sweep_utils.tensor_setup routes device-bound tensor setup through the host while device perf
is measured, leaves the measuring bracket and every opt-out alone, and restores ttnn on exit.
Host-only: ttnn is imported for its real signatures but every device call is faked."""

import inspect
import os
import sys
import types
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ttnn  # noqa: E402
from tests.ttnn import utils_for_testing  # noqa: E402
from sweep_utils import tensor_setup  # noqa: E402
from sweep_utils.tensor_setup import host_side_tensor_construction, setup_context  # noqa: E402


@pytest.fixture
def fake_ttnn(monkeypatch):
    calls = []

    def from_torch(tensor, dtype=None, **kwargs):
        calls.append(("from_torch", tensor, dtype, dict(kwargs)))
        return types.SimpleNamespace(kind="host", memory_config=kwargs.get("memory_config"))

    def to_device(tensor, device, memory_config=None, **kwargs):
        calls.append(("to_device", device, memory_config, kwargs.get("queue_id")))
        return types.SimpleNamespace(kind="device", device=lambda: device, memory_config=memory_config)

    def to_memory_config(tensor, memory_config, dtype=None, **kwargs):
        calls.append(("to_memory_config", memory_config, dtype, dict(kwargs)))
        return types.SimpleNamespace(kind="device", memory_config=memory_config)

    def from_device(tensor, **kwargs):
        calls.append(("from_device",))
        return types.SimpleNamespace(kind="host")

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "to_device", to_device)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "from_device", from_device)
    monkeypatch.setattr(ttnn, "is_tensor_storage_on_device", lambda t: getattr(t, "kind", None) == "device")
    monkeypatch.setattr(tensor_setup, "_announced", True)
    return calls


def device_tensor(dev=None):
    return types.SimpleNamespace(kind="device", device=lambda: dev)


# --- from_torch ---------------------------------------------------------------------------


def test_from_torch_with_device_builds_on_host_and_forwards_every_other_kwarg(fake_ttnn):
    dev, mc, mapper = object(), object(), object()
    with host_side_tensor_construction():
        out = ttnn.from_torch("t", "bf16", layout="TILE", device=dev, memory_config=mc, cq_id=1, mesh_mapper=mapper)
    assert out.kind == "device" and out.memory_config is mc
    assert fake_ttnn == [
        ("from_torch", "t", "bf16", {"layout": "TILE", "mesh_mapper": mapper}),
        ("to_device", dev, mc, 1),
    ]


def test_from_torch_without_device_is_untouched(fake_ttnn):
    with host_side_tensor_construction():
        out = ttnn.from_torch("t", "bf16", layout="TILE")
    assert out.kind == "host"
    assert fake_ttnn == [("from_torch", "t", "bf16", {"layout": "TILE"})]


def test_from_torch_none_tensor_passes_through_without_a_write(fake_ttnn):
    with host_side_tensor_construction():
        ttnn.from_torch(None, device=object())
    assert [c[0] for c in fake_ttnn] == ["from_torch"]


def test_from_torch_spec_supplies_the_placement(fake_ttnn):
    dev, mc = object(), object()
    spec = types.SimpleNamespace(memory_config=mc)
    with host_side_tensor_construction():
        ttnn.from_torch("t", spec=spec, device=dev)
    assert fake_ttnn == [("from_torch", "t", None, {"spec": spec}), ("to_device", dev, mc, None)]


def test_from_torch_spec_with_memory_config_is_forwarded_for_from_torch_to_reject(fake_ttnn):
    spec, mc = types.SimpleNamespace(memory_config=object()), object()
    with host_side_tensor_construction():
        ttnn.from_torch("t", spec=spec, memory_config=mc, device=object())
    assert fake_ttnn[0][3] == {"spec": spec, "memory_config": mc}


# --- to_memory_config -----------------------------------------------------------------------


def test_to_memory_config_on_device_tensor_round_trips_through_host(fake_ttnn):
    dev, mc = object(), object()
    with host_side_tensor_construction():
        out = ttnn.to_memory_config(device_tensor(dev), mc)
    assert out.memory_config is mc
    assert fake_ttnn == [("from_device",), ("to_device", dev, mc, None)]


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((device_tensor(), "mc", "bf8"), {}),
        ((device_tensor(), "mc"), {"output_tensor": "preallocated"}),
        ((types.SimpleNamespace(kind="host"), "mc"), {}),
    ],
    ids=["dtype-change", "output_tensor", "host-tensor"],
)
def test_to_memory_config_falls_through_when_the_round_trip_cannot_reproduce_it(fake_ttnn, args, kwargs):
    with host_side_tensor_construction():
        ttnn.to_memory_config(*args, **kwargs)
    assert fake_ttnn[0][0] == "to_memory_config"


def test_to_memory_config_inside_the_measuring_bracket_runs_on_device_while_from_torch_stays_rerouted(fake_ttnn):
    dev = object()
    with host_side_tensor_construction():
        t0 = utils_for_testing.start_measuring_time()
        ttnn.to_memory_config(device_tensor(dev), "mc")
        ttnn.from_torch("t", device=dev)
        utils_for_testing.stop_measuring_time(t0)
        ttnn.to_memory_config(device_tensor(dev), "mc")
    assert [c[0] for c in fake_ttnn] == ["to_memory_config", "from_torch", "to_device", "from_device", "to_device"]


# --- lifecycle and opt-outs ---------------------------------------------------------------


def test_originals_and_listener_restored_after_exit_and_on_error(fake_ttnn):
    orig = (ttnn.from_torch, ttnn.to_memory_config)
    with host_side_tensor_construction():
        assert ttnn.from_torch is not orig[0]
        assert tensor_setup._window_listener in utils_for_testing._measuring_window_listeners
    assert (ttnn.from_torch, ttnn.to_memory_config) == orig
    assert tensor_setup._window_listener not in utils_for_testing._measuring_window_listeners
    with pytest.raises(RuntimeError):  # allow-pytest.raises: plain Python error, no device error text for CI triage
        with host_side_tensor_construction():
            utils_for_testing.start_measuring_time()
            raise RuntimeError("boom")
    assert (ttnn.from_torch, ttnn.to_memory_config) == orig
    assert tensor_setup._in_measured_window is False


@pytest.mark.parametrize(
    "config, module",
    [
        (types.SimpleNamespace(measure_device_perf=False), types.SimpleNamespace()),
        (types.SimpleNamespace(measure_device_perf=True, device_side_setup=True), types.SimpleNamespace()),
        (types.SimpleNamespace(measure_device_perf=True), types.SimpleNamespace(_DEVICE_SIDE_SETUP=True)),
        (None, types.SimpleNamespace()),
    ],
    ids=["no-device-perf", "run-opt-out", "module-opt-out", "no-config"],
)
def test_setup_context_is_a_no_op_unless_device_perf_is_requested_without_opt_out(fake_ttnn, config, module):
    dev, mc = object(), object()
    orig = ttnn.from_torch
    with setup_context(module, config):
        assert ttnn.from_torch is orig
        ttnn.from_torch("t", device=dev, memory_config=mc)
    assert fake_ttnn == [("from_torch", "t", None, {"device": dev, "memory_config": mc})]


def test_setup_context_reroutes_when_device_perf_is_requested(fake_ttnn):
    config = types.SimpleNamespace(measure_device_perf=True)
    with setup_context(types.SimpleNamespace(), config):
        ttnn.from_torch("t", device=object())
    assert [c[0] for c in fake_ttnn] == ["from_torch", "to_device"]


# --- contract with the real ttnn bindings (no device needed) -----------------------------


def test_real_ttnn_signatures_match_what_the_wrappers_assume():
    positional = [
        p.name
        for p in inspect.signature(ttnn.from_torch.function).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["tensor", "dtype"], "wrapper takes dtype as the only positional after tensor"
    assert "queue_id" in ttnn.to_device.__doc__, "wrapper passes the queue as queue_id"
    assert "dtype" in ttnn.to_memory_config.__doc__, "wrapper takes dtype as the third positional"
    assert callable(ttnn.is_tensor_storage_on_device)
