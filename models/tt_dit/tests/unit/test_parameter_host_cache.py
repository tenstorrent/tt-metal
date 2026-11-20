# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-cache selection tests; no TT device is opened."""

from types import SimpleNamespace

import pytest

from ...layers import module


@pytest.fixture
def parameter(monkeypatch):
    monkeypatch.setattr(module, "_HOST_WEIGHT_CACHE", True)
    monkeypatch.setattr(module.Parameter, "_check_data", lambda self, value: None)
    monkeypatch.setattr(module.ttnn, "load_tensor", lambda path, device: path.read_bytes())
    monkeypatch.setattr(module.ttnn, "to_device", lambda value, device, memory_config: value)
    return module.Parameter(total_shape=(1,), device=SimpleNamespace(shape=(1, 1)))


def test_host_cache_reuses_unchanged_file(parameter, tmp_path, monkeypatch):
    path = tmp_path / "weight.tensorbin"
    path.write_bytes(b"original")
    parameter.load(path)
    monkeypatch.setattr(module.ttnn, "load_tensor", lambda *a, **kw: pytest.fail("unchanged weights reloaded"))
    parameter.load(path)
    assert parameter.data == b"original"


def test_host_cache_invalidates_for_a_different_path(parameter, tmp_path):
    first, second = tmp_path / "first.tensorbin", tmp_path / "second.tensorbin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    parameter.load(first)
    parameter.load(second)
    assert parameter.data == b"second"


def test_host_cache_invalidates_after_replacing_the_file(parameter, tmp_path):
    path, replacement = tmp_path / "weight.tensorbin", tmp_path / "replacement"
    path.write_bytes(b"old")
    parameter.load(path)
    replacement.write_bytes(b"new")
    replacement.replace(path)
    parameter.load(path)
    assert parameter.data == b"new"


def test_failed_reload_does_not_bind_old_weights_to_a_new_path(parameter, tmp_path, monkeypatch, expect_error):
    first, second = tmp_path / "first.tensorbin", tmp_path / "second.tensorbin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    parameter.load(first)
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("invalid tensorbin")

        patch.setattr(module.ttnn, "load_tensor", fail)
        with expect_error(module.LoadingError, "invalid tensorbin"):
            parameter.load(second)
    parameter.load(second)
    assert parameter.data == b"second"


@pytest.mark.parametrize("cache_enabled,on_host", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_load_restores_durable_layout(parameter, tmp_path, monkeypatch, cache_enabled, on_host, warm_cache):
    monkeypatch.setattr(module, "_HOST_WEIGHT_CACHE", cache_enabled)
    parameter.on_host = on_host
    parameter.device.shape = (2, 1)
    parameter.total_shape = (64,)
    path = tmp_path / "weight.tensorbin"
    path.write_bytes(b"original")
    if warm_cache:
        parameter.load(path)
    parameter.mesh_axes = (0,)
    parameter._data = b"sharded"
    freed = []
    monkeypatch.setattr(module.ttnn, "deallocate", freed.append)

    def load(path, device):
        assert not parameter.is_transiently_sharded()
        assert parameter._data is None
        return path.read_bytes()

    def to_device(value, device, memory_config):
        assert not parameter.is_transiently_sharded()
        assert parameter._data is None
        assert memory_config == parameter.memory_config
        return value

    monkeypatch.setattr(module.ttnn, "load_tensor", load)
    monkeypatch.setattr(module.ttnn, "to_device", to_device)
    parameter.load(path)
    assert freed == [b"sharded"]
    assert parameter.mesh_axes == parameter.durable_mesh_axes == (None,)
    assert parameter.local_shape == (64,)
    assert parameter.data == b"original"


@pytest.mark.parametrize("cache_enabled,on_host", [(True, False), (False, False), (True, True)])
def test_failed_sharded_reload_leaves_no_device_data(
    parameter, tmp_path, monkeypatch, cache_enabled, on_host, expect_error
):
    monkeypatch.setattr(module, "_HOST_WEIGHT_CACHE", cache_enabled)
    parameter.on_host = on_host
    parameter.mesh_axes = (0,)
    parameter._data = b"sharded"
    freed = []
    monkeypatch.setattr(module.ttnn, "deallocate", freed.append)
    path = tmp_path / "weight.tensorbin"
    path.write_bytes(b"invalid")

    def fail(*args, **kwargs):
        raise RuntimeError("invalid tensorbin")

    monkeypatch.setattr(module.ttnn, "load_tensor", fail)
    with expect_error(module.LoadingError, "invalid tensorbin"):
        parameter.load(path)
    assert freed == [b"sharded"]
    assert parameter.mesh_axes == parameter.durable_mesh_axes
    assert parameter._data is None
