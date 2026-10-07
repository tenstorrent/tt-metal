# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the device-write/migration notification boundary."""

from types import SimpleNamespace

import pytest

from models.demos.minimax_m3.tt.model import Model


def test_completion_observes_device_writes(monkeypatch):
    pending = []
    visible = []
    migrated = []
    mesh = object()
    hidden = object()

    def layer(index):
        def enqueue(value, **kwargs):
            pending.append(index)
            return value

        return enqueue

    def synchronize(device):
        assert device is mesh
        visible.extend(pending)
        pending.clear()

    def migrate(index):
        assert not pending
        assert visible == list(range(index + 1))
        migrated.append(index)

    monkeypatch.setattr("ttnn.synchronize_device", synchronize)
    model = SimpleNamespace(layers=[layer(i) for i in range(3)], mesh_device=mesh, is_last_rank=False)
    result = Model._forward_layers_and_head(model, hidden, None, None, on_layer_complete=migrate)
    assert result is hidden
    assert migrated == [0, 1, 2]


def test_device_failure_does_not_publish_completion(monkeypatch, expect_error):
    migrated = []

    def synchronize(device):
        raise RuntimeError("device failed")

    monkeypatch.setattr("ttnn.synchronize_device", synchronize)
    model = SimpleNamespace(layers=[lambda value, **kwargs: value], mesh_device=object(), is_last_rank=False)
    with expect_error(RuntimeError, "device failed"):
        Model._forward_layers_and_head(model, object(), None, None, on_layer_complete=migrated.append)
    assert migrated == []


def test_no_completion_consumer_keeps_asynchronous_path(monkeypatch):
    def unexpected_sync(device):
        pytest.fail("no completion consumer needs a host barrier")

    monkeypatch.setattr("ttnn.synchronize_device", unexpected_sync)
    hidden = object()
    model = SimpleNamespace(layers=[lambda value, **kwargs: value], mesh_device=object(), is_last_rank=False)
    assert Model._forward_layers_and_head(model, hidden, None, None) is hidden
