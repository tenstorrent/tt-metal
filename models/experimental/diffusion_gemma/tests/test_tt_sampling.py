# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.diffusion_gemma.tt import sampling as TS


class _FakeDevice:
    shape = (1, 4)

    def __init__(self, num_devices):
        self._num_devices = num_devices

    def get_num_devices(self):
        return self._num_devices


def test_rand_mesh_mapper_replicates_over_flattened_mesh(monkeypatch):
    calls = {}

    class _FakeTtnn:
        class PlacementReplicate:
            pass

        class MeshShape:
            def __init__(self, shape):
                self.shape = shape

        class MeshMapperConfig:
            def __init__(self, *, placements, mesh_shape_override=None):
                calls["placements"] = placements
                calls["mesh_shape_override"] = mesh_shape_override

    monkeypatch.setattr(TS, "ttnn", _FakeTtnn)

    mapper = TS._rand_mesh_mapper(_FakeDevice(num_devices=4))

    assert isinstance(mapper, _FakeTtnn.MeshMapperConfig)
    assert len(calls["placements"]) == 1
    assert isinstance(calls["placements"][0], _FakeTtnn.PlacementReplicate)
    assert calls["mesh_shape_override"].shape == [4]


def test_rand_mesh_mapper_single_device_returns_none():
    assert TS._rand_mesh_mapper(_FakeDevice(num_devices=1)) is None
