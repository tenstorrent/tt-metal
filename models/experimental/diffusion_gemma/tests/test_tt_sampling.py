# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

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


@pytest.mark.parametrize("seed", [0, -3])
def test_ttnn_gumbel_noise_helpers_reject_nonpositive_seed(seed):
    device = _FakeDevice(num_devices=1)

    with pytest.raises(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise((1, 1, 32, 32), device=device, seed=seed)
    with pytest.raises(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise_with_permuted_vocab((1, 1, 32, 32), device=device, seed=seed)
    with pytest.raises(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise_by_vocab_chunks((1, 1, 32, 32), device=device, seed=seed)


def test_softmax_uses_numeric_stable_default(monkeypatch):
    calls = {}
    logits = object()

    def fake_softmax(tensor, **kwargs):
        calls["tensor"] = tensor
        calls["kwargs"] = kwargs
        return "probs"

    monkeypatch.setattr(TS.ttnn, "softmax", fake_softmax)

    assert TS.softmax(logits) == "probs"
    assert calls == {
        "tensor": logits,
        "kwargs": {"dim": -1, "numeric_stable": True},
    }
