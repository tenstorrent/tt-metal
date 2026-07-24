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


class _FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        self.deallocated = force


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
def test_ttnn_gumbel_noise_helpers_reject_nonpositive_seed(seed, expect_error):
    device = _FakeDevice(num_devices=1)

    with expect_error(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise((1, 1, 32, 32), device=device, seed=seed)
    with expect_error(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise_with_permuted_vocab((1, 1, 32, 32), device=device, seed=seed)
    with expect_error(ValueError, match="positive nonzero"):
        TS.sample_gumbel_noise_by_vocab_chunks((1, 1, 32, 32), device=device, seed=seed)


@pytest.mark.parametrize("shape", [(), (1, 0, 32), (1, -2, 32)])
def test_ttnn_gumbel_noise_helpers_reject_bad_shape_dimensions(shape, expect_error):
    device = _FakeDevice(num_devices=1)

    with expect_error(ValueError, match="shape"):
        TS.sample_gumbel_noise(shape, device=device, seed=47472)
    with expect_error(ValueError, match="shape"):
        TS.sample_gumbel_noise_with_permuted_vocab(shape, device=device, seed=47472)
    with expect_error(ValueError, match="shape"):
        TS.sample_gumbel_noise_by_vocab_chunks(shape, device=device, seed=47472)


@pytest.mark.parametrize("shape", [(32,), (1,)])
def test_vocab_axis_gumbel_noise_helpers_require_sample_and_vocab_axes(shape, expect_error):
    device = _FakeDevice(num_devices=1)

    with expect_error(ValueError, match="sample axis and a vocab axis"):
        TS.sample_gumbel_noise_with_permuted_vocab(shape, device=device, seed=47472)
    with expect_error(ValueError, match="sample axis and a vocab axis"):
        TS.sample_gumbel_noise_by_vocab_chunks(shape, device=device, seed=47472)


def test_permuted_vocab_gumbel_noise_deallocates_pre_permute_tensor(monkeypatch):
    calls = {}
    raw = _FakeTensor("raw")
    permuted = _FakeTensor("permuted")
    reshaped = _FakeTensor("reshaped")

    class _FakeTtnn:
        TILE_LAYOUT = "tile"

        @staticmethod
        def rand(shape, **kwargs):
            calls["rand"] = (shape, kwargs)
            return raw

        @staticmethod
        def permute(tensor, order):
            calls["permute"] = (tensor, order)
            return permuted

        @staticmethod
        def reshape(tensor, shape):
            calls["reshape"] = (tensor, shape)
            return reshaped

    def fake_gumbel_from_uniform(tensor):
        calls["gumbel"] = tensor
        return "gumbel"

    monkeypatch.setattr(TS, "ttnn", _FakeTtnn)
    monkeypatch.setattr(TS, "_gumbel_from_uniform", fake_gumbel_from_uniform)

    out = TS.sample_gumbel_noise_with_permuted_vocab((2, 1, 4, 16), device="mesh", seed=47472, dtype="float32")

    assert out == "gumbel"
    # vocab (16) outermost, all non-vocab dims (2*1*4=8) collapsed into one tile-aligned axis;
    # no singleton axis lands in the tiled last-two dims (the 32x TILE-pad OOM fix).
    assert calls["rand"][0] == (16, 8)
    assert calls["permute"] == (raw, (1, 0))
    assert calls["reshape"] == (permuted, (2, 1, 4, 16))
    assert calls["gumbel"] is reshaped
    assert raw.deallocated is True
    assert permuted.deallocated is True
    assert reshaped.deallocated is False


def test_chunked_gumbel_noise_deallocates_parts_after_concat(monkeypatch):
    calls = {}
    uniforms = [_FakeTensor("u0"), _FakeTensor("u1")]
    parts = [_FakeTensor("part0"), _FakeTensor("part1")]
    concat = _FakeTensor("concat")

    class _FakeTtnn:
        TILE_LAYOUT = "tile"

        @staticmethod
        def rand(shape, **kwargs):
            calls.setdefault("rand", []).append((shape, kwargs))
            return uniforms[len(calls["rand"]) - 1]

        @staticmethod
        def concat(tensors, *, dim):
            calls["concat"] = (list(tensors), dim)
            return concat

    def fake_gumbel_from_uniform(tensor):
        return parts[uniforms.index(tensor)]

    monkeypatch.setattr(TS, "ttnn", _FakeTtnn)
    monkeypatch.setattr(TS, "_gumbel_from_uniform", fake_gumbel_from_uniform)

    out = TS.sample_gumbel_noise_by_vocab_chunks(
        (1, 1, 4, 5),
        device="mesh",
        seed=47472,
        vocab_chunk_size=3,
        dtype="float32",
    )

    assert out is concat
    assert [shape for shape, _ in calls["rand"]] == [(1, 1, 4, 3), (1, 1, 4, 2)]
    assert [kwargs["seed"] for _, kwargs in calls["rand"]] == [47472, 47475]
    assert calls["concat"] == (parts, -1)
    assert [part.deallocated for part in parts] == [True, True]
