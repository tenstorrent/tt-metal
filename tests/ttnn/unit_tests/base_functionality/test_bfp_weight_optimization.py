# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.weight_cache import mark_weight_cache_complete, marker_path, weight_cache_is_complete


DTYPES = [(ttnn.bfloat4_b, 0.13, 0.125, 0.875), (ttnn.bfloat8_b, 0.008, 0.0078125, 0.9921875)]


def _weights(small, shape=(32, 32), dtype=torch.float32):
    weights = torch.full(shape, small, dtype=dtype)
    weights[..., ::16] = 1.0
    return weights


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 32), (35, 47)])
def test_bfp_exponent_search(dtype, small, rounded, clipped, source_dtype, shape):
    weights = _weights(small, shape, source_dtype)
    original = weights.clone()
    ordinary = ttnn.to_torch(ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT))
    explicit_off = ttnn.to_torch(ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=False))
    optimized = ttnn.to_torch(ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True))
    assert torch.equal(weights, original)
    assert torch.equal(ordinary, explicit_off)
    expected = torch.full(shape, rounded)
    expected[..., ::16] = clipped
    assert torch.equal(optimized, expected)
    assert (optimized.double() - weights.double()).square().sum() < (
        ordinary.double() - weights.double()
    ).square().sum()


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
def test_bfp_cache_and_config(dtype, small, rounded, clipped, tmp_path, monkeypatch):
    weights = _weights(small)
    kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, cache_file_name=tmp_path / "weights")
    monkeypatch.setattr(ttnn.CONFIG, "enable_bfp_weight_optimization", False)
    ordinary = ttnn.to_torch(ttnn.as_tensor(weights, **kwargs))
    baseline_files = set(tmp_path.glob("*.tensorbin"))
    assert len(baseline_files) == 1
    monkeypatch.setattr(ttnn.CONFIG, "enable_bfp_weight_optimization", True)
    optimized = ttnn.to_torch(ttnn.as_tensor(weights, **kwargs))
    assert not torch.equal(ordinary, optimized)
    assert len(set(tmp_path.glob("*.tensorbin")) - baseline_files) == 1
    # Each mode must read its own cache file and ignore the new input values.
    assert torch.equal(optimized, ttnn.to_torch(ttnn.as_tensor(torch.zeros_like(weights), **kwargs)))
    assert torch.equal(ordinary, ttnn.to_torch(ttnn.as_tensor(torch.zeros_like(weights), optimize_bfp=False, **kwargs)))
    # The global setting must not change calls without a cache filename.
    assert torch.equal(ordinary, ttnn.to_torch(ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT)))
    assert torch.equal(ordinary, ttnn.to_torch(ttnn.as_tensor(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT)))
    assert torch.equal(
        optimized, ttnn.to_torch(ttnn.as_tensor(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True))
    )


def test_bfp_warm_cache_marker(tmp_path, monkeypatch):
    identity = dict(model_name="test", n_layers=1, mesh_shape=(1, 1))
    monkeypatch.setattr(ttnn.CONFIG, "enable_bfp_weight_optimization", False)
    old_marker = marker_path(tmp_path)
    (tmp_path / "weight.tensorbin").touch()
    mark_weight_cache_complete(tmp_path, {"weight": torch.ones(1)}, **identity)
    assert weight_cache_is_complete(tmp_path, **identity)
    monkeypatch.setattr(ttnn.CONFIG, "enable_bfp_weight_optimization", True)
    assert marker_path(tmp_path) != old_marker
    assert not weight_cache_is_complete(tmp_path, **identity)
    optimized_file = tmp_path / "weight_bfp_emax_minus1_v1.tensorbin"
    optimized_file.touch()
    mark_weight_cache_complete(tmp_path, {"weight": torch.ones(1)}, **identity)
    assert weight_cache_is_complete(tmp_path, **identity)
    optimized_file.unlink()
    assert not weight_cache_is_complete(tmp_path, **identity)
    monkeypatch.setattr(ttnn.CONFIG, "enable_bfp_weight_optimization", False)
    assert marker_path(tmp_path) == old_marker
    assert weight_cache_is_complete(tmp_path, **identity)


def test_bfp_invalid_options(expect_error):
    weights = torch.ones(32, 32)
    with expect_error(RuntimeError, "optimize_bfp requires"):
        ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, optimize_bfp=True)
    with expect_error(RuntimeError, "cannot be combined"):
        ttnn.from_torch(
            weights, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, optimize_bfp=True, enable_bfloat_opt=True
        )
    with expect_error(RuntimeError, "optimize_bfp requires"):
        ttnn.as_tensor(weights, dtype=ttnn.bfloat16, optimize_bfp=True)


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
def test_bfp_col_tilize(dtype, small, rounded, clipped):
    weights = _weights(small, (64, 32)).T.contiguous()
    optimized = ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT, col_tilize=True, optimize_bfp=True)
    expected = ttnn.from_torch(weights.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True)
    assert torch.equal(ttnn.to_torch(optimized), ttnn.to_torch(expected))


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
def test_bfp_device_roundtrip(device, dtype, small, rounded, clipped):
    weights = _weights(small)
    host = ttnn.from_torch(weights, dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True)
    on_device = ttnn.from_torch(
        weights,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        optimize_bfp=True,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert torch.equal(ttnn.to_torch(host), ttnn.to_torch(on_device))


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
def test_bfp_spec_and_small_tile(dtype, small, rounded, clipped):
    weights = _weights(small, (16, 16))
    spec = ttnn.TensorSpec(weights.shape, dtype, ttnn.TILE_LAYOUT, tile=ttnn.Tile([16, 16]))
    output = ttnn.to_torch(ttnn.from_torch(weights, spec=spec, optimize_bfp=True))
    expected = torch.full_like(weights, rounded)
    expected[:, 0] = clipped
    assert torch.equal(output, expected)


@pytest.mark.parametrize("dtype,small,rounded,clipped", DTYPES)
def test_bfp_mesh_sharding(mesh_device, dtype, small, rounded, clipped):
    num_devices = mesh_device.get_num_devices()
    # Divide the input inside a group of 16 values.
    # Search must use the new groups formed after this division.
    weights = _weights(small, (32, 18 * num_devices))
    output = ttnn.from_torch(
        weights,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        optimize_bfp=True,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    expected = torch.cat(
        [
            ttnn.to_torch(ttnn.from_torch(chunk.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True))
            for chunk in weights.chunk(num_devices, dim=1)
        ],
        dim=1,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b])
@pytest.mark.parametrize("pad_value", [0.0, 0.1])
@pytest.mark.parametrize("col_tilize", [False, True])
def test_bfp_bfloat16_matches_expanded_input(dtype, pad_value, col_tilize):
    generator = torch.Generator().manual_seed(42)
    weights = torch.randn((35, 47), generator=generator).to(torch.bfloat16)
    kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, optimize_bfp=True, pad_value=pad_value, col_tilize=col_tilize)
    direct = ttnn.to_torch(ttnn.from_torch(weights, **kwargs))
    expanded = ttnn.to_torch(ttnn.from_torch(weights.float(), **kwargs))
    assert torch.equal(direct, expanded)
