# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_as_tensor(device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [2**15])
@pytest.mark.parametrize("width", [2**16])
def test_allocate_large_tensor(device, height, width):
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    reshard_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, height, width]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        memory_config,
    )


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_as_tensor_with_cache(tmp_path, device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.float32)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.device() == device
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.device() == device
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_as_tensor_with_cache_local_dump(tmp_path, device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.float32)

    memory_config = ttnn.L1_MEMORY_CONFIG
    kwargs = dict(
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        cache_file_name=tmp_path / "cache_file",
        cache_dump_mode=ttnn.DumpTensorMode.LOCAL,
    )
    ttnn.as_tensor(torch_input_tensor, **kwargs)
    cached = list(tmp_path.iterdir())
    assert len(cached) == 1 and cached[0].suffix == ".tensorbin", "the dump should publish one cache file, no temp file"

    # A cache hit ignores the input, so the values must come from the file written above.
    tensor = ttnn.as_tensor(torch.zeros_like(torch_input_tensor), **kwargs)
    assert tensor.device() == device
    assert torch.allclose(torch_input_tensor, ttnn.to_torch(tensor))


def test_as_tensor_local_dump_cleans_up_on_failure(tmp_path, monkeypatch, expect_error):
    def failing_dump(file_name, tensor, mode):
        with open(file_name, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(ttnn._ttnn.tensor, "dump_tensor_flatbuffer", failing_dump)
    with expect_error(RuntimeError, "disk full"):
        ttnn.as_tensor(
            torch.rand((7, 3), dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=tmp_path / "cache_file",
            cache_dump_mode=ttnn.DumpTensorMode.LOCAL,
        )
    assert list(tmp_path.iterdir()) == [], "a failed dump must not leave a temp or partial cache file behind"


@pytest.fixture
def cache_with_failing_load(tmp_path, monkeypatch):
    kwargs = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, cache_file_name=tmp_path / "cache_file")
    ttnn.as_tensor(torch.rand((7, 3), dtype=torch.float32), **kwargs)
    (cache_path,) = tmp_path.iterdir()

    real_load = ttnn._ttnn.tensor.load_tensor_flatbuffer

    def failing_load(*args, **kwargs):
        raise RuntimeError("read error")

    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", failing_load)
    return kwargs, cache_path, real_load


def test_as_tensor_cache_load_failure_raises_and_keeps_file_without_repair(cache_with_failing_load, expect_error):
    kwargs, cache_path, _ = cache_with_failing_load
    good_bytes = cache_path.read_bytes()
    with ttnn.as_tensor_cache_repair(False):
        with expect_error(RuntimeError, "read error"):
            ttnn.as_tensor(torch.zeros((7, 3), dtype=torch.float32), **kwargs)
    assert cache_path.read_bytes() == good_bytes


def test_as_tensor_cache_repair_scope_restores_previous_setting(expect_error):
    with ttnn.as_tensor_cache_repair(False):
        with ttnn.as_tensor_cache_repair(True):
            assert ttnn.operations.core._as_tensor_cache_repair
        assert not ttnn.operations.core._as_tensor_cache_repair
    assert ttnn.operations.core._as_tensor_cache_repair
    with expect_error(ValueError, "boom"):
        with ttnn.as_tensor_cache_repair(False):
            raise ValueError("boom")
    assert ttnn.operations.core._as_tensor_cache_repair


def test_as_tensor_cache_load_failure_regenerates_file_with_repair(cache_with_failing_load, monkeypatch):
    kwargs, cache_path, real_load = cache_with_failing_load
    replacement = torch.rand((7, 3), dtype=torch.float32)
    tensor = ttnn.as_tensor(replacement, **kwargs)
    assert torch.equal(ttnn.to_torch(tensor), replacement)
    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", real_load)  # read it back with the real loader
    assert torch.equal(ttnn.to_torch(ttnn.load_tensor(cache_path)), replacement)
