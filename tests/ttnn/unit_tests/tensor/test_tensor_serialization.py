# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype, TORCH_INTEGER_DTYPES

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
def test_serialization(tmp_path, shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in TORCH_INTEGER_DTYPES:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.tensorbin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttnn.load_tensor(str(file_name)).to_torch()

    torch_tensor_from_file = torch_tensor_from_file.to(torch_tensor.dtype)

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing


def test_large_read_only_file_backed_tensor_upload(tmp_path, device):
    # Deliberately ungated. Uploading a read-only file mapping must produce the right tensor on every
    # system: with device-read-only pinning it takes the pinned path, and without it (older KMD, no
    # IOMMU, or pinning disabled) try_pin returns nullptr and the upload falls back to a copy. Gating
    # this on the KMD version would have left the fallback path -- the one every current CI runner
    # takes -- with no coverage at all.
    # 1024 * 9216 * 4 bytes = 36 MiB, above Metal's 32 MiB pinned H2D threshold.
    shape = (1, 1, 1024, 9216)
    # Position-dependent values, not a constant fill: a uniform buffer compares equal even if the pinned
    # path transfers the wrong offset, repeats a page, or drops the mapping's base offset.
    torch_tensor = torch.arange(1024 * 9216, dtype=torch.float32).reshape(shape)
    host_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.float32)
    file_name = tmp_path / "large_read_only.tensorbin"
    ttnn.dump_tensor(str(file_name), host_tensor)

    # load_tensor opens the file O_RDONLY and maps it PROT_READ | MAP_SHARED (MAP_PRIVATE where the
    # filesystem refuses a shared mapping) before uploading.
    device_tensor = ttnn.load_tensor(str(file_name), device=device)
    result = ttnn.to_torch(device_tensor)
    assert torch.equal(result, torch_tensor)


core_ranges = ttnn.num_cores_to_corerangeset(56, [8, 7], True)


@pytest.mark.parametrize(
    "tensor_spec",
    [
        ttnn.TensorSpec((1, 2, 3, 4), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, tile=ttnn.Tile([16, 16])),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1),
        ttnn.TensorSpec(
            (2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims_except([0], core_ranges),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).block_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).height_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).width_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).sharded(
            (1, 37, 37), core_ranges, ttnn.ShardShapeAlignment.RECOMMENDED
        ),
    ],
)
def test_sharded_tensor_serialization(tmp_path, device, tensor_spec):
    torch.manual_seed(0)
    dtype = tt_dtype_to_torch_dtype[tensor_spec.dtype]
    py_tensor = torch.rand(list(tensor_spec.shape), dtype=dtype)
    tt_tensor = ttnn.from_torch(py_tensor, spec=tensor_spec, device=device)
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    ttnn_tensor_from_file = ttnn.load_tensor(str(file_name), device=device)
    assert ttnn_tensor_from_file.spec == tensor_spec
    torch_tensor_from_file = ttnn.to_torch(ttnn_tensor_from_file)
    assert torch.allclose(py_tensor, torch_tensor_from_file)


def _dump_host_tensor(tmp_path, name="tensor.tensorbin"):
    torch_tensor = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    tt_tensor = ttnn.Tensor(torch_tensor, ttnn.float32)
    file_name = tmp_path / name
    ttnn.dump_tensor(str(file_name), tt_tensor)
    return file_name, torch_tensor


def test_dump_tensor_leaves_only_the_final_file(tmp_path):
    """dump_tensor writes through a temporary sibling and renames it into place, so a reader
    never sees a half-written file; nothing but the final file may remain."""
    file_name, torch_tensor = _dump_host_tensor(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [file_name.name]
    assert torch.equal(ttnn.to_torch(ttnn.load_tensor(str(file_name))), torch_tensor)


def test_dump_tensor_replaces_an_existing_file_atomically(tmp_path):
    """Replacing a file must publish the new content without touching the inode a loaded tensor
    is still mapped from; an in-place writer would truncate under that reader."""
    file_name, torch_tensor = _dump_host_tensor(tmp_path)
    original = ttnn.load_tensor(str(file_name))
    replacement = torch_tensor + 1
    ttnn.dump_tensor(str(file_name), ttnn.Tensor(replacement, ttnn.float32))
    assert sorted(p.name for p in tmp_path.iterdir()) == [file_name.name]
    assert torch.equal(ttnn.to_torch(original), torch_tensor)
    assert torch.equal(ttnn.to_torch(ttnn.load_tensor(str(file_name))), replacement)


@pytest.mark.parametrize("mode", [0o600, 0o664])
def test_dump_tensor_keeps_the_replaced_files_permissions(tmp_path, mode):
    """The temporary file gets the destination's exact mode, so under umask 022 replacing a 0600
    cache file does not publish it as 0644, and replacing a 0664 one does not drop group write;
    a new file still follows the umask."""
    file_name, torch_tensor = _dump_host_tensor(tmp_path)
    file_name.chmod(mode)
    old_umask = os.umask(0o022)
    try:
        _dump_host_tensor(tmp_path)
        fresh, _ = _dump_host_tensor(tmp_path, name="fresh.tensorbin")
    finally:
        os.umask(old_umask)
    assert file_name.stat().st_mode & 0o777 == mode
    assert fresh.stat().st_mode & 0o777 == 0o644
    assert torch.equal(ttnn.to_torch(ttnn.load_tensor(str(file_name))), torch_tensor)


@pytest.mark.parametrize("keep_fraction", [0.999, 0.5])
def test_load_tensor_rejects_a_file_truncated_after_its_header(tmp_path, keep_fraction, expect_error):
    """A header-complete file whose data section is cut short must raise, not load a buffer that
    runs past the mapping (the segfault seen on cold multi-process cache builds)."""
    file_name, _ = _dump_host_tensor(tmp_path)
    data = file_name.read_bytes()
    truncated = tmp_path / "truncated.tensorbin"
    truncated.write_bytes(data[: int(len(data) * keep_fraction)])
    with expect_error(RuntimeError, "truncated or corrupt"):
        ttnn.load_tensor(str(truncated))


def test_as_tensor_regenerates_a_truncated_cache_file(tmp_path):
    torch_tensor = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    cache_stem = tmp_path / "weight"
    cached = ttnn.as_tensor(torch_tensor, dtype=ttnn.float32, cache_file_name=str(cache_stem))
    (cache_file,) = tmp_path.glob("weight_*.tensorbin")
    assert torch.equal(ttnn.to_torch(cached), torch_tensor)
    data = cache_file.read_bytes()
    cache_file.write_bytes(data[: len(data) // 2])
    regenerated = ttnn.as_tensor(torch_tensor, dtype=ttnn.float32, cache_file_name=str(cache_stem))
    assert torch.equal(ttnn.to_torch(regenerated), torch_tensor)
    assert cache_file.read_bytes() == data
    assert sorted(p.name for p in tmp_path.iterdir()) == [cache_file.name]
