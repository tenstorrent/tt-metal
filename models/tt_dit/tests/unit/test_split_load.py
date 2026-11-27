# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the split load: which tensors split, and that a slice of the cache file is the tensor's bytes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn

from models.tt_dit.utils import split_load

MESH = (4, 8)
MIB = 1 << 20


@pytest.mark.parametrize(
    ("shape", "mesh_axes", "dtype", "layout", "expected"),
    [
        # replicated everywhere: the first dim that divides over both axes (32 chips x 32-row tiles)
        ((1, 1, 4096, 2048), [None] * 4, ttnn.bfloat16, ttnn.TILE_LAYOUT, (2, [0, 1])),
        # split over mesh axis 1 already: only axis 0 is free
        ((1, 1, 4096, 2048), [None, None, None, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, (2, [0])),
        # both axes taken: nothing to split
        ((1, 1, 4096, 2048), [None, None, 0, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, None),
        # too small
        ((1, 1, 256, 256), [None] * 4, ttnn.bfloat16, ttnn.TILE_LAYOUT, None),
        # block-float dtypes have no host_tensor_from_bytes
        ((1, 1, 4096, 2048), [None] * 4, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, None),
        # 3 x 4096 rows: a leading dim > 1 would make the slice non-contiguous, the next dim cannot split
        ((3, 4096, 2048), [None] * 3, ttnn.bfloat16, ttnn.TILE_LAYOUT, None),
        # rows do not divide over 32 tiles but do over 8: one axis alone
        ((1, 1, 1024, 4096), [None] * 4, ttnn.bfloat16, ttnn.TILE_LAYOUT, (2, [0, 1])),
        ((1, 1, 256, 8192), [None] * 4, ttnn.bfloat16, ttnn.TILE_LAYOUT, (2, [1])),
        # padded tile dims keep the plain load
        ((1, 1, 4096, 2000), [None] * 4, ttnn.bfloat16, ttnn.TILE_LAYOUT, None),
        # row-major: never cut the last dim
        ((3, 4096, 64), [None] * 3, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, None),
        ((64, 4096, 64), [None] * 3, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, [0, 1])),
        ((1, 4096, 4096), [None] * 3, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, (1, [0, 1])),
    ],
)
def test_plan_for(shape, mesh_axes, dtype, layout, expected):
    assert split_load.plan_for(shape, mesh_axes, MESH, dtype, layout) == expected


def test_plan_for_host_tensor_is_never_split():
    assert split_load.plan_for((1, 1, 4096, 2048), [None] * 4, MESH, ttnn.bfloat16, ttnn.TILE_LAYOUT, on_host=True) is None


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)],
)
def test_file_slices_are_the_tensor_bytes(tmp_path, dtype, torch_dtype):
    """Every 1/N byte range of a replicated mesh tensorbin is the matching rows of the tensor, bit for bit."""
    rows, cols = 2048, 1024
    source = torch.randn(1, 1, rows, cols).to(torch_dtype)
    host = ttnn.from_torch(source, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    path = tmp_path / "t.tensorbin"
    ttnn.dump_tensor(str(path), ttnn.from_host_shards([host] * 32, ttnn.MeshShape(*MESH)))

    mm, data_start, shards = split_load.read_shards(path)
    assert sorted(shards) == [(r, c) for r in range(MESH[0]) for c in range(MESH[1])]
    assert len(set(shards.values())) == 1  # replicated shards share one buffer
    (offset, size), = set(shards.values())
    full_bytes = split_load.slice_bytes(source.shape, dtype, ttnn.TILE_LAYOUT)
    assert size == full_bytes

    dim, axes = split_load.plan_for(source.shape, [None] * 4, MESH, dtype, ttnn.TILE_LAYOUT)
    n = MESH[0] * MESH[1]
    piece_shape = list(source.shape)
    piece_shape[dim] //= n
    piece = split_load.slice_bytes(piece_shape, dtype, ttnn.TILE_LAYOUT)
    spec = ttnn.TensorSpec(ttnn.Shape(piece_shape), dtype, ttnn.TILE_LAYOUT)
    want = ttnn.to_torch(host)
    for k in range(n):
        view = np.frombuffer(mm, dtype=np.uint8, count=piece, offset=data_start + offset + k * piece)
        got = ttnn.to_torch(ttnn.host_tensor_from_bytes(view, spec))
        assert torch.equal(got, want[:, :, k * piece_shape[2] : (k + 1) * piece_shape[2]])


def test_truncated_file_is_refused(tmp_path):
    host = ttnn.from_torch(torch.randn(1, 1, 2048, 1024).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    path = tmp_path / "t.tensorbin"
    ttnn.dump_tensor(str(path), host)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="truncated"):
        split_load.read_shards(path)


def test_host_slices_of_a_column_split_tensor(tmp_path):
    """A tensor already split over mesh axis 1: each chip gets its own column's slice; the slices of a column,
    stacked in row order, are that column's tensor (what the axis-0 all-gather builds)."""
    import types
    from pathlib import Path

    shape = [1, 1, 4096, 1024]
    columns = [torch.randn(*shape).to(torch.bfloat16) for _ in range(MESH[1])]
    hosts = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for t in columns]
    path = tmp_path / "t.tensorbin"
    shard_list = [hosts[c] for r in range(MESH[0]) for c in range(MESH[1])]
    ttnn.dump_tensor(str(path), ttnn.from_host_shards(shard_list, ttnn.MeshShape(*MESH)))

    dim, axes = split_load.plan_for(shape, (None, None, None, 1), MESH, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    assert (dim, axes) == (2, [0])
    param = types.SimpleNamespace(
        local_shape=shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=types.SimpleNamespace(shape=MESH),
    )
    part, _mm = split_load.SplitLoad(None)._host_slices(param, Path(path), dim, axes)
    pieces = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(part)]
    for c in range(MESH[1]):
        stacked = torch.cat([pieces[r * MESH[1] + c] for r in range(MESH[0])], dim=dim)
        assert torch.equal(stacked, columns[c]), f"column {c}"


def test_split_load_runs_under_the_host_weight_cache(tmp_path, monkeypatch):
    """With TT_DIT_HOST_WEIGHT_CACHE=1 (every LTX worker) a parameter still loads through the active split
    load, and keeps the host copy its later reloads push from."""
    import types

    from models.tt_dit.layers import module

    t = torch.randn(1, 1, 64, 64).to(torch.bfloat16)
    path = tmp_path / "t.tensorbin"
    ttnn.dump_tensor(str(path), ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
    monkeypatch.setattr(module, "_HOST_WEIGHT_CACHE", True)
    calls = []
    loader = types.SimpleNamespace(load=lambda param, p: calls.append(p) or True)
    param = object.__new__(module.Parameter)  # no device: the split loader stands in for the upload
    param.on_host, param._host_data, param.mesh_axes, param.durable_mesh_axes = False, None, (None,) * 4, (None,) * 4
    token = split_load._active.set(loader)
    try:
        module.Parameter.load(param, path)
    finally:
        split_load._active.reset(token)
    assert calls == [path]
    assert torch.equal(ttnn.to_torch(param._host_data), t)


def test_ltx_pipeline_loads_pass_a_ccl_manager():
    """Every load_model call in the LTX pipelines hands over a ccl_manager, so split load can run there."""
    import ast
    import pathlib

    ltx = pathlib.Path(__file__).resolve().parents[2] / "pipelines" / "ltx"
    missing = []
    for path in sorted(ltx.glob("pipeline_ltx*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "load_model":
                if "ccl_manager" not in {kw.arg for kw in node.keywords}:
                    missing.append(f"{path.name}:{node.lineno}")
    assert not missing, missing
