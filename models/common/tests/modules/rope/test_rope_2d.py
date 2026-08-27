# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rope.rope_2d import RotarySetup2D, RotarySetup2DConfig, prepare_rot_idxs
from models.common.tensor_utils import parse_shard_dims_from_mesh_mapper_config


class _ShapeOnly:
    def __init__(self, *shape):
        self.shape = shape


def _galaxy(arch=ttnn.device.Arch.WORMHOLE_B0, shape=(8, 4), devices=32):
    mesh = MagicMock()
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(8, 10)
    return mesh


def _grid_resources(rows=8):
    core_grid = ttnn.CoreCoord(8, 10)
    return core_grid, ttnn.num_cores_to_corerangeset(rows, core_grid, row_wise=True)


@pytest.mark.parametrize(
    "table_len,theta,scale,original_context",
    [(16384, 500000.0, 8.0, 8192), (40960, 1000000.0, None, None)],
    ids=["llama-scaled", "qwen"],
)
def test_rope_2d_resolves_llama_and_qwen_data(table_len, theta, scale, original_context):
    mesh = _galaxy()
    cos = LazyWeight(source=_ShapeOnly(1, 1, table_len, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, table_len, 128), device=mesh)
    config = RotarySetup2DConfig(
        cos,
        sin,
        max_batch_size=32,
        rope_theta=theta,
        rope_scaling_factor=scale,
        original_context_len=original_context,
        core_grid=_grid_resources()[0],
        batch_grid=_grid_resources()[1],
    )

    module = RotarySetup2D.from_config(config)

    assert module.config.is_resolved()
    assert module.config.head_dim == 128
    assert module.config.users_per_column == 8
    assert module.config.rope_theta == theta
    assert module.config.cos_matrix.mesh_mapper_config is None
    assert module.config.sin_matrix.mesh_mapper_config is None
    assert module.config.cos_matrix._value is None
    assert parse_shard_dims_from_mesh_mapper_config(module.config.decode_index_mapper_config) == [0]
    assert "PlacementReplicate" in repr(module.config.decode_index_mapper_config)
    assert tuple(module.config._decode_trans_mat.source.shape) == (1, 1, 256, 32)
    # (1, 1, 32, 32), not (1, 1, head_dim, head_dim). This assertion used to read
    # 128 and was wrong: `rotary_embedding_llama` applies the transformation one
    # tile at a time and rejects anything else --
    #     TT_FATAL ... Transformation matrix must have 4th dim equal to TILE_WIDTH
    # measured on `(8, 4)` at prefill 128. `get_rot_transformation_mat`'s own
    # docstring says "Must equal TILE_SIZE", and the qualified 1D reference
    # hard-forces `dhead = 32  # ROPE op uses a single tile`. Nothing had ever
    # driven the prefill matrix through the op, so the shape went unchallenged.
    assert tuple(module.config._prefill_trans_mat.source.shape) == (1, 1, 32, 32)
    assert not hasattr(RotarySetup2D, "from_model_args")


def test_rope_2d_fused_qk_keeps_users_grouped_per_column():
    mesh = _galaxy()
    cos = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    core_grid, batch_grid = _grid_resources(16)
    module = RotarySetup2D.from_config(
        RotarySetup2DConfig(cos, sin, 32, use_qk_fused=True, core_grid=core_grid, batch_grid=batch_grid)
    )
    assert tuple(module.config._decode_trans_mat.source.shape) == (1, 1, 512, 32)


def test_rope_2d_config_is_immutable():
    config = RotarySetup2DConfig(
        LazyWeight(source=_ShapeOnly(1, 1, 4096, 128)), LazyWeight(source=_ShapeOnly(1, 1, 4096, 128)), 32
    )
    with pytest.raises(FrozenInstanceError):
        config.head_dim = 128


@pytest.mark.parametrize(
    "shape,devices,arch,error",
    [
        ((4, 8), 32, ttnn.device.Arch.WORMHOLE_B0, r"shape \(8, 4\)"),
        ((8, 4), 16, ttnn.device.Arch.WORMHOLE_B0, "exactly 32"),
        ((8, 4), 32, ttnn.device.Arch.BLACKHOLE, "Wormhole only"),
    ],
)
def test_rope_2d_fails_closed_on_platform(shape, devices, arch, error):
    mesh = _galaxy(arch=arch, shape=shape, devices=devices)
    cos = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    with pytest.raises(ValueError, match=error):
        RotarySetup2D(cos, sin, 32)


def test_rope_2d_rejects_mismatched_tables_and_bad_batch_group():
    mesh = _galaxy()
    cos = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, 2048, 128), device=mesh)
    with pytest.raises(ValueError, match="identical shape"):
        RotarySetup2D(cos, sin, 32)

    sin = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    with pytest.raises(ValueError, match="8 users"):
        RotarySetup2D.from_config(RotarySetup2DConfig(cos, sin, 32, users_per_column=4))


def test_rope_2d_requires_explicit_fabric_safe_grids():
    mesh = _galaxy()
    cos = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)

    with pytest.raises(ValueError, match="explicit fabric-safe"):
        RotarySetup2D.from_config(RotarySetup2DConfig(cos, sin, 32))

    core_grid, wrong_batch_grid = _grid_resources(7)
    with pytest.raises(ValueError, match="exactly 8 cores"):
        RotarySetup2D.from_config(RotarySetup2DConfig(cos, sin, 32, core_grid=core_grid, batch_grid=wrong_batch_grid))


def test_rope_2d_prepares_grouped_decode_indices_with_resolved_mapper(monkeypatch):
    mesh = _galaxy()
    core_grid, batch_grid = _grid_resources()
    cos = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    sin = LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh)
    module = RotarySetup2D.from_config(RotarySetup2DConfig(cos, sin, 32, core_grid=core_grid, batch_grid=batch_grid))
    mapper = object()
    captured = {}
    monkeypatch.setattr(ttnn, "create_mesh_mapper", lambda device, config: mapper)

    def fake_as_tensor(source, **kwargs):
        captured.update(source=source, kwargs=kwargs)
        return object()

    monkeypatch.setattr(ttnn, "as_tensor", fake_as_tensor)
    positions = torch.arange(32, dtype=torch.int64)

    prepared = prepare_rot_idxs(module.config, positions, on_host=True)

    assert prepared is not None
    assert captured["source"].shape == (32, 32)
    assert torch.equal(captured["source"][:, 0], positions)
    assert torch.count_nonzero(captured["source"][:, 1:]) == 0
    assert captured["kwargs"]["mesh_mapper"] is mapper
    assert "device" not in captured["kwargs"]


@pytest.mark.parametrize("positions", [torch.arange(31), torch.cat((torch.arange(31), torch.tensor([-1])))])
def test_rope_2d_rejects_invalid_decode_indices(positions):
    mesh = _galaxy()
    core_grid, batch_grid = _grid_resources()
    module = RotarySetup2D.from_config(
        RotarySetup2DConfig(
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            32,
            core_grid=core_grid,
            batch_grid=batch_grid,
        )
    )

    with pytest.raises(ValueError):
        prepare_rot_idxs(module.config, positions)


def test_rope_2d_release_is_repeatable(monkeypatch):
    mesh = _galaxy()
    core_grid, batch_grid = _grid_resources()
    module = RotarySetup2D.from_config(
        RotarySetup2DConfig(
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            32,
            core_grid=core_grid,
            batch_grid=batch_grid,
        )
    )
    values = [object() for _ in range(4)]
    weights = (
        module.config.cos_matrix,
        module.config.sin_matrix,
        module.config._decode_trans_mat,
        module.config._prefill_trans_mat,
    )
    for weight, value in zip(weights, values):
        weight._value = value
    module._device_weights_loaded = True
    deallocated = []
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    module.release()
    module.release()

    assert deallocated == values
    assert not module._device_weights_loaded


def test_rope_2d_prefill_table_copy_is_tilized_and_decode_table_is_not(monkeypatch):
    """Each mode's table has exactly one legal layout, and they differ.

    Decode reads the table through `ttnn.embedding`, which needs a *row-major*
    weight table; prefill slices the table and hands the slice to
    `rotary_embedding_llama`, which rejects anything but TILE:

        TT_FATAL ... cos tensor to rotary embedding must be tilized

    measured on `(8, 4)` at prefill 128. So the second copy
    `_materialize_table_copy` makes must be tilized even though the table it
    copies is not, and that must not silently regress to `table.layout`.
    """

    from models.common.modules.rope import rope_2d

    mesh = _galaxy()
    core_grid, batch_grid = _grid_resources()
    module = RotarySetup2D.from_config(
        RotarySetup2DConfig(
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            LazyWeight(source=_ShapeOnly(1, 1, 4096, 128), device=mesh),
            32,
            core_grid=core_grid,
            batch_grid=batch_grid,
        )
    )
    assert module.config.cos_matrix.layout is ttnn.ROW_MAJOR_LAYOUT

    requested = []

    def fake_get_device_weight(self):
        requested.append(self.layout)
        return object()

    monkeypatch.setattr(LazyWeight, "get_device_weight", fake_get_device_weight)
    module.load_device_weights()

    # cos, sin, decode transform, prefill transform, and the two prefill copies.
    assert requested.count(ttnn.TILE_LAYOUT) >= 2
    assert rope_2d._materialize_table_copy.__doc__ is not None
