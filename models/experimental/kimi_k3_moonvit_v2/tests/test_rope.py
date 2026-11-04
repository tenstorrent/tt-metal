# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Rope` tables, and `rotary_embedding_llama` with them against the reference `apply_rope`.

The tables are written in place, so each test first fills them for another image."""

from __future__ import annotations

import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import apply_rope
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    MeshRows,
    P150_AND_RING,
    REFERENCE,
    SIMULATOR_GRID,
    SIMULATOR,
    reference_rope,
    ring_parallel,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope


def build(mesh_device, sp_axis, num_links, config) -> TT_Rope:
    config = device_config(config, sp_axis)
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    return TT_Rope(config, mesh_device=mesh_device, parallel=parallel)


def write_tables(tt_rope: TT_Rope, config, rows: MeshRows, h: int, w: int):
    """The mesh's rows of `host_prep.rope_tables` for an (h, w) image."""
    return tt_rope.write(*(rows.held(t) for t in host_prep.rope_tables(config, h, w)))


def check_rope_tables(mesh_device, sp_axis, num_links, config, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    h, w = check_grid(torch.tensor(grid), config)
    rows = MeshRows(h * w, config.max_patches, sp_axis)
    tt_rope = build(mesh_device, sp_axis, num_links, config)
    previous = write_tables(tt_rope, config, rows, 8, 8)
    rope = write_tables(tt_rope, config, rows, h, w)
    assert all(a is b for a, b in zip(rope, previous))
    cos, sin = host_prep.rope_cos_sin(config, h, w)
    cos, sin = (rows.take(host_prep.to_window_order(t, h, w, config.merge_kernel_size)) for t in (cos, sin))
    assert_quality(cos, rows.gather(rope.cos, mesh_device), pcc=0.9999, relative_rmse=0.003, label="cos")
    assert_quality(sin, rows.gather(rope.sin, mesh_device), pcc=0.9999, relative_rmse=0.003, label="sin")


def check_rope_apply(mesh_device, sp_axis, num_links, config, reference, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    rows = MeshRows(h * w, config.max_patches, sp_axis)
    q = random_tensor(h * w, config.num_attention_heads, config.head_dim)
    _, _, rope_freqs_cis = reference_rope(reference, grid_thws)
    golden, _ = apply_rope(q, q, rope_freqs_cis)
    golden = host_prep.to_window_order(golden, h, w, config.merge_kernel_size)

    tt_rope = build(mesh_device, sp_axis, num_links, config)
    write_tables(tt_rope, config, rows, 8, 8)
    rope = write_tables(tt_rope, config, rows, h, w)
    q_window = host_prep.to_window_order(q, h, w, config.merge_kernel_size)
    q_tt = rows.shard(q_window, mesh_device)  # [1, heads, S/sp, head_dim]
    out = ttnn.experimental.rotary_embedding_llama(q_tt, rope.cos, rope.sin, rope.trans_mat, is_decode_mode=False)
    assert_quality(rows.take(golden), rows.gather(out, mesh_device), relative_rmse=0.008)


@P150_AND_RING
@ROW_GRIDS
def test_rope_tables(mesh_device, sp_axis, num_links, config, size):
    check_rope_tables(mesh_device, sp_axis, num_links, config, row_grid(size))


@SIMULATOR
def test_rope_tables_small(mesh_device, sp_axis, num_links, config):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_rope_tables(mesh_device, sp_axis, num_links, simulator_config(config), SIMULATOR_GRID)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_rope_apply(mesh_device, sp_axis, num_links, config, reference, size):
    check_rope_apply(mesh_device, sp_axis, num_links, config, reference, row_grid(size))


@SIMULATOR
def test_rope_apply_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_rope_apply(mesh_device, sp_axis, num_links, simulator_config(config), reference, SIMULATOR_GRID)
