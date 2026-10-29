# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of the vision encoder's weight-matmul program configs (tt/vision/common.py, linear_program_config).
No device: the configs are built with the real ttnn classes and checked on the host.

- At 672x672 (2304 patches) on a 12x10 grid every linear gets the config pinned in CONFIGS_672.
- At the smaller sizes of CONFIGS_MID on 12x10, a linear whose transposed split would have fewer than
  LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core gets the untransposed split or keeps ttnn's own config, and one
  with exactly that many gets the transposed split.
- At 256 patches (the CI grid h16w16), on 11x10, 12x10 and 13x10, the tower's linears get a config and the merge
  linears keep ttnn's own, so the 1-card device tests run both paths; linear_1 and linear_2 get a config on 13x10 only.
  At 16 patches, the smallest image, every linear keeps ttnn's own.
- The helper takes its three limits as arguments: with other limits, linear_2 at 1024 patches gets the transposed split
  and fc1 at 2304 patches other K blocks.
- For every patch count an image can have (multiples of 4 up to 2304), on the 11x10, 12x10 and 13x10 core grids that
  Blackhole chips have depending on harvesting, every config passes the checks the matmul device op and its 2D
  multicast factory make, the factory's circular buffers fit in LINEAR_L1_BYTES, and no config uses transpose_mcast
  with fewer than LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core.

The 1-card leg selects these tests by name (vision_linear_program_config), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_linear_program_config.py
"""
from __future__ import annotations

import math

import pytest

import ttnn
from models.demos.minimax_m3.tt.vision.common import (
    LINEAR_L1_BYTES,
    LINEAR_MIN_TRANSPOSED_PER_CORE_M,
    _linear_blocking,
    hifi4_compute_config,
    linear_compute_config,
    linear_program_config,
)
from models.demos.minimax_m3.tt.vision.model_config import MiniMaxM3VLModelArgs

ARGS = MiniMaxM3VLModelArgs()
HEADS_SIZE = ARGS.num_attention_heads * ARGS.padded_head_dim  # q, k and v each, with the heads padded (1536)
MERGE = ARGS.spatial_merge_size**2
# The encoder's weight matmuls: (in_features, out_features, patches per row).
LINEARS = {
    "patch_embed": (ARGS.patch_flat_dim, ARGS.hidden_size, 1),
    "qkv": (ARGS.hidden_size, 3 * HEADS_SIZE, 1),
    "out_proj": (HEADS_SIZE, ARGS.hidden_size, 1),
    "fc1": (ARGS.hidden_size, ARGS.intermediate_size, 1),
    "fc2": (ARGS.intermediate_size, ARGS.hidden_size, 1),
    "linear_1": (ARGS.hidden_size, ARGS.projector_hidden_size, 1),
    "linear_2": (ARGS.projector_hidden_size, ARGS.text_hidden_size, 1),
    "merge_linear_1": (ARGS.merged_hidden_size, ARGS.projector_hidden_size, MERGE),
    "merge_linear_2": (ARGS.projector_hidden_size, ARGS.text_hidden_size, MERGE),
}
FIELDS = ("transpose_mcast", "per_core_M", "per_core_N", "in0_block_w", "out_subblock_h", "out_subblock_w")
# At 2304 patches on a 12x10 grid.
CONFIGS_672 = {
    "patch_embed": (True, 6, 4, 1, 2, 2),
    "qkv": (True, 6, 15, 4, 3, 1),
    "out_proj": (True, 6, 4, 8, 2, 2),
    "fc1": (True, 6, 16, 4, 2, 2),
    "fc2": (True, 6, 4, 8, 2, 2),
    "linear_1": (True, 6, 20, 2, 2, 2),
    "linear_2": (True, 6, 20, 2, 2, 2),  # K blocks of 4 ran faster in a matmul sweep but overflow LINEAR_L1_BYTES
    "merge_linear_1": (False, 2, 16, 8, 2, 2),
    "merge_linear_2": (False, 2, 16, 8, 2, 2),
}
# (linear, patches): the config on a 12x10 grid, None for ttnn's own. The comments give the image and the per-core
# block (M x N tiles) of the transposed split.
CONFIGS_MID = {
    ("qkv", 324): None,  # 252x252; 1x15
    ("fc2", 324): (False, 2, 4, 8, 2, 2),  # 1x4
    ("linear_2", 768): None,  # 336x448; 2x20
    ("linear_2", 1024): None,  # 448x448; 3x20
    ("merge_linear_2", 1296): None,  # 504x504; 1x20
    ("fc1", 1296): (True, 4, 16, 5, 2, 2),  # 4x16
}
# Ids without "AxB": the 1-card leg selects "1x1", which "11x10" would contain.
GRIDS = {"x11y10": (11, 10), "x12y10": (12, 10), "x13y10": (13, 10)}


class _Device:
    """Stands in for the mesh device: linear_program_config only asks it for its compute grid."""

    def __init__(self, x: int, y: int):
        self.grid = ttnn.CoreCoord(x, y)

    def compute_with_storage_grid_size(self):
        return self.grid


def _config(name, patches, grid=(12, 10), compute_kernel_config=None):
    """linear_program_config for one of LINEARS in the forward of an image of `patches` patches."""
    in_features, out_features, patches_per_row = LINEARS[name]
    if compute_kernel_config is None:
        compute_kernel_config = linear_compute_config()
    return linear_program_config(
        _Device(*grid), patches // patches_per_row, in_features, out_features, compute_kernel_config
    )


def _assert_valid(config, rows, in_features, out_features, grid, fp32_dest_acc):
    """The checks the matmul device op and its 2D multicast factory make on this config, for operands and output in
    DRAM (matmul_device_operation.cpp, matmul_multicore_reuse_mcast_2d_program_factory.cpp)."""
    m_tiles, k_tiles, n_tiles = (math.ceil(d / ttnn.TILE_SIZE) for d in (rows, in_features, out_features))
    assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == grid
    assert k_tiles % config.in0_block_w == 0
    assert config.out_block_h % config.out_subblock_h == 0 and config.out_block_w % config.out_subblock_w == 0
    assert config.per_core_M % config.out_block_h == 0 and config.per_core_N % config.out_block_w == 0
    assert config.out_subblock_h * config.out_subblock_w <= (4 if fp32_dest_acc else 8)  # half the dest register
    assert config.per_core_M <= m_tiles
    blocks_x, blocks_y = math.ceil(n_tiles / config.per_core_N), math.ceil(m_tiles / config.per_core_M)
    if config.transpose_mcast:
        blocks_x, blocks_y = blocks_y, blocks_x
    assert blocks_x <= grid[0] and blocks_y <= grid[1]
    assert blocks_x >= 2 and blocks_y >= 2  # a 2D split, as ttnn's own 2D configs for DRAM operands
    assert not config.fuse_batch and config.fused_activation is None
    # The factory's circular buffers, bf16 tiles of 2 KB: the in0 and in1 K blocks (double-buffered when K takes
    # more than one), the output block, a bias row, and the output block's partial sums, which with fp32
    # accumulation are fp32 tiles in a buffer of their own.
    depth = 2 if k_tiles > config.in0_block_w else 1
    block = config.out_block_h * config.out_block_w
    cb_bytes = 2048 * (depth * config.in0_block_w * (config.out_block_h + config.out_block_w) + block)
    cb_bytes += 2048 * config.out_block_w + (4096 * block if fp32_dest_acc else 0)
    assert cb_bytes <= LINEAR_L1_BYTES


def test_vision_linear_program_config_pinned():
    """The configs at 672x672 and at the sizes of CONFIGS_MID on 12x10; which linears keep ttnn's own config at 256
    patches on each grid and at 16 patches; the helper's choice with other limits."""
    for name, want in CONFIGS_672.items():
        config = _config(name, 2304)
        assert config is not None, name
        assert tuple(getattr(config, f) for f in FIELDS) == want, f"{name}: {config}"
        assert (config.out_block_h, config.out_block_w) == (config.per_core_M, config.per_core_N), name
        in_features, out_features, patches_per_row = LINEARS[name]
        _assert_valid(config, 2304 // patches_per_row, in_features, out_features, (12, 10), fp32_dest_acc=True)
    for (name, patches), want in CONFIGS_MID.items():
        config = _config(name, patches)
        got = None if config is None else tuple(getattr(config, f) for f in FIELDS)
        assert got == want, f"{name} at {patches} patches: {config}"
    tower = {"patch_embed", "qkv", "out_proj", "fc1", "fc2"}
    # On 13x10 linear_1 and linear_2 get 1x15 tiles per core at 256 patches, fewer than a 1D multicast's 16.
    for grid, with_config in (((11, 10), tower), ((12, 10), tower), ((13, 10), tower | {"linear_1", "linear_2"})):
        assert {name for name in LINEARS if _config(name, 256, grid) is not None} == with_config, grid
    assert all(_config(name, 16) is None for name in LINEARS)
    # The limits are arguments of the cached helper. In tiles (M, K, N): linear_2 at 1024 patches with a limit of 3
    # tiles of M per core, and fc1 at 2304 with 1.4 MB of circular buffers and with K blocks of at most 2.
    assert _linear_blocking(32, 192, 192, 12, 10, True, min_transposed_per_core_m=3) == (3, 20, 6, 1, 4, True)
    assert _linear_blocking(72, 40, 160, 12, 10, True, l1_bytes=1_400_000) == (6, 16, 8, 2, 2, True)
    assert _linear_blocking(72, 40, 160, 12, 10, True, max_in0_block_w=2) == (6, 16, 2, 2, 2, True)


@pytest.mark.parametrize("fp32_dest_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("grid", list(GRIDS.values()), ids=list(GRIDS))
def test_vision_linear_program_config_valid(grid, fp32_dest_acc):
    """Every config for every image size, as the device op and the 2D factory check it, and transpose_mcast only with
    at least LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M per core."""
    compute_kernel_config = hifi4_compute_config(fp32_dest_acc=fp32_dest_acc, packer_l1_acc=True)
    configs = transposed = 0
    for name, (in_features, out_features, patches_per_row) in LINEARS.items():
        for patches in range(16, 2304 + 1, 4):
            config = _config(name, patches, grid, compute_kernel_config)
            if config is not None:
                _assert_valid(config, patches // patches_per_row, in_features, out_features, grid, fp32_dest_acc)
                if config.transpose_mcast:
                    assert config.per_core_M >= LINEAR_MIN_TRANSPOSED_PER_CORE_M, f"{name} at {patches} patches"
                    transposed += 1
                configs += 1
    assert configs > 0 and transposed > 0
