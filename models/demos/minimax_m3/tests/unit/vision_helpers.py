# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parts of the vision encoder unit tests: for the device tests (test_vision_*_vs_ref.py) the image grids, the
host to device copies and the comparison with the reference; for the host tests a stand-in for the mesh device
(grid_device), and the weight matmul's checks and circular buffers, for the tests of its program configs and of the
L1 budget. The reference model and its activations are fixtures in tests/conftest.py.
tests/perf/test_vision_encoder_timing.py also uses VISION_GRIDS and to_host."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3.tt.vision import common
from tests.ttnn.utils_for_testing import comp_relative_frobenius

# Patch grid (t, h, w) of one image: its size in pixels divided by the 14 pixel patch. The ids name the grid in
# patches, never as AxB: the Galaxy leg selects -k "8x4", which the pixel size "448x448" would contain.
VISION_GRIDS = {
    "h16w16": (1, 16, 16),  # 224x224: 256 patches, 64 vision tokens
    "h32w32": (1, 32, 32),  # 448x448: 1024 patches
    "h48w48": (1, 48, 48),  # 672x672, the largest the processor makes: 2304 patches, 576 tokens
    "h34w46": (1, 34, 46),  # 640x480 like models/sample_data/huggingface_cat_image.jpg: 1564 patches, not whole tiles
    "h40w54": (1, 40, 54),  # 4:3 photos of 800x600 and larger, resized to 756x560: 2160 patches, not whole tiles
    "h36w36": (1, 36, 36),  # 504x504: 1296 patches, not whole tiles; 1 tile of rows on a transposed split's last core
    "h18w18": (1, 18, 18),  # 252x252: 324 patches, not whole tiles, where QKV and fc1 keep ttnn's own config
    # 168x168: 144 patches, where only the patch embed, the out projection and fc2 get a program config, and SDPA
    # masks 112 of its 256 keys
    "h12w12": (1, 12, 12),
    # 56x56, the smallest the processor makes: 16 patches and 4 tokens, every linear on ttnn's own config, and SDPA
    # with one 128-key chunk, 112 of its keys masked
    "h4w4": (1, 4, 4),
}
# The CI entry "MiniMax-M3 vision unit tests (1-card)", which has a 5 minute timeout, runs h16w16 and h40w54. h40w54
# runs what h16w16 does not: the RoPE op's reload kernels, partial row tiles and, on 12x10, the transposed matmul split
# with a partial last block and program configs for the merge linears. The P150 runners in main's CI logs have 11x10
# grids, where h40w54 takes neither: there the transposed split comes at h36w36 and h48w48, and the merge linears keep
# ttnn's own config at every grid here. M3_VISION_ALL_GRIDS=1 adds the other grids. They are left out rather than
# skipped, so the leg shows no skipped vision tests.
CI_GRIDS = ("h16w16", "h40w54")
# The attention test also runs h18w18 in CI. SDPA pads the keys to a multiple of its 128-key chunk and masks the padded
# ones: 16 of 2176 at 2160 patches, too few for a broken mask to move the output past its bound, and 60 of 384 at 324.
ATTENTION_CI_GRIDS = CI_GRIDS + ("h18w18",)
# The tower test of activations in L1 against DRAM runs h16w16 and h36w36 in CI: both keep the activations in L1 on
# 11x10, 12x10 and 13x10, and h36w36 also the MLP's hidden activations in block shards. h36w36 adds a partial tile row,
# one tile row on the last core of every block linear's split, so a partial last shard, and on 12x10 and 13x10 splits
# that leave cores out.
L1_VS_DRAM_CI_GRIDS = ("h16w16", "h36w36")
# With M3_VISION_ALL_GRIDS=1 that test also runs these, so that on 11x10, 12x10 and 13x10 its images take every layout of
# the MLP's block shards the rule admits: whole shards on 11 columns (1408 patches), whole shards on 13 columns of 13x10
# (1664) and a partial last shard on 12 columns of 12x10 and 13x10 (1440).
L1_VS_DRAM_ALL_GRIDS = {
    **VISION_GRIDS,
    "h32w44": (1, 32, 44),  # 448x616
    "h32w52": (1, 32, 52),  # 448x728
    "h36w40": (1, 36, 40),  # 504x560
}
# The grids of common.L1_GRIDS by test id, without "AxB": the 1-card leg selects "1x1", which "11x10" would contain.
L1_GRID_IDS = {f"x{x}y{y}": (x, y) for x, y in common.L1_GRIDS}
# Twice the largest relative L2 error per test, measured on one P150 with a 12x10 compute grid at the grids of
# VISION_GRIDS from 256 patches up, each logged 4-place value taken at the top of its rounding (0.0059 as 0.00595).
MAX_REL_L2 = {
    "patch_embed": 0.0035,
    "layernorm": 0.0039,
    "attention": 0.0119,
    "mlp": 0.0061,
    "block": 0.0083,
    "projector": 0.0081,
    "tower": 0.0125,
    "final": 0.0149,
}


def parametrize_vision_grids(ci_grids=CI_GRIDS, grids=VISION_GRIDS):
    """Parametrize grid_thw over ci_grids, or over all of grids when M3_VISION_ALL_GRIDS=1. Raises ValueError for an
    id of ci_grids that grids lacks, whose tests would otherwise drop out of the leg without a sign."""
    unknown = sorted(set(ci_grids) - grids.keys())
    if unknown:
        raise ValueError(f"grid ids {unknown} of ci_grids are not among the {len(grids)} grids to take them from")
    run_all = os.getenv("M3_VISION_ALL_GRIDS") == "1"
    params = [pytest.param(grid, id=grid_id) for grid_id, grid in grids.items() if run_all or grid_id in ci_grids]
    return pytest.mark.parametrize("grid_thw", params)


def grid_name(grid_thw) -> str:
    t, h, w = grid_thw
    return f"h{h}w{w}" if t == 1 else f"t{t}h{h}w{w}"


# A check's bound at a grid where its relative L2 error passes MAX_REL_L2. At h4w4 SDPA attends over 16 keys (112 of
# its one 128-key chunk masked), so each output row averages few bf16 products: on a P150 the attention's relL2 was
# 0.0135 there (PCC 0.999915), against 0.0048 to 0.0070 from 144 to 2304 patches.
MAX_REL_L2_AT_GRID = {("attention", "h4w4"): 0.027}


def max_rel_l2(check: str, grid_thw) -> float:
    """The relative L2 bound of check (a key of MAX_REL_L2) at grid_thw: its MAX_REL_L2_AT_GRID bound, if any."""
    return MAX_REL_L2_AT_GRID.get((check, grid_name(grid_thw)), MAX_REL_L2[check])


def rows_to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """(L, D) host rows -> (1, 1, L, D) bf16 tile tensor in DRAM, replicated over the mesh (common.to_device)."""
    return common.to_device(x.reshape(1, 1, *x.shape), mesh_device)


def grid_device(x: int, y: int):
    """Stands in for a mesh device in host tests: the program configs only ask it for its compute grid, x by y cores."""
    return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(x, y))


def to_host(x: ttnn.Tensor, mesh_device, shape) -> torch.Tensor:
    """The first device's copy of x, as an fp32 host tensor of the given shape."""
    out = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:1].reshape(shape).float()


def check_close(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Compares out with the reference ref. Returns the line to log, "[name] <comp_allclose> <PCC> relL2=<relative L2
    error>", and whether out passes: no NaN or Inf, a PCC of at least min_pcc and a relative L2 error of at most
    max_rel_l2. PCC alone passes NaN (comp_pcc zeroes it) and any scale. The line of a failure adds the bounds, the
    relative L2 error to 6 places, the NaN and Inf counts and the least-squares gain of out over ref, which shows a
    scale error.

    With the bounds of MAX_REL_L2 the relative L2 error decides. A NaN or Inf value makes it NaN or Inf, which fails,
    and for the tests' outputs 1 - PCC is about relL2**2 / 2, so the PCC bars of 0.9999 and 0.9998 bind only at errors
    past every bound."""
    pcc_ok, pcc = comp_pcc(ref, out, min_pcc)
    rel_l2, _ = comp_relative_frobenius(ref, out)
    line = f"[{name}] {comp_allclose(ref, out)} {pcc} relL2={rel_l2:.4f}"
    num_nan, num_inf = int(out.isnan().sum()), int(out.isinf().sum())
    passed = pcc_ok and rel_l2 <= max_rel_l2 and not num_nan and not num_inf
    if not passed:
        gain = ((out * ref).sum() / (ref * ref).sum()).item()
        line += (
            f", needs PCC >= {min_pcc} and relL2 <= {max_rel_l2}: relL2 {rel_l2:.6f}, gain {gain:.4f}, {num_nan} NaN"
            f" and {num_inf} Inf in {out.numel()} values"
        )
    return line, passed


def assert_close(name: str, ref: torch.Tensor, out: torch.Tensor, min_pcc: float, max_rel_l2: float):
    """Logs check_close's line and asserts that out passes it."""
    line, passed = check_close(name, ref, out, min_pcc, max_rel_l2)
    logger.info(line)
    assert passed, line


def matmul_cb_bytes(config, k_tiles: int, fp32_dest_acc: bool, bias: bool = True, out_sharded: bool = False) -> int:
    """The circular buffers per core of the 2D multicast matmul factory for a linear_program_config config, bf16 tiles
    of 2 KB (matmul_multicore_reuse_mcast_2d_program_factory.cpp): the in0 and in1 K blocks, double-buffered when K
    takes more than one, the output block, a bias row, and the output block's partial sums, which with fp32
    accumulation are fp32 tiles in a buffer of their own. A block-sharded input keeps its K-block buffers (the shard is
    a buffer of its own, on the tensor). A block-sharded output is the output block's buffer, so that block takes no
    room; the partial sums keep theirs with fp32 accumulation and share the shard without."""
    depth = 2 if k_tiles > config.in0_block_w else 1
    block = config.out_block_h * config.out_block_w
    tiles = depth * config.in0_block_w * (config.out_block_h + config.out_block_w)
    if not out_sharded:
        tiles += block
    if bias:
        tiles += config.out_block_w
    return 2048 * tiles + (4096 * block if fp32_dest_acc else 0)


def matmul_sharded_output(config, m_tiles: int, n_tiles: int):
    """The memory config ttnn gives a matmul's output in L1 block shards for this config, as it derives it
    (matmul_device_operation.cpp, the output spec of a 2D multicast config): blocks of per_core_M x per_core_N tiles
    from core (0, 0), M along the grid's columns with transpose_mcast (COL_MAJOR), along its rows without (ROW_MAJOR).
    """
    blocks_m, blocks_n = math.ceil(m_tiles / config.per_core_M), math.ceil(n_tiles / config.per_core_N)
    columns, rows = (blocks_m, blocks_n) if config.transpose_mcast else (blocks_n, blocks_m)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, rows - 1))})
    orientation = ttnn.ShardOrientation.COL_MAJOR if config.transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR
    shape = (config.per_core_M * ttnn.TILE_SIZE, config.per_core_N * ttnn.TILE_SIZE)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(cores, shape, orientation)
    )


def matmul_config_problems(
    config, rows, in_features, out_features, grid, fp32_dest_acc, in0_shards=None, out_sharded=False
) -> list[str]:
    """The checks the matmul device op and its 2D multicast factory make on this config (matmul_device_operation.cpp,
    matmul_multicore_reuse_mcast_2d_program_factory.cpp), and that its circular buffers fit in LINEAR_L1_BYTES: the
    ones it fails, by name, none where it passes them all. The input and the output are interleaved, in DRAM or L1,
    which those checks do not tell apart, or in L1 block shards: in0_shards is the input's memory config then (with its
    shard spec, as ttnn gave it), and out_sharded is true for an output in block shards. For a sharded input it also
    checks what matmul_cb_bytes takes: the input's shards are whole along K and hold exactly the split's blocks, so no
    core holds a slice of K without an output block, which would take a relay buffer of its own."""
    m_tiles, k_tiles, n_tiles = (math.ceil(d / ttnn.TILE_SIZE) for d in (rows, in_features, out_features))
    blocks_m, blocks_n = math.ceil(m_tiles / config.per_core_M), math.ceil(n_tiles / config.per_core_N)
    blocks_x, blocks_y = (blocks_m, blocks_n) if config.transpose_mcast else (blocks_n, blocks_m)
    grid_size = config.compute_with_storage_grid_size
    checks = {
        "the config's grid is the device's": (grid_size.x, grid_size.y) == grid,
        "K blocks divide K": k_tiles % config.in0_block_w == 0,
        "subblocks divide the output block": config.out_block_h % config.out_subblock_h == 0
        and config.out_block_w % config.out_subblock_w == 0,
        "output blocks divide the core's block": config.per_core_M % config.out_block_h == 0
        and config.per_core_N % config.out_block_w == 0,
        "a subblock fits half the dest register": config.out_subblock_h * config.out_subblock_w
        <= (4 if fp32_dest_acc else 8),
        "per_core_M at most M": config.per_core_M <= m_tiles,
        "the split fits the grid": blocks_x <= grid[0] and blocks_y <= grid[1],
        # As ttnn's own 2D configs for DRAM operands.
        "a 2D split": blocks_x >= 2 and blocks_y >= 2,
        # A sharded input needs fused batches, which with one batch change nothing; the encoder passes it only then.
        "fuse_batch for a sharded input only": config.fuse_batch == (in0_shards is not None),
        "no fused activation": config.fused_activation is None,
        "the buffers fit LINEAR_L1_BYTES": matmul_cb_bytes(config, k_tiles, fp32_dest_acc, out_sharded=out_sharded)
        <= common.LINEAR_L1_BYTES,
    }
    if in0_shards is not None:
        spec = in0_shards.shard_spec
        box = spec.grid.bounding_box()
        size = box.grid_size()
        shard_m, shard_k = spec.shape[0] // ttnn.TILE_SIZE, spec.shape[1] // ttnn.TILE_SIZE
        shards_m, shards_k = (size.x, size.y) if config.transpose_mcast else (size.y, size.x)
        checks.update(
            {
                "the input in L1 block shards": in0_shards.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
                and in0_shards.buffer_type == ttnn.BufferType.L1,
                "COL_MAJOR shards with transpose_mcast, else ROW_MAJOR": (
                    spec.orientation == ttnn.ShardOrientation.COL_MAJOR
                )
                == config.transpose_mcast,
                "per_core_M is the shard height": config.per_core_M == shard_m,
                "K blocks divide the shard width": shard_k % config.in0_block_w == 0,
                "the shards from core (0, 0), inside the grid": (box.start.x, box.start.y) == (0, 0)
                and box.end.x < grid[0]
                and box.end.y < grid[1],
                "one shard per block of M": shards_m == blocks_m,
                "one K shard per block of N, no relay buffer": shards_k == blocks_n,
                "whole K shards": shards_k * shard_k == k_tiles,
            }
        )
    if out_sharded:
        checks.update(
            {
                "a sharded output's subblock one tile high or as wide as the block": config.out_subblock_w
                == config.per_core_N
                or config.out_subblock_h == 1,
                "a sharded output's block one tile high or as wide as the core's": config.out_block_w
                == config.per_core_N
                or config.out_block_h == 1,
                # The factory's limit on a sharded output's tiles.
                "a sharded output's block at most 65535 tiles": config.per_core_M * config.per_core_N <= 65535,
            }
        )
    return [name for name, passed in checks.items() if not passed]


def assert_matmul_config_valid(
    config, rows, in_features, out_features, grid, fp32_dest_acc, in0_shards=None, out_sharded=False
):
    """Asserts that the config passes matmul_config_problems's checks."""
    problems = matmul_config_problems(
        config, rows, in_features, out_features, grid, fp32_dest_acc, in0_shards=in0_shards, out_sharded=out_sharded
    )
    assert not problems, f"{config} fails: {problems}"
