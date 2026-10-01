# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the prefill SDPA, matmul and RMSNorm config choices."""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gemma4_d_p.tt.attention.operations import projection_math_fidelity
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_sdpa_chunk_sizes
from models.demos.gemma4_d_p.tt.matmul_config import prefill_1d_matmul_program_config
from models.demos.gemma4_d_p.tt.rms_norm import _block_shard_geometry

GRID = SimpleNamespace(x=11, y=10)


@pytest.mark.parametrize(
    "slab, sliding, expected",
    [
        (256, False, (64, 256, 3, True)),
        (512, False, (128, 256, 3, True)),
        (1024, False, (96, 256, 1, True)),
        (256, True, (128, 128, 1, False)),
        # A quarter slab that is not whole tiles falls back to one tile.
        (64, False, (32, 256, 3, True)),
        (160, False, (32, 256, 3, True)),
        # Segmented accumulation needs one Q chunk per core: 8 heads x ceil(slab / q) <= 110 cores.
        (1536, False, (128, 256, 1, True)),
        (1664, False, (128, 256, 1, True)),
        # Too long for any q tried (chunk 16384: 16 chunks of q 128 per head): q 96 without segments.
        (2048, False, (96, 256, 1, False)),
        (4096, False, (96, 256, 1, False)),
    ],
)
def test_ring_sdpa_chunk_sizes(slab, sliding, expected):
    assert ring_sdpa_chunk_sizes(slab, sliding, num_heads=8, num_cores=110) == expected


@pytest.mark.parametrize(
    "rows, expected",
    [
        (256, ttnn.MathFidelity.HiFi2),
        (511, ttnn.MathFidelity.HiFi2),
        (512, ttnn.MathFidelity.LoFi),
        (1024, ttnn.MathFidelity.LoFi),
    ],
)
def test_projection_math_fidelity(rows, expected):
    assert projection_math_fidelity(rows) == expected


def _tensor(rows, cols, shard_cols=None):
    """A host stand-in for a tensor; shard_cols makes it width-sharded with that shard width."""
    shard_spec = SimpleNamespace(shape=(rows, shard_cols))
    return SimpleNamespace(
        padded_shape=(1, 1, rows, cols),
        is_sharded=lambda: shard_cols is not None,
        memory_config=lambda: SimpleNamespace(shard_spec=shard_spec),
    )


@pytest.mark.parametrize("n", [2048, 5376])
def test_1d_matmul_config_at_short_m(n):
    config = prefill_1d_matmul_program_config(_tensor(256, 5376), _tensor(5376, n), GRID)
    assert config is not None
    assert (config.per_core_M, config.per_core_N) == (8, 2)
    assert 5376 // 32 % config.in0_block_w == 0 and config.in0_block_w <= 16


def test_1d_matmul_config_reads_sharded_activation_one_shard_per_k_block():
    # Chunk 2048's MLP down: 8 tile rows, K 5376 in 8-tile shards, 4 weight columns per core.
    config = prefill_1d_matmul_program_config(
        _tensor(256, 5376, shard_cols=256), _tensor(5376, 5376), GRID, per_core_n=4
    )
    assert (config.in0_block_w, config.per_core_N) == (8, 4)
    # fp32 dest: at most 4 tiles per output subblock.
    assert config.out_subblock_h * config.out_subblock_w <= 4


@pytest.mark.parametrize(
    "rows, n",
    [(512, 2048), (256, 2 * 32 * 111), (256, 3 * 32)],  # M above 8 tiles, more columns than cores, odd width
)
def test_1d_matmul_config_falls_back(rows, n):
    assert prefill_1d_matmul_program_config(_tensor(rows, 5376), _tensor(5376, n), GRID) is None


# Rows per device after the TP row split: 64 / 128 / 256 at chunk 2048 / 4096 / 8192.
@pytest.mark.parametrize(
    "rows, expected", [(64, (2, 14, 1, 12)), (128, (2, 14, 2, 12)), (256, (2, 14, 4, 12)), (1024, (4, 21, 8, 8))]
)
def test_rms_norm_block_shard_geometry(rows, expected):
    assert _block_shard_geometry(rows, 5376) == expected


def test_rms_norm_block_shard_falls_back_when_blocks_overflow_l1():
    assert _block_shard_geometry(2048, 5376) is None
