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
        (256, False, (64, 256, 3)),
        (512, False, (128, 256, 3)),
        (1024, False, (96, 256, 1)),
        (256, True, (128, 128, 1)),
        # A quarter slab that is not whole tiles falls back to one tile.
        (64, False, (32, 256, 3)),
        (160, False, (32, 256, 3)),
    ],
)
def test_ring_sdpa_chunk_sizes(slab, sliding, expected):
    assert ring_sdpa_chunk_sizes(slab, sliding) == expected


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


def _tensor(rows, cols):
    return SimpleNamespace(padded_shape=(1, 1, rows, cols))


@pytest.mark.parametrize("n", [2048, 5376])
def test_1d_matmul_config_at_short_m(n):
    config = prefill_1d_matmul_program_config(_tensor(256, 5376), _tensor(5376, n), GRID)
    assert config is not None
    assert (config.per_core_M, config.per_core_N) == (8, 2)
    assert 5376 // 32 % config.in0_block_w == 0 and config.in0_block_w <= 16


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
