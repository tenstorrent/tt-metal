# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Contracts: (1) every case the codegen gate rejects falls back to native under ordinary
# routing; (2) an accepted case dispatched twice under forced codegen stays a program-cache hit
# and rebinds its buffers; (3) the forced-codegen entry refuses an out-of-scope case rather than
# falling back. The block below is generated from the op's coverage data; hand-add off-grid
# regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.concat` takes no implementation argument -- it routes on its own. The forced legs therefore
# come from the verification-only entries in the private module; see concat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.concat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.concat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _inputs(shapes, dtype, layout, device):
    xs = [ttnn.from_torch(_make_input(s, dtype), dtype=dtype, layout=layout, device=device) for s in shapes]
    return xs


_ROUTING = [
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.float32,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.int32,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
    ),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
]
_ROUTING_IDS = [
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|bfloat16|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|float32|row_major",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|float32|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|int32|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|uint32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|bfloat16|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|float32|row_major",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|float32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|int32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|uint32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat16|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|row_major",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|int32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|uint32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat16|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|row_major",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|int32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|uint32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|bfloat16|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|float32|row_major",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|float32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|int32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|uint32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|bfloat16|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|float32|row_major",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|float32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|int32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|uint32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|bfloat16|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|bfloat8_b|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|float32|row_major",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|float32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|int32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|uint32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|bfloat16|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|bfloat8_b|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|float32|row_major",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|float32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|int32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|uint32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat16|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat8_b|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|float32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|float32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|int32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|uint32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|bfloat16|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|bfloat8_b|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|float32|row_major",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|float32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|int32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|uint32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|bfloat16|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|bfloat8_b|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|float32|row_major",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|float32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|int32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|uint32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|bfloat16|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|bfloat8_b|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|float32|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|float32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|int32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|uint32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|bfloat16|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|bfloat8_b|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|float32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|float32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|int32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|uint32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|bfloat16|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|bfloat8_b|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|float32|row_major",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|float32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|int32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|uint32|tile",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _ROUTING, ids=_ROUTING_IDS)
def test_concat_codegen_routing(device, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xs, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat16|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_program_cache_hit(device, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    ys = _inputs(shapes, dtype, layout, device)
    second_golden = ttnn.to_torch(_force_native(ys, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(ys, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


# One case per rejection clause the gate owns, so a forced leg cannot quietly serve native.
_OUT_OF_SCOPE = [
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_OUT_OF_SCOPE_IDS = ["tile_layout", "unsupported_dtype", "single_input"]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _OUT_OF_SCOPE, ids=_OUT_OF_SCOPE_IDS)
def test_forced_codegen_refuses_out_of_scope_case(device, expect_error, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, **kwargs)


# --- hand-added below the generated block ---

# The factory dispatches on input count and on whether the concat dim is the last one, and each of
# the four builders assembles its own runtime-argument list with its own buffer bindings. The
# generated leg above parametrizes over layout, which for this op is one value, so it reaches only
# the N-way width builder; a stale-address regression in any of the other three would return the
# first allocation's data on a cache hit with nothing to catch it. Those three are the cases here.
_CACHE_HIT_BRANCHES = [
    ([[1, 32, 32], [1, 64, 32]], {"dim": 1}, "two_input_nonwidth"),
    ([[1, 32, 32], [1, 32, 64]], {"dim": 2}, "two_input_width"),
    ([[1, 32, 32], [1, 64, 32], [1, 32, 32]], {"dim": 1}, "nway_nonwidth"),
]


@pytest.mark.parametrize(
    "shapes,kwargs", [(c[0], c[1]) for c in _CACHE_HIT_BRANCHES], ids=[c[2] for c in _CACHE_HIT_BRANCHES]
)
def test_concat_codegen_program_cache_hit_every_builder(device, shapes, kwargs):
    dtype, layout = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    ys = _inputs(shapes, dtype, layout, device)
    second_golden = ttnn.to_torch(_force_native(ys, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(ys, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


def test_concat_codegen_declines_host_tensors(expect_error):
    # The gate reads page sizes and alignments off buffers a host tensor does not own, so it has to
    # decline before asking anything else -- otherwise routing dereferences null ahead of the
    # validation that would have named the real problem.
    xs = [ttnn.from_torch(torch.rand(1, 32, 32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def test_concat_codegen_declines_zero_volume(device, expect_error):
    # No builder has a zero-work path, and a zero-width output makes the stick count a division by
    # zero, so these shapes have to stay on native.
    xs = _inputs([[1, 32, 0], [1, 32, 0]], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)
