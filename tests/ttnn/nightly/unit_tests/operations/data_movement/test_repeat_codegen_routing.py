# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing-fallback coverage: every case the codegen gate rejects must fall back to native.
# The generated block below is emitted from the port's coverage ledger; hand-add off-grid
# regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.repeat` takes no implementation argument -- it routes on its own. The forced-native golden
# leg therefore comes from the verification-only entry in the private module; see repeat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.repeat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_DTYPES = [ttnn.bfloat16]
_DTYPE_IDS = ["bfloat16"]

_ROUTING = [
    # TILE H-dim repeat on non-tile-aligned H=1 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=10 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=12 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 12, 24], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=14 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 14, 28], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=16 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 16, 32], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=18 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 18, 36], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=20 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 20, 40], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=22 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 22, 44], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=4 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 4], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 4, 8], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=6 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 6, 12], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
    # TILE H-dim repeat on non-tile-aligned H=8 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 10, 20])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 12, 24])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 14, 28])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 16, 32])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 18, 36])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 20, 40])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 22, 44])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 4, 8])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 6, 12])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([1, 3, 8, 16])}, ttnn.TILE_LAYOUT),
    ([1, 2, 8, 16], {"repeat_dims": ttnn.Shape([2, 2, 2, 2])}, ttnn.TILE_LAYOUT),
]
_ROUTING_IDS = [
    # TILE H-dim repeat on non-tile-aligned H=1 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 1, 1, 1]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=10 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 10, 20]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=12 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 12, 24]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 12, 24]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=14 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 14, 28]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 14, 28]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=16 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 16, 32]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 16, 32]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=18 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 18, 36]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 18, 36]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=20 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 20, 40]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 20, 40]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=22 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 22, 44]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 22, 44]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=4 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 4, 4]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 4, 4]|repeat_dims=[2, 2, 2, 2]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 4, 8]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 4, 8]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=6 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 6, 12]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 6, 12]|repeat_dims=[2, 2, 2, 2]|tile",
    # TILE H-dim repeat on non-tile-aligned H=8 (H % 32 != 0) is not tile-page representable: RepeatCodegen repeats in tile-page space (ceil(H/32) tiles), which shape-mismatches / folds pad rows for a sub-tile H repeat
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 10, 20]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 12, 24]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 14, 28]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 16, 32]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 18, 36]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 20, 40]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 22, 44]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 4, 8]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 6, 12]|tile",
    "[1, 2, 8, 16]|repeat_dims=[1, 3, 8, 16]|tile",
    "[1, 2, 8, 16]|repeat_dims=[2, 2, 2, 2]|tile",
]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("shape,kwargs,layout", _ROUTING, ids=_ROUTING_IDS)
def test_repeat_codegen_routing(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    # The golden call warms the native program, so a correct fallback leaves the cache
    # unchanged; only a mis-route to codegen compiles a new program and grows it.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---

# Mixed placement (interleaved DRAM input, interleaved L1 output requested via
# memory_config) must route to native: the RM factories derive CB slot sizes and
# per-page transfer sizes from one side's aligned page size, and DRAM/L1 page
# alignments differ, so a cross-placement call would overrun destination pages or
# CB slots. All three cases are otherwise in codegen scope, so the placement gate
# is the only thing demoting them.
_MIXED_PLACEMENT = [
    # higher-dim RM: a writer paced by the wider DRAM pitch would overrun narrower L1 pages
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 1, 1])}, ttnn.ROW_MAJOR_LAYOUT),
    # last-dim RM: a reader paced by the wider DRAM pitch would overrun out-pitched CB slots
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 1, 1, 3])}, ttnn.ROW_MAJOR_LAYOUT),
    # TILE: page size is placement-agnostic today; kept so relaxing the gate is a conscious act
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 1, 1])}, ttnn.TILE_LAYOUT),
]

_MIXED_PLACEMENT_IDS = [
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 1, 1]|row_major|dram_to_l1",
    "[1, 2, 10, 20]|repeat_dims=[1, 1, 1, 3]|row_major|dram_to_l1",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 1, 1]|tile|dram_to_l1",
]


@pytest.mark.parametrize("shape,kwargs,layout", _MIXED_PLACEMENT, ids=_MIXED_PLACEMENT_IDS)
def test_repeat_codegen_routing_mixed_placement(device, shape, kwargs, layout):
    l1_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs, memory_config=l1_mc))
    # Same cache-growth route assertion as the generated matrix above.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, **kwargs, memory_config=l1_mc)
    assert_equal(golden, ttnn.to_torch(out))
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    msg = "auto routed a mixed-placement case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# Both RM factories size their CB as kRepeatCbDepth (8) slots of one stick, and a stick scales with
# the tensor's width, so a wide enough input projects a CB no core's L1 can hold. Such a case is
# otherwise fully in codegen scope; without the capacity gate it routes to codegen and then throws
# out of circular-buffer allocation at program-compile time rather than falling back.
#
# 131072 bf16 elements is a 256 KiB stick, so the projected CB is 2 MiB -- past L1 on every arch
# (1.5 MiB at most) with room to spare, so the case stays out of scope without tracking the exact
# per-core budget, which moves with whatever else is allocated.
_L1_OVERFLOW_WIDTH = 131072


@pytest.mark.parametrize(
    "repeat_dims",
    [ttnn.Shape([1, 3, 1, 1]), ttnn.Shape([1, 1, 3, 1])],
    ids=["repeat_dims=[1, 3, 1, 1]", "repeat_dims=[1, 1, 3, 1]"],
)
def test_repeat_codegen_routing_wide_rm_exceeds_l1(device, repeat_dims):
    shape = [1, 2, 2, _L1_OVERFLOW_WIDTH]
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    golden = ttnn.to_torch(_force_native(xt, repeat_dims))
    # Same cache-growth route assertion as the generated matrix above.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, repeat_dims)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an L1-overflowing case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_a_wide_rm_case_that_exceeds_l1(device, expect_error):
    x = _make_input([1, 2, 2, _L1_OVERFLOW_WIDTH], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 3, 1, 1]))


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. A TILE H-dim repeat needs a tile-aligned H, and H=10 is not.
    x = _make_input([1, 2, 10, 20], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 1, 3, 1]))


# Hand-added: tile-geometry routing, over both axes a tile varies on. An off-default *shape* changes
# the page count and the page size, and the host-side page map feeding the codegen prim derives Ht/Wt
# from the 32x32 constants. A transposed 32x32 leaves those two quantities alone -- so no
# page-geometry check can see it -- but the datums inside the page are swizzled, and the codegen
# output spec is derived from the layout alone and so comes back with the flags cleared. Both have to
# reach native. H is tile-aligned and the dtype/layout are in scope, so only the tile drives the
# route.
#
# Route only, no value assertion: native does not serve either tile correctly -- for 16x16 two native
# calls on the same input disagree with each other, and for a transposed 32x32 native and codegen
# both differ from torch. What this port owes is that it declines the case instead of answering it
# wrongly in its own way. Native's answer is not a reference here.
_OFF_DEFAULT_TILES = [ttnn.Tile([16, 16]), ttnn.Tile([32, 32], transpose_tile=True)]
_OFF_DEFAULT_TILE_IDS = ["shape_16x16", "transposed_32x32"]


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_repeat_non_default_tile_routes_to_native(device, tile):
    shape = [1, 1, 32, 64]
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    repeat_dims = ttnn.Shape([1, 1, 3, 1])
    # Primes the cache with the native program, so an unchanged count means native served the call.
    _force_native(xt, repeat_dims)
    entries_before = device.num_program_cache_entries()
    ttnn.repeat(xt, repeat_dims)
    msg = "auto routed a non-default tile to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_forced_codegen_refuses_non_default_tile(device, expect_error, tile):
    x = _make_input([1, 1, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 1, 3, 1]))
