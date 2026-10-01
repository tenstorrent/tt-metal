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

# Sub-tile TILE H/W repeats run row-major between one untilize and one retilize, so they route to
# codegen. The golden is torch: a repeat is a pure copy.
_ROUTING = [
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
    out, grew = _auto_route_grows_cache(device, xt, **kwargs)
    assert_equal(x.repeat(*kwargs["repeat_dims"]), ttnn.to_torch(out))
    assert grew, "auto served a sub-tile TILE H/W repeat on native; expected the codegen round trip"


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---

# Mixed placement (interleaved DRAM input, interleaved L1 output requested via
# memory_config) routes to codegen: DRAM/L1 page alignments differ, so every RM slot
# is sized as the larger of the two aligned pages (rm_slot_bytes). A slot sized from
# one side would overrun destination pages or CB slots and show up as a wrong answer.
_MIXED_PLACEMENT = [
    # higher-dim RM: a writer paced by the wider DRAM pitch would overrun narrower L1 pages
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 3, 1, 1])}, ttnn.ROW_MAJOR_LAYOUT),
    # last-dim RM: a reader paced by the wider DRAM pitch would overrun out-pitched CB slots
    ([1, 2, 10, 20], {"repeat_dims": ttnn.Shape([1, 1, 1, 3])}, ttnn.ROW_MAJOR_LAYOUT),
    # TILE: page size is placement-agnostic
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
    out, grew = _auto_route_grows_cache(device, xt, **kwargs, memory_config=l1_mc)
    assert_equal(x.repeat(*kwargs["repeat_dims"]), ttnn.to_torch(out))
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert grew, "auto served a DRAM->L1 repeat on native; expected codegen"


# A row-major leg pages one stick per CB slot, the slot holds the larger of the leg's input and
# output sticks, and routing sends a leg to codegen only when two slots fit the static L1 window. A
# case where they do not is otherwise fully in codegen scope; without the capacity gate it
# routes to codegen and then throws out of circular-buffer allocation instead of falling back.
#
# 131072 bf16 elements is a 256 KiB input stick. A last-dim x3 repeat makes the output stick
# 768 KiB, so two slots are 1.5 MiB -- the whole of L1 on the largest arch, hence past the window
# on every arch. Native's last-dim CBs hold the input stick, so native still serves it.
_L1_OVERFLOW_WIDTH = 131072


@pytest.mark.parametrize(
    "repeat_dims",
    [ttnn.Shape([1, 1, 1, 3])],
    ids=["repeat_dims=[1, 1, 1, 3]"],
)
def test_repeat_codegen_routing_wide_rm_exceeds_l1(device, repeat_dims):
    shape = [1, 2, 2, _L1_OVERFLOW_WIDTH]
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, repeat_dims))
    # The golden call warms the native program, so only a codegen route grows the cache.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, repeat_dims)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an L1-overflowing case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_a_wide_rm_case_that_exceeds_l1(device, expect_error):
    x = _make_input([1, 2, 2, _L1_OVERFLOW_WIDTH], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 1, 1, 3]))


def _pin_l1_headroom(device, headroom_bytes):
    """Lowers the live L1 frontier to about `headroom_bytes` above the CB base on every bank.

    Interleaved L1 spreads pages round-robin over the banks, so N tiles per bank lower the lowest
    occupied address by the same amount on all of them; only the occupancy matters.
    """
    info = ttnn._ttnn.reports.get_device_info(device)
    tiles_per_bank = (info.cb_limit - headroom_bytes) // (32 * 32 * 2)
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    return resident, resident.buffer_address() - info.address_at_first_l1_cb_buffer


def _wide_last_dim_case(device, num_repeats):
    """A row-major last-dim repeat whose codegen CB slot, the output stick, is a quarter of clear L1."""
    info = ttnn._ttnn.reports.get_device_info(device)
    width = (info.cb_limit // (8 * num_repeats)) // 32 * 32
    x = _make_input([1, 1, 4, width], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return xt, ttnn.Shape([1, 1, 1, num_repeats]), x.repeat(1, 1, 1, num_repeats), 2 * width * num_repeats


def test_repeat_codegen_rm_cb_plan_follows_live_l1(device):
    # Routing budgets two slots against the static L1 window, but the CB depth comes from the L1 free
    # at dispatch. A case warmed on a clear device and repeated with the free window pinned between one
    # and two slots must compile a single-slot program rather than replay the cached double-buffered one,
    # whose CBs would overlap the pinned buffer.
    xt, repeat_dims, expected, slot_bytes = _wide_last_dim_case(device, 2)
    out, grew = _auto_route_grows_cache(device, xt, repeat_dims)
    assert_equal(expected, ttnn.to_torch(out))
    assert grew, "auto served the warm-up on native; expected codegen"
    entries_after_warmup = device.num_program_cache_entries()

    resident, headroom = _pin_l1_headroom(device, 3 * slot_bytes // 2)
    try:
        assert slot_bytes <= headroom < 2 * slot_bytes, f"pinned headroom {headroom} B is not 1-2 slots"
        assert_equal(expected, ttnn.to_torch(ttnn.repeat(xt, repeat_dims)))
        msg = "the pressured call hit the double-buffered program; the CB plan is not in the cache key"
        assert device.num_program_cache_entries() > entries_after_warmup, msg
    finally:
        ttnn.deallocate(resident)


def test_repeat_codegen_falls_back_when_free_l1_holds_no_slot(device):
    # With less than one output stick of L1 free, codegen has no CB plan at all. Native's last-dim CBs
    # stage the input stick, a quarter of it at x4, so the call must route there instead of failing.
    xt, repeat_dims, expected, slot_bytes = _wide_last_dim_case(device, 4)
    device.clear_program_cache()
    resident, headroom = _pin_l1_headroom(device, 3 * slot_bytes // 4)
    try:
        assert slot_bytes // 2 <= headroom < slot_bytes, f"pinned headroom {headroom} B is not 1/2-1 slot"
        assert_equal(expected, ttnn.to_torch(ttnn.repeat(xt, repeat_dims)))
    finally:
        ttnn.deallocate(resident)


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. A sub-tile TILE H-dim repeat needs a row-major leg, and bfloat8_b has none.
    x = _make_input([1, 2, 10, 20], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
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
    device.clear_program_cache()
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


# --- Routing of the sharded and perf-demoted configs ---


def _auto_route_grows_cache(device, xt, repeat_dims, **kwargs):
    """Runs ttnn.repeat after priming the native program on an empty cache; returns (out, grew).

    Clearing first makes "grew" mean "auto compiled a program native does not use" regardless of what
    earlier tests left cached, so a codegen route is observable, not only a native fallback.
    """
    device.clear_program_cache()
    _force_native(xt, repeat_dims, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, repeat_dims, **kwargs)
    return out, device.num_program_cache_entries() > entries_before


_DEMOTED = [
    (
        [1, 2, 128, 64],
        {
            "repeat_dims": ttnn.Shape([1, 1, 1, 2]),
            "memory_config": ttnn.create_sharded_memory_config(
                shape=(1, 2, 128, 128),
                core_grid=ttnn.CoreGrid(x=4, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
        },
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {
            "repeat_dims": ttnn.Shape([1, 1, 1, 2]),
            "memory_config": ttnn.create_sharded_memory_config(
                shape=(1, 2, 128, 128),
                core_grid=ttnn.CoreGrid(x=4, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
        },
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 256, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 256, 128),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 256, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 256, 128),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {
            "repeat_dims": ttnn.Shape([2, 1, 1, 1]),
            "memory_config": ttnn.create_sharded_memory_config(
                shape=(2, 2, 64, 128),
                core_grid=ttnn.CoreGrid(x=4, y=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
        },
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {
            "repeat_dims": ttnn.Shape([2, 1, 1, 1]),
            "memory_config": ttnn.create_sharded_memory_config(
                shape=(2, 2, 64, 128),
                core_grid=ttnn.CoreGrid(x=4, y=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
        },
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
]
_DEMOTED_IDS = [
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input+output|repeat_dims=[1, 1, 1, 2]|float32|row_major",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input+output|repeat_dims=[1, 1, 1, 2]|bfloat16|row_major",
    "[1, 2, 256, 128]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 256, 128]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input+output|repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input+output|repeat_dims=[2, 1, 1, 1]|float32|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _DEMOTED, ids=_DEMOTED_IDS)
def test_repeat_codegen_demotion(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg
    expected = x.repeat(*kwargs["repeat_dims"]).to(golden.dtype)
    assert_equal(expected, ttnn.to_torch(out))
    # A demotion trades speed only: codegen still serves the case, and correctly.
    assert_equal(expected, ttnn.to_torch(_force_codegen(xt, **kwargs)))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
@pytest.mark.parametrize(
    "shape,shard_grid,out_mc,expect_codegen",
    [
        ([1, 2, 256, 128], ttnn.CoreGrid(y=8, x=1), ttnn.L1_MEMORY_CONFIG, True),
        ([1, 2, 256, 128], ttnn.CoreGrid(y=1, x=8), ttnn.L1_MEMORY_CONFIG, False),
        ([1, 5, 256, 128], ttnn.CoreGrid(y=1, x=8), ttnn.L1_MEMORY_CONFIG, False),
        ([1, 1, 256, 128], ttnn.CoreGrid(y=1, x=4), ttnn.L1_MEMORY_CONFIG, True),
        ([1, 2, 256, 128], ttnn.CoreGrid(y=1, x=8), ttnn.DRAM_MEMORY_CONFIG, False),
        ([1, 2, 256, 128], ttnn.CoreGrid(y=1, x=4), ttnn.DRAM_MEMORY_CONFIG, True),
    ],
    ids=[
        "[1,2]-CoreGrid(y=8,x=1)-l1",
        "[1,2]-CoreGrid(y=1,x=8)-l1",
        "[1,5]-CoreGrid(y=1,x=8)-l1",
        "[1,1]-CoreGrid(y=1,x=4)-l1",
        "[1,2]-CoreGrid(y=1,x=8)-dram",
        "[1,2]-CoreGrid(y=1,x=4)-dram",
    ],
)
def test_repeat_codegen_shard_row_read_hotspot(device, shape, shard_grid, out_mc, expect_codegen, dtype):
    # A single shard row of 8+ cores demotes on any arch and either output; shorter rows and columns stay.
    repeat_dims = ttnn.Shape([2, 1, 1, 1])
    x = _make_input(shape, dtype)
    placement = ttnn.create_sharded_memory_config(
        shape=tuple(shape),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=placement)
    out, grew = _auto_route_grows_cache(device, xt, repeat_dims, memory_config=out_mc)
    out = ttnn.to_torch(out)
    assert_equal(x.repeat(*repeat_dims).to(out.dtype), out)
    assert grew == expect_codegen, f"expected {'codegen' if expect_codegen else 'native'}, cache grew={grew}"


_CACHE_HIT = [
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 2, 1, 1])}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    (
        [1, 1, 1, 1],
        {"repeat_dims": ttnn.Shape([1, 3, 10, 20])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
]
_CACHE_HIT_IDS = [
    "[1, 1, 1, 1]|repeat_dims=[1, 2, 1, 1]|bfloat16|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 10, 20]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|tile",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|tile",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|row_major",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_repeat_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    second_golden = ttnn.to_torch(_force_native(yt, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


def _sharded(shape, x, y, strategy):
    return ttnn.create_sharded_memory_config(
        shape=tuple(shape),
        core_grid=ttnn.CoreGrid(x=x, y=y),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )


_H = ttnn.ShardStrategy.HEIGHT
_W = ttnn.ShardStrategy.WIDTH

# Cases codegen serves faster than native, on device and on wall time; a perf demotion must not catch them.
# Each: (shape, repeat_dims, dtype, layout, input placement, output memory_config or None).
_CODEGEN_WINS = [
    *[
        ([1, 2, h, w], [1, 2, 1, 1], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, None)
        for h, w in [(4, 4), (6, 12), (8, 16), (10, 20), (12, 24), (14, 28), (16, 32), (18, 36), (20, 40), (22, 44)]
    ],
    ([1, 1, 1, 1], [1, 2, 1, 1], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, None),
    (
        [1, 2, 64, 128],
        [2, 2, 1, 1],
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        _sharded([1, 2, 64, 128], 4, 1, _W),
        _sharded([2, 4, 64, 128], 4, 1, _W),
    ),
]


def _codegen_win_id(case):
    shape, repeat_dims, dtype, layout, placement, out_mc = case
    where = "dram" if placement == ttnn.DRAM_MEMORY_CONFIG else str(placement.memory_layout).split(".")[-1]
    return f"{shape}x{repeat_dims}-{dtype}-{layout}-{where}"


@pytest.mark.parametrize(
    "shape,repeat_dims,dtype,layout,placement,out_mc", _CODEGEN_WINS, ids=[_codegen_win_id(c) for c in _CODEGEN_WINS]
)
def test_repeat_codegen_win_routes_to_codegen(device, shape, repeat_dims, dtype, layout, placement, out_mc):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    kwargs = {} if out_mc is None else {"memory_config": out_mc}
    out, grew = _auto_route_grows_cache(device, xt, ttnn.Shape(repeat_dims), **kwargs)
    assert_equal(x.repeat(repeat_dims).to(ttnn.to_torch(out).dtype), ttnn.to_torch(out))
    assert grew, "auto routed a codegen-winning case to native (program cache did not grow)"
