# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.common.tensor_utils import align_shape_to_tile
from models.common.tests.utils import stable_model_seed


def test_stable_model_seed_deterministic() -> None:
    assert stable_model_seed("llama-3") == stable_model_seed("llama-3")


def test_stable_model_seed_distinct() -> None:
    assert stable_model_seed("llama-3") != stable_model_seed("mistral-7b")


def test_stable_model_seed_uint32_range() -> None:
    seed = stable_model_seed("llama-3")
    assert 0 <= seed < 2**32


def test_align_shape_to_tile():
    """Test align_shape_to_tile (replacement for deprecated ttnn.pad_to_tile_shape)."""

    assert align_shape_to_tile([1, 2, 3, 4]) == [1, 2, 32, 32]
    assert align_shape_to_tile([1, 1, 32, 32]) == [1, 1, 32, 32]
    assert align_shape_to_tile([1, 1, 33, 65]) == [1, 1, 64, 96]
    assert align_shape_to_tile([1, 1, 1, 1]) == [1, 1, 32, 32]
    assert align_shape_to_tile([1, 384, 49, 96]) == [1, 384, 64, 96]
    assert align_shape_to_tile([1, 9, 49, 768]) == [1, 9, 64, 768]
    assert align_shape_to_tile([2, 4, 64, 128]) == [2, 4, 64, 128]
    assert align_shape_to_tile((1, 1, 10, 10)) == [1, 1, 32, 32]
    assert align_shape_to_tile([7]) == [32]
    assert align_shape_to_tile([7, 50]) == [32, 64]
    assert align_shape_to_tile([2, 3, 4, 5, 6, 7]) == [2, 3, 4, 5, 32, 32]
    assert align_shape_to_tile([1, 1, 10, 10], tile_size=16) == [1, 1, 16, 16]
    assert align_shape_to_tile([1, 1, 17, 33], tile_size=16) == [1, 1, 32, 48]

    original = [1, 1, 10, 10]
    align_shape_to_tile(original)
    # ensure input is not mutated
    assert original == [1, 1, 10, 10]
