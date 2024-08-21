# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_shape_equality_operator():
    input_shape_a = ttnn.Shape((1, 2, 3, 4))
    assert input_shape_a == (1, 2, 3, 4)
    assert (1, 2, 3, 4) == input_shape_a
    assert input_shape_a.with_tile_padding() == (1, 2, 3, 4)
    assert (1, 2, 3, 4) == input_shape_a.with_tile_padding()

    input_shape_b = ttnn.Shape((1, 2, 3, 4))

    assert input_shape_a == input_shape_b

    input_shape_c = ttnn.Shape((1, 2, 3, 4), (1, 2, 32, 32))
    assert input_shape_a != input_shape_c
    assert input_shape_b != input_shape_c
    assert input_shape_c == (1, 2, 3, 4)
    assert (1, 2, 3, 4) == input_shape_c
    assert input_shape_c.with_tile_padding() == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_c.with_tile_padding()

    input_shape_d = ttnn.Shape((1, 2, 32, 32))
    assert input_shape_a != input_shape_d
    assert input_shape_b != input_shape_d
    assert input_shape_c != input_shape_d
    assert input_shape_d == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_d
    assert input_shape_d.with_tile_padding() == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_d.with_tile_padding()

    input_shape_e = ttnn.Shape((1, 2, 3, 4), (1, 2, 32, 32))
    assert input_shape_a != input_shape_e
    assert input_shape_b != input_shape_e
    assert input_shape_c == input_shape_e
    assert input_shape_d != input_shape_e
    assert input_shape_e == (1, 2, 3, 4)
    assert (1, 2, 3, 4) == input_shape_e
    assert (1, 2, 32, 32) == input_shape_d
    assert input_shape_e.with_tile_padding() == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_e.with_tile_padding()

    input_shape_f = ttnn.Shape((1, 2, 4, 4), (1, 2, 32, 32))
    assert input_shape_a != input_shape_f
    assert input_shape_b != input_shape_f
    assert input_shape_c != input_shape_f
    assert input_shape_d != input_shape_f
    assert input_shape_e != input_shape_f
    assert input_shape_f == (1, 2, 4, 4)
    assert (1, 2, 4, 4) == input_shape_f
    assert input_shape_f.with_tile_padding() == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_f.with_tile_padding()

    input_shape_g = ttnn.Shape((2, 4, 4), (2, 32, 32))
    assert input_shape_a != input_shape_g
    assert input_shape_b != input_shape_g
    assert input_shape_c != input_shape_g
    assert input_shape_d != input_shape_g
    assert input_shape_e != input_shape_g
    assert input_shape_e != input_shape_g
    assert input_shape_f != input_shape_g
    assert input_shape_g == (2, 4, 4)
    assert (2, 4, 4) == input_shape_g
    assert input_shape_g.with_tile_padding() == (2, 32, 32)
    assert (2, 32, 32) == input_shape_g.with_tile_padding()
