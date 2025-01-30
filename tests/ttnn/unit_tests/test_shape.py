# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_shape_equality_operator():
    input_shape_a = ttnn.Shape((1, 2, 3, 4))
    assert input_shape_a == (1, 2, 3, 4)
    assert (1, 2, 3, 4) == input_shape_a

    input_shape_b = ttnn.Shape((1, 2, 3, 4))

    assert input_shape_a == input_shape_b

    input_shape_d = ttnn.Shape((1, 2, 32, 32))
    assert input_shape_a != input_shape_d
    assert input_shape_b != input_shape_d
    assert input_shape_d == (1, 2, 32, 32)
    assert (1, 2, 32, 32) == input_shape_d
