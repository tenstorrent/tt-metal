import numpy as np

import ttnn


def test_quantize_uint8_saturates_both_bounds_exactly():
    actual = ttnn.quantize([-2.0, -0.6, 0.0, 1.4, 300.0], 1.0, 0, np.uint8)
    np.testing.assert_array_equal(actual, [0, 0, 0, 1, 255])


def test_requantize_uint8_saturates_both_bounds_exactly():
    actual = ttnn.requantize([-100, 0, 2, 300], 1.0, 1.0, 0, 0, np.uint8)
    np.testing.assert_array_equal(actual, [0, 0, 2, 255])


def test_uint8_negative_values_do_not_wrap_or_mirror():
    positive = ttnn.quantize([5.0], 1.0, 0, np.uint8)
    negative = ttnn.quantize([-5.0], 1.0, 0, np.uint8)
    np.testing.assert_array_equal(positive, [5])
    np.testing.assert_array_equal(negative, [0])


def test_int8_behavior_is_saturated_and_preserved():
    actual = ttnn.quantize([-200, -3, 4, 200], 1.0, 0, np.int8)
    np.testing.assert_array_equal(actual, [-128, -3, 4, 127])
