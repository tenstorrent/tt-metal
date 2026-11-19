# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.init's RNG state: ``set_rng_state`` continues from a state ``get_rng_state`` returned, and a state it refuses
leaves the generator in place. Device-free."""

import numpy as np
import pytest

import ttml


@pytest.fixture(autouse=True)
def _keep_init_rng():
    """The generator later tests would have drawn from, put back after each test here."""
    saved = ttml.init._rng
    yield
    ttml.init._rng = saved


def test_set_rng_state_continues_from_a_state_get_rng_state_returned():
    ttml.init.manual_seed(3)
    ttml.init._rng.integers(0, 2**32, dtype=np.uint32)  # mid-stream: half of a 64-bit output is cached
    state = ttml.init.get_rng_state()
    assert state["bit_generator"] == "PCG64" and state["has_uint32"] == 1
    expected = (ttml.init._rng.integers(0, 2**32, dtype=np.uint32), ttml.init._rng.random(4).tolist())

    ttml.init.manual_seed(4)
    ttml.init.set_rng_state(state)
    assert ttml.init.get_rng_state() == state
    assert (ttml.init._rng.integers(0, 2**32, dtype=np.uint32), ttml.init._rng.random(4).tolist()) == expected


def test_a_refused_state_leaves_the_generator_in_place(expect_error):
    ttml.init.manual_seed(5)
    before = ttml.init.get_rng_state()
    with expect_error(ValueError, "not a numpy bit generator's state: bit_generator='seed'"):
        ttml.init.set_rng_state(dict(before, bit_generator="seed"))  # np.random.seed is a function
    with expect_error(OverflowError, "too large"):
        ttml.init.set_rng_state(dict(before, state={"state": 2**200, "inc": 1}))
    assert ttml.init.get_rng_state() == before
