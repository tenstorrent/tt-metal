# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Makes this directory's helper (``dcp_oracle``) importable whether pytest collects this directory alone or the whole
``tt-train/tests/python`` tree, and puts the process-wide generators back after every test here, since these tests
reseed and restore them."""

import os
import random
import sys

import numpy as np
import pytest

import ttml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _restore_process_generators():
    """The ttml, ttml.init, python, numpy and torch generators as they were before the test, so tests collected after
    this directory start from the state they would have had without it. Snapshotted through each generator's own API,
    not through the checkpoint code these tests check."""
    ctx = ttml.autograd.AutoContext.get_instance()
    torch = sys.modules.get("torch")
    init_rng = ttml.init._rng
    saved = (
        ctx.get_generator_state(),
        random.getstate(),
        np.random.get_state(),
        init_rng.bit_generator.state,
        torch.get_rng_state() if torch is not None else None,
    )
    yield
    ctx.set_generator_state(saved[0])
    random.setstate(saved[1])
    np.random.set_state(saved[2])
    init_rng.bit_generator.state = saved[3]
    ttml.init._rng = init_rng  # a restore or manual_seed replaces the object
    if torch is not None:
        torch.set_rng_state(saved[4])
