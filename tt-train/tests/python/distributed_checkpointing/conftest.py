# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Makes this directory's helper (``dcp_oracle``) and the parent directory's (``tp_block``) importable whether pytest
collects this directory alone or the whole ``tt-train/tests/python`` tree, and puts the process-wide generators back
after every test here, since these tests reseed and restore them."""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.dirname(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(autouse=True)
def _restore_process_generators():
    """The ttml, ttml.init, python, numpy and torch generators as they were before the test, so tests collected after
    this directory start from the state they would have had without it."""
    from ttml.distributed_checkpointing._train_state import capture_process_rng, restore_process_rng

    state = capture_process_rng()
    yield
    restore_process_rng(state)
