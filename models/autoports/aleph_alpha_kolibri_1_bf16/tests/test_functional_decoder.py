# SPDX-License-Identifier: Apache-2.0
"""CI uses deterministic weights sampled from the checked-in real tensor stats.

Run from the source-built tt-metal Python environment with
TT_METAL_TRACE_ALLOC_TRACKING=1. Real checkpoint, batch and full-context runners
are separate CLI commands documented in doc/functional_decoder/README.md.
"""

import os

import pytest

from .run_coverage import run


@pytest.mark.parametrize("layer", [0, 4], ids=["sliding_attention", "full_attention"])
def test_functional_decoder(layer):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    run(layer, synthetic=True, output=f"synthetic_{layer}.json")
