# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device integration tests use the immutable target config and measured weight stats.

Run on a host with chip 0 assigned to this test. For the complete evidence battery
use battery.py and long_context.py; the ordinary pytest path needs no HF weights.
"""

from types import SimpleNamespace

import pytest

from models.autoports.qwen_qwen3_6_35b_a3b.tests.battery import run


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
def test_synthetic_target_shapes(layer_idx, tmp_path):
    run(
        SimpleNamespace(
            layer=layer_idx, synthetic=True, mode="boundaries", batch=2, output=tmp_path / f"{layer_idx}.json"
        )
    )
