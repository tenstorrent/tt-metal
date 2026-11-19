# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The fused implementation is selected explicitly, including in ordinary pytest."""
from types import SimpleNamespace

import pytest

from models.autoports.qwen_qwen3_6_35b_a3b.tests import battery
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
def test_fused_target_shapes(layer_idx, tmp_path, monkeypatch):
    monkeypatch.setattr(battery, "FunctionalDecoder", FusedDecoder)
    monkeypatch.setenv("QWEN_DECODER", "fused")
    assert battery.FunctionalDecoder is FusedDecoder
    battery.run(
        SimpleNamespace(
            layer=layer_idx, synthetic=True, mode="boundaries", batch=2, output=tmp_path / f"fused_{layer_idx}.json"
        )
    )
