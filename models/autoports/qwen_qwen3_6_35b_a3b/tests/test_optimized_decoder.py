# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint boundary tests explicitly exercising the optimized decoder."""

import json
from types import SimpleNamespace

import pytest

from models.autoports.qwen_qwen3_6_35b_a3b.tests import battery
from models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_optimized_decoder import assert_optimized_path
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


@pytest.mark.parametrize("layer_idx", [0, 3, 34], ids=["linear_early", "full_attention", "linear_late"])
def test_optimized_target_shapes(layer_idx, tmp_path, monkeypatch):
    source = assert_optimized_path()
    assert OptimizedDecoder._moe is not FusedDecoder._moe, "Optimized MoE must not be a fused fallback"
    monkeypatch.setattr(battery, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setenv("QWEN_DECODER", "optimized")
    assert battery.FunctionalDecoder is OptimizedDecoder
    output = tmp_path / f"optimized_{layer_idx}.json"
    battery.run(SimpleNamespace(layer=layer_idx, synthetic=False, mode="boundaries", batch=2, output=output))
    data = json.loads(output.read_text())
    assert data["synthetic"] is False
    data.update(decoder="OptimizedDecoder", measured_decoder_source=str(source))
    output.write_text(json.dumps(data, indent=2) + "\n")
