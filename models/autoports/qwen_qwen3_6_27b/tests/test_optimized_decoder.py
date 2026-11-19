# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI smoke exercises the delivered optimized default with real checkpoint weights.

Run serialized with no other device owner. The standalone evidence runners add
full context, resume controls, real weights, allocation tracking and watcher.
"""

import sys

import pytest

from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import main


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_optimized_decoder(layer, batch, monkeypatch, tmp_path):
    from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

    monkeypatch.setenv("QWEN_DECODER", "optimized")
    monkeypatch.delenv("QWEN_FUSION_CANDIDATE", raising=False)
    monkeypatch.delenv("QWEN_OPT_CANDIDATE", raising=False)
    monkeypatch.delenv("QWEN_OPT_POLICY", raising=False)
    assert decoder_class() is OptimizedDecoder
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decoder",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--length",
            "33",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    main()
