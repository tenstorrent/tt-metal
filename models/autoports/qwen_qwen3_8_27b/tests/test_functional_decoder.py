# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape synthetic CI coverage; real-checkpoint runner evidence is separate."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.autoports.qwen_qwen3_8_27b.tests.run_decoder import run

EVIDENCE = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"


@pytest.mark.parametrize("layer", [0, 3])
def test_functional_decoder_boundaries(layer, tmp_path):
    run(
        SimpleNamespace(
            snapshot=EVIDENCE / "hf_config.json",
            synthetic_stats=EVIDENCE / "weight_stats.json",
            layer=layer,
            batch=2,
            length=32,
            lengths="1,31,32,33,127,128,129,257,31",
            profile=False,
            continuation=True,
            output=tmp_path / f"layer_{layer}.json",
        )
    )
