# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-path regression coverage at actual target geometry."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_27b.tests.run_fused_decoder import run
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder

EVIDENCE = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"


@pytest.mark.parametrize("layer,batch", [(0, 2), (3, 2), (0, 3)])
def test_fused_decoder_boundaries(layer, batch, tmp_path):
    # A regression in runner dispatch must fail instead of exercising baseline.
    with patch.object(FunctionalDecoder, "from_state_dict", side_effect=AssertionError("Functional fallback")):
        run(
            SimpleNamespace(
                snapshot=EVIDENCE / "hf_config.json",
                synthetic_stats=EVIDENCE / "weight_stats.json",
                layer=layer,
                batch=batch,
                length=32,
                lengths="1,2,3,31,32,33,127,128,129,257,31",
                profile=False,
                continuation=True,
                baseline=False,
                benchmark=False,
                compare_dir=None,
                output=tmp_path / f"fused_layer_{layer}.json",
            )
        )
