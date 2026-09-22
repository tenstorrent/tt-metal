# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight optimized-default regressions, including batch and chunk tails.

Record fixtures with record_decoder_activations.py before running this suite.
TT_OPTIMIZED_ACTIVATIONS and QWEN38_SNAPSHOT can override the local fixtures.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import run
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder

SNAPSHOT = Path(
    os.environ.get(
        "QWEN38_SNAPSHOT",
        "/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    )
)
ACTIVATIONS = Path(
    os.environ.get(
        "TT_OPTIMIZED_ACTIVATIONS",
        "/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations_long",
    )
)


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize(
    "batch,lengths",
    [
        (1, "1,2,3,31,32,33,127,128,129,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,4097,31"),
        (2, "1,31,33,129,257,31"),
        (3, "257,31"),
        (8, "31"),
        (16, "31"),
        (32, "257,31"),
    ],
)
def test_optimized_default(layer, batch, lengths, tmp_path):
    assert (ACTIVATIONS / f"layer{layer}.pt").exists(), "Record real HF input activations first"
    with (
        patch.object(FunctionalDecoder, "from_state_dict", side_effect=AssertionError("Functional fallback")),
        patch.object(FusedDecoder, "from_state_dict", side_effect=AssertionError("Fused fallback")),
    ):
        run(
            SimpleNamespace(
                snapshot=SNAPSHOT,
                synthetic_stats=None,
                activations=ACTIVATIONS,
                policy="{}",
                policy_file=None,
                layer=layer,
                batch=batch,
                length=32,
                lengths=lengths,
                profile=False,
                continuation=True,
                baseline=False,
                benchmark=True,
                compare_dir=None,
                output=tmp_path / f"optimized_l{layer}_b{batch}.json",
            )
        )
