# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shipped TTNN configuration, pinned.

Every value here is a decision with a measurement behind it, recorded in the module that owns it.
The modules are heavily commented and those comments quote numbers; this test is what stops the
comments and the code drifting apart, and what makes an accidental default flip fail loudly
instead of quietly changing what everyone benchmarks.

Needs no device and no checkpoint -- it only imports the modules.

    pytest -svv models/experimental/voxtral_tts/tests/test_tt_defaults.py
"""

import os

import pytest

# The env knobs each module reads at import. Clear them so the test sees SHIPPED defaults even if
# the shell has leftovers from an experiment -- which is exactly when a drift check matters.
for _v in ("VOXTRAL_BACKBONE", "VOXTRAL_GPT_WEIGHTS", "VOXTRAL_GPT_TRACE", "VOXTRAL_GPT_WINDOW",
           "VOXTRAL_GPT_DECODE_NATIVE", "VOXTRAL_GPT_FASTNORM", "VOXTRAL_SDPA",
           "VOXTRAL_FLOW_TRACE", "VOXTRAL_FLOW_WEIGHTS", "VOXTRAL_FLOW_FIDELITY",
           "VOXTRAL_FLOW_FP32ACC", "VOXTRAL_CLEAR_PCACHE"):
    os.environ.pop(_v, None)

ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipe  # noqa: E402


def test_block1_is_the_default_backbone():
    """Ours, not the tt_transformers wrapper: prefill 0.999881 vs 0.999564, decode 0.99991 vs
    0.981, 34.9 ms/frame vs 48."""
    assert pipe.BACKBONE_IMPL == "gpt"


def test_block1_weight_precision_is_mixed():
    """BFP8 on FF1_FF3 ONLY. All-BFP8 is faster but triggers the card-wedging hang AND a
    repetition loop on fixture case 4; adding FF2 alone is enough to bring the hang back."""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat16
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b
    assert gpt.FF2_WEIGHT_DTYPE is None, "FF2 in BFP8 reintroduces the hang"


def test_block1_decode_native_is_off():
    """OFF, but provisionally. It is 6.6 ms/frame faster at the same decode PCC. It was turned
    off after fixture case 4 collapsed with it on -- then case 4 turned out to be unstable in
    every implementation including the fp32 CPU reference (81/8/57 frames for one word), so it
    cannot discriminate between implementations. Needs a re-gate excluding case 4."""
    assert gpt.DECODE_NATIVE is False


def test_block2_weights_are_bfp8_but_fidelity_stays_high():
    """BFP8 weights are 1.23x for one extra differing code in 222. Lowering the MATH fidelity is
    the opposite trade -- ~4 ms for 10-20x the code errors -- so HiFi4 + fp32 accumulation stay."""
    assert flow.WEIGHT_DTYPE == ttnn.bfloat8_b
    assert flow._FID == ttnn.MathFidelity.HiFi4
    assert flow._FP32ACC is True


def test_experimental_paths_are_off():
    """Each is implemented and correct, and each measured slower or worse. They stay available
    for re-testing on other silicon, but must not be on by accident."""
    assert gpt.USE_TRACE is False, "Block 1 trace: 0.7 ms slower at equal window"
    assert flow.USE_TRACE is False, "dual trace: bit-identical but ~6 ms/frame slower"
    assert gpt.SDPA == set(), "sdpa in prefill costs PCC for nothing measurable"
    assert gpt._FASTNORM is False, "cheap RMSNorm drops model decode PCC to 0.992"


def test_program_cache_clearing_is_off():
    """It was a mitigation for the hang; mixed precision removed the need. Keep the switch -- the
    underlying ttnn failure (a silent hang, not an error) is still unfixed upstream."""
    assert pipe.CLEAR_PROGRAM_CACHE is False


def test_padding_multiples_stay_on_the_tile_grid():
    """prefill's mask and decode's cache slice are cut at these boundaries; a ragged value would
    misalign them silently rather than raise."""
    assert gpt.PREFILL_MULTIPLE % gpt.TILE == 0
    assert gpt.DECODE_WINDOW % gpt.TILE == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
