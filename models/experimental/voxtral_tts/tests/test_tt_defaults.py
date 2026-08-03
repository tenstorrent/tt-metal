# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shipped TTNN configuration, pinned.

There are no runtime toggles left -- every alternative that was measured and rejected has been
deleted rather than parked behind a flag, so what the code does is what it does. What survives is
a handful of CONSTANTS, and each one encodes a measurement that is expensive to rediscover. This
test is what makes an accidental edit to one of them fail loudly instead of quietly changing the
model. The reason for each lives in the module that owns it; the one-liners here are pointers.

Needs no device and no checkpoint -- it only imports the modules.

    pytest -svv models/experimental/voxtral_tts/tests/test_tt_defaults.py
"""

import pytest

ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402


def test_block1_weight_precision_is_mixed():
    """BFP8 on FF1_FF3 ONLY. All-BFP8 is ~5 ms/frame faster and triggers a card-wedging hang in
    multi-utterance runs; putting FF2 in BFP8 too is enough to bring it back."""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat16
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b


def test_block1_math_config_keeps_fp32_accumulation():
    """RMSNorm's mean-of-squares needs it. Dropping the compute config makes that op 2.4x faster
    and takes model decode PCC from 0.99991 to 0.992, worst sample 1.7% -> 18.9%."""
    assert gpt.COMPUTE_CONFIG.fp32_dest_acc_en is True
    assert gpt.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4


def test_block2_weights_are_bfp8_but_fidelity_stays_high():
    """BFP8 weights are 1.23x for one extra differing code in 222. Lowering the MATH fidelity is
    the opposite trade -- ~4 ms for 10-20x the code errors."""
    assert flow.WEIGHT_DTYPE == ttnn.bfloat8_b
    assert flow.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4
    assert flow.COMPUTE_CONFIG.fp32_dest_acc_en is True


def test_prefill_padding_stays_on_the_tile_grid():
    """Prefill's causal mask is cut at this boundary; a ragged value would misalign it silently
    rather than raise. 128 itself is a kernel-shape-churn choice, not a hardware limit."""
    assert gpt.PREFILL_MULTIPLE % gpt.TILE == 0


def test_fused_qkv_width_matches_the_head_config():
    """Decode's head op reads ONE fused projection and splits it by head count; a mismatch would
    mis-slice q/k/v rather than raise."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        HEAD_DIM,
        N_HEADS,
        N_KV_HEADS,
    )

    assert gpt._QKV_WIDTH == (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
