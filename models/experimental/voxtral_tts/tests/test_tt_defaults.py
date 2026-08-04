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


def test_block1_weights_are_all_bfp8():
    """ALL of Block 1's matmul weights are BFP8 now, including w2. Decode is bandwidth-limited on
    weight bytes, so this is the single biggest lever and every matrix that can be halved has been.

    w2 was blocked for months by a card-wedging hang. That hang was in ttnn's conv `halo_gather`
    kernel, reached via the CODEC's output projection -- not Block 1 at all. It is fixed by
    computing that projection as matmuls (STATUS.md 6.12-6.13), so if someone reverts the codec
    change, this must revert too or the hang comes back."""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat8_b       # w2
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b    # FF1, FF3
    assert gpt.ATTN_WEIGHT_DTYPE == ttnn.bfloat8_b  # wqkv, wo


def test_codec_output_projection_does_not_use_conv1d():
    """The pair to the test above. The codec's output projection must NOT call ttnn.conv1d: its
    halo_gather kernel issues an out-of-range NOC write on the second execution of that shape and
    hangs the card, which is what blocked w2 in BFP8. It is computed as matmuls instead, which needs
    the per-tap weights below. STATUS.md 6.12-6.13."""
    import inspect

    from models.experimental.voxtral_tts.tt import ttnn_voxtral_codec as codec

    src = inspect.getsource(codec.TtVoxtralCodecDecoder._graph)
    init = inspect.getsource(codec.TtVoxtralCodecDecoder.__init__)
    assert '_conv1d(x, "out"' not in src, "the output projection is back on ttnn.conv1d -- see 6.13"
    assert "_out_taps" in init
    # And its prefix must come from ttnn.gather, not _pad_causal's six single-row slices: one such
    # slice of the 16 MiB input costs 0.381 ms, so the six of them were 2.28 of the op's 6.26 ms.
    # Bit-identical either way, so only the clock catches a revert. STATUS.md 6.14.
    assert "self._pad_causal(" not in src, "the projection is back on the slice-built pad -- see 6.14"
    assert "ttnn.gather(" in src and "_out_prefix_idx" in init


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


def test_block2_semantic_head_stays_fp32():
    """It produces an INDEX, not a value. Measured over 64 hidden states, bf16 weights pick a
    DIFFERENT index on 4 of them; fp32 matches the host answer on all 64, for 0.2 ms."""
    assert flow.SEMANTIC_DTYPE == ttnn.float32


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
