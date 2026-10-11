# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The FP64 KDA golden against Flash Linear Attention's independently written naive recurrence.

Both run in FP64, so any difference above ~1e-12 is a semantic disagreement (gate sign, beta
placement, q scale, state layout), not rounding. Key and value widths differ so a transposed
``[K, V]`` state cannot pass.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference, l2_norm_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.fla_naive_kda import naive_chunk_kda, naive_recurrent_kda
from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import slice_nrmse

FP64_AGREEMENT = 1e-12
BATCH, HEADS, KEY_DIM, VALUE_DIM = 2, 3, 16, 24


def make_inputs(sequence: int, gate_regime: str, seed: int) -> dict[str, torch.Tensor]:
    """Raw (un-normalized) q/k plus K3-style bounded gates in one of three decay regimes."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    gate_logits = randn(BATCH, sequence, HEADS, KEY_DIM)
    gate = {
        # K3's bounded form, -5·sigmoid(x), spread over its whole range.
        "bounded": -5.0 * torch.sigmoid(2.0 * gate_logits),
        # Long memory: decays so close to 1 that FP32 cannot distinguish many of them from 1.
        "long_memory": -1e-4 * torch.sigmoid(gate_logits),
        # Saturated at the K3 bound: |G_last| reaches 160 per 32 tokens.
        "saturated": torch.full_like(gate_logits, -5.0),
    }[gate_regime]
    return {
        "q": randn(BATCH, sequence, HEADS, KEY_DIM),
        "k": randn(BATCH, sequence, HEADS, KEY_DIM),
        "v": randn(BATCH, sequence, HEADS, VALUE_DIM),
        "gate": gate,
        "beta": torch.sigmoid(randn(BATCH, sequence, HEADS)),
        "initial_state": 0.1 * randn(BATCH, HEADS, KEY_DIM, VALUE_DIM),
    }


@pytest.mark.parametrize("gate_regime", ["bounded", "long_memory", "saturated"])
@pytest.mark.parametrize("sequence", [1, 17, 65])
def test_golden_matches_fla_naive_recurrence_in_float64(sequence: int, gate_regime: str) -> None:
    inputs = make_inputs(sequence, gate_regime, seed=sequence)
    golden_output, golden_state = kda_recurrent_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["gate"],
        inputs["beta"],
        inputs["initial_state"],
        dtype=torch.float64,
    )
    fla_output, fla_state = naive_recurrent_kda(
        l2_norm_reference(inputs["q"], dtype=torch.float64),
        l2_norm_reference(inputs["k"], dtype=torch.float64),
        inputs["v"],
        inputs["gate"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        compute_dtype=torch.float64,
    )

    output_error = slice_nrmse(fla_output, golden_output, keep_dims=(2,))  # per head
    state_error = slice_nrmse(fla_state, golden_state, keep_dims=(0, 1))  # per batch and head
    assert output_error.max() <= FP64_AGREEMENT, output_error
    assert state_error.max() <= FP64_AGREEMENT, state_error


@pytest.mark.parametrize("gate_regime", ["bounded", "long_memory", "saturated"])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_token_recurrence_matches_chunked_form_in_float64(chunk_size: int, gate_regime: str) -> None:
    """The token-ordered golden equals FLA's chunked (WY) form, an independent algorithm, at every chunk size.

    Two chunks exercise the carry from one chunk into the next; FLA's naive chunked form needs whole chunks.
    """
    inputs = make_inputs(2 * chunk_size, gate_regime, seed=chunk_size)
    golden_output, golden_state = kda_recurrent_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["gate"],
        inputs["beta"],
        inputs["initial_state"],
        dtype=torch.float64,
    )
    chunked_output, chunked_state = naive_chunk_kda(
        l2_norm_reference(inputs["q"], dtype=torch.float64),
        l2_norm_reference(inputs["k"], dtype=torch.float64),
        inputs["v"],
        inputs["gate"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        chunk_size=chunk_size,
        compute_dtype=torch.float64,
    )

    output_error = slice_nrmse(chunked_output, golden_output, keep_dims=(2,))
    state_error = slice_nrmse(chunked_state, golden_state, keep_dims=(0, 1))
    assert output_error.max() <= FP64_AGREEMENT, output_error
    assert state_error.max() <= FP64_AGREEMENT, state_error
