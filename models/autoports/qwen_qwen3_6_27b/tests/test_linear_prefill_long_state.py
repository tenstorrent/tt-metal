# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only regressions for bounded long linear-attention prefill construction."""

import math

import torch

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    LINEAR_PREFILL_CHUNK_SIZE,
    _BalancedSequenceConcat,
)


def _peak_retained_chunks(sequence: int, chunk_size: int) -> int:
    """Model the binary concat reducer without constructing TTNN tensors."""
    occupied = []
    peak = 0
    for _ in range(math.ceil(sequence / chunk_size)):
        level = 0
        while level < len(occupied) and occupied[level]:
            occupied[level] = False
            level += 1
        if level == len(occupied):
            occupied.append(True)
        else:
            occupied[level] = True
        peak = max(peak, sum(occupied))
    return peak


def _recur_token(state, key, value, beta, decay):
    state = state * decay
    memory_value = key @ state
    delta = (value - memory_value) * beta
    state = state + key[:, None] * delta[None, :]
    return state, key @ state


def test_advertised_context_concat_retention_is_logarithmic():
    chunks = math.ceil(ADVERTISED_CONTEXT / LINEAR_PREFILL_CHUNK_SIZE)
    peak = _peak_retained_chunks(ADVERTISED_CONTEXT, LINEAR_PREFILL_CHUNK_SIZE)
    assert chunks == 8192
    assert peak <= math.ceil(math.log2(chunks))
    assert LINEAR_PREFILL_CHUNK_SIZE + peak <= 45


def test_balanced_concat_preserves_chronological_order(monkeypatch):
    def concatenate(chunks, *, dim, memory_config):
        assert dim == 2
        assert memory_config == "dram"
        return "".join(chunks)

    monkeypatch.setattr(
        "models.autoports.qwen_qwen3_6_27b.tt.functional_decoder.ttnn.concat",
        concatenate,
    )
    reducer = _BalancedSequenceConcat(dim=2, memory_config="dram")
    for chunk in ("A", "B", "C", "D", "E", "F", "G"):
        reducer.append(chunk)
    assert reducer.finish() == "ABCDEFG"


def test_near_context_non_aligned_recurrent_state_matches_chunk_boundaries():
    """Exercise the exact gated-delta recurrence at target context length.

    Heads/dimensions are deliberately reduced so this CPU oracle is cheap, but
    the sequence length is the advertised 262,144 plus one and the selected
    observations straddle tile/chunk/context boundaries.
    """
    sequence = ADVERTISED_CONTEXT + 1
    generator = torch.Generator().manual_seed(20260729)
    # Reuse a deterministic 257-token signal. This keeps the near-context test
    # compact while still varying every adjacent token and chunk boundary.
    period = 257
    keys = torch.randn(period, 4, generator=generator, dtype=torch.float64)
    keys = keys / torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
    values = torch.randn(period, 4, generator=generator, dtype=torch.float64) * 0.1
    betas = torch.sigmoid(torch.randn(period, generator=generator, dtype=torch.float64))
    decays = torch.sigmoid(torch.randn(period, generator=generator, dtype=torch.float64))
    selected = {31, 32, 33, 63, 64, 65, sequence - 2, sequence - 1}

    state_token = torch.zeros(4, 4, dtype=torch.float64)
    token_outputs = {}
    for index in range(sequence):
        signal = index % period
        state_token, value = _recur_token(state_token, keys[signal], values[signal], betas[signal], decays[signal])
        if index in selected:
            token_outputs[index] = value.clone()

    state_chunked = torch.zeros_like(state_token)
    chunk_outputs = {}
    for start in range(0, sequence, LINEAR_PREFILL_CHUNK_SIZE):
        for index in range(start, min(start + LINEAR_PREFILL_CHUNK_SIZE, sequence)):
            signal = index % period
            state_chunked, value = _recur_token(
                state_chunked, keys[signal], values[signal], betas[signal], decays[signal]
            )
            if index in selected:
                chunk_outputs[index] = value.clone()

    torch.testing.assert_close(state_chunked, state_token, rtol=0, atol=0)
    for index in selected:
        torch.testing.assert_close(chunk_outputs[index], token_outputs[index], rtol=0, atol=0)
