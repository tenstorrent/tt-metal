# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for multi-head attention, including the cached decode path."""

from dataclasses import replace

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import compute_metrics
from models.demos.time_series_transformer.tt.attention import KeyValueCache, MultiHeadAttention
from models.demos.time_series_transformer.tt.config import get_ttnn_dtype
from models.demos.time_series_transformer.tt.ops import make_causal_mask, to_device, to_torch
from models.demos.time_series_transformer.tt.weights import substate

PCC_THRESHOLD = 0.999


def torch_attention(
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Reference implementation mirroring TimeSeriesTransformerAttention."""
    batch, seq, d_model = hidden_states.shape
    head_dim = d_model // num_heads
    scaling = head_dim**-0.5

    def project(x, name):
        return torch.nn.functional.linear(x, state[f"{name}.weight"], state[f"{name}.bias"])

    def split(x):
        return x.view(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)

    query = split(project(hidden_states, "q_proj"))
    key = split(project(key_value_states, "k_proj"))
    value = split(project(key_value_states, "v_proj"))

    scores = (query @ key.transpose(-1, -2)) * scaling
    if causal:
        scores = scores + torch.full((seq, seq), float("-inf")).triu(1)
    context = torch.softmax(scores, dim=-1) @ value
    context = context.transpose(1, 2).reshape(batch, seq, d_model)
    return torch.nn.functional.linear(context, state["out_proj.weight"], state["out_proj.bias"])


@pytest.fixture(scope="module")
def encoder_attn_state(hf_state):
    return {k: v.float() for k, v in substate(hf_state, "model.encoder.layers.0.self_attn").items()}


class TestMultiHeadAttention:
    def test_self_attention_pcc(self, device, config, encoder_attn_state):
        dtype = get_ttnn_dtype(config.dtype)
        attention = MultiHeadAttention(config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        hidden = torch.randn(2, config.context_length, config.d_model)
        expected = torch_attention(hidden, hidden, encoder_attn_state, num_heads=config.encoder_attention_heads)
        actual = to_torch(attention(to_device(hidden, device=device, dtype=dtype)))

        mse, mae, pcc = compute_metrics(expected, actual)
        assert pcc > PCC_THRESHOLD, f"self-attention PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_cross_attention_pcc(self, device, config, encoder_attn_state):
        dtype = get_ttnn_dtype(config.dtype)
        attention = MultiHeadAttention(config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        query = torch.randn(2, config.prediction_length, config.d_model)
        memory = torch.randn(2, config.context_length, config.d_model)
        expected = torch_attention(query, memory, encoder_attn_state, num_heads=config.encoder_attention_heads)
        actual = to_torch(
            attention(
                to_device(query, device=device, dtype=dtype),
                to_device(memory, device=device, dtype=dtype),
            )
        )

        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > PCC_THRESHOLD, f"cross-attention PCC {pcc:.6f}"

    def test_causal_mask_pcc(self, device, config, encoder_attn_state):
        dtype = get_ttnn_dtype(config.dtype)
        attention = MultiHeadAttention(config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        hidden = torch.randn(2, config.prediction_length, config.d_model)
        expected = torch_attention(
            hidden, hidden, encoder_attn_state, num_heads=config.encoder_attention_heads, causal=True
        )
        mask = make_causal_mask(config.prediction_length, device=device, dtype=dtype, mask_value=config.attn_mask_value)
        actual = to_torch(attention(to_device(hidden, device=device, dtype=dtype), attention_mask=mask))

        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > PCC_THRESHOLD, f"causal attention PCC {pcc:.6f}"

    @pytest.mark.parametrize(
        "exact_softmax, row_sum_tolerance",
        [
            # The composed reduction normalizes rows to ~1e-3.
            (True, 2e-3),
            # ttnn.softmax leaves rows a few percent off. That is tolerated because the error
            # is close to a uniform per-row scale factor, which the layer norm after the
            # residual removes -- see test_e2e_model for the end-to-end evidence.
            (False, 5e-2),
        ],
        ids=["exact", "fused_kernel"],
    )
    def test_attention_probs_are_a_distribution(
        self, device, config, encoder_attn_state, exact_softmax, row_sum_tolerance
    ):
        """Softmax rows must sum to ~1 -- catches masking applied on the wrong axis."""
        local_config = replace(config, use_exact_softmax=exact_softmax)
        dtype = get_ttnn_dtype(local_config.dtype)
        attention = MultiHeadAttention(local_config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        hidden = torch.randn(2, local_config.context_length, local_config.d_model)
        attention(to_device(hidden, device=device, dtype=dtype))
        probs = to_torch(attention.last_attention_probs)

        assert probs.shape == (
            2,
            local_config.encoder_attention_heads,
            local_config.context_length,
            local_config.context_length,
        )
        row_sums = probs.sum(-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=row_sum_tolerance, rtol=row_sum_tolerance)


class TestKeyValueCache:
    """Stepping one token at a time through the cache must equal one full-sequence pass."""

    def test_cached_decode_matches_full_pass(self, device, config, encoder_attn_state):
        dtype = get_ttnn_dtype(config.dtype)
        attention = MultiHeadAttention(config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        steps = 6
        hidden = torch.randn(2, steps, config.d_model)
        expected = torch_attention(
            hidden, hidden, encoder_attn_state, num_heads=config.encoder_attention_heads, causal=True
        )

        cache = KeyValueCache()
        outputs = []
        for step in range(steps):
            token = to_device(hidden[:, step : step + 1, :], device=device, dtype=dtype)
            outputs.append(to_torch(attention(token, cache=cache)))

        actual = torch.cat(outputs, dim=1)
        assert cache.length == steps
        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > PCC_THRESHOLD, f"cached decode PCC {pcc:.6f}"

    def test_cross_attention_cache_is_reused(self, device, config, encoder_attn_state):
        dtype = get_ttnn_dtype(config.dtype)
        attention = MultiHeadAttention(config, device=device, dtype=dtype)
        attention.load_hf_state_dict(encoder_attn_state, strict=True)

        memory = torch.randn(2, config.context_length, config.d_model)
        tt_memory = to_device(memory, device=device, dtype=dtype)
        cache = KeyValueCache()

        first = to_torch(
            attention(to_device(torch.randn(2, 1, config.d_model), device=device, dtype=dtype), tt_memory, cache=cache)
        )
        assert cache.is_filled
        assert cache.length == config.context_length
        cached_key = cache.key

        query = torch.randn(2, 1, config.d_model)
        second = to_torch(attention(to_device(query, device=device, dtype=dtype), tt_memory, cache=cache))
        # The stored keys must be the exact same tensor -- not recomputed.
        assert cache.key is cached_key

        uncached = to_torch(attention(to_device(query, device=device, dtype=dtype), tt_memory))
        _, _, pcc = compute_metrics(uncached, second)
        assert pcc > 0.9999, f"cross-attention cache diverged: PCC {pcc:.6f}"
        assert first.shape == second.shape
