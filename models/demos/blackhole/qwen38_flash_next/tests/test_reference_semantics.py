# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.reference import (
    build_ngram_hash_spec,
    causal_depthwise_conv1d,
    gated_delta_recurrent,
    gated_residual_read,
    gated_residual_write,
    mtp_input_fusion,
    ngram_token_ids,
    qsa_shared_draft_indices,
    router_topk,
    speculative_commit_state,
    zero_centered_rms_norm,
)


def test_zero_centered_grouped_rms_norm_is_per_branch():
    x = torch.tensor([[3.0, 4.0, 0.0, 2.0, 0.0, 0.0]])
    weight = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.1, 0.2])
    got = zero_centered_rms_norm(x, weight, eps=1e-6, group_size=3)

    grouped = x.reshape(1, 2, 3)
    expected = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected.flatten(-2) * (1.0 + weight)
    torch.testing.assert_close(got, expected)

    whole_stream = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1.0 + weight)
    assert not torch.allclose(got, whole_stream)


def test_gated_residual_read_and_write_match_official_equations():
    torch.manual_seed(11)
    batch, branches, hidden, rank = 2, 4, 3, 2
    residual = torch.randn(batch, branches * hidden)
    norm_weight = torch.randn(branches * hidden)
    down = torch.randn(rank, branches * hidden)
    up = torch.randn(branches * hidden, rank)
    inject = torch.randn(branches, branches * hidden)
    block = torch.randn(batch, hidden)

    got_read, got_normed = gated_residual_read(
        residual,
        norm_weight=norm_weight,
        down_weight=down,
        up_weight=up,
        hc_count=branches,
        hidden_size=hidden,
        eps=1e-6,
    )
    expected_normed = zero_centered_rms_norm(residual, norm_weight, 1e-6, group_size=hidden)
    expected_gate = torch.sigmoid(F.linear(F.silu(F.linear(expected_normed, down) / branches), up))
    expected_read = (
        expected_gate.reshape(batch, branches, hidden) * expected_normed.reshape(batch, branches, hidden)
    ).mean(dim=-2)
    torch.testing.assert_close(got_normed, expected_normed)
    torch.testing.assert_close(got_read, expected_read)

    got_write = gated_residual_write(
        block,
        residual=residual,
        normalized_residual=got_normed,
        inject_weight=inject,
        hc_count=branches,
        hidden_size=hidden,
    )
    coeff = 2 * torch.sigmoid(F.linear(expected_normed, inject) / branches)
    expected_write = (residual.reshape(batch, branches, hidden) + coeff.unsqueeze(-1) * block.unsqueeze(-2)).flatten(-2)
    torch.testing.assert_close(got_write, expected_write)


def test_mtp_input_fusion_globally_normalizes_then_projects_each_gr_branch_with_shared_weight():
    torch.manual_seed(23)
    batch, branches, hidden = 3, 4, 5
    embedding = torch.randn(batch, hidden)
    hidden_state = torch.randn(batch, branches * hidden)
    embedding_norm = torch.randn(hidden)
    hidden_norm = torch.randn(branches * hidden)
    fc_embedding = torch.randn(hidden, hidden)
    fc_hidden = torch.randn(hidden, hidden)

    got = mtp_input_fusion(
        embedding,
        hidden_state,
        embedding_norm_weight=embedding_norm,
        hidden_norm_weight=hidden_norm,
        fc_embedding_weight=fc_embedding,
        fc_hidden_weight=fc_hidden,
        hc_count=branches,
        hidden_size=hidden,
        eps=1e-6,
    )

    e = F.linear(zero_centered_rms_norm(embedding, embedding_norm, 1e-6), fc_embedding)
    h = zero_centered_rms_norm(hidden_state, hidden_norm, 1e-6)
    h = F.linear(h.reshape(batch, branches, hidden), fc_hidden)
    expected = (h + e.unsqueeze(-2)).flatten(-2)
    torch.testing.assert_close(got, expected)
    assert got.shape == hidden_state.shape

    grouped = zero_centered_rms_norm(hidden_state, hidden_norm, 1e-6, group_size=hidden)
    grouped = F.linear(grouped.reshape(batch, branches, hidden), fc_hidden)
    grouped = (grouped + e.unsqueeze(-2)).flatten(-2)
    assert not torch.allclose(got, grouped)


def test_ngram_hash_is_split_invariant_and_resets_history_at_eos():
    spec = build_ngram_hash_spec(
        unigram_vocab_size=97,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=101,
        ple_layer_index=0,
        seed=1234,
        divisible_by=8,
    )
    eos = 96
    tokens = torch.tensor([[1, 2, 3, eos, 4, 5, 6], [9, eos, 10, 11, 12, 13, 14]])

    all_ids, all_state = ngram_token_ids(tokens, previous_context=None, eos_token_id=eos, spec=spec)
    first_ids, first_state = ngram_token_ids(tokens[:, :4], previous_context=None, eos_token_id=eos, spec=spec)
    second_ids, second_state = ngram_token_ids(tokens[:, 4:], previous_context=first_state, eos_token_id=eos, spec=spec)

    torch.testing.assert_close(torch.cat([first_ids, second_ids], dim=1), all_ids)
    torch.testing.assert_close(second_state, all_state)
    assert all_ids.shape == (2, 7, 4)
    assert spec.padded_vocab_size % 8 == 0

    after_eos, _ = ngram_token_ids(torch.tensor([[4]]), torch.tensor([[3, eos]]), eos, spec)
    fresh, _ = ngram_token_ids(torch.tensor([[4]]), None, eos, spec)
    torch.testing.assert_close(after_eos, fresh)


def test_gated_delta_recurrent_matches_tokenwise_decode_and_keeps_fp32_state():
    torch.manual_seed(37)
    batch, seq, heads, key_dim, value_dim = 2, 5, 3, 4, 6
    q = torch.randn(batch, seq, heads, key_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq, heads, value_dim, dtype=torch.bfloat16)
    log_decay = -torch.rand(batch, seq, heads)
    beta = torch.sigmoid(torch.randn(batch, seq, heads))
    initial = torch.randn(batch, heads, key_dim, value_dim, dtype=torch.float32)

    full_out, full_state = gated_delta_recurrent(q, k, v, log_decay, beta, initial)
    state = initial.clone()
    pieces = []
    for index in range(seq):
        out, state = gated_delta_recurrent(
            q[:, index : index + 1],
            k[:, index : index + 1],
            v[:, index : index + 1],
            log_decay[:, index : index + 1],
            beta[:, index : index + 1],
            state,
        )
        pieces.append(out)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_out)
    torch.testing.assert_close(state, full_state)
    assert full_state.dtype == torch.float32


def test_causal_depthwise_conv_sequence_matches_incremental_updates():
    torch.manual_seed(41)
    batch, channels, seq, kernel = 2, 3, 7, 4
    x = torch.randn(batch, channels, seq)
    weight = torch.randn(channels, kernel)
    initial = torch.randn(batch, channels, kernel)

    full, full_state = causal_depthwise_conv1d(x, weight, initial_state=initial, activation="silu")
    state = initial.clone()
    pieces = []
    for index in range(seq):
        out, state = causal_depthwise_conv1d(x[..., index : index + 1], weight, initial_state=state, activation="silu")
        pieces.append(out)

    torch.testing.assert_close(torch.cat(pieces, dim=-1), full)
    torch.testing.assert_close(state, full_state)


def test_router_scores_are_fp32_softmax_topk_then_renormalized():
    logits = torch.tensor([[12.0, 11.0, -8.0, 10.0]], dtype=torch.bfloat16)
    scores, indices = router_topk(logits, top_k=3)
    probs = torch.softmax(logits.float(), dim=-1)
    expected_scores, expected_indices = torch.topk(probs, 3, dim=-1)
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(scores, expected_scores.to(logits.dtype))


@pytest.mark.parametrize("accepted_drafts", range(5))
def test_speculative_commit_selects_current_plus_accepted_draft_state(accepted_drafts):
    # Five verifier positions: the current target token followed by four drafts.
    per_step = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    accepted = torch.tensor([accepted_drafts, 4 - accepted_drafts])
    got = speculative_commit_state(per_step, accepted)
    expected = torch.stack([per_step[0, accepted_drafts], per_step[1, 4 - accepted_drafts]])
    torch.testing.assert_close(got, expected)


def test_qsa_shared_draft_indices_append_only_the_causal_tail():
    captured = torch.tensor([[8, 9, 0, 1], [20, 21, 12, 13]], dtype=torch.int32)
    captured_len = torch.tensor([30, 40], dtype=torch.int32)
    positions = torch.tensor([32, 41], dtype=torch.int32)
    got = qsa_shared_draft_indices(captured, captured_len, positions, tail_width=4)
    expected = torch.tensor(
        [[8, 9, 0, 1, 30, 31, 32, -1], [20, 21, 12, 13, 40, 41, -1, -1]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(got, expected)


def test_ngram_multiplier_values_are_stable():
    spec = build_ngram_hash_spec(248320, 3, 8, 20_000_000, 0, 1234, 128)
    # Pin the exact SplitMix64-derived values so an apparently harmless hash rewrite fails loudly.
    assert spec.layer_multipliers.tolist() == [23703573157769, 20109073645365, 8052911324071]
    assert len(spec.head_vocab_sizes) == 16
    assert spec.head_vocab_sizes[0] > 20_000_000
    assert math.prod([spec.ngram_size, spec.heads_per_ngram]) > 0


def test_ngram_nonzero_ple_layer_uses_global_prime_head_numbers():
    first = build_ngram_hash_spec(97, 3, 2, 101, 0, 1234, 8)
    second = build_ngram_hash_spec(97, 3, 2, 101, 1, 1234, 8)
    assert len(second.head_vocab_sizes) == 4
    assert second.head_vocab_sizes[0] > first.head_vocab_sizes[-1]
    assert second.head_offsets[0] == 0
